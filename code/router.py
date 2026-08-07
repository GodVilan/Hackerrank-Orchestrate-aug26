"""The single structured model call. Produces a reading; decides nothing.

    route(message_row, features, pool, media) -> ModelRead

One Anthropic call per row, using ``schema.ROUTE_MESSAGE_TOOL`` with
``tool_choice`` pinned to that tool, so the response shape is guaranteed rather
than parsed.

**There is no JSON parsing or repair path, and none may be added.** If the
response carries no ``tool_use`` block, the call is retried once and then
:class:`ToolOutputMissing` is raised. ``main.py`` converts that into the
``DIGEST_INSUFFICIENT_SIGNAL`` fallback row with ``unresolved: true``. A model
that failed to answer in the required shape is a failure to record, not a string
to salvage.

Caching, concurrency, and per-row statistics all live here:

* **Cache key** — ``sha256(message_id + rendered feature block + sorted
  candidate ids + model + PROMPT_VERSION)``. Bumping ``PROMPT_VERSION``
  invalidates everything, because a changed prompt is a different question.
* **Concurrency** — ``ThreadPoolExecutor`` with a shared rate limiter and
  exponential backoff on 429 and 5xx.
* **Statistics** — every row records model, token counts, latency, cache hit,
  and retry count into ``run_stats.json``, written from the first run.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import anthropic

import config
import prompts
import schema
from schema import ModelRead


class ToolOutputMissing(RuntimeError):
    """The response carried no tool_use block, after one retry.

    Raised rather than recovered. ``main.py`` turns this into the
    DIGEST_INSUFFICIENT_SIGNAL fallback and marks the row unresolved.
    """


@dataclass
class RowStat:
    """One row's operational record, written to run_stats.json."""

    message_id: str
    model: str = config.PRIMARY_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    cache_hit: bool = False
    retries: int = 0
    error: str | None = None


@dataclass
class _Stats:
    rows: dict[str, RowStat] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, stat: RowStat) -> None:
        with self.lock:
            self.rows[stat.message_id] = stat

    def dump(self, extra: dict | None = None) -> dict:
        with self.lock:
            rows = {k: asdict(v) for k, v in sorted(self.rows.items())}
        totals = {
            "rows": len(rows),
            "model_calls": sum(1 for r in rows.values() if not r["cache_hit"]),
            "cache_hits": sum(1 for r in rows.values() if r["cache_hit"]),
            "input_tokens": sum(r["input_tokens"] for r in rows.values()),
            "output_tokens": sum(r["output_tokens"] for r in rows.values()),
            "cache_read_tokens": sum(r["cache_read_tokens"] for r in rows.values()),
            "cache_write_tokens": sum(r["cache_write_tokens"] for r in rows.values()),
            "total_retries": sum(r["retries"] for r in rows.values()),
            "errors": sum(1 for r in rows.values() if r["error"]),
            "wall_clock_s": round(sum(r["latency_s"] for r in rows.values()), 2),
        }
        payload = {
            "model": config.PRIMARY_MODEL,
            "prompt_version": config.PROMPT_VERSION,
            "concurrency": config.ROUTER_MAX_WORKERS,
            "totals": totals,
            **(extra or {}),
            "rows": rows,
        }
        return payload

    def write(self, extra: dict | None = None) -> None:
        config.RUN_STATS.write_text(
            json.dumps(self.dump(extra), indent=2) + "\n", encoding="utf-8"
        )


STATS = _Stats()


class _RateLimiter:
    """Shared minimum gap between requests across all worker threads."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if sleep_for:
            time.sleep(sleep_for)


_LIMITER = _RateLimiter(config.ROUTER_MIN_INTERVAL_S)
_client_lock = threading.Lock()
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    with _client_lock:
        if _client is None:
            _client = anthropic.Anthropic(
                api_key=config.require_api_key(), max_retries=0
            )
    return _client


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def cache_key(message_row, features, pool) -> str:
    """sha256 over everything that could change the answer."""
    digest = hashlib.sha256()
    digest.update(message_row["message_id"].encode("utf-8"))
    digest.update(prompts._render_facts(features).encode("utf-8"))
    digest.update(";".join(sorted(c.message_id for c in pool)).encode("utf-8"))
    digest.update(config.PRIMARY_MODEL.encode("utf-8"))
    digest.update(str(config.PROMPT_VERSION).encode("utf-8"))
    # Media and transcript are part of the question, not metadata about it.
    digest.update((message_row.get("media_id") or "").encode("utf-8"))
    digest.update((message_row.get("_transcript") or "").encode("utf-8"))
    return digest.hexdigest()


def _cache_path(key: str):
    config.ROUTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.ROUTER_CACHE_DIR / f"{key}.json"


def _cache_read(key: str) -> dict | None:
    path = _cache_path(key)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cache_write(key: str, payload: dict) -> None:
    try:
        _cache_path(key).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # a cache we cannot write is a slow run, not a failed one


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _call_once(message_row, features, pool, media):
    _LIMITER.wait()
    return _get_client().messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        system=prompts.system_blocks(),
        tools=[schema.ROUTE_MESSAGE_TOOL],
        tool_choice={"type": "tool", "name": schema.ROUTE_MESSAGE_TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": prompts.build_user_content(
                    message_row, features, pool, media
                ),
            }
        ],
    )


def _extract_tool_input(response) -> dict | None:
    for block in response.content:
        if block.type == "tool_use" and block.name == schema.ROUTE_MESSAGE_TOOL_NAME:
            return dict(block.input)
    return None


def _schema_problems(payload: dict) -> list[str]:
    """Validate a tool payload against the enums the tool schema declares.

    Necessary because ``tool_choice`` guarantees the tool is *called*, not that
    its arguments respect the declared enums. Without ``strict: true`` — which is
    not documented as supported on this model — an enum is a hint the model
    usually honours rather than a constraint the API enforces. Observed in
    practice: a ``message_type`` of ``"marketplace"``, which is a ``group_type``
    value copied out of the deterministic facts block.

    A violation is treated exactly like a missing tool block: retried, then
    surfaced as ToolOutputMissing so the row degrades visibly rather than
    carrying an out-of-vocabulary value into ``output.csv``.
    """
    props = schema.ROUTE_MESSAGE_TOOL["input_schema"]["properties"]
    problems: list[str] = []
    for name in schema.ROUTE_MESSAGE_TOOL["input_schema"]["required"]:
        if name not in payload:
            problems.append(f"missing {name}")
    for name, value in payload.items():
        spec = props.get(name)
        if spec is None:
            problems.append(f"unexpected property {name}")
            continue
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{name}={value!r} not in {spec['enum']}")
    evidence = payload.get("evidence_message_ids")
    if evidence is not None and not isinstance(evidence, list):
        problems.append("evidence_message_ids is not a list")
    return problems


def route(message_row, features, pool, media=None) -> ModelRead:
    """One structured reading for one row. Raises ToolOutputMissing on failure."""
    message_id = message_row["message_id"]
    key = cache_key(message_row, features, pool)

    cached = _cache_read(key)
    if cached is not None:
        # A cached payload is revalidated, not trusted. An entry written before
        # the enum check existed — or by an older prompt version — can be
        # poisoned, and serving it would reintroduce the exact failure the
        # check exists to catch.
        if not _schema_problems(cached):
            STATS.record(RowStat(message_id=message_id, cache_hit=True))
            return ModelRead.from_tool_input(cached)
        _cache_path(key).unlink(missing_ok=True)

    stat = RowStat(message_id=message_id)
    started = time.monotonic()
    last_error: Exception | None = None

    # One attempt, then MAX_RETRIES more. Transport failures and a missing
    # tool block are both retried; the retry count is recorded either way.
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = _call_once(message_row, features, pool, media)
            usage = response.usage
            stat.input_tokens = getattr(usage, "input_tokens", 0)
            stat.output_tokens = getattr(usage, "output_tokens", 0)
            stat.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            stat.cache_write_tokens = (
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )

            payload = _extract_tool_input(response)
            if payload is None:
                last_error = ToolOutputMissing(
                    f"{message_id}: no tool_use block "
                    f"(stop_reason={response.stop_reason})"
                )
            else:
                problems = _schema_problems(payload)
                if problems:
                    last_error = ToolOutputMissing(
                        f"{message_id}: tool output violates the schema: "
                        f"{'; '.join(problems)}"
                    )
                else:
                    stat.latency_s = round(time.monotonic() - started, 3)
                    STATS.record(stat)
                    _cache_write(key, payload)
                    return ModelRead.from_tool_input(payload)
        except Exception as exc:  # noqa: BLE001 — classified immediately below
            last_error = exc
            if not _is_retryable(exc):
                break

        if attempt < config.MAX_RETRIES:
            stat.retries += 1
            delay = min(
                config.ROUTER_BACKOFF_BASE_S * (2**attempt),
                config.ROUTER_BACKOFF_MAX_S,
            )
            time.sleep(delay + random.uniform(0, delay * 0.1))

    stat.latency_s = round(time.monotonic() - started, 3)
    stat.error = f"{type(last_error).__name__}: {last_error}"
    STATS.record(stat)
    raise ToolOutputMissing(stat.error) from last_error


def route_many(jobs) -> dict[str, ModelRead | ToolOutputMissing]:
    """Route many rows concurrently.

    ``jobs`` is an iterable of ``(message_row, features, pool, media)``. Returns
    a dict keyed by message_id; a row that exhausted its retries maps to the
    ToolOutputMissing exception rather than raising, so one bad row cannot end
    the run.
    """
    jobs = list(jobs)
    results: dict[str, ModelRead | ToolOutputMissing] = {}

    def _run(job):
        message_row = job[0]
        try:
            return message_row["message_id"], route(*job)
        except ToolOutputMissing as exc:
            return message_row["message_id"], exc

    with ThreadPoolExecutor(max_workers=config.ROUTER_MAX_WORKERS) as pool_exec:
        for message_id, outcome in pool_exec.map(_run, jobs):
            results[message_id] = outcome
    return results
