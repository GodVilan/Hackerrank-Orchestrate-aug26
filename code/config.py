"""Single source of truth for paths and model settings.

Import this module from anywhere in ``code/`` rather than hardcoding paths or
model names. Every path below is absolute and derived from this file's own
location, so the entry points work regardless of the current directory.

Media paths
-----------
The ``file_path`` column in ``images.csv`` and ``voice_notes.csv`` is relative
to :data:`DATASET_DIR`, **not** to the repo root. The stored values look like
``media/images/img_001.jpg`` and ``media/audio/vn_001.mp3``, so resolve them
with :func:`resolve_media_path` (or ``DATASET_DIR / file_path``). Joining them
against the repo root or the process CWD will not find the file.

Secrets
-------
``ANTHROPIC_API_KEY`` is read from the environment only. As a developer
convenience, a git-ignored ``code/.env`` is parsed on import and used to fill
in variables that are *not* already set. A real environment variable always
wins over the ``.env`` file. No secret is ever hardcoded here.

Feature constraint
------------------
``message_id`` ordinal position, numeric suffix, and row order are never
features. Nothing in this package may key off the numeric part of a
``message_id`` or its position in a file.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

DATASET_DIR = REPO_ROOT / "dataset"

# Participant-facing inputs.
MESSAGES_CSV = DATASET_DIR / "messages.csv"
SAMPLE_MESSAGES_CSV = DATASET_DIR / "sample_messages.csv"
USERS_CSV = DATASET_DIR / "users.csv"
GROUPS_CSV = DATASET_DIR / "groups.csv"
GROUP_MEMBERS_CSV = DATASET_DIR / "group_members.csv"
BUSINESS_ACCOUNTS_CSV = DATASET_DIR / "business_accounts.csv"
USER_BUSINESS_HISTORY_CSV = DATASET_DIR / "user_business_history.csv"
MESSAGE_HISTORY_CSV = DATASET_DIR / "message_history.csv"
MESSAGE_EVENTS_CSV = DATASET_DIR / "message_events.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
VOICE_NOTES_CSV = DATASET_DIR / "voice_notes.csv"
DAILY_NOTIFICATION_SUMMARY_CSV = DATASET_DIR / "daily_notification_summary.csv"

# Blank submission template shipped with the dataset. Read-only reference for
# the column contract; predictions are written to OUTPUT_CSV instead.
OUTPUT_TEMPLATE_CSV = DATASET_DIR / "output.csv"

# Media roots. Prefer resolve_media_path() over joining these by hand.
MEDIA_DIR = DATASET_DIR / "media"
IMAGES_DIR = MEDIA_DIR / "images"
AUDIO_DIR = MEDIA_DIR / "audio"

# Generated artifacts (git-ignored).
OUTPUT_CSV = REPO_ROOT / "output.csv"
CACHE_DIR = CODE_DIR / ".cache"

# Committed cache. Distinct from CACHE_DIR: `.cache/` holds regenerable model
# responses and is git-ignored, while `cache/` holds artifacts that ship with
# the submission so the pipeline runs end to end on a machine that has no ASR
# model weights installed.
COMMITTED_CACHE_DIR = CODE_DIR / "cache"
TRANSCRIPT_CACHE = COMMITTED_CACHE_DIR / "transcripts.json"

# Required output columns, in the exact required order.
OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)


def resolve_media_path(file_path: str) -> Path:
    """Resolve a ``file_path`` from images.csv / voice_notes.csv to disk.

    Those columns are relative to :data:`DATASET_DIR`, not the repo root.
    """
    return DATASET_DIR / file_path


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

ENV_FILE = CODE_DIR / ".env"


def _hydrate_env_from_file(path: Path) -> None:
    """Fill in unset environment variables from a KEY=VALUE file.

    Real environment variables always win: a key already present in os.environ
    is never overwritten. Blank lines, ``#`` comments, and lines without ``=``
    are skipped. Surrounding single or double quotes on the value are stripped.
    Missing or unreadable files are ignored so importing this module never
    fails because of a developer-local file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


_hydrate_env_from_file(ENV_FILE)

# Read from the environment only. None when unset — callers that need the key
# should fail with a clear message rather than this module raising on import,
# so that offline tooling can still import config.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def require_api_key() -> str:
    """Return ANTHROPIC_API_KEY, raising a clear error when it is unset."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it, or copy code/.env.example "
            "to code/.env and fill it in."
        )
    return ANTHROPIC_API_KEY


# --------------------------------------------------------------------------
# Model settings
# --------------------------------------------------------------------------

# Bumped by hand whenever prompts.py changes. It is part of the router cache
# key, so bumping it invalidates every cached response — which is the point: a
# changed prompt is a different question and must not be answered from cache.
PROMPT_VERSION = 1

PRIMARY_MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
ENABLE_PROMPT_CACHING = True
MAX_RETRIES = 3

# --------------------------------------------------------------------------
# Ladder thresholds
# --------------------------------------------------------------------------
#
# None of these values were optimized against an accuracy metric. Each sits
# inside an interval where the observed distribution has no mass at all, so
# every value in that interval produces identical behaviour on this corpus.
# The comments record the observed distribution and the interval; they do not
# record a search. Reproduce all of them with `python code/fit_thresholds.py`.

# Layer 1 — sender report rate. Read GLOBALLY (across every recipient), gated
# at global_sender_n >= 3, with a separate n_prior > 0 clause that suppresses
# first contacts. See README "Decision 4" for why global rather than per-pair.
#
# Observed global rates at that floor, one value per sender:
#   0.0 (u_047 and others) · 0.0833 (u_041) | 0.4545 (u_050) · 0.5625 (u_052)
#   · 0.6667 (u_049) · 1.0 (u_053)
# The set is bimodal with nothing in (0.0833, 0.4545). Any threshold inside
# that interval selects exactly the same senders. 0.30 sits inside it.
#
# The same constant is also read as a tier signal inside the scam branch, where
# it is applied to the per-pair rate. That use records corroboration for a
# decision already made; it never decides an action on its own.
T_REPORT = 0.30

# Layer 3 — sender dismiss rate, per (recipient, sender), gated at n_prior >= 5.
# Observed value set at that floor is again bimodal with an empty band:
#   {0.0, 0.0435, 0.1667} and {1.0}
# Any threshold in (0.1667, 1.0) selects exactly the same 16 of 110 rows.
# 0.50 sits inside that band. Note the upper mode is a single value: the
# senders this fires on dismissed every prior message, not merely most.
T_DISMISS = 0.50

# Layer 3 — per-message forwarded_count, mute branch.
#
# One labeled row makes this branch load-bearing. Of the three gold mute rows
# with forwarded_count > 0:
#   sample_msg_013  fwd=6,  n_prior=11, dismiss_rate=1.0  -> caught earlier by
#                   the Layer 3 dismiss branch; this threshold is not what
#                   decides it
#   sample_msg_014  fwd=11, n_prior=2                     -> falls below the
#                   dismiss branch's n_prior >= 5 gate, so THIS branch is the
#                   only thing that mutes it. Gold labels it mute.
#   sample_msg_015  fwd=3,  business, message_type=promotion -> an opt-out case
#                   caught at Layer 2, and excluded from this branch by the
#                   message_type conjunct regardless of threshold
#
# So gold constrains the threshold from above only: it must be <= 11 to catch
# sample_msg_014. Nothing labeled distinguishes 3 from 6. 6 is chosen as the
# narrower of the two, on a branch that cannot be overridden downstream.
T_FWD_MESSAGE = 6

# Layer 3 — per-sender mean_forwarded_count. Selects between
# MUTE_FORWARD_PATTERN and MUTE_SIMILAR_IGNORED; never decides the action.
# Observed means across all 14 senders: {0.0, 0.04, 0.05, 0.26, 1.0, 1.19, 7.83}
# — one sender at 7.83 and nothing else above 1.19. Any threshold in
# (1.19, 7.83) yields identical sub-rule selection on this corpus.
T_FWD_SENDER_MEAN = 3

# --------------------------------------------------------------------------
# Impersonation signature (Layer 1)
# --------------------------------------------------------------------------
# All four conjuncts are required together:
#   verified == 0
#   AND official_domain is non-empty      (an absent domain is not a mismatch)
#   AND official_domain != domain_used_by_sender, casefolded
#   AND (account_age_days < 60 OR domain_used_by_sender_age_days < 30)
#   AND report_rate_per_1k > T_REPORT_PER_1K
#
# Observed report_rate_per_1k across all 110 businesses is empty across the
# interval (6.48, 11.25). The boundary businesses are:
#   business_008  Swiggy, verified, domain matches      6.48  (highest below)
#   business_047  IRCTC, unverified, domain mismatched  11.25 (lowest above)
# Any threshold inside that interval fires on exactly the same 21 businesses,
# so every value in it is behaviourally identical on this corpus. 10 is a round
# value within the band; it carries no other justification.
#
# The rate conjunct is retained even though age + mismatch alone already
# exclude every verified brand here — that exclusion holds because all verified
# brands in this sample happen to be 1000+ days old, which is a property of the
# sample rather than one the term can be relied on to have on the hidden set.
T_REPORT_PER_1K = 10
IMPERSONATION_MAX_ACCOUNT_AGE_DAYS = 60
IMPERSONATION_MAX_DOMAIN_AGE_DAYS = 30

# --------------------------------------------------------------------------
# ASR provider (faster-whisper)
# --------------------------------------------------------------------------
# Pinned explicitly so a transcript can be reproduced from the audio. Changing
# any value here changes the cache key and invalidates every cached transcript.
#
# Determinism: greedy decoding (beam_size 1), temperature 0.0 with no fallback
# ladder, and VAD disabled — faster-whisper's VAD introduces segment boundaries
# that shift with model version, which would make transcripts non-reproducible.
# `task` is pinned rather than left to the library default. faster-whisper
# defaults to "transcribe", but a default is not a contract: "translate" would
# silently rewrite non-English audio into English before routing, altering
# message content. Verified empirically on vn_015 — transcribe and translate
# produce different text, so this build transcribes.
ASR_TASK = "transcribe"
ASR_MODEL = "base"
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"
ASR_BEAM_SIZE = 1
ASR_TEMPERATURE = 0.0
ASR_VAD_FILTER = False
ASR_CONDITION_ON_PREVIOUS_TEXT = False
#: None lets the model detect the language. Detection is deterministic for a
#: fixed model and input, and the corpus mixes English with Hindi/Urdu.
ASR_LANGUAGE = None

# --------------------------------------------------------------------------
# Router concurrency and backoff
# --------------------------------------------------------------------------
# 110 rows is the entire scale story. Four workers keeps well inside rate
# limits while finishing a full run in about a minute.
ROUTER_MAX_WORKERS = 4
ROUTER_MIN_INTERVAL_S = 0.15   # shared rate limiter: min gap between requests
ROUTER_BACKOFF_BASE_S = 1.0    # exponential backoff on 429 / 5xx
ROUTER_BACKOFF_MAX_S = 30.0
RUN_STATS = CODE_DIR / "run_stats.json"
ROUTER_CACHE_DIR = CACHE_DIR / "router"
