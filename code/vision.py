"""Image preparation for the router. Normalizes and encodes; interprets nothing.

    image_block(media_id) -> dict     # an Anthropic image content block

This module resizes and base64-encodes. It does not describe, classify, or read
text out of an image — that is the model's job, and the result comes back in
``media_summary`` on the tool output.

Images are downscaled so the long edge is at most 1568px. ``img_012`` is 1.9MB
at full size; sending it unresized costs tokens for detail the model does not
use. Encoded blocks are cached by content hash under ``code/cache/images/``, so
the resize and encode happen once per distinct image rather than once per row —
``img_008`` alone appears on three rows.

**The declared ``media_type`` is derived from the bytes, never the filename.**
Every image in this dataset carries a ``.jpg`` extension, but only 10 of 20 are
JPEG: 7 are PNG, 2 WEBP, and 1 AVIF. Labelling by extension declared PNG and
WEBP payloads as ``image/jpeg``, which the API validates and rejects. The format
is sniffed from magic bytes by :func:`_sniff_format`; anything outside
:data:`API_SUPPORTED` is re-encoded rather than relabelled.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import config
import data_layer

#: Anthropic's recommended maximum long edge. Above this the image is
#: downscaled server-side anyway, so sending more is paid-for waste.
MAX_LONG_EDGE = 1568

#: The container formats the Anthropic image API accepts. AVIF is deliberately
#: absent: the dataset contains one AVIF file and the API will reject it, so it
#: is re-encoded to JPEG rather than declared with a format the API cannot read.
API_SUPPORTED = {"jpeg", "png", "gif", "webp"}


class UnsupportedImageFormat(RuntimeError):
    """The bytes are not an API-supported format and could not be converted.

    Raised rather than returning a payload with a fabricated label. A wrong
    media_type is rejected by the API at request time, which is a worse failure
    than an explicit one here.
    """


def _sniff_format(raw: bytes) -> str | None:
    """Identify the container from magic bytes. Returns None if unrecognized.

    Deliberately does not take a filename and does not use Pillow. The filename
    is not evidence about content: every image in this dataset carries a ``.jpg``
    extension and only half of them are JPEG.
    """
    if raw[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    if raw[4:8] == b"ftyp" and raw[8:12] in (b"avif", b"avis"):
        return "avif"
    return None


def _cache_dir() -> Path:
    path = config.COMMITTED_CACHE_DIR / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _content_hash(raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(raw)
    digest.update(str(MAX_LONG_EDGE).encode("utf-8"))
    return digest.hexdigest()[:32]


def _prepare(raw: bytes) -> tuple[bytes, str]:
    """Return the bytes to send and the media_type that truthfully describes them.

    The media_type is decided from the bytes, never from the filename. Four
    outcomes:

    * format not in :data:`API_SUPPORTED` (or unrecognized) — re-encode to JPEG
      regardless of pixel dimensions, because the API cannot read the original
      at all
    * supported and within :data:`MAX_LONG_EDGE` — pass the original bytes
      through with their true label
    * supported and oversized — downscale and re-encode to JPEG
    * Pillow missing, or the re-encode fails — pass supported bytes through
      unchanged (costs tokens, stays correct); for an unsupported format there
      is nothing truthful to return, so raise
    """
    fmt = _sniff_format(raw)
    needs_conversion = fmt not in API_SUPPORTED

    try:
        from PIL import Image
    except ImportError as exc:
        if needs_conversion:
            raise UnsupportedImageFormat(
                f"image is {fmt or 'an unrecognized format'}, which the API does "
                "not accept, and Pillow is not installed to convert it"
            ) from exc
        return raw, f"image/{fmt}"

    try:
        with Image.open(io.BytesIO(raw)) as image:
            oversized = max(image.size) > MAX_LONG_EDGE
            if not oversized and not needs_conversion:
                return raw, f"image/{fmt}"

            converted = image.convert("RGB")
            if oversized:
                scale = MAX_LONG_EDGE / max(image.size)
                new_size = (
                    round(image.width * scale),
                    round(image.height * scale),
                )
                converted = converted.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue(), "image/jpeg"
    except Exception as exc:  # noqa: BLE001 — classified immediately below
        if needs_conversion:
            raise UnsupportedImageFormat(
                f"image is {fmt or 'an unrecognized format'}, which the API does "
                f"not accept, and re-encoding failed: {type(exc).__name__}: {exc}"
            ) from exc
        # A supported format that would not re-encode still ships correctly
        # labelled. That costs tokens, not correctness.
        return raw, f"image/{fmt}"


def image_block(media_id: str) -> dict:
    """Build the Anthropic image content block for one ``media_id``.

    Raises FileNotFoundError (naming the media_id) if the image is absent —
    a missing file referenced by ``images.csv`` is a dataset problem, not
    something to paper over with an empty block.
    """
    path = data_layer.resolve_media_path("image", media_id)
    raw = path.read_bytes()

    key = _content_hash(raw)
    cached = _cache_dir() / f"{key}.b64"
    meta = _cache_dir() / f"{key}.type"

    if cached.exists() and meta.exists():
        data = cached.read_text(encoding="ascii")
        media_type = meta.read_text(encoding="ascii").strip()
    else:
        payload, media_type = _prepare(raw)
        data = base64.standard_b64encode(payload).decode("ascii")
        cached.write_text(data, encoding="ascii")
        meta.write_text(media_type, encoding="ascii")

    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }
