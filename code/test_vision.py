"""Contract tests for vision.py — does the declared media_type match the bytes?

    python code/test_vision.py

**The oracle is PIL, not our own sniffer.** Asserting
``media_type == "image/" + _sniff_format(returned_bytes)`` would compare the
sniffer to itself: a misclassification moves both sides together and the test
passes while the API still 400s. Every assertion below decodes the returned
bytes with ``PIL.Image.open`` and compares against *that*, which is an
independent implementation. A separate test then checks our sniffer against the
same oracle, so a sniffer bug fails loudly on its own line.

Tests use an isolated cache directory. They neither depend on nor pollute
``code/cache/images/``.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import config  # noqa: E402
import data_layer  # noqa: E402
import vision  # noqa: E402

#: PIL format name -> the media_type the API expects for those bytes.
PIL_FORMAT_TO_MEDIA_TYPE = {
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",  # multi-picture JPEG; still JPEG bytes
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "AVIF": "image/avif",
}


def _oracle_media_type(payload: bytes) -> str:
    """What these bytes actually are, according to PIL. Independent of vision.py."""
    with Image.open(io.BytesIO(payload)) as image:
        fmt = image.format
    return PIL_FORMAT_TO_MEDIA_TYPE.get(fmt, f"UNKNOWN/{fmt}")


def _blocks_with_isolated_cache():
    """image_block() for all 20 images, against a throwaway cache directory."""
    original = config.COMMITTED_CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.COMMITTED_CACHE_DIR = Path(tmp)
        try:
            out = {}
            for media_id in sorted(data_layer.index_images()):
                block = vision.image_block(media_id)
                payload = __import__("base64").b64decode(block["source"]["data"])
                out[media_id] = (block["source"]["media_type"], payload)
            return out
        finally:
            config.COMMITTED_CACHE_DIR = original


def test_declared_media_type_matches_actual_bytes() -> None:
    """The core contract. Oracle is PIL, not our sniffer."""
    mismatches = []
    for media_id, (declared, payload) in _blocks_with_isolated_cache().items():
        truth = _oracle_media_type(payload)
        if declared != truth:
            mismatches.append(f"{media_id}: declared {declared}, bytes are {truth}")
    assert not mismatches, (
        f"{len(mismatches)} image(s) declare a media_type that does not match "
        f"their bytes; the API validates this and will reject them:\n    "
        + "\n    ".join(mismatches)
    )


def test_img_020_avif_is_actually_reencoded() -> None:
    """AVIF is outside the API's accepted set, so it must be converted, not relabelled."""
    original = config.COMMITTED_CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        config.COMMITTED_CACHE_DIR = Path(tmp)
        try:
            block = vision.image_block("img_020")
        finally:
            config.COMMITTED_CACHE_DIR = original
    payload = __import__("base64").b64decode(block["source"]["data"])

    assert block["source"]["media_type"] == "image/jpeg", block["source"]["media_type"]
    assert _oracle_media_type(payload) == "image/jpeg", (
        "img_020 is declared image/jpeg but its bytes are "
        f"{_oracle_media_type(payload)} — it was relabelled, not re-encoded"
    )
    # And the source really is AVIF, so this test is exercising the branch it claims to.
    source = data_layer.resolve_media_path("image", "img_020").read_bytes()
    assert _oracle_media_type(source) == "image/avif", _oracle_media_type(source)


def test_declared_media_type_is_api_supported() -> None:
    """Nothing may be sent with a media_type the image API does not accept."""
    accepted = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    bad = [
        f"{mid}: {declared}"
        for mid, (declared, _) in _blocks_with_isolated_cache().items()
        if declared not in accepted
    ]
    assert not bad, f"unsupported media_type declared: {bad}"


def test_sniffer_agrees_with_pil() -> None:
    """Guards the sniffer itself, against the same independent oracle."""
    sniff = getattr(vision, "_sniff_format", None)
    if sniff is None:
        print("      (skipped — vision._sniff_format does not exist yet)")
        return
    wrong = []
    for media_id in sorted(data_layer.index_images()):
        raw = data_layer.resolve_media_path("image", media_id).read_bytes()
        truth = _oracle_media_type(raw).split("/", 1)[1]
        got = sniff(raw)
        if got != truth:
            wrong.append(f"{media_id}: sniffed {got!r}, PIL says {truth!r}")
    assert not wrong, "sniffer disagrees with PIL:\n    " + "\n    ".join(wrong)


def test_filename_suffix_is_not_consulted_for_media_type() -> None:
    """Every file here is .jpg; if the suffix drove the label, all 20 would be jpeg."""
    declared = {mid: d for mid, (d, _) in _blocks_with_isolated_cache().items()}
    suffixes = {
        data_layer.resolve_media_path("image", mid).suffix for mid in declared
    }
    assert suffixes == {".jpg"}, f"corpus assumption broken: {suffixes}"
    assert len(set(declared.values())) > 1, (
        "all 20 images declared the same media_type despite differing content — "
        "the label is still being derived from the filename"
    )


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
