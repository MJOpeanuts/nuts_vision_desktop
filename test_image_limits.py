"""Tests for the central image size policy (no huge image is ever allocated)."""
import io
import struct
import sys
import warnings
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_limits import (  # noqa: E402
    DEFAULT_MAX_IMAGE_PIXELS, ENV_VAR, ImageTooLargeError, InvalidImageError,
    get_max_image_pixels, open_image,
)
from PIL import Image  # noqa: E402


def fake_png(width: int, height: int) -> io.BytesIO:
    """PNG with a real header but a tiny, truncated body (never decoded)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                      + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b""))


def test_default_limit_allows_108mp(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert get_max_image_pixels() == DEFAULT_MAX_IMAGE_PIXELS == 120_000_000
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any DecompressionBombWarning would fail
        img = open_image(fake_png(12000, 9000))  # 108 MP header only
    assert img.size == (12000, 9000)


def test_image_over_limit_rejected_before_decoding(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(ImageTooLargeError, match="too large"):
        open_image(fake_png(13000, 10000))  # 130 MP


def test_huge_image_beyond_pillow_bomb_limit_is_clean_error():
    with pytest.raises(ImageTooLargeError):
        open_image(fake_png(40000, 40000))


def test_limit_configurable(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1000")
    with pytest.raises(ImageTooLargeError):
        open_image(fake_png(100, 100))
    assert open_image(fake_png(20, 20)).size == (20, 20)


def test_invalid_env_value(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "abc")
    with pytest.raises(ValueError):
        get_max_image_pixels()


def test_invalid_image(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        open_image(bad)
    with pytest.raises(InvalidImageError):
        open_image(tmp_path / "missing.jpg")


def test_global_pillow_settings_untouched():
    assert Image.MAX_IMAGE_PIXELS is not None
