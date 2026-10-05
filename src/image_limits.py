#!/usr/bin/env python3
"""
Central image size policy for nuts_vision.

Every image opened through Pillow must go through :func:`open_image`.  The
pixel count is read from the file header (Pillow decodes lazily) and compared
with an explicit limit *before* any full decoding.

The limit defaults to 120 000 000 pixels (a 108 MP photo is accepted) and can
be changed with the NUTS_VISION_MAX_IMAGE_PIXELS environment variable.

Pillow's own DecompressionBombWarning threshold (~89 MP) is bypassed only for
the header read performed here, in a local ``warnings.catch_warnings`` block;
``Image.MAX_IMAGE_PIXELS`` and the global warning filters are never modified.
The image resolution is never reduced.
"""

import os
import warnings
from typing import IO, Union

from PIL import Image, UnidentifiedImageError

DEFAULT_MAX_IMAGE_PIXELS = 120_000_000
ENV_VAR = "NUTS_VISION_MAX_IMAGE_PIXELS"


class InvalidImageError(ValueError):
    """The file is not a readable image."""


class ImageTooLargeError(InvalidImageError):
    """The image exceeds the configured pixel limit."""


def get_max_image_pixels() -> int:
    """Configured pixel limit (env override, else the default)."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return DEFAULT_MAX_IMAGE_PIXELS
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(f"{ENV_VAR} must be a positive integer (got {raw!r}).")
    return value


def _too_large(pixels: int, limit: int) -> ImageTooLargeError:
    return ImageTooLargeError(
        f"Image too large: {pixels / 1e6:.1f} MP exceeds the limit of "
        f"{limit / 1e6:.1f} MP. Use a smaller image or raise {ENV_VAR}."
    )


def open_image(source: Union[str, os.PathLike, IO[bytes]]) -> Image.Image:
    """
    Open an image lazily after checking its dimensions against the limit.

    Raises:
        ImageTooLargeError: the image has more pixels than the limit
            (raised before the pixel data is decoded).
        InvalidImageError: the file is missing, corrupt or not an image.
    """
    limit = get_max_image_pixels()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(source)
    except Image.DecompressionBombError:
        # Pillow refuses images above twice its own threshold.
        raise ImageTooLargeError(
            f"Image too large: more than {2 * Image.MAX_IMAGE_PIXELS / 1e6:.0f} MP "
            f"(limit {limit / 1e6:.1f} MP). Use a smaller image or raise {ENV_VAR}."
        ) from None
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise InvalidImageError(f"Invalid or unreadable image: {e}") from e

    width, height = img.size
    if width * height > limit:
        img.close()
        raise _too_large(width * height, limit)
    return img
