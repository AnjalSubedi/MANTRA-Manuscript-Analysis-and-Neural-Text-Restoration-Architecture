# -*- coding: utf-8 -*-
"""
MANTRA Stage 2.3 — Image Transforms for HTR Line Images.

Deterministic preprocessing only (no augmentation).
Converts raw PNG line images into grayscale float32 tensors [1, H, W]
with aspect-ratio-preserving resize to a fixed height.
"""
import logging
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def load_image(path: str) -> Image.Image:
    """Open an image file and convert to 8-bit grayscale ('L' mode).

    Args:
        path: Absolute or project-relative path to a PNG file.

    Returns:
        PIL Image in mode 'L'.

    Raises:
        FileNotFoundError: if the path does not exist.
        Exception: for corrupt / unreadable files.
    """
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img


def resize_to_height(img: Image.Image, target_height: int) -> Image.Image:
    """Resize so height == target_height while preserving aspect ratio.

    Width is scaled proportionally using high-quality Lanczos resampling.
    The original aspect ratio is preserved exactly — no stretching or
    squeezing in either dimension.

    Args:
        img: Grayscale PIL Image.
        target_height: Desired output height in pixels (e.g. 64).

    Returns:
        Resized PIL Image with height == target_height.
    """
    orig_w, orig_h = img.size
    if orig_h == target_height:
        return img
    scale = target_height / orig_h
    new_w = max(1, round(orig_w * scale))
    return img.resize((new_w, target_height), Image.LANCZOS)


def image_to_tensor(img: Image.Image, invert: bool = False) -> torch.Tensor:
    """Convert a grayscale PIL Image to a float32 tensor [1, H, W] in [0, 1].

    Pixel convention (default, invert=False):
        Background → high values (≈1.0, white)
        Text       → low values  (≈0.0, dark)

    If ``invert=True`` the pixel values are flipped (1 − pixel) so that
    text becomes bright and background becomes dark.

    Args:
        img: Grayscale PIL Image in mode 'L'.
        invert: Whether to invert pixel values.

    Returns:
        torch.Tensor of shape [1, H, W], dtype float32, range [0, 1].
    """
    # Convert to float [0, 1]
    tensor = torch.from_numpy(
        # numpy array is uint8 [H, W]
        __import__("numpy").array(img, dtype="float32") / 255.0
    )
    if invert:
        tensor = 1.0 - tensor
    # Add channel dimension → [1, H, W]
    return tensor.unsqueeze(0)


def preprocess_image(
    path: str,
    target_height: int = 64,
    max_width: int | None = None,
    invert: bool = False,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """End-to-end image preprocessing pipeline.

    Steps:
        1. Load image and convert to grayscale.
        2. Resize to ``target_height`` preserving aspect ratio.
        3. If ``max_width`` is set and the resized width exceeds it,
           the sample is **skipped** (returns None) — the image is never
           cropped or horizontally squeezed.
        4. Convert to float32 tensor [1, H, W] in [0, 1].

    Args:
        path: Path to the image file.
        target_height: Fixed output height in pixels.
        max_width: Optional width cap.  Images wider than this after
            height-based resize are **rejected** (function returns None
            for the tensor) so the caller can skip/report them.
            ``None`` means no limit.
        invert: Whether to invert pixel values.

    Returns:
        A 3-tuple ``(tensor, original_size, processed_size)`` where sizes
        are ``(width, height)`` pairs.  If the image exceeds ``max_width``,
        the tensor is ``None`` and ``processed_size`` is ``(0, 0)``.
    """
    img = load_image(path)
    original_size = img.size  # (W, H)

    img = resize_to_height(img, target_height)
    processed_w, processed_h = img.size

    # Width cap — skip over-wide images instead of distorting them
    if max_width is not None and processed_w > max_width:
        logger.warning(
            "Image %s exceeds max_width after resize (%d > %d) — skipped.",
            path, processed_w, max_width,
        )
        return None, original_size, (0, 0)

    tensor = image_to_tensor(img, invert=invert)
    return tensor, original_size, (processed_w, processed_h)
