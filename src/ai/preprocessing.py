"""
WATERSCOPE-AI — Preprocessing Utilities
Robust image loading, validation, letterbox resizing, and normalization for CV models.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import cv2
import numpy as np

def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Loads an image from a filepath or validates an existing numpy ndarray.

    Args:
        image_input: File path (str/Path) or numpy array.

    Returns:
        np.ndarray: BGR image array.

    Raises:
        FileNotFoundError: If the specified image path does not exist.
        ValueError: If the file is not a valid image or cannot be decoded.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at path: {path}")
        
        # cv2.imread handles standard image formats
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to decode image from path: {path}")
        return img
    elif isinstance(image_input, np.ndarray):
        validate_image(image_input)
        return image_input.copy()
    else:
        raise TypeError(f"Expected str, Path, or np.ndarray, got {type(image_input).__name__}")

def validate_image(img: np.ndarray) -> bool:
    """
    Validates that a numpy array is a valid non-empty 2D/3D image.

    Args:
        img: Numpy array to validate.

    Returns:
        bool: True if valid.

    Raises:
        ValueError: If image dimensions or sizes are invalid.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}")
    if img.size == 0:
        raise ValueError("Image array is empty (size 0).")
    if img.ndim not in (2, 3):
        raise ValueError(f"Image must have 2 or 3 dimensions, got {img.ndim}.")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"Invalid image dimensions: {img.shape}")
    return True

def resize_image_aspect_ratio(
    img: np.ndarray,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes an image preserving aspect ratio and adds letterbox padding to reach target square size.

    Args:
        img: Source image (BGR or grayscale).
        target_size: Square target dimension (e.g., 640).
        pad_color: Color tuple for padding (default neutral gray (114, 114, 114)).

    Returns:
        Tuple containing:
            - padded_img: Padded image of shape (target_size, target_size, C)
            - scale: Scale factor applied to original image
            - (pad_w, pad_h): Padding added to width and height (half-pad on each side)
    """
    validate_image(img)
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    # Pad top, bottom, left, right
    top = pad_h
    bottom = target_size - new_h - top
    left = pad_w
    right = target_size - new_w - left

    padded_img = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded_img, scale, (pad_w, pad_h)

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image from uint8 [0, 255] to float32 [0.0, 1.0].

    Args:
        img: Input image array.

    Returns:
        np.ndarray: float32 normalized image.
    """
    validate_image(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif np.issubdtype(img.dtype, np.floating):
        if img.max() > 1.0:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)
    return (img.astype(np.float32) / 255.0)
