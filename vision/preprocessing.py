"""
preprocessing.py — Image Enhancement Utilities

Uses OpenCV to pre-process grocery images for improved barcode scanning
and object detection, especially in low-light conditions.

Techniques applied:
  • Grayscale conversion
  • CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  • Gaussian blur for noise reduction
  • Adaptive thresholding for barcode emphasis
  • Optional sharpening kernel
"""

import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk with validation.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        BGR image as a NumPy array.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Args:
        image: BGR image array.

    Returns:
        Single-channel grayscale image.
    """
    if len(image.shape) == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE for adaptive contrast enhancement.

    Particularly effective for images captured in dim grocery aisles.

    Args:
        gray: Grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_size: Size of the grid for histogram equalization.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce high-frequency noise.

    Args:
        image: Input image (grayscale or BGR).
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adaptive_threshold(gray: np.ndarray, block_size: int = 11,
                       constant: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to binarize the image.

    Useful for isolating barcode bars from the background.

    Args:
        gray: Grayscale image.
        block_size: Neighbourhood size for threshold calculation.
        constant: Constant subtracted from the mean.

    Returns:
        Binary image.
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, constant
    )


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply an unsharp-mask style sharpening kernel.

    Args:
        image: Input image (grayscale or BGR).

    Returns:
        Sharpened image.
    """
    kernel = np.array([
        [0,  -1,  0],
        [-1,  5, -1],
        [0,  -1,  0]
    ], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def enhance_image(image_path: str, for_barcode: bool = False) -> np.ndarray:
    """Full pre-processing pipeline for a grocery image.

    Steps:
        1. Load image
        2. Convert to grayscale
        3. Apply CLAHE for contrast enhancement
        4. Gaussian blur for noise reduction
        5. (Optional) Adaptive threshold for barcode mode

    Args:
        image_path: Path to the input image.
        for_barcode: If True, also applies adaptive thresholding
                     to create a clean binary image for barcode decoding.

    Returns:
        Pre-processed image as a NumPy array.
    """
    image = load_image(image_path)
    gray = to_grayscale(image)
    enhanced = apply_clahe(gray)
    denoised = reduce_noise(enhanced)

    if for_barcode:
        denoised = adaptive_threshold(denoised)

    return denoised


def enhance_for_detection(image_path: str) -> np.ndarray:
    """Pre-process an image for YOLO object detection.

    Object detection models expect colour images, so this pipeline
    enhances the original BGR image without converting to grayscale.

    Steps:
        1. Load image
        2. Sharpen to restore edge detail
        3. Light noise reduction

    Args:
        image_path: Path to the input image.

    Returns:
        Enhanced BGR image suitable for YOLO inference.
    """
    image = load_image(image_path)
    sharpened = sharpen(image)
    denoised = reduce_noise(sharpened, kernel_size=3)
    return denoised
