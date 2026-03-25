"""
barcode_scanner.py — Barcode / QR Code Decoder

Uses the pyzbar library to detect and decode 1D barcodes (EAN-13, UPC-A,
Code 128, etc.) and 2D codes (QR) from grocery product images.
"""

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

from vision.preprocessing import load_image, enhance_image


# ── Barcode type groups ──────────────────────────────────────────────
BARCODE_1D_TYPES = {"EAN13", "EAN8", "UPCA", "UPCE", "CODE128", "CODE39", "I25"}
BARCODE_2D_TYPES = {"QRCODE"}


def _decode_image(image: np.ndarray) -> list:
    """Low-level decode wrapper around pyzbar.

    Args:
        image: Grayscale or BGR NumPy array.

    Returns:
        List of pyzbar Decoded objects.
    """
    if not PYZBAR_AVAILABLE:
        raise ImportError(
            "pyzbar is not installed.  Run:  pip install pyzbar\n"
            "On Windows you may also need to install the Visual C++ "
            "Redistributable."
        )
    return pyzbar_decode(image)


def scan_barcode(image_input, enhance: bool = True) -> list[dict]:
    """Detect and decode barcodes / QR codes from an image.

    The function tries multiple pre-processing strategies to maximise
    detection rate:
        1. Enhanced grayscale (CLAHE + denoise)
        2. Raw loaded image (no processing)
        3. Adaptive-thresholded binary image

    Args:
        image_input: Either a file path (str) or a NumPy image array.
        enhance: Whether to apply pre-processing pipeline.

    Returns:
        List of dicts, each containing:
            - ``data``  : decoded string (e.g. ``"5901234123457"``)
            - ``type``  : barcode symbology (e.g. ``"EAN13"``, ``"QRCODE"``)
            - ``rect``  : bounding rectangle ``(x, y, w, h)``
            - ``category``: ``"1D"`` or ``"2D"``
    """
    results = []
    seen = set()  # avoid duplicates across strategies

    # Build a list of images to try
    images_to_try = []

    if isinstance(image_input, str):
        # Strategy 1: enhanced grayscale
        if enhance:
            images_to_try.append(enhance_image(image_input, for_barcode=False))
        # Strategy 2: raw image
        images_to_try.append(load_image(image_input))
        # Strategy 3: binary threshold
        if enhance:
            images_to_try.append(enhance_image(image_input, for_barcode=True))
    elif isinstance(image_input, np.ndarray):
        images_to_try.append(image_input)
    else:
        raise TypeError(
            f"image_input must be a file path (str) or ndarray, "
            f"got {type(image_input).__name__}"
        )

    for img in images_to_try:
        decoded_objects = _decode_image(img)
        for obj in decoded_objects:
            barcode_data = obj.data.decode("utf-8", errors="replace")
            barcode_type = obj.type

            # De-duplicate
            key = (barcode_data, barcode_type)
            if key in seen:
                continue
            seen.add(key)

            category = "2D" if barcode_type in BARCODE_2D_TYPES else "1D"
            rect = obj.rect  # named tuple: (left, top, width, height)

            results.append({
                "data": barcode_data,
                "type": barcode_type,
                "rect": (rect.left, rect.top, rect.width, rect.height),
                "category": category,
            })

    return results


def scan_barcode_from_path(image_path: str) -> list[dict]:
    """Convenience wrapper — scan barcodes from a file path.

    Args:
        image_path: Path to an image file.

    Returns:
        Same as :func:`scan_barcode`.
    """
    return scan_barcode(image_path, enhance=True)


def draw_detections(image_path: str, output_path: str | None = None) -> np.ndarray:
    """Annotate an image with detected barcode bounding boxes and labels.

    Useful for debugging and visual verification.

    Args:
        image_path: Path to the source image.
        output_path: If provided, saves the annotated image to this path.

    Returns:
        Annotated BGR image.
    """
    image = load_image(image_path)
    detections = scan_barcode(image_path)

    for det in detections:
        x, y, w, h = det["rect"]
        label = f'{det["type"]}: {det["data"]}'

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    if output_path:
        cv2.imwrite(output_path, image)

    return image
