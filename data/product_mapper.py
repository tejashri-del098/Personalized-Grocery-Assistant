"""
product_mapper.py — Detection-to-Data Mapping Layer

The "brain" that connects Phase 1 (vision) outputs to Phase 2 (data)
lookups.  Given barcode scan results and/or object detection results,
this module produces unified ``ProductInfo`` dicts containing:
  • Product identity (name, barcode, detection source)
  • Nutritional data (from OpenFoodFacts)
  • Price comparison (from local SQLite DB)
"""

from data.nutrition_api import fetch_nutrition, search_product
from data.price_db import get_price_comparison


def map_barcode(barcode_data: str) -> dict:
    """Map a scanned barcode to nutritional + price data.

    Args:
        barcode_data: Decoded barcode string (e.g. ``"5901234123457"``).

    Returns:
        Unified product info dict.
    """
    nutrition = fetch_nutrition(barcode_data)
    product_name = nutrition["name"] if nutrition else "Unknown Product"

    prices = get_price_comparison(product_name)

    return {
        "source": "barcode",
        "barcode": barcode_data,
        "name": product_name,
        "nutrition": nutrition,
        "prices": prices,
    }


def map_detection(detection: dict) -> dict:
    """Map a YOLO detection to price data (and optional nutrition).

    Since produce items typically don't have barcodes, this function
    uses the detected label to look up prices in the local database
    and optionally searches OpenFoodFacts for generic nutrition info.

    Args:
        detection: Dict from :func:`vision.object_detector.detect_products`
                   with at least a ``label`` key.

    Returns:
        Unified product info dict.
    """
    label = detection.get("label", "unknown")
    confidence = detection.get("confidence", 0.0)

    # Price lookup (fuzzy match on label)
    prices = get_price_comparison(label)

    # Try to get generic nutrition info from OpenFoodFacts search
    nutrition = None
    search_results = search_product(label, page_size=1)
    if search_results:
        best_match = search_results[0]
        nutrition = fetch_nutrition(best_match["barcode"])

    return {
        "source": "detection",
        "label": label,
        "confidence": confidence,
        "name": label.capitalize(),
        "nutrition": nutrition,
        "prices": prices,
    }


def map_all(barcode_results: list[dict],
            detection_results: list[dict]) -> list[dict]:
    """Map all scan + detection results to unified product info.

    Barcodes are processed first (higher data fidelity), followed by
    object detections.

    Args:
        barcode_results: Output of :func:`vision.barcode_scanner.scan_barcode`.
        detection_results: Output of :func:`vision.object_detector.detect_products`.

    Returns:
        List of unified product info dicts.
    """
    products = []

    # Process barcodes
    for bc in barcode_results:
        product = map_barcode(bc["data"])
        product["barcode_type"] = bc.get("type", "N/A")
        products.append(product)

    # Process object detections
    seen_labels = set()
    for det in detection_results:
        label = det.get("label", "").lower()
        if label in seen_labels:
            continue  # avoid duplicate lookups for same product type
        seen_labels.add(label)
        products.append(map_detection(det))

    # If nothing was found at all, provide a helpful fallback
    if not products:
        products.append({
            "source": "none",
            "name": "No products detected",
            "nutrition": None,
            "prices": [],
        })

    return products
