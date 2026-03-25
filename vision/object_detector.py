"""
object_detector.py — YOLOv8 Product Detection

Uses the Ultralytics YOLOv8-nano model to identify "unlabeled" grocery
products such as fruits, vegetables, and packaged goods that lack barcodes.

The model is filtered to a curated set of grocery-relevant COCO classes
so that irrelevant detections (person, car, etc.) are suppressed.
"""

import os
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from vision.preprocessing import load_image, enhance_for_detection


# ── Grocery-relevant COCO class IDs and names ───────────────────────
# Full COCO has 80 classes; we keep only those plausible in a grocery.
GROCERY_CLASSES = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
}

# Default model path (auto-downloads on first use)
DEFAULT_MODEL = "yolov8n.pt"


class ProductDetector:
    """Wrapper around YOLOv8 for grocery product detection.

    Attributes:
        model: Loaded YOLO model instance.
        grocery_ids: Set of COCO class IDs relevant to groceries.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL):
        """Initialise the detector with a YOLOv8 model.

        Args:
            model_path: Path to a ``.pt`` weight file.  Defaults to
                ``yolov8n.pt`` which is auto-downloaded by Ultralytics.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.  Run:  pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self.grocery_ids = set(GROCERY_CLASSES.keys())

    def detect(self, image_input, confidence: float = 0.5,
               filter_grocery: bool = True) -> list[dict]:
        """Run inference on an image and return detections.

        Args:
            image_input: File path (str) or BGR NumPy array.
            confidence: Minimum confidence threshold (0–1).
            filter_grocery: If True, only return grocery-relevant classes.

        Returns:
            List of dicts with keys:
                - ``label``: human-readable class name
                - ``confidence``: detection confidence (float)
                - ``bbox``: bounding box ``[x1, y1, x2, y2]``
                - ``class_id``: COCO class ID
        """
        # Prepare image
        if isinstance(image_input, str):
            image = enhance_for_detection(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError(
                f"image_input must be str or ndarray, "
                f"got {type(image_input).__name__}"
            )

        # Run YOLOv8 inference
        results = self.model(image, conf=confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]

                # Filter to grocery classes if requested
                if filter_grocery and class_id not in self.grocery_ids:
                    continue

                label = GROCERY_CLASSES.get(
                    class_id,
                    result.names.get(class_id, f"class_{class_id}")
                )

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(c, 1) for c in bbox],
                    "class_id": class_id,
                })

        return detections


# ── Module-level convenience function ────────────────────────────────
_detector_instance = None


def detect_products(image_input, confidence: float = 0.5) -> list[dict]:
    """Detect grocery products in an image (module-level convenience).

    Lazily initialises a shared :class:`ProductDetector` instance so
    the model is loaded only once across multiple calls.

    Args:
        image_input: File path or NumPy image.
        confidence: Minimum confidence threshold.

    Returns:
        List of detection dicts (see :meth:`ProductDetector.detect`).
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ProductDetector()
    return _detector_instance.detect(image_input, confidence=confidence)
