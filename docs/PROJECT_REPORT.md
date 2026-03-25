# 📝 Project Report — Personalized Grocery Assistant

## 1. Project Overview

The **Personalized Grocery Assistant** is a command-line Python application that leverages computer vision to identify grocery products from images and provides actionable insights including nutritional information and price comparisons across stores.

The system processes a single input image through a multi-stage pipeline:

1. **Image Pre-processing** — Enhances image quality for reliable detection
2. **Barcode Scanning** — Decodes 1D/2D barcodes to identify packaged products
3. **Object Detection** — Identifies unpackaged produce via deep learning
4. **Data Integration** — Fetches nutrition data and compares store prices
5. **Presentation** — Displays structured, colour-coded results in the terminal

---

## 2. Syllabus Coverage

### 2.1 Image Processing

| Technique | Location | Purpose |
|---|---|---|
| Grayscale conversion | `vision/preprocessing.py` → `to_grayscale()` | Reduces 3-channel colour images to single-channel for efficient processing |
| CLAHE | `vision/preprocessing.py` → `apply_clahe()` | Contrast-Limited Adaptive Histogram Equalization enhances local contrast — critical for dimly-lit grocery aisles |
| Gaussian blur | `vision/preprocessing.py` → `reduce_noise()` | Suppresses high-frequency noise that can cause false barcode reads |
| Adaptive thresholding | `vision/preprocessing.py` → `adaptive_threshold()` | Binarises images for barcode isolation, adapting to varying illumination |
| Sharpening kernel | `vision/preprocessing.py` → `sharpen()` | Restores edge detail in blurry images before YOLO inference |

### 2.2 Feature Extraction

| Technique | Location | Purpose |
|---|---|---|
| Barcode pattern decoding | `vision/barcode_scanner.py` → `scan_barcode()` | Extracts structured data (EAN-13, UPC-A, QR) from barcode patterns using `pyzbar`'s ZBar engine |
| Multi-strategy extraction | `vision/barcode_scanner.py` | Tries 3 pre-processing strategies (enhanced, raw, binary) to maximise detection rate |
| Bounding box extraction | `vision/object_detector.py` | Extracts object locations `[x1, y1, x2, y2]` and class confidence from YOLO predictions |
| Class filtering | `vision/object_detector.py` → `GROCERY_CLASSES` | Filters COCO detections to a curated set of grocery-relevant categories |

### 2.3 Model Inference

| Model | Location | Details |
|---|---|---|
| YOLOv8-nano | `vision/object_detector.py` → `ProductDetector` | Pre-trained on COCO (80 classes), filtered to 17 grocery-relevant classes |
| Confidence thresholding | `main.py --confidence` | User-configurable minimum confidence (default 0.5) to control precision/recall trade-off |
| Lazy loading | `detect_products()` | Model loaded once on first call, reused for subsequent inferences |

---

## 3. Architecture

```
User Image
    │
    ▼
┌───────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Pre-processing│────▶│  Barcode Scanner │────▶│  Nutrition   │
│  (OpenCV)     │     │  (pyzbar)        │     │  API Lookup  │
└───────────────┘     └──────────────────┘     │(OpenFoodFacts│
    │                                          └──────┬───────┘
    │                                                 │
    ▼                                                 ▼
┌───────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Enhancement   │────▶│ Object Detector  │────▶│  Price DB    │
│  for YOLO     │     │  (YOLOv8-nano)   │     │  (SQLite)    │
└───────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                                                      ▼
                                               ┌─────────────┐
                                               │  Formatted   │
                                               │  CLI Output  │
                                               └─────────────┘
```

---

## 4. Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| Image Processing | OpenCV 4.8+ | Industry-standard, extensive algorithm library, fast C++ backend |
| Barcode Decoding | pyzbar (ZBar) | Proven multi-format decoder, supports EAN-13/UPC-A/QR |
| Object Detection | Ultralytics YOLOv8-nano | State-of-the-art accuracy with minimal model size (~6 MB) |
| Nutrition Data | OpenFoodFacts API | Free, open-source food database with 3M+ products |
| Price Storage | SQLite | Zero-configuration, file-based, perfect for local demos |
| CLI Framework | argparse | Python standard library, no extra dependencies |
| Output Formatting | ANSI escape codes | Universal terminal support, no additional dependencies |

---

## 5. Key Design Decisions

1. **Multi-strategy barcode scanning**: Rather than a single pass, the scanner tries enhanced → raw → binarised images. This dramatically improves detection rates for damaged or partially obscured barcodes.

2. **Grocery class filtering**: YOLO's full 80-class COCO vocabulary includes irrelevant classes (person, car, etc.). We curate 17 grocery-specific classes to reduce false positives.

3. **Lazy model initialization**: The YOLOv8 model is loaded only on first use and reused, avoiding unnecessary startup cost when running in barcode-only mode (`--no-detect`).

4. **Idempotent database seeding**: `price_db.init_db()` uses `INSERT OR IGNORE` so it can be called multiple times without duplicating data.

5. **Graceful degradation**: If `pyzbar` or `ultralytics` is not installed, the corresponding feature is skipped with a warning rather than crashing.

---

## 6. Challenges & Solutions

| Challenge | Solution |
|---|---|
| Low-light barcode scanning | CLAHE + multi-strategy pre-processing pipeline |
| YOLO detecting non-grocery items | Curated whitelist of 17 COCO class IDs |
| Missing products in OpenFoodFacts | Fallback search-by-name + graceful "not found" response |
| Cross-platform terminal colours | ANSI escape codes with reset sequences |

---

## 7. Future Enhancements

- **Real-time camera feed**: Replace single-image mode with live webcam scanning
- **Allergen alerts**: Cross-reference detected allergens with a user allergy profile
- **Shopping list integration**: Track prices over time and suggest optimal purchase timing
- **Custom model fine-tuning**: Train YOLOv8 on a grocery-specific dataset for higher accuracy
- **Web/mobile interface**: Flask/FastAPI backend with React frontend

---

## 8. References

1. Ultralytics YOLOv8 Documentation — https://docs.ultralytics.com/
2. OpenFoodFacts API — https://wiki.openfoodfacts.org/API
3. pyzbar Documentation — https://github.com/NaturalHistoryMuseum/pyzbar
4. OpenCV Documentation — https://docs.opencv.org/
5. COCO Dataset Classes — https://cocodataset.org/

---

*Report prepared for Computer Vision course submission.*
