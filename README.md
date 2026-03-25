# 🛒 Personalized Grocery Assistant

A Python CLI tool that uses **computer vision** to identify grocery products from images — via **barcode scanning** and **YOLOv8 object detection** — then fetches **nutritional information** from OpenFoodFacts and compares **prices** across stores.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Barcode Scanning** | Decodes EAN-13, UPC-A, QR, and other 1D/2D barcodes via `pyzbar` |
| **Object Detection** | Identifies unlabeled produce (bananas, apples, etc.) using YOLOv8-nano |
| **Image Pre-processing** | CLAHE, Gaussian blur, adaptive thresholding for low-light images |
| **Nutrition Lookup** | Fetches ingredients, allergens, and Nutri-Score from OpenFoodFacts API |
| **Price Comparison** | Compares prices across 3 stores (SQLite database) |
| **CLI Interface** | Clean argparse-based command-line tool with formatted output |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py (CLI)                      │
├────────────┬──────────────────┬──────────────────────────┤
│  Phase 1   │     Phase 2      │        Phase 3           │
│  Vision    │  Data Integration│     Presentation         │
├────────────┼──────────────────┼──────────────────────────┤
│ preprocess │ nutrition_api.py │   formatter.py           │
│ barcode    │ price_db.py      │   (ANSI-colored          │
│ _scanner   │ product_mapper   │    console output)       │
│ object     │   .py            │                          │
│ _detector  │                  │                          │
└──────┬─────┴────────┬─────────┴──────────────────────────┘
       │              │
   OpenCV         OpenFoodFacts API
   pyzbar         SQLite Database
   YOLOv8
```

---

## 🚀 How to Run

### Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- On Windows: [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (required by `pyzbar`)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/Personalized-Grocery-Assistant.git
cd Personalized-Grocery-Assistant
```

### Step 2 — Create a Virtual Environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Generate Sample Images

```bash
python generate_samples.py
```

This creates `samples/sample_barcode.png` and `samples/sample_produce.jpg`.

### Step 5 — Run the Assistant

```bash
# Full analysis (nutrition + price comparison)
python main.py --image samples/sample_barcode.png --mode full

# Nutrition info only
python main.py --image samples/sample_barcode.png --mode nutrition

# Price comparison only
python main.py --image samples/sample_produce.jpg --mode price

# Custom confidence threshold
python main.py --image photo.jpg --confidence 0.4

# Skip object detection (barcode only)
python main.py --image barcode.png --no-detect

# Skip barcode scanning (detection only)
python main.py --image produce.jpg --no-barcode
```

---

## 📋 CLI Arguments

| Argument | Short | Type | Default | Description |
|---|---|---|---|---|
| `--image` | `-i` | `str` | *required* | Path to the grocery product image |
| `--mode` | `-m` | `str` | `full` | Display mode: `nutrition`, `price`, or `full` |
| `--confidence` | `-c` | `float` | `0.5` | YOLO detection confidence threshold (0–1) |
| `--no-detect` | — | flag | — | Skip object detection |
| `--no-barcode` | — | flag | — | Skip barcode scanning |

---

## 📂 Project Structure

```
Personalized Grocery Assistant/
├── main.py                  # CLI entry point (argparse)
├── requirements.txt         # Python dependencies
├── generate_samples.py      # Sample image generator
├── README.md                # This file
│
├── vision/                  # Phase 1: Computer Vision
│   ├── __init__.py
│   ├── preprocessing.py     # OpenCV image enhancement
│   ├── barcode_scanner.py   # pyzbar barcode decoding
│   └── object_detector.py   # YOLOv8 object detection
│
├── data/                    # Phase 2: Data Integration
│   ├── __init__.py
│   ├── nutrition_api.py     # OpenFoodFacts API client
│   ├── price_db.py          # SQLite price comparison DB
│   └── product_mapper.py    # Detection → Data mapping
│
├── utils/                   # Phase 3: Presentation
│   ├── __init__.py
│   └── formatter.py         # Console output formatting
│
├── samples/                 # Test images (generated)
├── db/                      # SQLite database (auto-created)
├── models/                  # YOLO weights (auto-downloaded)
└── docs/
    └── PROJECT_REPORT.md    # Syllabus coverage report
```

---

## 📊 Sample Output

```
  ╔══════════════════════════════════════════════════════╗
  ║       🛒  Personalized Grocery Assistant  🛒        ║
  ║      Smart Vision  •  Nutrition  •  Prices          ║
  ╚══════════════════════════════════════════════════════╝

  📷  Analysing: sample_barcode.png
  📂  Mode: full  |  Confidence: 0.5

══════════════  SCAN RESULTS  ══════════════

  ▸ Barcodes detected: 1
    • [EAN13] 3017620422003

══════════  NUTRITION — Nutella  ══════════

  Product:  Nutella
  Brand:    Ferrero
  Barcode:  3017620422003

  Nutri-Score:   E
  [A]  [B]  [C]  [D]  [E]

  Nutriments (per 100 g):
  ────────────────────────────────────────
  Energy             539 kcal
  Fat                30.9 g
  Sugars             56.3 g
  Proteins            6.3 g
  ────────────────────────────────────────

  ⚠ Allergens: en:nuts, en:milk, en:soybeans
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **OpenCV** | Image pre-processing (CLAHE, blur, thresholding) |
| **pyzbar** | 1D/2D barcode decoding |
| **Ultralytics YOLOv8** | Real-time object detection |
| **OpenFoodFacts API** | Nutritional data retrieval |
| **SQLite** | Local price comparison database |
| **argparse** | Command-line interface |
| **Python 3.10+** | Core language |

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

[Your Name]

---

*Built as a Computer Vision course project demonstrating Image Processing, Feature Extraction, and Model Inference.*
