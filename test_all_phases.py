#!/usr/bin/env python3
"""
test_all_phases.py — Comprehensive verification script.

Tests all 4 phases of the Personalized Grocery Assistant to ensure
every module works correctly end-to-end.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

results = []


def test(name, func):
    try:
        func()
        print(f"  {PASS}  {name}")
        results.append(("PASS", name))
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results.append(("FAIL", name))


def test_warn(name, func):
    """Test that may warn instead of fail (e.g. network-dependent)."""
    try:
        func()
        print(f"  {PASS}  {name}")
        results.append(("PASS", name))
    except Exception as e:
        print(f"  {WARN}  {name}: {e}")
        results.append(("WARN", name))


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 1: Core Vision & Detection")
print("=" * 60)

# 1a. Preprocessing
def test_preprocessing():
    from vision.preprocessing import (
        load_image, to_grayscale, apply_clahe,
        reduce_noise, adaptive_threshold, sharpen,
        enhance_image, enhance_for_detection
    )
    img = load_image("samples/sample_barcode.png")
    assert img is not None and img.shape == (250, 400, 3), f"Bad shape: {img.shape}"

    gray = to_grayscale(img)
    assert len(gray.shape) == 2, "Grayscale should be 2D"

    clahe = apply_clahe(gray)
    assert clahe.shape == gray.shape, "CLAHE shape mismatch"

    denoised = reduce_noise(clahe)
    assert denoised.shape == gray.shape, "Denoise shape mismatch"

    thresh = adaptive_threshold(denoised)
    assert thresh.shape == gray.shape, "Threshold shape mismatch"

    sharp = sharpen(img)
    assert sharp.shape == img.shape, "Sharpen shape mismatch"

    enh = enhance_image("samples/sample_barcode.png", for_barcode=True)
    assert enh is not None, "enhance_image returned None"

    det_img = enhance_for_detection("samples/sample_barcode.png")
    assert det_img.shape == img.shape, "enhance_for_detection shape mismatch"

test("1a. Preprocessing — all 8 functions", test_preprocessing)


# 1b. Barcode scanner
def test_barcode_scanner():
    from vision.barcode_scanner import scan_barcode, scan_barcode_from_path
    results = scan_barcode_from_path("samples/sample_barcode.png")
    assert len(results) > 0, "No barcodes detected!"
    bc = results[0]
    assert bc["data"] == "3017620422003", f"Wrong barcode: {bc['data']}"
    assert bc["type"] in ("EAN13", "EAN8", "CODE128"), f"Unexpected type: {bc['type']}"
    assert bc["category"] in ("1D", "2D"), f"Bad category: {bc['category']}"
    assert len(bc["rect"]) == 4, "Rect should have 4 elements"
    print(f"        Decoded: [{bc['type']}] {bc['data']}")

test("1b. Barcode Scanner — EAN-13 decode", test_barcode_scanner)


# 1c. Object detector
def test_object_detector():
    from vision.object_detector import detect_products, GROCERY_CLASSES
    assert len(GROCERY_CLASSES) == 17, f"Expected 17 grocery classes, got {len(GROCERY_CLASSES)}"
    # Run detection (may or may not detect objects in synthetic image)
    detections = detect_products("samples/sample_produce.jpg", confidence=0.3)
    assert isinstance(detections, list), "Should return a list"
    print(f"        Grocery classes: {len(GROCERY_CLASSES)}")
    print(f"        Detections on synthetic image: {len(detections)}")
    if detections:
        for d in detections:
            print(f"        -> {d['label']} ({d['confidence']*100:.1f}%)")

test("1c. Object Detector — YOLOv8 load + inference", test_object_detector)


# 1d. Error handling
def test_preprocessing_errors():
    from vision.preprocessing import load_image
    try:
        load_image("nonexistent_file.jpg")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # expected

test("1d. Error handling — FileNotFoundError", test_preprocessing_errors)


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 2: Data Integration & API Logic")
print("=" * 60)

# 2a. Price database
def test_price_db():
    from data.price_db import init_db, get_price_comparison, get_all_products, get_cheapest_store
    init_db()
    products = get_all_products()
    assert len(products) == 15, f"Expected 15 products, got {len(products)}"
    print(f"        Products seeded: {len(products)}")

    prices = get_price_comparison("banana")
    assert len(prices) == 3, f"Expected 3 store prices, got {len(prices)}"
    for p in prices:
        assert "store" in p and "price" in p, "Missing keys in price row"
    price_strs = []
    for p in prices:
        store = p["store"]
        price = p["price"]
        price_strs.append(f"{store}=${price:.2f}")
    print(f"        Banana prices: {', '.join(price_strs)}")

    cheapest = get_cheapest_store("banana")
    assert cheapest is not None, "Cheapest store should not be None"
    print(f"        Cheapest: {cheapest['store']} @ ${cheapest['price']:.2f}")

test("2a. Price Database — init, query, cheapest", test_price_db)


# 2b. Nutrition API
def test_nutrition_api():
    from data.nutrition_api import fetch_nutrition
    data = fetch_nutrition("3017620422003")  # Nutella
    assert data is not None, "Nutrition data should not be None"
    assert data["name"] != "Unknown Product", f"Bad product name: {data['name']}"
    assert data["barcode"] == "3017620422003"
    assert data["nutriscore"] in ("A", "B", "C", "D", "E", "N/A")
    assert "nutriments" in data
    print(f"        Product: {data['name']} (brand: {data['brand']})")
    print(f"        Nutri-Score: {data['nutriscore']}")
    print(f"        Energy: {data['nutriments']['energy_kcal']} kcal/100g")

test_warn("2b. Nutrition API — OpenFoodFacts lookup", test_nutrition_api)


# 2c. Product mapper
def test_product_mapper():
    from data.product_mapper import map_barcode, map_detection, map_all

    # Map barcode
    bc_product = map_barcode("3017620422003")
    assert bc_product["source"] == "barcode"
    assert bc_product["barcode"] == "3017620422003"
    print(f"        Barcode mapped: {bc_product['name']}")

    # Map detection label
    det = {"label": "banana", "confidence": 0.95}
    det_product = map_detection(det)
    assert det_product["source"] == "detection"
    assert len(det_product["prices"]) > 0, "Should find banana prices"
    print(f"        Detection mapped: {det_product['name']} ({len(det_product['prices'])} stores)")

    # Map all
    barcodes = [{"data": "3017620422003", "type": "EAN13"}]
    detections = [{"label": "apple", "confidence": 0.9}]
    all_products = map_all(barcodes, detections)
    assert len(all_products) >= 2, f"Expected >= 2 products, got {len(all_products)}"
    print(f"        map_all: {len(all_products)} products mapped")

test_warn("2c. Product Mapper — barcode + detection mapping", test_product_mapper)


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 3: CLI Development")
print("=" * 60)

# 3a. Argparse
def test_argparse():
    from main import build_parser
    parser = build_parser()

    # Test valid args
    args = parser.parse_args(["--image", "test.jpg", "--mode", "nutrition", "--confidence", "0.4"])
    assert args.image == "test.jpg"
    assert args.mode == "nutrition"
    assert args.confidence == 0.4
    assert args.no_detect is False
    assert args.no_barcode is False

    # Test defaults
    args2 = parser.parse_args(["--image", "test.jpg"])
    assert args2.mode == "full"
    assert args2.confidence == 0.5

    # Test flags
    args3 = parser.parse_args(["--image", "test.jpg", "--no-detect", "--no-barcode"])
    assert args3.no_detect is True
    assert args3.no_barcode is True

    print(f"        Modes: nutrition, price, full")
    print(f"        Default confidence: {args2.confidence}")
    print(f"        Flags: --no-detect, --no-barcode")

test("3a. Argparse — argument parsing & defaults", test_argparse)


# 3b. Formatter
def test_formatter():
    from utils.formatter import (
        print_banner, print_detection_summary, print_nutrition,
        print_price_comparison, print_all_products, print_error
    )
    # Just verify they don't crash with sample data
    print_banner()

    print_detection_summary(
        [{"type": "EAN13", "data": "3017620422003"}],
        [{"label": "banana", "confidence": 0.95}]
    )

    product = {
        "name": "Test Product",
        "nutrition": {
            "name": "Test", "brand": "TestBrand", "barcode": "123",
            "nutriscore": "B", "allergens": "None listed",
            "ingredients": "Test ingredients",
            "nutriments": {
                "energy_kcal": 200, "fat_g": 10, "saturated_fat_g": 3,
                "sugars_g": 15, "salt_g": 0.5, "proteins_g": 8, "fiber_g": 2
            }
        },
        "prices": [
            {"store": "FreshMart", "price": 2.49, "currency": "USD", "product_name": "Test"},
            {"store": "GreenGrocer", "price": 2.29, "currency": "USD", "product_name": "Test"},
            {"store": "MegaMart", "price": 2.69, "currency": "USD", "product_name": "Test"},
        ]
    }
    print_nutrition(product)
    print_price_comparison(product)

test("3b. Formatter — all output functions", test_formatter)


# 3c. Requirements
def test_requirements():
    with open("requirements.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    expected = ["opencv-python", "numpy", "Pillow", "pyzbar", "ultralytics", "requests", "tabulate"]
    for pkg in expected:
        found = any(pkg.lower() in l.lower() for l in lines)
        assert found, f"Missing package: {pkg}"
    print(f"        Packages listed: {len(lines)}")
    print(f"        All required packages present: {', '.join(expected)}")

test("3c. Requirements.txt — all dependencies listed", test_requirements)


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PHASE 4: Documentation & Packaging")
print("=" * 60)

# 4a. README
def test_readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    checks = {
        "Project title": "Personalized Grocery Assistant" in content,
        "How to Run section": "How to Run" in content,
        "Clone instructions": "git clone" in content,
        "Venv instructions": "venv" in content,
        "pip install": "pip install" in content,
        "Usage examples": "python main.py" in content,
        "Project structure": "Project Structure" in content,
        "CLI arguments table": "--image" in content and "--mode" in content,
    }
    for check, passed in checks.items():
        assert passed, f"README missing: {check}"
    print(f"        All {len(checks)} README checks passed")

test("4a. README.md — completeness check", test_readme)


# 4b. Project report
def test_report():
    with open("docs/PROJECT_REPORT.md", encoding="utf-8") as f:
        content = f.read()
    checks = {
        "Image Processing": "Image Processing" in content,
        "Feature Extraction": "Feature Extraction" in content,
        "Model Inference": "Model Inference" in content,
        "Architecture": "Architecture" in content,
        "Technology Stack": "Technology" in content,
        "CLAHE": "CLAHE" in content,
        "YOLOv8": "YOLOv8" in content,
        "OpenFoodFacts": "OpenFoodFacts" in content,
    }
    for check, passed in checks.items():
        assert passed, f"Report missing: {check}"
    print(f"        All {len(checks)} report coverage checks passed")

test("4b. PROJECT_REPORT.md — syllabus coverage", test_report)


# 4c. Gitignore
def test_gitignore():
    with open(".gitignore") as f:
        content = f.read()
    assert "venv" in content, ".gitignore missing venv"
    assert "__pycache__" in content, ".gitignore missing pycache"
    assert "*.db" in content, ".gitignore missing db files"
    assert "*.pt" in content, ".gitignore missing model weights"

test("4c. .gitignore — essential exclusions", test_gitignore)


# 4d. Project structure
def test_project_structure():
    required = [
        "main.py", "requirements.txt", "README.md", ".gitignore",
        "generate_samples.py",
        "vision/__init__.py", "vision/preprocessing.py",
        "vision/barcode_scanner.py", "vision/object_detector.py",
        "data/__init__.py", "data/nutrition_api.py",
        "data/price_db.py", "data/product_mapper.py",
        "utils/__init__.py", "utils/formatter.py",
        "docs/PROJECT_REPORT.md",
        "samples/sample_barcode.png", "samples/sample_produce.jpg",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    assert len(missing) == 0, f"Missing files: {missing}"
    print(f"        All {len(required)} required files present")

test("4d. Project structure — all files exist", test_project_structure)


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)

passes = sum(1 for s, _ in results if s == "PASS")
warns = sum(1 for s, _ in results if s == "WARN")
fails = sum(1 for s, _ in results if s == "FAIL")
total = len(results)

print(f"\n  Total: {total} tests")
print(f"  {PASS}: {passes}")
if warns:
    print(f"  {WARN}: {warns} (network-dependent)")
if fails:
    print(f"  {FAIL}: {fails}")
print()

if fails == 0:
    print(f"  \033[92m\033[1m{'=' * 50}")
    print(f"  ALL PHASES VERIFIED SUCCESSFULLY!")
    print(f"  {'=' * 50}\033[0m\n")
else:
    print(f"  \033[91m\033[1m{fails} test(s) failed — see above.\033[0m\n")

sys.exit(1 if fails > 0 else 0)
