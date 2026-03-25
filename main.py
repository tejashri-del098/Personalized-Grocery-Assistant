#!/usr/bin/env python3
"""
main.py — Personalized Grocery Assistant CLI

Entry point for the command-line interface.  Orchestrates the full
pipeline: image pre-processing → barcode scanning → object detection
→ data lookup → formatted console output.

Usage examples:
    python main.py --image samples/sample_barcode.png --mode nutrition
    python main.py --image samples/sample_produce.jpg --mode price
    python main.py --image samples/sample_barcode.png --mode full
    python main.py --image path/to/photo.jpg --confidence 0.4
"""

import argparse
import sys
import os

# Ensure the project root is on sys.path so that local packages resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.barcode_scanner import scan_barcode
from vision.object_detector import detect_products
from vision.preprocessing import load_image
from data.product_mapper import map_all
from utils.formatter import (
    print_banner,
    print_detection_summary,
    print_all_products,
    print_error,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Personalized Grocery Assistant",
        description=(
            "🛒  Identify grocery products from images using barcode scanning "
            "and object detection, then fetch nutrition info and compare prices."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --image sample_cereal.jpg --mode nutrition\n"
            "  python main.py --image photo.jpg --mode price\n"
            "  python main.py --image photo.jpg --mode full --confidence 0.4\n"
        ),
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the grocery product image to analyse.",
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["nutrition", "price", "full"],
        default="full",
        help=(
            "What information to display. "
            "'nutrition' = nutritional facts only, "
            "'price' = price comparison only, "
            "'full' = both. (default: full)"
        ),
    )

    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for object detection (0–1). "
             "(default: 0.5)",
    )

    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip YOLO object detection (barcode scanning only).",
    )

    parser.add_argument(
        "--no-barcode",
        action="store_true",
        help="Skip barcode scanning (object detection only).",
    )

    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Banner ──
    print_banner()

    # ── Validate image path ──
    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        print_error(f"Image file not found: {image_path}")
        return 1

    print(f"  📷  Analysing: {os.path.basename(image_path)}")
    print(f"  📂  Mode: {args.mode}  |  Confidence: {args.confidence}")
    print()

    # ── Phase 1: Vision ──
    barcode_results = []
    detection_results = []

    # Barcode scanning
    if not args.no_barcode:
        print("  🔍  Scanning for barcodes...")
        try:
            barcode_results = scan_barcode(image_path)
        except ImportError as e:
            print(f"  ⚠  Barcode scanning unavailable: {e}")
        except Exception as e:
            print(f"  ⚠  Barcode scan error: {e}")

    # Object detection
    if not args.no_detect:
        print("  🤖  Running object detection...")
        try:
            detection_results = detect_products(
                image_path, confidence=args.confidence
            )
        except ImportError as e:
            print(f"  ⚠  Object detection unavailable: {e}")
        except Exception as e:
            print(f"  ⚠  Object detection error: {e}")

    # ── Print detection summary ──
    print_detection_summary(barcode_results, detection_results)

    # ── Phase 2: Data integration ──
    print("  📡  Fetching product data...")
    products = map_all(barcode_results, detection_results)

    # ── Phase 3: Display ──
    print_all_products(products, mode=args.mode)

    print(f"\n  ✅  Analysis complete — {len(products)} product(s) processed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
