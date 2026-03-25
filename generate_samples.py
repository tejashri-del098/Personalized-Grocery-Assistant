#!/usr/bin/env python3
"""
generate_samples.py — Generate sample test images for the Grocery Assistant.

Creates:
  1. A barcode image (EAN-13 for Nutella: 3017620422003)
  2. A simple produce-like test image (colored rectangles as stand-ins)

Run once:  python generate_samples.py
"""

import os
import sys
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def generate_barcode_image():
    """Generate a synthetic EAN-13 barcode image using OpenCV drawing.

    Encodes barcode 3017620422003 (Nutella) as a visual barcode pattern.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    if not CV2_AVAILABLE:
        print("[!] OpenCV not available — skipping barcode image generation.")
        return

    # Create a white canvas
    width, height = 400, 250
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # EAN-13 encoding tables
    barcode = "3017620422003"

    # L-code patterns (0 = white, 1 = black)
    L_PATTERNS = {
        '0': [0,0,0,1,1,0,1], '1': [0,0,1,1,0,0,1],
        '2': [0,0,1,0,0,1,1], '3': [0,1,1,1,1,0,1],
        '4': [0,1,0,0,0,1,1], '5': [0,1,1,0,0,0,1],
        '6': [0,1,0,1,1,1,1], '7': [0,1,1,1,0,1,1],
        '8': [0,1,1,0,1,1,1], '9': [0,0,0,1,0,1,1],
    }
    # R-code patterns
    R_PATTERNS = {
        '0': [1,1,1,0,0,1,0], '1': [1,1,0,0,1,1,0],
        '2': [1,1,0,1,1,0,0], '3': [1,0,0,0,0,1,0],
        '4': [1,0,1,1,1,0,0], '5': [1,0,0,1,1,1,0],
        '6': [1,0,1,0,0,0,0], '7': [1,0,0,0,1,0,0],
        '8': [1,0,0,1,0,0,0], '9': [1,1,1,0,1,0,0],
    }
    # G-code patterns
    G_PATTERNS = {
        '0': [0,1,0,0,1,1,1], '1': [0,1,1,0,0,1,1],
        '2': [0,0,1,1,0,1,1], '3': [0,1,0,0,0,0,1],
        '4': [0,0,1,1,1,0,1], '5': [0,1,1,1,0,0,1],
        '6': [0,0,0,0,1,0,1], '7': [0,0,1,0,0,0,1],
        '8': [0,0,0,1,0,0,1], '9': [0,0,1,0,1,1,1],
    }

    # Parity patterns for first digit
    PARITY = {
        '0': 'LLLLLL', '1': 'LLGLGG', '2': 'LLGGLG', '3': 'LLGGGL',
        '4': 'LGLLGG', '5': 'LGGLLG', '6': 'LGGGLL', '7': 'LGLGLG',
        '8': 'LGLGGL', '9': 'LGGLGL',
    }

    first_digit = barcode[0]
    parity = PARITY[first_digit]

    # Build bar sequence
    bars = [1, 0, 1]  # start guard

    for i in range(6):
        digit = barcode[i + 1]
        if parity[i] == 'L':
            bars.extend(L_PATTERNS[digit])
        else:
            bars.extend(G_PATTERNS[digit])

    bars.extend([0, 1, 0, 1, 0])  # center guard

    for i in range(6):
        digit = barcode[i + 7]
        bars.extend(R_PATTERNS[digit])

    bars.extend([1, 0, 1])  # end guard

    # Draw bars
    bar_width = 3
    bar_height = 150
    x_start = (width - len(bars) * bar_width) // 2
    y_start = 30

    for i, bar in enumerate(bars):
        if bar == 1:
            x = x_start + i * bar_width
            cv2.rectangle(img, (x, y_start), (x + bar_width - 1, y_start + bar_height),
                         (0, 0, 0), -1)

    # Add barcode number text below
    cv2.putText(img, barcode, (x_start + 20, y_start + bar_height + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add label
    cv2.putText(img, "Sample Barcode (Nutella)", (50, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    output_path = os.path.join(SAMPLES_DIR, "sample_barcode.png")
    cv2.imwrite(output_path, img)
    print(f"  ✓ Created: {output_path}")


def generate_produce_image():
    """Generate a simple test image with colored shapes representing produce."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    if not CV2_AVAILABLE:
        print("[!] OpenCV not available — skipping produce image generation.")
        return

    width, height = 640, 480
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # light gray bg

    # Draw "banana" — yellow ellipse
    cv2.ellipse(img, (200, 200), (120, 40), -30, 0, 360, (0, 220, 255), -1)
    cv2.putText(img, "Banana", (140, 280), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 2)

    # Draw "apple" — red circle
    cv2.circle(img, (450, 200), 70, (0, 0, 220), -1)
    cv2.circle(img, (450, 200), 70, (0, 0, 180), 2)
    # Apple stem
    cv2.line(img, (450, 130), (455, 110), (0, 100, 0), 3)
    cv2.putText(img, "Apple", (410, 300), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 2)

    # Draw "orange" — orange circle
    cv2.circle(img, (200, 400), 55, (0, 140, 255), -1)
    cv2.putText(img, "Orange", (155, 475), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 2)

    # Draw "broccoli" — green circle cluster
    for offset in [(-20, -15), (20, -15), (0, -30), (-10, 0), (10, 0)]:
        cv2.circle(img, (450 + offset[0], 390 + offset[1]), 20,
                  (0, 150, 0), -1)
    cv2.rectangle(img, (443, 400), (457, 440), (0, 120, 0), -1)
    cv2.putText(img, "Broccoli", (395, 475), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 2)

    # Title
    cv2.putText(img, "Sample Produce Image", (180, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    output_path = os.path.join(SAMPLES_DIR, "sample_produce.jpg")
    cv2.imwrite(output_path, img)
    print(f"  ✓ Created: {output_path}")


if __name__ == "__main__":
    print("\n  Generating sample images...\n")
    generate_barcode_image()
    generate_produce_image()
    print("\n  Done! Sample images are in the 'samples/' directory.\n")
