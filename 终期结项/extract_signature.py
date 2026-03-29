import argparse
from pathlib import Path

import cv2
import numpy as np


def find_signature_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Find the main handwritten signature region in a photo of paper.

    Heuristic tuned for scenes like the provided image:
    - search only the lower-middle area to avoid page edges / printed text
    - use adaptive threshold + connected components
    - merge components that look like ink strokes
    """
    h, w = gray.shape

    # Lower-middle ROI: avoids top printed text and page borders.
    rx1, rx2 = int(w * 0.25), int(w * 0.75)
    ry1, ry2 = int(h * 0.35), int(h * 0.75)
    roi = gray[ry1:ry2, rx1:rx2]

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    xs, ys, xes, yes = [], [], [], []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        # Keep only components that look like pen strokes.
        if area >= 40 and cw >= 6 and ch >= 6:
            xs.append(x)
            ys.append(y)
            xes.append(x + cw)
            yes.append(y + ch)

    if not xs:
        # Fallback: center crop
        return int(w * 0.35), int(h * 0.42), int(w * 0.65), int(h * 0.62)

    x1 = min(xs) + rx1
    y1 = min(ys) + ry1
    x2 = max(xes) + rx1
    y2 = max(yes) + ry1

    pad_x = 20
    pad_y = 15
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return x1, y1, x2, y2


def clean_binary_mask(gray_crop: np.ndarray) -> np.ndarray:
    """Create a bold, clean binary signature mask."""
    # Slight denoise while keeping edges.
    blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)

    # Adaptive threshold handles uneven lighting on paper photos.
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    # Remove tiny dust/noise.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= 18 and not (w <= 2 and h <= 2):
            cleaned[labels == i] = 255

    # Connect broken strokes and make them slightly bolder.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)

    return cleaned


def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to transparent black RGBA PNG."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    return rgba


def extract_signature(input_path: str, output_path: str) -> Path:
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = find_signature_bbox(gray)
    gray_crop = gray[y1:y2, x1:x2]

    mask = clean_binary_mask(gray_crop)
    rgba = mask_to_rgba(mask)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), rgba)
    if not ok:
        raise RuntimeError(f"Failed to write output: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bold transparent signature PNG from a photo.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output PNG path")
    args = parser.parse_args()

    out = extract_signature(args.input, args.output)
    print(f"Saved: {out}")