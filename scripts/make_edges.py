import os, glob
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

IN_DIR = Path("frames_tga")
OUT_DIR = Path("edges")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canny thresholds
LOW, HIGH = 100, 200

files = sorted(glob.glob(str(IN_DIR / "*.tga")) + glob.glob(str(IN_DIR / "*.TGA")))
if not files:
    raise SystemExit(f"No TGA files found in {IN_DIR.resolve()}")

def read_image_any(f: str):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    if img is not None:
        return img  # BGR

    # Pillow fallback
    try:
        im = Image.open(f).convert("RGB")
        arr = np.array(im)          # RGB
        arr = arr[:, :, ::-1].copy()  # to BGR
        return arr
    except Exception as e:
        return None

bad = []
for f in tqdm(files, desc="Edges"):
    img = read_image_any(f)
    if img is None:
        bad.append(f)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, LOW, HIGH)

    out_path = OUT_DIR / (Path(f).stem + ".png")
    cv2.imwrite(str(out_path), edges)

print(f"✅ Wrote edges for {len(files) - len(bad)} / {len(files)} frames to: {OUT_DIR.resolve()}")
if bad:
    print("⚠️ Failed to read these files:")
    for b in bad[:10]:
        print("  ", b)
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more")
