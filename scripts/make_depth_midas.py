import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IN_DIR = Path("frames_tga")
OUT_DIR = Path("depth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# MiDaS models: DPT_Large (best), DPT_Hybrid (faster)
model_type = "DPT_Hybrid"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform

files = sorted(
    glob.glob(str(IN_DIR / "*.tga")) +
    glob.glob(str(IN_DIR / "*.TGA")) +
    glob.glob(str(IN_DIR / "*.png"))
)
if not files:
    raise SystemExit(f"No frames found in {IN_DIR.resolve()}")

def read_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)  # RGB uint8

def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-8)
    return (d * 255.0).clip(0, 255).astype(np.uint8)

with torch.no_grad():
    for f in tqdm(files, desc="Depth"):
        rgb = read_rgb(f)

        inp = transform(rgb).to(device)
        pred = midas(inp)

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = pred.cpu().numpy()
        depth_u8 = depth_to_uint8(depth)

        out_path = OUT_DIR / (Path(f).stem + ".png")
        cv2.imwrite(str(out_path), depth_u8)

print(f"✅ Wrote depth maps to: {OUT_DIR.resolve()}")
