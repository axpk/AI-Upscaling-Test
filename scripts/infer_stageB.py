import glob
from pathlib import Path
from typing import List

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# Config
DATA_ROOT = Path("dataset/seq01")
RGB_DIR = DATA_ROOT / "rgb"
DEPTH_DIR = DATA_ROOT / "depth"

CKPT_PATH = Path("out/stageB/model/stageB_last.pt")
OUT_DIR = Path("out/stageB/frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Debug slicing (set to None for full)
START_FRAME = 0
MAX_FRAMES = None

USE_AMP = True
DELTA_SCALE = 0.6   # MUST match training
HISTORY = 2
IN_CH = 3 + 1 + HISTORY * (3 + 1)
OUT_CH = 3


# IO helpers
def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def to_float01_u8(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0

def list_names_sorted(dirp: Path) -> List[str]:
    return sorted([Path(p).name for p in glob.glob(str(dirp / "*.png"))])


def calc_flow(prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray  = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    dis.setGradientDescentIterations(40)

    flow = dis.calc(prev_gray, cur_gray, None)
    flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (0, 0), 1.5)
    flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (0, 0), 1.5)
    return flow

def warp_bgr(img_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(
        img_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped

def trust_mask(prev_bgr: np.ndarray, cur_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cur_gray  = cv2.cvtColor(cur_bgr,  cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    diff = np.abs(cur_gray - prev_gray)
    mag  = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2).astype(np.float32)

    w_diff = np.clip(1.0 - (diff / 0.10), 0.0, 1.0)
    w_mag  = np.clip(1.0 - (mag / 12.0), 0.0, 1.0)

    mask = w_diff * w_mag
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
    return mask.astype(np.float32)  # (H,W)

def flow_consistency_mask(flow_fw: np.ndarray, flow_bw: np.ndarray) -> np.ndarray:
    h, w = flow_fw.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_fw[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow_fw[:, :, 1]).astype(np.float32)

    flow_bw_at_fw_x = cv2.remap(flow_bw[:, :, 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flow_bw_at_fw_y = cv2.remap(flow_bw[:, :, 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flow_bw_at_fw = np.stack([flow_bw_at_fw_x, flow_bw_at_fw_y], axis=-1)

    inconsistency = np.sqrt((flow_fw[:, :, 0] + flow_bw_at_fw[:, :, 0])**2 +
                            (flow_fw[:, :, 1] + flow_bw_at_fw[:, :, 1])**2)
    mask = np.clip(1.0 - (inconsistency / 1.5), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    return mask.astype(np.float32)

def build_warp_and_mask(prev_rgb: np.ndarray, cur_rgb: np.ndarray, prev_for_warp_bgr: np.ndarray):
    flow_fw = calc_flow(prev_rgb[:, :, ::-1], cur_rgb[:, :, ::-1])
    flow_bw = calc_flow(cur_rgb[:, :, ::-1], prev_rgb[:, :, ::-1])

    warped_prev_bgr = warp_bgr(prev_for_warp_bgr, flow_fw)
    mask_trust = trust_mask(prev_rgb[:, :, ::-1], cur_rgb[:, :, ::-1], flow_fw)
    mask_occ = flow_consistency_mask(flow_fw, flow_bw)
    mask = (mask_trust * mask_occ).astype(np.float32)
    return warped_prev_bgr, mask


# (must match training)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
    def forward(self, x):
        h = self.block(x)
        d = self.down(h)
        return h, d

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))

class SmallUNet(nn.Module):
    def __init__(self, in_ch=IN_CH, out_ch=3, base=48):
        super().__init__()
        self.in_conv = ConvBlock(in_ch, base)
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*3)
        self.d3 = Down(base*3, base*4)
        self.mid = ConvBlock(base*4, base*4)
        self.u3 = Up(base*4, base*4, base*3)
        self.u2 = Up(base*3, base*3, base*2)
        self.u1 = Up(base*2, base*2, base)
        self.out = nn.Conv2d(base + base, out_ch, 1)

    def forward(self, x_in):
        s0 = self.in_conv(x_in)
        s1, d1 = self.d1(s0)
        s2, d2 = self.d2(d1)
        s3, d3 = self.d3(d2)
        m = self.mid(d3)
        x = self.u3(m, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = torch.cat([x, s0], dim=1)

        delta = self.out(x)
        delta = DELTA_SCALE * torch.tanh(delta)

        rgb_in = x_in[:, :3, :, :]
        return torch.clamp(rgb_in + delta, 0.0, 1.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AMP_ENABLED = (USE_AMP and device == "cuda")
    print("Device:", device, "AMP:", AMP_ENABLED)

    # Load checkpoint
    if not CKPT_PATH.exists():
        raise SystemExit(f"Missing checkpoint: {CKPT_PATH}")

    model = SmallUNet(in_ch=IN_CH, out_ch=OUT_CH, base=48).to(device)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    names = list_names_sorted(RGB_DIR)
    names = [n for n in names if (DEPTH_DIR / n).exists()]
    if not names:
        raise SystemExit("No frames found.")

    names = names[START_FRAME:]
    if MAX_FRAMES is not None:
        names = names[:MAX_FRAMES]

    first = names[0]
    rgb_prev = load_rgb(RGB_DIR / first)
    pred_prev_bgr = rgb_prev[:, :, ::-1].copy()  # BGR
    pred_prev2_bgr = pred_prev_bgr.copy()

    Image.fromarray(rgb_prev).save(OUT_DIR / first)

    with torch.no_grad():
        for i in tqdm(range(1, len(names)), desc="Infer StageB"):
            cur_name  = names[i]

            rgb_cur  = load_rgb(RGB_DIR / cur_name)
            depth_cur = load_gray(DEPTH_DIR / cur_name)

            rgb_cur_f = to_float01_u8(rgb_cur)
            depth_f   = to_float01_u8(depth_cur)[..., None]
            parts = [rgb_cur_f, depth_f]

            for h in range(1, HISTORY + 1):
                idx_h = i - h
                if idx_h < 0:
                    warped_prev_bgr = rgb_cur[:, :, ::-1].copy()
                    mask = np.zeros(rgb_cur.shape[:2], dtype=np.float32)
                else:
                    prev_name = names[idx_h]
                    rgb_prev = load_rgb(RGB_DIR / prev_name)
                    prev_for_warp = pred_prev_bgr if h == 1 else pred_prev2_bgr
                    warped_prev_bgr, mask = build_warp_and_mask(rgb_prev, rgb_cur, prev_for_warp)

                warped_f = warped_prev_bgr[:, :, ::-1].astype(np.float32) / 255.0
                mask_f = mask[..., None]
                parts.extend([warped_f, mask_f])

            x_np = np.concatenate(parts, axis=-1).astype(np.float32)
            x = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                pred = model(x)[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

            out = (pred * 255.0).astype(np.uint8)
            Image.fromarray(out).save(OUT_DIR / cur_name)

            pred_prev2_bgr = pred_prev_bgr
            pred_prev_bgr = out[:, :, ::-1].copy()

    print("✅ Wrote:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
