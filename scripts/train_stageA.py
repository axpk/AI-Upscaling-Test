import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Config
DATA_ROOT = Path("dataset/seq01")
RGB_DIR = DATA_ROOT / "rgb"
DEPTH_DIR = DATA_ROOT / "depth"
TARGET_DIR = DATA_ROOT / "teacher"

OUT_DIR = Path("out/stageA/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = OUT_DIR / "stageA_last.pt"

# Training
SEED = 123
EPOCHS = 10
BATCH_SIZE = 1  # 16GB VRAM
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True
GRAD_ACCUM_STEPS = 6
GRAD_LOSS_W = 0.15
RESET_TRAINING = True


# Split
VAL_EVERY_N = 20
LOG_EVERY = 50

USE_AMP = True

# Input setup: rgb(3) + depth(1) = 4 channels
IN_CH = 4
OUT_CH = 3


# Utils
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def to_float01(x_u8: np.ndarray) -> np.ndarray:
    return (x_u8.astype(np.float32) / 255.0)

def list_ids(rgb_dir: Path) -> List[str]:
    files = sorted([p.name for p in rgb_dir.glob("*.png")])
    return files


# Dataset
class FrameDataset(Dataset):
    def __init__(self, ids: List[str], augment: bool = False):
        self.ids = ids
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        rgb = load_rgb(RGB_DIR / name)          # (H,W,3)
        depth = load_gray(DEPTH_DIR / name)     # (H,W)
        tgt = load_rgb(TARGET_DIR / name)       # (H,W,3)


        if self.augment and np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])
            tgt = np.ascontiguousarray(tgt[:, ::-1, :])

        rgb_f = to_float01(rgb)                 # (H,W,3)
        depth_f = to_float01(depth)[..., None]  # (H,W,1)
        x = np.concatenate([rgb_f, depth_f], axis=-1)  # (H,W,4)
        y = to_float01(tgt)                     # (H,W,3)


        x_t = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y_t = torch.from_numpy(y).permute(2, 0, 1).contiguous()

        return x_t, y_t, name


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
    def forward(self, x):
        return self.net(x)

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
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class SmallUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base=48):
        super().__init__()
        self.in_conv = ConvBlock(in_ch, base)

        self.d1 = Down(base, base*2)    # 48 -> 96
        self.d2 = Down(base*2, base*3)  # 96 -> 144
        self.d3 = Down(base*3, base*4)  # 144 -> 192

        self.mid = ConvBlock(base*4, base*4)

        self.u3 = Up(base*4, base*4, base*3)   # 192 + 192 -> 144
        self.u2 = Up(base*3, base*3, base*2)   # 144 + 144 -> 96
        self.u1 = Up(base*2, base*2, base)     # 96 + 96 -> 48

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
        delta = self.out(x)                 # (B,3,H,W)

        delta = 0.2 * torch.tanh(delta)

        rgb_in = x_in[:, :3, :, :]          # assumes first 3 channels are RGB in [0,1]
        pred = torch.clamp(rgb_in + delta, 0.0, 1.0)
        return pred


def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def sobel_mag(img: torch.Tensor) -> torch.Tensor:
    # luminance
    gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ids = list_ids(RGB_DIR)
    if not ids:
        raise SystemExit(f"No pngs found in {RGB_DIR.resolve()}")

    # filter to those with depth + teacher
    valid = []
    for n in ids:
        if (DEPTH_DIR / n).exists() and (TARGET_DIR / n).exists():
            valid.append(n)
    ids = valid
    print("Frames:", len(ids))

    # split
    val_ids = ids[::VAL_EVERY_N]
    train_ids = [x for x in ids if x not in set(val_ids)]
    print("Train:", len(train_ids), "Val:", len(val_ids))

    train_ds = FrameDataset(train_ids, augment=True)
    val_ds = FrameDataset(val_ids, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = SmallUNet(in_ch=IN_CH, out_ch=OUT_CH, base=48).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    start_epoch = 0

    if RESET_TRAINING and CKPT_PATH.exists():
        print(f"Deleting existing checkpoint: {CKPT_PATH}")
        CKPT_PATH.unlink()

    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print("Resuming from epoch", start_epoch)

    def run_val():
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc="Val", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                    pred = model(x)
                    loss_rec = charbonnier(pred, y)
                    loss_grad = F.l1_loss(sobel_mag(pred), sobel_mag(y))
                    loss = loss_rec + GRAD_LOSS_W * loss_grad

                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    global_step = 0

    # Epochs
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        opt.zero_grad(set_to_none=True)

        for step, (x, y, _) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                pred = model(x)

                loss_rec = charbonnier(pred, y)
                loss_grad = F.l1_loss(sobel_mag(pred), sobel_mag(y))

                loss_color = F.l1_loss(pred.mean(dim=1, keepdim=True), y.mean(dim=1, keepdim=True))
                loss = (loss_rec + 0.15*loss_grad + 0.05*loss_color) / GRAD_ACCUM_STEPS


            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=f"{(loss.item() * GRAD_ACCUM_STEPS):.4f}")

        remainder = (step + 1) % GRAD_ACCUM_STEPS
        if remainder != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        val_loss = run_val()
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
        }, CKPT_PATH)
        print("Saved:", CKPT_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
