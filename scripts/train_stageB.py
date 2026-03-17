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
from torch.utils.data import Dataset, DataLoader


# Config
DATA_ROOT = Path("dataset/seq01")
RGB_DIR = DATA_ROOT / "rgb"
DEPTH_DIR = DATA_ROOT / "depth"
TEACHER_DIR = DATA_ROOT / "teacher"

OUT_DIR = Path("out/stageB/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT_DIR / "stageB_last.pt"

# Stage B
CLIP_LEN = 4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 6
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
PIN_MEMORY = True
USE_AMP = True

# Loss weights
GRAD_LOSS_W = 0.15
TEMP_LOSS_W = 0.20
TEMP_GRAD_LOSS_W = 0.08
TEMP_LF_LOSS_W = 0.06
STATS_LOSS_W = 0.10

P_USE_STUDENT_PREV = 0.50

# must match Stage A
DELTA_SCALE = 0.6


HISTORY = 2

IN_CH = 3 + 1 + HISTORY * (3 + 1)
OUT_CH = 3

RESET_TRAINING = True  # delete existing checkpoints in OUT_DIR


# helpers
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
    return flow  # (H,W,2)

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

def collate_clip(batch):
    if len(batch) == 1:
        return batch[0]
    return batch

def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def sobel_mag(img: torch.Tensor) -> torch.Tensor:
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

def stats_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pred_mu = pred.mean(dim=(2,3))
    tgt_mu  = tgt.mean(dim=(2,3))
    pred_sd = pred.std(dim=(2,3), unbiased=False)
    tgt_sd  = tgt.std(dim=(2,3), unbiased=False)
    return F.l1_loss(pred_mu, tgt_mu) + F.l1_loss(pred_sd, tgt_sd)

def lowfreq(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=k, stride=k)


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

        rgb_in = x_in[:, :3, :, :]  # first 3 channels are rgb_t
        return torch.clamp(rgb_in + delta, 0.0, 1.0)


class ClipDataset(Dataset):
    def __init__(self, names: List[str], clip_len: int):
        self.names = names
        self.clip_len = clip_len

    def __len__(self):
        return len(self.names) - self.clip_len

    def __getitem__(self, idx: int):
        return self.names[idx: idx + self.clip_len]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AMP_ENABLED = (USE_AMP and device == "cuda")

    print("Device:", device)
    print("AMP enabled:", AMP_ENABLED)

    if device == "cuda":
        torch.cuda.empty_cache()

    if RESET_TRAINING and CKPT_PATH.exists():
        print(f"Deleting existing checkpoint: {CKPT_PATH}")
        CKPT_PATH.unlink()

    names = list_names_sorted(RGB_DIR)
    names = [n for n in names if (DEPTH_DIR / n).exists() and (TEACHER_DIR / n).exists()]
    if len(names) < CLIP_LEN + 1:
        raise SystemExit("Not enough frames for clip training.")

    ds = ClipDataset(names, CLIP_LEN)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_clip
    )


    model = SmallUNet(in_ch=IN_CH, out_ch=OUT_CH, base=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    AMP_ENABLED = (USE_AMP and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    print("AMP enabled:", AMP_ENABLED, "device:", device)

    DID_SCALE = False


    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"StageB Epoch {epoch+1}/{EPOCHS}")

        opt.zero_grad(set_to_none=True)
        last_step = -1

        for step, clip_names in enumerate(pbar):
            last_step = step

            clip = clip_names

            name0 = clip[0]
            rgb0 = load_rgb(RGB_DIR / name0)
            pred_prev_bgr = rgb0[:, :, ::-1].copy()
            pred_prev2_bgr = pred_prev_bgr.copy()

            loss_total = 0.0

            for t in range(1, len(clip)):
                cur_name  = clip[t]

                rgb_cur  = load_rgb(RGB_DIR / cur_name)
                depth_cur = load_gray(DEPTH_DIR / cur_name)
                teacher_cur = load_rgb(TEACHER_DIR / cur_name)

                history_warped = []
                history_masks = []

                for h in range(1, HISTORY + 1):
                    idx_h = t - h
                    if idx_h < 0:
                        warped_prev_bgr = rgb_cur[:, :, ::-1].copy()
                        mask = np.zeros(rgb_cur.shape[:2], dtype=np.float32)
                    else:
                        prev_name = clip[idx_h]
                        rgb_prev = load_rgb(RGB_DIR / prev_name)

                        use_student = (np.random.rand() < P_USE_STUDENT_PREV)
                        if use_student:
                            prev_for_warp = pred_prev_bgr if h == 1 else pred_prev2_bgr
                        else:
                            prev_for_warp = load_rgb(TEACHER_DIR / prev_name)[:, :, ::-1].copy()

                        warped_prev_bgr, mask = build_warp_and_mask(rgb_prev, rgb_cur, prev_for_warp)

                    history_warped.append(warped_prev_bgr)
                    history_masks.append(mask)

                rgb_cur_f = to_float01_u8(rgb_cur)                       # (H,W,3)
                depth_f   = to_float01_u8(depth_cur)[..., None]          # (H,W,1)

                parts = [rgb_cur_f, depth_f]
                for warped_prev_bgr, mask in zip(history_warped, history_masks):
                    warped_f = warped_prev_bgr[:, :, ::-1].astype(np.float32) / 255.0  # (H,W,3) RGB
                    mask_f = mask[..., None]                                           # (H,W,1)
                    parts.extend([warped_f, mask_f])

                x_np = np.concatenate(parts, axis=-1).astype(np.float32)
                y_np = to_float01_u8(teacher_cur).astype(np.float32)                                       # (H,W,3)

                x = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
                y = torch.from_numpy(y_np).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)

                mask_t = torch.from_numpy(history_masks[0]).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
                warped_prev_t = torch.from_numpy(
                    history_warped[0][:, :, ::-1].astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    pred = model(x)  # (1,3,H,W)

                    loss_rec = charbonnier(pred, y)
                    loss_grad = F.l1_loss(sobel_mag(pred), sobel_mag(y))
                    loss_stats = stats_loss(pred, y)

                    loss_temp = (mask_t * torch.abs(pred - warped_prev_t)).mean()
                    loss_temp_grad = (mask_t * torch.abs(sobel_mag(pred) - sobel_mag(warped_prev_t))).mean()

                    pred_lf = lowfreq(pred)
                    warp_lf = lowfreq(warped_prev_t)
                    mask_lf = lowfreq(mask_t)
                    loss_temp_lf = (mask_lf * torch.abs(pred_lf - warp_lf)).mean()

                    loss = (
                        loss_rec +
                        GRAD_LOSS_W * loss_grad +
                        STATS_LOSS_W * loss_stats +
                        TEMP_LOSS_W * loss_temp +
                        TEMP_GRAD_LOSS_W * loss_temp_grad +
                        TEMP_LF_LOSS_W * loss_temp_lf
                    )
                    loss = loss / GRAD_ACCUM_STEPS

                if AMP_ENABLED:
                    scaler.scale(loss).backward()
                    DID_SCALE = True
                else:
                    loss.backward()

                loss_total += float(loss.item() * GRAD_ACCUM_STEPS)

                pred_prev = (
                    pred[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
                ).astype(np.uint8)
                pred_prev2_bgr = pred_prev_bgr
                pred_prev_bgr = pred_prev[:, :, ::-1].copy()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                if AMP_ENABLED:
                    if DID_SCALE:
                        scaler.step(opt)
                        scaler.update()
                    DID_SCALE = False
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)


            pbar.set_postfix(loss=f"{loss_total:.4f}")

        if last_step >= 0:
            remainder = (step + 1) % GRAD_ACCUM_STEPS
            if remainder != 0:
                if AMP_ENABLED:
                    if DID_SCALE:
                        scaler.step(opt)
                        scaler.update()
                    DID_SCALE = False
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)


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
