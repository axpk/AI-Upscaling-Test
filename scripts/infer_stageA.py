import glob
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match training
IN_CH = 4
OUT_CH = 3

DATA_ROOT = Path("dataset/seq01")
RGB_DIR = DATA_ROOT / "rgb"
DEPTH_DIR = DATA_ROOT / "depth"

CKPT_PATH = Path("out/stageA/model/stageA_last.pt")
OUT_DIR = Path("out/stageA/frames")

# Clear out output folder after every run
if OUT_DIR.exists():
    for f in OUT_DIR.glob("*.png"):
        f.unlink()
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


USE_AMP = True

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
    def __init__(self, in_ch=4, out_ch=3, base=48):
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
        delta = self.out(x)                 # (B,3,H,W)

        delta = 0.2 * torch.tanh(delta)

        rgb_in = x_in[:, :3, :, :]          # assumes first 3 channels are RGB in [0,1]
        pred = torch.clamp(rgb_in + delta, 0.0, 1.0)
        return pred

def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

def load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda") else torch.float32

    model = SmallUNet(in_ch=IN_CH, out_ch=OUT_CH, base=48).to(device)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    names = sorted([Path(p).name for p in glob.glob(str(RGB_DIR / "*.png"))])
    if not names:
        raise SystemExit("No rgb frames found.")

    with torch.no_grad():
        for n in tqdm(names, desc="Infer"):
            rgb = load_rgb(RGB_DIR / n).astype(np.float32) / 255.0
            depth = (load_gray(DEPTH_DIR / n).astype(np.float32) / 255.0)[..., None]
            x = np.concatenate([rgb, depth], axis=-1)  # (H,W,4)
            x_t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                pred = model(x_t)[0].permute(1,2,0).detach().cpu().numpy()

            out = (pred * 255.0).clip(0,255).astype(np.uint8)
            Image.fromarray(out).save(OUT_DIR / n)

    print("✅ Wrote:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
