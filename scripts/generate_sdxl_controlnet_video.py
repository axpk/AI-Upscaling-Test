import os
import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    AutoencoderKL,
)

from config import START_FRAME, MAX_FRAMES, OUT_W, OUT_H, SEQ_NAME, FPS


# Paths
FRAMES_DIR = Path("frames_tga")
EDGES_DIR  = Path("edges")
DEPTH_DIR  = Path("depth")
OUT_DIR = Path("out/teacher_frames")

# Clear out output folder after every run
if OUT_DIR.exists():
    for f in OUT_DIR.glob("*.png"):
        f.unlink()
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


PROMPT = (
    "cinematic realism, photorealistic, real-world scene, physically based materials, "
    "realistic specular highlights, natural global illumination, balanced exposure, "
    "28mm lens, soft contrast, subtle halation, high dynamic range, "
    "true-to-life colors, fine surface texture detail, subtle film grain"
)
NEG_PROMPT = (
    "cartoon, anime, illustration, painting, CGI, stylized, plastic, waxy, "
    "overly smooth, lowres, blurry, oversharpened, noisy, deformed geometry, "
    "extra objects, text, watermark, unrealistic lighting, overexposed, "
    "underexposed, oversaturated"
)

# Diffusion settings
SEED = 1234
STEPS = 18
GUIDANCE = 5.0
STRENGTH = 0.40


# ControlNet strengths
DEPTH_SCALE = 0.95
CANNY_SCALE = 0.50


TEMPORAL_BLEND = 0.50
LOCK_COLOR_TO_FIRST = True


# Helpers
def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def resize_pil(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.BICUBIC)

def match_color(source_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """
    Simple per-channel mean/std match (fast, decent).
    """
    src = source_bgr.astype(np.float32)
    ref = reference_bgr.astype(np.float32)

    for c in range(3):
        s_mean, s_std = src[:, :, c].mean(), src[:, :, c].std() + 1e-6
        r_mean, r_std = ref[:, :, c].mean(), ref[:, :, c].std() + 1e-6
        src[:, :, c] = (src[:, :, c] - s_mean) * (r_std / s_std) + r_mean

    return np.clip(src, 0, 255).astype(np.uint8)


def load_frame_triplet(frame_path: Path):
    stem = frame_path.stem
    edge_path = EDGES_DIR / f"{stem}.png"
    depth_path = DEPTH_DIR / f"{stem}.png"
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge: {edge_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth: {depth_path}")
    return pil_rgb(frame_path), pil_rgb(edge_path), pil_rgb(depth_path)

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1].copy()  # BGR

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")

def warp_prev_output(prev_out_bgr: np.ndarray, prev_frame_bgr: np.ndarray, cur_frame_bgr: np.ndarray):
    """
    Optical-flow warping using OpenCV DIS flow + light smoothing.
    Returns (warped_prev, flow).
    """
    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray  = cv2.cvtColor(cur_frame_bgr,  cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    dis.setGradientDescentIterations(40)

    flow = dis.calc(prev_gray, cur_gray, None)

    flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (0, 0), 1.5)
    flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (0, 0), 1.5)

    h, w = prev_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)

    warped = cv2.remap(
        prev_out_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped, flow

def compute_trust_mask(prev_frame_bgr: np.ndarray, cur_frame_bgr: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Returns mask in [0,1] where 1 = trust warped previous output.
    """
    prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cur_gray  = cv2.cvtColor(cur_frame_bgr,  cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # photometric difference
    diff = np.abs(cur_gray - prev_gray)

    # motion magnitude
    mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2).astype(np.float32)

    # turn into soft weights
    w_diff = np.clip(1.0 - (diff / 0.10), 0.0, 1.0)     # tolerate ~10% brightness change
    # mag: large motion less reliable
    w_mag  = np.clip(1.0 - (mag / 12.0), 0.0, 1.0)      # tolerate up to ~12px motion

    mask = w_diff * w_mag

    # blur mask for smooth transitions
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)

    return mask  # (H,W) float in [0,1]


# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print("Device:", device, "dtype:", dtype)
MODEL_VARIANT = "fp16" if dtype == torch.float16 else None

try:
    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=dtype,
        use_safetensors=True
    )
    controlnet_canny = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=dtype,
        use_safetensors=True
    )

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype,
        use_safetensors=True
    )

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "RunDiffusion/Juggernaut-XL-v9",
        controlnet=[controlnet_depth, controlnet_canny],
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant=MODEL_VARIANT,
    )
except OSError as e:
    msg = str(e)
    if "model.safetensors" in msg or "pytorch_model.bin" in msg:
        raise SystemExit(
            "Missing safetensors files for one or more model components. "
            "Please ensure the model repo provides .safetensors for all parts "
            "(UNet, VAE, and text encoders), or download a safetensors-only "
            "variant of the checkpoint."
        ) from e
    raise
pipe = pipe.to(device)
pipe.enable_attention_slicing()
if device == "cuda":
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

generator = torch.Generator(device=device).manual_seed(SEED)

# Gather frames
frame_files = sorted(
    glob.glob(str(FRAMES_DIR / "*.png")) +
    glob.glob(str(FRAMES_DIR / "*.tga")) +
    glob.glob(str(FRAMES_DIR / "*.TGA"))
)
if not frame_files:
    raise SystemExit(f"No frames found in {FRAMES_DIR.resolve()}")

# Debug
frame_files = frame_files[START_FRAME:]
if MAX_FRAMES is not None:
    frame_files = frame_files[:MAX_FRAMES]

prev_out_bgr = None
prev_frame_bgr = None

for idx, f in enumerate(tqdm(frame_files, desc="Generating")):
    frame_path = Path(f)
    out_path = OUT_DIR / (frame_path.stem + ".png")
    if out_path.exists():
        continue

    frame_img, edge_img, depth_img = load_frame_triplet(frame_path)

    frame_img = resize_pil(frame_img, OUT_W, OUT_H)
    edge_img  = resize_pil(edge_img,  OUT_W, OUT_H)
    depth_img = resize_pil(depth_img, OUT_W, OUT_H)

    if TEMPORAL_BLEND > 0 and prev_out_bgr is not None and prev_frame_bgr is not None:
        cur_frame_bgr = pil_to_bgr(frame_img)

        warped_prev, flow = warp_prev_output(prev_out_bgr, prev_frame_bgr, cur_frame_bgr)
        trust = compute_trust_mask(prev_frame_bgr, cur_frame_bgr, flow)  # (H,W) in [0,1]

        alpha = (TEMPORAL_BLEND * trust)[..., None]  # (H,W,1)

        blended_init = (1.0 - alpha) * cur_frame_bgr + alpha * warped_prev
        blended_init = np.clip(blended_init, 0, 255).astype(np.uint8)

        init_img = bgr_to_pil(blended_init)
    else:
        init_img = frame_img

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=init_img,
        control_image=[depth_img, edge_img],
        controlnet_conditioning_scale=[DEPTH_SCALE, CANNY_SCALE],
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        strength=STRENGTH,
        generator=generator,
    ).images[0]

    result_bgr = pil_to_bgr(result)

    if idx == 0:
        first_out_bgr = result_bgr.copy()
    elif LOCK_COLOR_TO_FIRST:
        result_bgr = match_color(result_bgr, first_out_bgr)
        result = bgr_to_pil(result_bgr)


    result.save(out_path)

    prev_out_bgr = result_bgr
    prev_frame_bgr = pil_to_bgr(frame_img)

print("Done. Outputs in:", OUT_DIR.resolve())
