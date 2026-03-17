import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from config import START_FRAME, MAX_FRAMES, OUT_W, OUT_H

# ---- inputs ----
FRAMES_DIR = Path("frames_tga")
DEPTH_DIR  = Path("depth")
EDGES_DIR  = Path("edges")
TEACHER_DIR = Path("out/teacher_frames") # diffusion outputs

# ---- output ----
OUT_ROOT = Path("dataset/seq01")
OUT_RGB = OUT_ROOT / "rgb"
OUT_DEPTH = OUT_ROOT / "depth"
OUT_EDGES = OUT_ROOT / "edges"
OUT_TEACHER = OUT_ROOT / "teacher"
CLEAR_OUTPUT = True



def clear_dir(d: Path):
    if d.exists():
        for f in d.glob("*.png"):
            f.unlink()

def ensure_dirs():
    dirs = [OUT_RGB, OUT_DEPTH, OUT_EDGES, OUT_TEACHER]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    if CLEAR_OUTPUT:
        print("Clearing existing dataset folders...")
        for d in dirs:
            clear_dir(d)

def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def load_gray(p: Path) -> Image.Image:
    return Image.open(p).convert("L")

def resize(img: Image.Image) -> Image.Image:
    return img.resize((OUT_W, OUT_H), Image.BICUBIC)

def main():
    ensure_dirs()

    frame_files = sorted(
        glob.glob(str(FRAMES_DIR / "*.png")) +
        glob.glob(str(FRAMES_DIR / "*.tga")) +
        glob.glob(str(FRAMES_DIR / "*.TGA"))
    )
    if not frame_files:
        raise SystemExit(f"No frames found in {FRAMES_DIR.resolve()}")

    frame_files = frame_files[START_FRAME:]
    if MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]

    # Only keep frames that also have teacher output
    valid = []
    for f in frame_files:
        stem = Path(f).stem
        if (TEACHER_DIR / f"{stem}.png").exists():
            valid.append(Path(f))
    if not valid:
        raise SystemExit("No overlapping (frame, teacher) pairs found. Check your folders.")

    for i, frame_path in enumerate(tqdm(valid, desc="Export")):
        stem = frame_path.stem
        depth_path = DEPTH_DIR / f"{stem}.png"
        edges_path = EDGES_DIR / f"{stem}.png"
        teacher_path = TEACHER_DIR / f"{stem}.png"

        if not depth_path.exists() or not edges_path.exists() or not teacher_path.exists():
            continue

        idx = i + 1
        name = f"{idx:06d}.png"

        rgb = resize(load_rgb(frame_path))
        depth = resize(load_gray(depth_path))
        edges = resize(load_gray(edges_path))
        teacher = resize(load_rgb(teacher_path))

        rgb.save(OUT_RGB / name)
        depth.save(OUT_DEPTH / name)
        edges.save(OUT_EDGES / name)
        teacher.save(OUT_TEACHER / name)

    print("✅ Exported to:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
