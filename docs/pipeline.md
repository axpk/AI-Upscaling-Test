# Pipeline Overview

This project takes raw Half‑Life 2 demo frames and distills them into a fast student model
that produces photorealistic frames with temporal stability.

## Inputs

- `frames_tga/` — extracted HL2 frames (`.tga` or `.png`)

## Derived guides

- `edges/` — Canny edges per frame (`scripts/make_edges.py`)
- `depth/` — MiDaS depth per frame (`scripts/make_depth_midas.py`)

## Teacher (slow, high-quality)

- `scripts/generate_sdxl_controlnet_video.py`
- Output: `out/` (photoreal teacher frames)

Optional: encode a teacher video

- `./make_video.sh`

## Dataset export

Creates normalized naming across inputs and teacher outputs.

- `scripts/export_dataset.py`
- Output: `dataset/seq01/{rgb,depth,edges,teacher}/000001.png ...`

## Student Stage A (single-frame)

Fast U‑Net model that predicts a residual over RGB using depth as extra input.

- Train: `scripts/train_stageA.py`
- Output: `student_stageA/stageA_last.pt`

Inference:

- `scripts/infer_stageA.py` → `student_out/`
- `./make_student_video.sh`

## Student Stage B (temporal)

Temporal student model that uses warped previous outputs + trust/occlusion masks.

- Train: `scripts/train_stageB.py`
- Output: `student_stageB/stageB_last.pt`

Inference:

- `scripts/infer_stageB.py` → `studentB_out/`
- `./make_studentB_video.sh`

## Temporal evaluation

Lightweight temporal stability metric (warped vs unwarped difference).

- `scripts/eval_temporal.py --input studentB_out`

