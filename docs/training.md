# Training and Inference Notes

## Environment

```
pip install numpy pillow opencv-python tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
pip install python-dotenv
```

## Stage A (single-frame)

Train:

```
python scripts/train_stageA.py
```

Output:

```
student_stageA/stageA_last.pt
```

Infer:

```
python scripts/infer_stageA.py
./make_student_video.sh
```

## Stage B (temporal)

Train:

```
python scripts/train_stageB.py
```

Output:

```
student_stageB/stageB_last.pt
```

Infer:

```
python scripts/infer_stageB.py
./make_studentB_video.sh
```

## Temporal stability evaluation

```
python scripts/eval_temporal.py --input studentB_out
```

## Notes

- Stage B must be retrained if input channel count changes (e.g., history length).
- Ensure `dataset/seq01` has matching RGB, depth, and teacher frames.
- If you change `OUT_W/OUT_H` in `scripts/config.py`, regenerate the dataset.
