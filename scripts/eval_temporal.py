import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def calc_flow(prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> np.ndarray:
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

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
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    diff = np.abs(cur_gray - prev_gray)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).astype(np.float32)

    w_diff = np.clip(1.0 - (diff / 0.10), 0.0, 1.0)
    w_mag = np.clip(1.0 - (mag / 12.0), 0.0, 1.0)

    mask = w_diff * w_mag
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
    return mask.astype(np.float32)


def flow_consistency_mask(flow_fw: np.ndarray, flow_bw: np.ndarray) -> np.ndarray:
    h, w = flow_fw.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_fw[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow_fw[:, :, 1]).astype(np.float32)

    flow_bw_at_fw_x = cv2.remap(flow_bw[:, :, 0], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flow_bw_at_fw_y = cv2.remap(flow_bw[:, :, 1], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flow_bw_at_fw = np.stack([flow_bw_at_fw_x, flow_bw_at_fw_y], axis=-1)

    inconsistency = np.sqrt((flow_fw[:, :, 0] + flow_bw_at_fw[:, :, 0]) ** 2 +
                            (flow_fw[:, :, 1] + flow_bw_at_fw[:, :, 1]) ** 2)
    mask = np.clip(1.0 - (inconsistency / 1.5), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    return mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Simple temporal stability evaluation.")
    parser.add_argument("--input", type=str, default="out/stageB/frames", help="Folder of PNG frames.")
    parser.add_argument("--max_frames", type=int, default=None, help="Max number of frames to evaluate.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    frames = sorted([p for p in input_dir.glob("*.png")])
    if not frames:
        raise SystemExit(f"No frames found in {input_dir.resolve()}")

    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    unwarped_err = []
    warped_err = []

    for i in tqdm(range(1, len(frames)), desc="Eval"):
        prev = load_rgb(frames[i - 1])
        cur = load_rgb(frames[i])

        prev_bgr = prev[:, :, ::-1]
        cur_bgr = cur[:, :, ::-1]

        # Raw frame-to-frame difference
        diff_raw = np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32))) / 255.0
        unwarped_err.append(diff_raw)

        # Warped difference with trust/occlusion mask
        flow_fw = calc_flow(prev_bgr, cur_bgr)
        flow_bw = calc_flow(cur_bgr, prev_bgr)
        warped_prev = warp_bgr(prev_bgr, flow_fw)[:, :, ::-1].astype(np.float32) / 255.0
        cur_f = cur.astype(np.float32) / 255.0

        mask_trust = trust_mask(prev_bgr, cur_bgr, flow_fw)
        mask_occ = flow_consistency_mask(flow_fw, flow_bw)
        mask = (mask_trust * mask_occ).astype(np.float32)

        diff_warp = np.abs(cur_f - warped_prev).mean(axis=2)
        weighted = (diff_warp * mask).sum() / (mask.sum() + 1e-6)
        warped_err.append(float(weighted))

    print(f"Frames: {len(frames)}")
    print(f"Unwarped MAE (avg): {np.mean(unwarped_err):.4f}")
    print(f"Warped+masked MAE (avg): {np.mean(warped_err):.4f}")


if __name__ == "__main__":
    main()
