#!/usr/bin/env python3
"""Run the published LAF-Net weights on FFS exports and filter aligned depth."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from confidence_geometry import rectification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laf_root", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--confidence_dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")

    sys.path.insert(0, str(args.laf_root))
    import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.device = device
    model = models.LAFNet_CVPR2019().to(device)
    weights = args.laf_root / "saved_models/LAFNet_CVPR2019_networks.pth"
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    depth_paths = sorted((args.input_root / "raw_depth").glob("*.png"))
    if not depth_paths:
        raise FileNotFoundError(f"No raw depth PNGs in {args.input_root / 'raw_depth'}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.confidence_dir.mkdir(parents=True, exist_ok=True)
    inverse_map = None
    for index, depth_path in enumerate(depth_paths, 1):
        stem = depth_path.stem
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        disparity = cv2.imread(str(args.input_root / "disparity" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(args.input_root / "laf_rgb" / f"{stem}.png"))
        cost = np.load(args.input_root / "topk_cost" / f"{stem}.npy")
        if depth is None or disparity is None or rgb is None or cost.shape != (7, *disparity.shape):
            raise ValueError(f"Invalid LAF inputs for frame {stem}")
        if disparity.shape != rgb.shape[:2] or depth.dtype != np.uint16:
            raise ValueError(f"Mismatched LAF inputs for frame {stem}")

        with torch.inference_mode():
            confidence = model(
                torch.from_numpy(cost.copy()).unsqueeze(0).to(device),
                torch.from_numpy(disparity.astype(np.float32) / (256.0 * 228.0))[None, None].to(device),
                torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 256.0)
                .permute(2, 0, 1).unsqueeze(0).to(device),
            )[0, 0].cpu().numpy()

        height, width = depth.shape
        model_height, model_width = disparity.shape
        scale = min(model_width / width, model_height / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        left = (model_width - resized_width) // 2
        top = (model_height - resized_height) // 2
        confidence = cv2.resize(
            confidence[top:top + resized_height, left:left + resized_width],
            (width, height), interpolation=cv2.INTER_LINEAR,
        )
        if args.aligned:
            if inverse_map is None:
                inverse_map = rectification(args.calibration, width, height)[0]
            confidence = cv2.remap(
                confidence, inverse_map, None, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        confidence[depth == 0] = 0
        confidence_u16 = np.rint(np.clip(confidence, 0, 1) * 65535).astype(np.uint16)
        filtered = depth.copy()
        filtered[confidence_u16 < np.ceil(args.threshold * 65535)] = 0
        if not cv2.imwrite(str(args.confidence_dir / depth_path.name), confidence_u16):
            raise OSError(f"Cannot write LAF confidence for {stem}")
        if not cv2.imwrite(str(args.output_dir / depth_path.name), filtered):
            raise OSError(f"Cannot write LAF filtered depth for {stem}")
        print(f"[LAF] {index}/{len(depth_paths)} {stem}: valid={np.count_nonzero(filtered)}", flush=True)


if __name__ == "__main__":
    main()
