#!/usr/bin/env python3
"""Run static OTB ConfNet on FFS disparity and filter aligned raw depth."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from confidence_geometry import rectification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confnet_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--right_subdir", required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")

    depths = sorted((args.input_root / "raw_depth").glob("*.png"))
    if not depths:
        raise FileNotFoundError(f"No raw depth images in {args.input_root / 'raw_depth'}")
    confidence_root = args.input_root / "confnet_otb"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(depths[0]), cv2.IMREAD_UNCHANGED)
    height, width = sample.shape
    inverse_left, right_maps = rectification(args.calibration, width, height)
    disp_sample = cv2.imread(str(args.input_root / "disparity" / depths[0].name), cv2.IMREAD_UNCHANGED)
    model_height, model_width = disp_sample.shape
    scale = min(model_width / width, model_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (model_width - resized_width) // 2
    top = (model_height - resized_height) // 2

    if not all((confidence_root / path.stem / "otb.png").is_file() for path in depths):
        stage = args.input_root / "confnet_inputs"
        right_dir = stage / "right"
        right_dir.mkdir(parents=True, exist_ok=True)
        for depth_path in depths:
            right_path = args.dataset_root / args.right_subdir / depth_path.name
            right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if right is None or right.shape[:2] != (height, width):
                raise ValueError(f"Missing or mismatched right image: {right_path}")
            right = cv2.remap(right, *right_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right = cv2.resize(right, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            right = cv2.copyMakeBorder(
                right, top, model_height - top - resized_height,
                left, model_width - left - resized_width, cv2.BORDER_REPLICATE,
            )
            if not cv2.imwrite(str(right_dir / depth_path.name), right):
                raise OSError(f"Cannot save prepared right image: {depth_path.name}")

        filelist = stage / "filelist.txt"
        filelist.write_text("".join(f"{path.name}\n" for path in depths), encoding="utf-8")
        command = [
            sys.executable, str(args.confnet_root / "main.py"),
            "--mode", "otb", "--cpu", "--checkpoint_path", str(args.checkpoint),
            "--output_path", str(confidence_root),
            "--left_dir", str(args.input_root / "laf_rgb") + "/",
            "--right_dir", str(right_dir) + "/",
            "--disp_dir", str(args.input_root / "disparity") + "/",
            "--dataset", str(filelist), "--image_height", str(model_height),
            "--image_width", str(model_width),
        ]
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        subprocess.run(command, cwd=args.confnet_root, env=environment, check=True)

    for index, depth_path in enumerate(depths, 1):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence_path = confidence_root / depth_path.stem / "otb.png"
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if depth is None or confidence is None or confidence.shape != (model_height, model_width):
            raise ValueError(f"Missing or mismatched ConfNet output: {confidence_path}")
        confidence = cv2.resize(
            confidence[top:top + resized_height, left:left + resized_width],
            (width, height), interpolation=cv2.INTER_LINEAR,
        )
        if args.aligned:
            confidence = cv2.remap(
                confidence, inverse_left, None, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        filtered = depth.copy()
        filtered[confidence < np.ceil(args.threshold * 65535)] = 0
        if not cv2.imwrite(str(args.output_dir / depth_path.name), filtered):
            raise OSError(f"Cannot save ConfNet filtered depth: {depth_path.name}")
        print(f"[ConfNet OTB] {index}/{len(depths)} {depth_path.stem}: valid={np.count_nonzero(filtered)}", flush=True)


if __name__ == "__main__":
    main()
