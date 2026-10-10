"""Stereo rectification shared by confidence-filtered depth exporters."""

from pathlib import Path

import cv2
import numpy as np


def rectification(calibration_path: Path, width: int, height: int):
    values = np.fromstring(
        " ".join(line.split("#", 1)[0] for line in calibration_path.read_text().splitlines()),
        sep=" ",
    )
    if values.size < 41:
        raise ValueError(f"Expected at least 41 calibration values: {calibration_path}")
    left_k = values[:9].reshape(3, 3)
    left_dist = values[10:15]
    right_k = values[15:24].reshape(3, 3)
    right_dist = values[24:29]
    rotation = values[29:38].reshape(3, 3)
    translation = values[38:41]
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        left_k, left_dist, right_k, right_dist, (width, height), rotation,
        translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
        newImageSize=(width, height),
    )
    pixels = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
    inverse_left = cv2.undistortPoints(
        pixels.astype(np.float32).reshape(-1, 1, 2), left_k, left_dist,
        R=r1, P=p1,
    ).reshape(height, width, 2)
    right_maps = cv2.initUndistortRectifyMap(
        right_k, right_dist, r2, p2, (width, height), cv2.CV_32FC1,
    )
    return inverse_left, right_maps
