#!/usr/bin/env python3
"""Generate a static textured plane for a fixed-disparity stereo background."""
from pathlib import Path

import cv2
import numpy as np


def main():
    rng = np.random.default_rng(4605)
    height, width = 480, 640
    noise = rng.integers(0, 256, (height, width), np.uint8)
    broad = cv2.GaussianBlur(noise, (0, 0), 9)
    fine = cv2.GaussianBlur(noise, (0, 0), 1.2)
    value = np.clip(115 + (broad.astype(np.float32) - 127) * 2.5
                    + (fine.astype(np.float32) - 127) * 0.9, 35, 215).astype(np.uint8)
    image = cv2.merge((value, value, value))
    for y in range(28, height, 56):
        for x in range(28, width, 56):
            cv2.circle(image, (x, y), 4, (50, 86, 160), -1, cv2.LINE_AA)
    output = Path(__file__).resolve().parent / "assets" / "stereo_background.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(output)


if __name__ == "__main__":
    main()
