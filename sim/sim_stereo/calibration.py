from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np


_FLOAT_RE = re.compile(
    r"[-+]?(?:\d*\.\d+|\d+\.?(?:\d*)?)(?:[eE][-+]?\d+)?"
)


@dataclass
class StereoCalibration:
    k_left: np.ndarray
    baseline: float
    d_left: np.ndarray
    k_right: np.ndarray
    d_right: np.ndarray
    r: np.ndarray
    t: np.ndarray


def _parse_floats(text: str) -> list[float]:
    values: list[float] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        values.extend(float(x) for x in _FLOAT_RE.findall(line))
    return values


def load_calibration(path: str | Path) -> StereoCalibration:
    values = _parse_floats(Path(path).read_text())
    expected = 9 + 1 + 5 + 9 + 5 + 9 + 3
    if len(values) < expected:
        raise ValueError(
            f"Calibration file {path} has {len(values)} numeric values, expected at least {expected}"
        )

    idx = 0

    def take(n: int) -> np.ndarray:
        nonlocal idx
        chunk = np.asarray(values[idx : idx + n], dtype=np.float64)
        idx += n
        return chunk

    k_left = take(9).reshape(3, 3)
    baseline = float(take(1)[0])
    d_left = take(5)
    k_right = take(9).reshape(3, 3)
    d_right = take(5)
    r = take(9).reshape(3, 3)
    t = take(3).reshape(3, 1)
    return StereoCalibration(
        k_left=k_left,
        baseline=baseline,
        d_left=d_left,
        k_right=k_right,
        d_right=d_right,
        r=r,
        t=t,
    )


def load_transform(path: str | Path) -> np.ndarray:
    values = _parse_floats(Path(path).read_text())
    if len(values) < 16:
        raise ValueError(f"Transform file {path} has {len(values)} numeric values, expected 16")
    return np.asarray(values[:16], dtype=np.float64).reshape(4, 4)
