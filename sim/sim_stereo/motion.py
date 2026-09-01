from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _deg2rad(values: Iterable[float]) -> np.ndarray:
    return np.deg2rad(np.asarray(list(values), dtype=np.float64))


def _make_transform(translation: np.ndarray, rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy_rad
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rot = rz @ ry @ rx

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = translation
    return transform


@dataclass
class MotionProfile:
    base_translation: np.ndarray
    base_rpy_rad: np.ndarray
    trans_amp: np.ndarray
    trans_freq: np.ndarray
    rot_amp_rad: np.ndarray
    rot_freq: np.ndarray

    @classmethod
    def default(cls) -> "MotionProfile":
        return cls(
            base_translation=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            base_rpy_rad=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            trans_amp=np.zeros(3, dtype=np.float64),
            trans_freq=np.zeros(3, dtype=np.float64),
            rot_amp_rad=np.zeros(3, dtype=np.float64),
            rot_freq=np.zeros(3, dtype=np.float64),
        )

    @classmethod
    def from_user_input(cls) -> "MotionProfile":
        def ask_vector(prompt: str, default: list[float], scale_deg: bool = False) -> np.ndarray:
            suffix = " (deg)" if scale_deg else ""
            try:
                text = input(f"{prompt}{suffix} [{', '.join(str(x) for x in default)}]: ").strip()
            except EOFError:
                print(f"\nNo interactive input available, using default {prompt}: {default}")
                text = ""
            if not text:
                values = default
            else:
                values = [float(x) for x in text.split()]
            arr = np.asarray(values, dtype=np.float64)
            if scale_deg:
                arr = np.deg2rad(arr)
            return arr

        base_translation = ask_vector("Initial translation x y z", [0.0, 0.0, 0.8])
        base_rpy_rad = ask_vector("Initial rotation roll pitch yaw", [0.0, 0.0, 0.0], scale_deg=True)
        trans_amp = ask_vector("Translation amplitude x y z", [0.0, 0.0, 0.0])
        trans_freq = ask_vector("Translation frequency x y z", [0.0, 0.0, 0.0])
        rot_amp_rad = ask_vector("Rotation amplitude roll pitch yaw", [0.0, 0.0, 0.0], scale_deg=True)
        rot_freq = ask_vector("Rotation frequency roll pitch yaw", [0.0, 0.0, 0.0])
        return cls(
            base_translation=base_translation,
            base_rpy_rad=base_rpy_rad,
            trans_amp=trans_amp,
            trans_freq=trans_freq,
            rot_amp_rad=rot_amp_rad,
            rot_freq=rot_freq,
        )

    def pose_at(self, t_sec: float) -> np.ndarray:
        trans_offset = self.trans_amp * np.sin(2.0 * math.pi * self.trans_freq * t_sec)
        rot_offset = self.rot_amp_rad * np.sin(2.0 * math.pi * self.rot_freq * t_sec)
        translation = self.base_translation + trans_offset
        rpy = self.base_rpy_rad + rot_offset
        return _make_transform(translation, rpy)
