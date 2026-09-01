from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh

from .calibration import StereoCalibration, load_transform


def _cv_to_gl_transform() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform_to_pose(transform_cv: np.ndarray) -> np.ndarray:
    s = _cv_to_gl_transform()
    return s @ transform_cv @ s


class StereoRenderer:
    def __init__(
        self,
        mesh_path: str | Path,
        calibration: StereoCalibration,
        width: int,
        height: int,
        mesh_to_origin_path: str | Path | None = None,
    ) -> None:
        self.calibration = calibration
        self.width = int(width)
        self.height = int(height)

        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        if mesh_to_origin_path:
            mesh.apply_transform(load_transform(mesh_to_origin_path))
        if getattr(mesh.visual, "kind", None) == "texture" and hasattr(mesh.visual, "to_color"):
            mesh.visual = mesh.visual.to_color()

        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.scene = pyrender.Scene(bg_color=np.array([18, 18, 18, 255], dtype=np.uint8), ambient_light=np.array([0.35, 0.35, 0.35]))
        self.object_node = self.scene.add(self.mesh, pose=np.eye(4, dtype=np.float64))

        self.left_camera = pyrender.IntrinsicsCamera(
            fx=float(calibration.k_left[0, 0]),
            fy=float(calibration.k_left[1, 1]),
            cx=float(calibration.k_left[0, 2]),
            cy=float(calibration.k_left[1, 2]),
        )
        self.right_camera = pyrender.IntrinsicsCamera(
            fx=float(calibration.k_right[0, 0]),
            fy=float(calibration.k_right[1, 1]),
            cx=float(calibration.k_right[0, 2]),
            cy=float(calibration.k_right[1, 2]),
        )

        self.left_camera_pose = np.eye(4, dtype=np.float64)
        right_pose_cv = np.eye(4, dtype=np.float64)
        right_pose_cv[:3, :3] = calibration.r.T
        right_pose_cv[:3, 3] = (-calibration.r.T @ calibration.t).reshape(3)
        self.right_camera_pose = _transform_to_pose(right_pose_cv)

        light_positions = [
            np.eye(4, dtype=np.float64),
            self._pose_from_translation([0.3, 0.2, -0.2]),
            self._pose_from_translation([-0.3, 0.2, -0.2]),
            self._pose_from_translation([0.0, -0.4, -0.2]),
        ]
        for pose in light_positions:
            self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.8), pose=pose)

        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
        self.left_camera_node = self.scene.add(self.left_camera, pose=self.left_camera_pose)
        self.right_camera_node = self.scene.add(self.right_camera, pose=self.right_camera_pose)

    @staticmethod
    def _pose_from_translation(translation: list[float] | np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(translation, dtype=np.float64)
        return _transform_to_pose(pose)

    def render(self, object_pose_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        object_pose_gl = _transform_to_pose(object_pose_cv)
        self.object_node.matrix = object_pose_gl
        self.scene.main_camera_node = self.left_camera_node
        left_color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        self.scene.main_camera_node = self.right_camera_node
        right_color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        left_color = cv2.cvtColor(left_color, cv2.COLOR_RGB2BGR)
        right_color = cv2.cvtColor(right_color, cv2.COLOR_RGB2BGR)
        return left_color, right_color

    def close(self) -> None:
        self.renderer.delete()
