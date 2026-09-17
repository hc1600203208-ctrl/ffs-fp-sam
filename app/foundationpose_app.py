#!/usr/bin/env python3

import argparse
import copy
import csv
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from PyQt5 import QtCore, QtGui, QtWidgets
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    REPO_ROOT
    / "fp/src/foundationpose_cpp/config/"
    "foundationpose_stereo_tracker_multi_bluepink_example.yaml"
)


@dataclass(frozen=True)
class ObjectSpec:
    index: int
    name: str
    topic_suffix: str
    model_path: str
    mesh_path: str
    mask_name: str


def sanitize_name(name):
    result = []
    for character in name:
        if character.isascii() and character.isalnum():
            result.append(character.lower())
        elif character in "_-":
            result.append(character)
        else:
            result.append("_")
    return "".join(result) or "object"


def quaternion_to_rpy_degrees(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _node_parameters(data, node_name):
    try:
        parameters = data[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing ROS parameters for {node_name}") from error
    if not isinstance(parameters, dict):
        raise ValueError(f"invalid ROS parameters for {node_name}")
    return parameters


def load_pipeline_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("parameter file must contain a YAML mapping")

    sam = _node_parameters(data, "grounded_sam_multi_first_mask_node")
    tracker = _node_parameters(data, "foundationpose_stereo_tracker_multi_node")
    sam_names = list(sam.get("object_names", []))
    tracker_names = list(tracker.get("tracked_object_names", []))
    if not sam_names or sam_names != tracker_names:
        raise ValueError("SAM object_names must match tracker tracked_object_names")

    count = len(sam_names)
    aligned = {
        "mask_output_names": sam.get("mask_output_names", []),
        "tracked_mesh_paths": tracker.get("tracked_mesh_paths", []),
        "tracked_mask_image_names": tracker.get("tracked_mask_image_names", []),
    }
    for key in ("dino_engines", "dino_onnxs"):
        values = sam.get(key, [])
        if values:
            aligned[key] = values
    for key, values in aligned.items():
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"{key} must contain {count} entries")
    if aligned["mask_output_names"] != aligned["tracked_mask_image_names"]:
        raise ValueError("SAM and tracker mask names must match")

    detector_paths = aligned.get("dino_onnxs") or aligned.get("dino_engines")
    if not detector_paths:
        raise ValueError("dino_engines or dino_onnxs must be configured")
    objects = [
        ObjectSpec(
            index=index,
            name=name,
            topic_suffix=sanitize_name(name),
            model_path=str(detector_paths[index]),
            mesh_path=str(aligned["tracked_mesh_paths"][index]),
            mask_name=str(aligned["mask_output_names"][index]),
        )
        for index, name in enumerate(sam_names)
    ]
    if len({item.topic_suffix for item in objects}) != len(objects):
        raise ValueError("object names must produce unique ROS topic suffixes")
    return data, objects


def build_session_config(data, selected_indices):
    selected_indices = list(selected_indices)
    if not selected_indices:
        raise ValueError("select at least one object")
    result = copy.deepcopy(data)
    sam = _node_parameters(result, "grounded_sam_multi_first_mask_node")
    tracker = _node_parameters(result, "foundationpose_stereo_tracker_multi_node")
    count = len(sam["object_names"])
    if any(index < 0 or index >= count for index in selected_indices):
        raise ValueError("selected object index is out of range")

    for key in ("object_names", "dino_engines", "dino_onnxs", "mask_output_names"):
        values = sam.get(key)
        if isinstance(values, list) and len(values) == count:
            sam[key] = [values[index] for index in selected_indices]
    for key in ("tracked_object_names", "tracked_mesh_paths", "tracked_mask_image_names"):
        values = tracker.get(key)
        if isinstance(values, list) and len(values) == count:
            tracker[key] = [values[index] for index in selected_indices]

    sam["publish_mask_topics"] = True
    tracker["publish_visualization"] = True
    tracker["show_visualization_window"] = False
    return result


def validate_selected_files(data, objects):
    sam = _node_parameters(data, "grounded_sam_multi_first_mask_node")
    tracker = _node_parameters(data, "foundationpose_stereo_tracker_multi_node")
    paths = [item.model_path for item in objects] + [item.mesh_path for item in objects]
    paths += [
        sam.get("sam_encoder_engine", ""),
        sam.get("sam_decoder_engine", ""),
        tracker.get("refiner_engine_path", ""),
        tracker.get("scorer_engine_path", ""),
        tracker.get("caminfo_path", ""),
    ]
    paths += list(tracker.get("stereo_engine_file_path", []))
    missing = [str(path) for path in paths if not path or not Path(path).expanduser().is_file()]
    if missing:
        raise FileNotFoundError("missing model/config files:\n" + "\n".join(missing))


def load_stereo_calibration(path):
    values = []
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.split("#", 1)[0]
            values.extend(float(value) for value in line.split())
    if len(values) < 41:
        raise ValueError(f"stereo calibration requires 41 values, got {len(values)}")
    offset = 0
    left_k = np.asarray(values[offset : offset + 9], np.float64).reshape(3, 3)
    offset += 9
    offset += 1  # Baseline is followed by full stereo extrinsics later in the file.
    left_dist = np.asarray(values[offset : offset + 5], np.float64)
    offset += 5
    right_k = np.asarray(values[offset : offset + 9], np.float64).reshape(3, 3)
    offset += 9
    right_dist = np.asarray(values[offset : offset + 5], np.float64)
    offset += 5
    rotation = np.asarray(values[offset : offset + 9], np.float64).reshape(3, 3)
    offset += 9
    translation = np.asarray(values[offset : offset + 3], np.float64)
    return left_k, left_dist, right_k, right_dist, rotation, translation


class StereoPreview:
    def __init__(self, calibration_path):
        self.calibration = load_stereo_calibration(calibration_path)
        self.image_size = None
        self.maps = None

    def _ensure_maps(self, width, height):
        size = (width, height)
        if self.image_size == size:
            return
        left_k, left_dist, right_k, right_dist, rotation, translation = self.calibration
        left_r, right_r, left_p, right_p, _, _, _ = cv2.stereoRectify(
            left_k,
            left_dist,
            right_k,
            right_dist,
            size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )
        left_maps = cv2.initUndistortRectifyMap(
            left_k, left_dist, left_r, left_p, size, cv2.CV_32FC1
        )
        right_maps = cv2.initUndistortRectifyMap(
            right_k, right_dist, right_r, right_p, size, cv2.CV_32FC1
        )
        self.image_size = size
        self.maps = (*left_maps, *right_maps)

    def compute(self, left_bgr, right_bgr):
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise ValueError("left and right image sizes differ")
        height, width = left_bgr.shape[:2]
        self._ensure_maps(width, height)
        left = cv2.remap(left_bgr, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        right = cv2.remap(right_bgr, self.maps[2], self.maps[3], cv2.INTER_LINEAR)
        if width > 640:
            scale = 640.0 / width
            output_size = (640, max(1, round(height * scale)))
            left = cv2.resize(left, output_size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, output_size, interpolation=cv2.INTER_AREA)
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        num_disparities = max(16, min(128, (left_gray.shape[1] // 4 // 16) * 16))
        block_size = 5
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        valid = np.isfinite(disparity) & (disparity > 0.0)
        grayscale = np.zeros(disparity.shape, np.uint8)
        if np.any(valid):
            low, high = np.percentile(disparity[valid], (2.0, 98.0))
            if high - low > 1e-6:
                normalized = np.clip((disparity - low) / (high - low), 0.0, 1.0)
                grayscale[valid] = np.round(normalized[valid] * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        return color


class AppRosNode(Node):
    def __init__(self, pose_callback):
        super().__init__("foundationpose_visual_app")
        self.bridge = CvBridge()
        self.pose_callback = pose_callback
        self.raw_enabled = False
        self.lock = threading.Lock()
        self.images = {}
        self.sequence = 0
        self.app_subscriptions = []

    def configure(self, data, objects):
        for subscription in self.app_subscriptions:
            self.destroy_subscription(subscription)
        self.app_subscriptions.clear()
        with self.lock:
            self.images.clear()
            self.sequence = 0

        tracker = _node_parameters(data, "foundationpose_stereo_tracker_multi_node")
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        image_topics = {
            "left": tracker.get("left_image_topic", "/left/image_raw"),
            "right": tracker.get("right_image_topic", "/right/image_raw"),
            "overlay": tracker.get(
                "visualization_topic", "/foundationpose/multi_visualization"
            ),
        }
        for key, topic in image_topics.items():
            subscription = self.create_subscription(
                Image,
                topic,
                lambda message, image_key=key: self._image_callback(image_key, message),
                image_qos,
            )
            self.app_subscriptions.append(subscription)

        pose_topic = tracker.get("pose_topic", "/foundationpose/pose")
        for item in objects:
            subscription = self.create_subscription(
                PoseStamped,
                f"{pose_topic}/{item.topic_suffix}",
                lambda message, name=item.name: self._pose_callback(name, message),
                pose_qos,
            )
            self.app_subscriptions.append(subscription)

    def set_raw_enabled(self, enabled):
        self.raw_enabled = bool(enabled)

    def _image_callback(self, key, message):
        if key != "overlay" and not self.raw_enabled:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:
            self.get_logger().warning(f"cannot convert {key} image: {error}")
            return
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        with self.lock:
            self.sequence += 1
            self.images[key] = (self.sequence, stamp_ns, image)

    def _pose_callback(self, name, message):
        pose = message.pose
        try:
            roll, pitch, yaw = quaternion_to_rpy_degrees(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        except ValueError:
            return
        self.pose_callback(
            name,
            {
                "stamp_sec": message.header.stamp.sec,
                "stamp_nanosec": message.header.stamp.nanosec,
                "frame_id": message.header.frame_id,
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "qx": pose.orientation.x,
                "qy": pose.orientation.y,
                "qz": pose.orientation.z,
                "qw": pose.orientation.w,
            },
        )

    def latest_images(self):
        with self.lock:
            return dict(self.images)


class ImagePanel(QtWidgets.QLabel):
    def __init__(self, empty_text):
        super().__init__(empty_text)
        self.empty_text = empty_text
        self.pixmap_source = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(240, 150)
        self.setStyleSheet("background:#101418;color:#87919b;border:1px solid #cbd1d6;")

    def set_frame(self, frame):
        if frame is None or frame.size == 0:
            return
        height, width = frame.shape[:2]
        if frame.ndim == 2:
            image = QtGui.QImage(
                frame.data, width, height, frame.strides[0], QtGui.QImage.Format_Grayscale8
            ).copy()
        else:
            image = QtGui.QImage(
                frame.data, width, height, frame.strides[0], QtGui.QImage.Format_BGR888
            ).copy()
        self.pixmap_source = QtGui.QPixmap.fromImage(image)
        self._render()

    def clear_frame(self, text=None):
        self.pixmap_source = None
        self.clear()
        self.setText(text or self.empty_text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def _render(self):
        if self.pixmap_source is not None:
            self.setPixmap(
                self.pixmap_source.scaled(
                    self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
            )


class MainWindow(QtWidgets.QMainWindow):
    pose_received = QtCore.pyqtSignal(str, object)
    process_message = QtCore.pyqtSignal(str)

    def __init__(self, ros_node, initial_config):
        super().__init__()
        self.ros_node = ros_node
        self.config_data = None
        self.objects = []
        self.object_checks = []
        self.table_rows = {}
        self.last_poses = {}
        self.last_image_sequences = {}
        self.process = None
        self.stop_deadline = None
        self.stop_stage = 0
        self.restart_pending = False
        self.accept_poses = False
        self.epoch = -1
        self.seen_in_epoch = set()
        self.pose_file = None
        self.pose_writer = None
        self.session = tempfile.TemporaryDirectory(prefix="foundationpose_app_")
        self.preview = None
        self.preview_executor = ThreadPoolExecutor(max_workers=1)
        self.preview_future = None
        self.last_preview_pair = None
        self.last_preview_submit = 0.0

        self.setWindowTitle("FoundationPose Multi-Object Tracker")
        self.resize(1480, 920)
        self._build_ui()
        self._apply_style()
        self.pose_received.connect(self._handle_pose)
        self.process_message.connect(self._handle_process_message)
        self._load_config(initial_config)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start(100)
        self.process_timer = QtCore.QTimer(self)
        self.process_timer.timeout.connect(self._poll_process)
        self.process_timer.start(250)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(10)

        controls = QtWidgets.QFrame()
        controls.setObjectName("controls")
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(7)

        config_row = QtWidgets.QHBoxLayout()
        config_row.addWidget(QtWidgets.QLabel("Parameter file"))
        self.config_path = QtWidgets.QLineEdit()
        self.config_path.setReadOnly(True)
        config_row.addWidget(self.config_path, 1)
        self.config_button = QtWidgets.QToolButton()
        self.config_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.config_button.setToolTip("Select parameter file")
        self.config_button.clicked.connect(self._browse_config)
        config_row.addWidget(self.config_button)
        config_row.addSpacing(12)
        config_row.addWidget(QtWidgets.QLabel("Tracked objects"))
        self.object_widget = QtWidgets.QWidget()
        self.object_layout = QtWidgets.QHBoxLayout(self.object_widget)
        self.object_layout.setContentsMargins(0, 0, 0, 0)
        self.object_layout.setSpacing(12)
        config_row.addWidget(self.object_widget)
        config_row.addStretch(1)
        controls_layout.addLayout(config_row)

        action_row = QtWidgets.QHBoxLayout()
        self.tracking_toggle = QtWidgets.QCheckBox("Start tracking")
        self.tracking_toggle.toggled.connect(self._tracking_toggled)
        action_row.addWidget(self.tracking_toggle)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self._reset_tracking)
        action_row.addWidget(self.reset_button)
        action_row.addSpacing(14)

        self.save_toggle = QtWidgets.QCheckBox("Save poses")
        self.save_toggle.toggled.connect(self._save_toggled)
        action_row.addWidget(self.save_toggle)
        self.save_path = QtWidgets.QLineEdit(
            str(Path.home() / f"foundationpose_poses_{datetime.now():%Y%m%d_%H%M%S}.csv")
        )
        action_row.addWidget(self.save_path, 1)
        self.save_button = QtWidgets.QToolButton()
        self.save_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_button.setToolTip("Select pose file")
        self.save_button.clicked.connect(self._browse_save_path)
        action_row.addWidget(self.save_button)
        action_row.addSpacing(14)

        self.raw_toggle = QtWidgets.QCheckBox("Show stereo images and disparity")
        self.raw_toggle.toggled.connect(self._raw_toggled)
        action_row.addWidget(self.raw_toggle)
        controls_layout.addLayout(action_row)
        root.addWidget(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        overlay_area = QtWidgets.QWidget()
        overlay_layout = QtWidgets.QVBoxLayout(overlay_area)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(4)
        overlay_layout.addWidget(QtWidgets.QLabel("Multi-object pose visualization"))
        self.overlay_panel = ImagePanel("Waiting for pose visualization")
        overlay_layout.addWidget(self.overlay_panel, 1)
        splitter.addWidget(overlay_area)

        table_area = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_area)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(4)
        table_layout.addWidget(QtWidgets.QLabel("6DoF (m / deg)"))
        self.pose_table = QtWidgets.QTableWidget(0, 10)
        self.pose_table.setHorizontalHeaderLabels(
            ["Object", "Status", "X", "Y", "Z", "Roll", "Pitch", "Yaw", "Frame", "Updated"]
        )
        self.pose_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.pose_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.pose_table.verticalHeader().setVisible(False)
        self.pose_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.pose_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.pose_table, 1)
        splitter.addWidget(table_area)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.raw_area = QtWidgets.QWidget()
        raw_layout = QtWidgets.QHBoxLayout(self.raw_area)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.setSpacing(8)
        self.left_panel = self._add_named_panel(raw_layout, "Left image", "Waiting for left image")
        self.right_panel = self._add_named_panel(raw_layout, "Right image", "Waiting for right image")
        self.disparity_panel = self._add_named_panel(
            raw_layout, "Color disparity (SGBM preview)", "Waiting for stereo images"
        )
        self.raw_area.setVisible(False)
        root.addWidget(self.raw_area)

        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setObjectName("status")
        self.statusBar().addPermanentWidget(self.status_label)

    @staticmethod
    def _add_named_panel(layout, title, empty_text):
        area = QtWidgets.QWidget()
        area_layout = QtWidgets.QVBoxLayout(area)
        area_layout.setContentsMargins(0, 0, 0, 0)
        area_layout.setSpacing(3)
        area_layout.addWidget(QtWidgets.QLabel(title))
        panel = ImagePanel(empty_text)
        panel.setMaximumHeight(260)
        area_layout.addWidget(panel)
        layout.addWidget(area, 1)
        return panel

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#eef1f3; color:#182026; font-size:13px; }
            QFrame#controls { background:#ffffff; border:1px solid #cbd1d6; border-radius:6px; }
            QLineEdit { background:#ffffff; border:1px solid #adb5bd; border-radius:4px; padding:5px 7px; }
            QPushButton, QToolButton { background:#ffffff; border:1px solid #9da7af; border-radius:4px; padding:5px 9px; }
            QPushButton:hover, QToolButton:hover { background:#e7ecef; }
            QPushButton:disabled, QToolButton:disabled { color:#929aa1; background:#e5e8ea; }
            QCheckBox { spacing:7px; }
            QCheckBox::indicator { width:17px; height:17px; }
            QTableWidget { background:#ffffff; alternate-background-color:#f5f7f8; border:1px solid #cbd1d6; gridline-color:#dde1e4; }
            QHeaderView::section { background:#e2e7ea; border:0; border-right:1px solid #c7cdd1; padding:6px; font-weight:600; }
            QStatusBar { background:#ffffff; border-top:1px solid #cbd1d6; }
            QLabel#status { color:#3d4952; font-weight:600; padding:2px 8px; }
            """
        )

    def _browse_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select parameter file", str(Path(self.config_path.text()).parent), "YAML (*.yaml *.yml)"
        )
        if path:
            self._load_config(path)

    def _load_config(self, path):
        try:
            data, objects = load_pipeline_config(path)
            validate_selected_files(data, objects)
            tracker = _node_parameters(data, "foundationpose_stereo_tracker_multi_node")
            preview = StereoPreview(tracker["caminfo_path"])
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Configuration error", str(error))
            return

        self.config_data = data
        self.objects = objects
        self.preview = preview
        self.config_path.setText(str(Path(path).expanduser().resolve()))
        while self.object_layout.count():
            item = self.object_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.object_checks.clear()
        for item in objects:
            checkbox = QtWidgets.QCheckBox(item.name)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._selection_changed)
            self.object_layout.addWidget(checkbox)
            self.object_checks.append(checkbox)
        self._selection_changed()
        self._set_status("Configuration ready", "#24735b")

    def _selected(self):
        return [
            item for item, checkbox in zip(self.objects, self.object_checks) if checkbox.isChecked()
        ]

    def _selection_changed(self):
        selected = self._selected()
        self._reset_pose_table(selected)
        if self.config_data is not None:
            self.ros_node.configure(self.config_data, selected)

    def _reset_pose_table(self, objects):
        self.pose_table.setRowCount(len(objects))
        self.table_rows = {}
        self.last_poses.clear()
        for row, item in enumerate(objects):
            self.table_rows[item.name] = row
            values = [item.name, "Waiting", "-", "-", "-", "-", "-", "-", "-", "-"]
            for column, value in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(value)
                if column >= 2:
                    cell.setTextAlignment(QtCore.Qt.AlignCenter)
                self.pose_table.setItem(row, column, cell)

    def _tracking_toggled(self, checked):
        if checked:
            self._start_tracking()
        else:
            self.restart_pending = False
            self._request_stop("Stopping")

    def _start_tracking(self):
        selected = self._selected()
        try:
            if self.process is not None and self.process.poll() is None:
                return
            validate_selected_files(self.config_data, selected)
            session_data = build_session_config(
                self.config_data, [item.index for item in selected]
            )
            session_root = Path(self.session.name)
            config_path = session_root / "params.yaml"
            mask_path = session_root / "masks"
            mask_path.mkdir(exist_ok=True)
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(session_data, stream, sort_keys=False, allow_unicode=True)
            ros2 = shutil.which("ros2")
            if not ros2:
                raise RuntimeError("ros2 executable is not available; source ROS 2 first")
            command = [
                ros2,
                "launch",
                "foundationpose_cpp",
                "foundationpose_stereo_tracker_multi.launch.py",
                f"params_file:={config_path}",
                f"first_mask_output_dir:={mask_path}",
                "run_first_mask:=true",
            ]
            self.ros_node.configure(session_data, selected)
            self.process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as error:
            self._set_toggle(self.tracking_toggle, False)
            self._set_status("Startup failed", "#a63d40")
            QtWidgets.QMessageBox.critical(self, "Startup failed", str(error))
            return

        self.epoch += 1
        self.seen_in_epoch.clear()
        self.accept_poses = True
        self.stop_deadline = None
        self.stop_stage = 0
        self.restart_pending = False
        self._set_controls_running(True)
        self._reset_pose_table(selected)
        self.overlay_panel.clear_frame()
        self._set_status("Waiting for mask / register", "#9a6700")
        threading.Thread(
            target=self._read_process_output, args=(self.process,), daemon=True
        ).start()

    def _read_process_output(self, process):
        if process.stdout is None:
            return
        keywords = ("ERROR", "FATAL", "All first-frame masks", "ready", "failed", "Failed")
        for line in process.stdout:
            if any(keyword in line for keyword in keywords):
                self.process_message.emit(line.strip())

    def _handle_process_message(self, line):
        self.status_label.setToolTip(line)
        if "All first-frame masks" in line:
            self._set_status("Registering", "#9a6700")
        elif "ERROR" in line or "FATAL" in line or "failed" in line.lower():
            self._set_status("Runtime error", "#a63d40")

    def _request_stop(self, status):
        self.accept_poses = False
        process = self.process
        if process is None or process.poll() is not None:
            self._finish_stopped()
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        self.stop_deadline = time.monotonic() + 8.0
        self.stop_stage = 1
        self.reset_button.setEnabled(False)
        self._set_status(status, "#6c757d")

    def _poll_process(self):
        process = self.process
        if process is None:
            return
        return_code = process.poll()
        if return_code is not None:
            restart = self.restart_pending
            self.process = None
            self.stop_deadline = None
            self.stop_stage = 0
            if restart:
                QtCore.QTimer.singleShot(250, self._start_tracking)
            else:
                self._finish_stopped(unexpected=self.tracking_toggle.isChecked())
            return
        if self.stop_deadline is not None and time.monotonic() >= self.stop_deadline:
            next_signal = signal.SIGTERM if self.stop_stage == 1 else signal.SIGKILL
            try:
                os.killpg(process.pid, next_signal)
            except ProcessLookupError:
                pass
            self.stop_stage += 1
            self.stop_deadline = time.monotonic() + 3.0

    def _finish_stopped(self, unexpected=False):
        self.process = None
        self.accept_poses = False
        self._set_toggle(self.tracking_toggle, False)
        self._set_controls_running(False)
        self._set_status("Unexpected exit" if unexpected else "Stopped", "#a63d40" if unexpected else "#3d4952")

    def _reset_tracking(self):
        if self.process is None or self.process.poll() is not None:
            return
        self.restart_pending = True
        self._request_stop("Resetting")

    def _set_controls_running(self, running):
        self.config_button.setEnabled(not running)
        for checkbox in self.object_checks:
            checkbox.setEnabled(not running)
        self.reset_button.setEnabled(running)

    @staticmethod
    def _set_toggle(widget, checked):
        widget.blockSignals(True)
        widget.setChecked(checked)
        widget.blockSignals(False)

    def _browse_save_path(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save poses", self.save_path.text(), "CSV (*.csv)"
        )
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            self.save_path.setText(path)

    def _save_toggled(self, checked):
        if not checked:
            self._close_pose_file()
            self.save_path.setEnabled(True)
            self.save_button.setEnabled(True)
            return
        try:
            path = Path(self.save_path.text()).expanduser()
            if not path.name:
                raise ValueError("Select a CSV file")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.pose_file = path.open("w", encoding="utf-8", newline="", buffering=1)
            self.pose_writer = csv.writer(self.pose_file)
            self.pose_writer.writerow(
                [
                    "epoch",
                    "mode",
                    "stamp_sec",
                    "stamp_nanosec",
                    "object",
                    "frame_id",
                    "x_m",
                    "y_m",
                    "z_m",
                    "roll_deg",
                    "pitch_deg",
                    "yaw_deg",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                ]
            )
        except Exception as error:
            self._close_pose_file()
            self._set_toggle(self.save_toggle, False)
            QtWidgets.QMessageBox.critical(self, "Cannot save", str(error))
            return
        self.save_path.setEnabled(False)
        self.save_button.setEnabled(False)

    def _close_pose_file(self):
        if self.pose_file is not None:
            self.pose_file.flush()
            self.pose_file.close()
        self.pose_file = None
        self.pose_writer = None

    def _raw_toggled(self, checked):
        self.raw_area.setVisible(checked)
        self.ros_node.set_raw_enabled(checked)
        if not checked:
            self.left_panel.clear_frame()
            self.right_panel.clear_frame()
            self.disparity_panel.clear_frame()

    def _handle_pose(self, name, data):
        if not self.accept_poses or name not in self.table_rows:
            return
        now = time.monotonic()
        self.last_poses[name] = (now, data)
        row = self.table_rows[name]
        values = [
                    "Live",
            f"{data['x']:.4f}",
            f"{data['y']:.4f}",
            f"{data['z']:.4f}",
            f"{data['roll']:.2f}",
            f"{data['pitch']:.2f}",
            f"{data['yaw']:.2f}",
            data["frame_id"],
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
        ]
        for column, value in enumerate(values, start=1):
            self.pose_table.item(row, column).setText(value)
        self._set_status("Tracking", "#24735b")

        if self.pose_writer is not None:
            mode = "register" if name not in self.seen_in_epoch else "track"
            self.seen_in_epoch.add(name)
            try:
                self.pose_writer.writerow(
                    [
                        self.epoch,
                        mode,
                        data["stamp_sec"],
                        data["stamp_nanosec"],
                        name,
                        data["frame_id"],
                        data["x"],
                        data["y"],
                        data["z"],
                        data["roll"],
                        data["pitch"],
                        data["yaw"],
                        data["qx"],
                        data["qy"],
                        data["qz"],
                        data["qw"],
                    ]
                )
            except Exception as error:
                self._close_pose_file()
                self._set_toggle(self.save_toggle, False)
                QtWidgets.QMessageBox.critical(self, "Save failed", str(error))

    def _refresh(self):
        images = self.ros_node.latest_images()
        overlay = images.get("overlay")
        if overlay and self.last_image_sequences.get("overlay") != overlay[0]:
            self.last_image_sequences["overlay"] = overlay[0]
            self.overlay_panel.set_frame(overlay[2])

        if self.raw_toggle.isChecked():
            for key, panel in (("left", self.left_panel), ("right", self.right_panel)):
                value = images.get(key)
                if value and self.last_image_sequences.get(key) != value[0]:
                    self.last_image_sequences[key] = value[0]
                    panel.set_frame(value[2])
            self._update_disparity(images.get("left"), images.get("right"))

        now = time.monotonic()
        for name, (updated_at, _) in self.last_poses.items():
            if now - updated_at > 2.0 and name in self.table_rows:
                self.pose_table.item(self.table_rows[name], 1).setText("Timed out")

    def _update_disparity(self, left, right):
        if self.preview_future is not None and self.preview_future.done():
            try:
                self.disparity_panel.set_frame(self.preview_future.result())
            except Exception as error:
                self.disparity_panel.clear_frame("Disparity preview unavailable")
                self.disparity_panel.setToolTip(str(error))
            self.preview_future = None
        if not left or not right or self.preview_future is not None:
            return
        if abs(left[1] - right[1]) > 100_000_000:
            return
        pair = (left[0], right[0])
        now = time.monotonic()
        if pair == self.last_preview_pair or now - self.last_preview_submit < 0.35:
            return
        self.last_preview_pair = pair
        self.last_preview_submit = now
        self.preview_future = self.preview_executor.submit(
            self.preview.compute, left[2], right[2]
        )

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{color};font-weight:600;padding:2px 8px;")

    def _stop_process_blocking(self):
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def closeEvent(self, event):
        self.restart_pending = False
        self.accept_poses = False
        self._close_pose_file()
        self._stop_process_blocking()
        self.preview_executor.shutdown(wait=False, cancel_futures=True)
        self.session.cleanup()
        event.accept()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="FoundationPose multi-object desktop app")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="multi-object ROS YAML")
    parser.add_argument(
        "--check", action="store_true", help="validate the configuration and exit"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if args.check:
        data, objects = load_pipeline_config(args.config)
        validate_selected_files(data, objects)
        tracker = _node_parameters(data, "foundationpose_stereo_tracker_multi_node")
        load_stereo_calibration(tracker["caminfo_path"])
        print("configuration valid:", ", ".join(item.name for item in objects))
        return 0

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("FoundationPose Multi-Object Tracker")
    rclpy.init(args=None)
    window_holder = {}

    def emit_pose(name, data):
        window = window_holder.get("window")
        if window is not None:
            window.pose_received.emit(name, data)

    ros_node = AppRosNode(emit_pose)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    window = MainWindow(ros_node, args.config)
    window_holder["window"] = window
    window.show()

    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal_timer = QtCore.QTimer()
    signal_timer.start(250)
    signal_timer.timeout.connect(lambda: None)
    try:
        return_code = application.exec_()
    finally:
        executor.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()
        executor_thread.join(timeout=2)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
