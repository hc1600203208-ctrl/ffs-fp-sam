#!/usr/bin/env python3
"""Ground control for the CPU satellite stereo renderer and byte receiver."""
import os
import math
from collections import OrderedDict
from html import escape
import shutil
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.exceptions import InvalidTopicNameException
from rclpy.node import Node

from sct_protocol import command, configuration, parse_ack
from sim_stereo_app import REPO_ROOT, read_background, resolve_mesh_path
from sim_stereo_multi_app import MultiSimWindow, load_objects
from foundationpose_app import quaternion_to_rpy_degrees

DEFAULT_CONFIG = REPO_ROOT / "fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml"
DEFAULT_BACKGROUND = REPO_ROOT / "app/assets/stereo_background.png"


class TrackingNode(Node):
    def __init__(self):
        super().__init__("satellite_stereo_app")
        self.poses = {}
        self.truth_by_stamp = OrderedDict()
        self.latest_truth = {}
        self.object_names = []
        self.bridge = CvBridge()
        self.frames = {}
        self.frame_lock = threading.Lock()
        self.mask_subscriptions = []
        self.mask_prefix = "/first_masks"
        self.view_subscriptions = []
        self.on_ack = None
        self.control_publisher = self.create_publisher(UInt8MultiArray, "/satellite/control_bytes", 10)
        self.create_subscription(UInt8MultiArray, "/satellite/control_ack_bytes", self._ack, 10)

    def _ack(self, message):
        try:
            reply = parse_ack(bytes(message.data))
            if self.on_ack is not None:
                self.on_ack(*reply)
        except (ValueError, UnicodeDecodeError) as error:
            self.get_logger().warning(f"Invalid SCT1 reply: {error}")

    def send_control(self, kind, request_id, payload=b""):
        self.control_publisher.publish(UInt8MultiArray(data=list(command(kind, request_id, payload))))

    def _pose(self, name, message):
        with self.frame_lock:
            self.poses[name] = (stamp_key(message.header.stamp), pose_values(message.pose))

    def _truth(self, message):
        with self.frame_lock:
            if len(message.poses) != len(self.object_names):
                return
            values = {name: pose_values(pose) for name, pose in zip(self.object_names, message.poses)}
            self.truth_by_stamp[stamp_key(message.header.stamp)] = values
            self.latest_truth = values
            while len(self.truth_by_stamp) > 256:
                self.truth_by_stamp.popitem(last=False)

    def _image(self, key, message):
        try:
            encoding = "mono8" if key.startswith("mask:") else "bgr8"
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding=encoding).copy()
            with self.frame_lock:
                self.frames[key] = image
        except Exception as error:
            self.get_logger().warning(f"Cannot convert {key} image: {error}")

    def configure_topics(self, parameters, mask_parameters):
        for subscription in self.view_subscriptions:
            self.destroy_subscription(subscription)
        self.view_subscriptions.clear()
        with self.frame_lock:
            self.frames.clear()
            self.poses.clear()
            self.truth_by_stamp.clear()
            self.latest_truth.clear()
        # Best effort accepts both reliable and best effort image publishers.
        qos = rclpy.qos.QoSProfile(depth=1, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        topic = parameters.get("combined_visualization_topic", "/foundationpose/combined_visualization")
        self.view_subscriptions.append(self.create_subscription(
            Image, topic, lambda msg: self._image("combined", msg), qos))
        self.view_subscriptions.append(self.create_subscription(
            Image, parameters.get("left_image_topic", "/left/image_raw"),
            lambda msg: self._image("left", msg), qos))
        self.view_subscriptions.append(self.create_subscription(
            PoseArray, "/satellite/ground_truth", self._truth, 10))
        self.mask_prefix = mask_parameters.get("mask_topic_prefix", "/first_masks").rstrip("/")
        self.pose_prefix = parameters.get("pose_topic", "/foundationpose/pose").rstrip("/")
        return topic

    def set_mask_objects(self, objects):
        for subscription in self.mask_subscriptions:
            self.destroy_subscription(subscription)
        self.mask_subscriptions.clear()
        with self.frame_lock:
            self.frames = {key: frame for key, frame in self.frames.items() if not key.startswith("mask:")}
        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        for name, _mesh in objects:
            topic = self.mask_prefix + "/" + sanitize_topic_name(name)
            try:
                self.mask_subscriptions.append(self.create_subscription(
                    Image, topic, lambda msg, key=f"mask:{name}": self._image(key, msg), qos))
            except InvalidTopicNameException as error:
                self.get_logger().warning(f"Cannot subscribe to mask for {name}: {error}")
        for subscription in getattr(self, "pose_subscriptions", []):
            self.destroy_subscription(subscription)
        self.pose_subscriptions = []
        with self.frame_lock:
            self.poses.clear()
            self.truth_by_stamp.clear()
            self.latest_truth.clear()
        for name, _mesh in objects:
            try:
                self.pose_subscriptions.append(self.create_subscription(
                    PoseStamped, self.pose_prefix + "/" + sanitize_topic_name(name),
                    lambda msg, name=name: self._pose(name, msg), 10))
            except InvalidTopicNameException as error:
                self.get_logger().warning(f"Cannot subscribe to pose for {name}: {error}")

    def set_truth_objects(self, names):
        with self.frame_lock:
            self.object_names = list(names)
            self.truth_by_stamp.clear()
            self.latest_truth.clear()

    def set_background(self, _image):
        pass


class SatelliteWindow(MultiSimWindow):
    control_response = QtCore.pyqtSignal(int, int, str)

    def __init__(self, ros_node):
        self.receiver = None
        self.pending = None
        self.request_id = 0
        super().__init__(ros_node)
        self.setWindowTitle("Satellite Stereo Control & Tracking")
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, screen.width()), min(960, screen.height()))
        self.start_button.setText("Start satellite renderer")
        self.reset_button.setText("Apply and restart")
        self.statusBar().showMessage("Waiting for satellite renderer")
        self.pose_timer = QtCore.QTimer(self)
        self.pose_timer.timeout.connect(self._show_pose)
        self.pose_timer.start(200)
        self.control_response.connect(self._on_control_response)
        self.ros_node.on_ack = self.control_response.emit

    def _build_ui(self):
        super()._build_ui()
        self.config_path.setText(str(DEFAULT_CONFIG))
        self.preview_labels = {}
        self._shown_frames = {}
        self._objects_signature = None
        self._loading_config = False
        self.mesh_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mesh-preview")
        self.mesh_jobs = {}
        self.mesh_selector = QtWidgets.QComboBox()
        self.motion_defaults = ([0.05, 0.05, 0.02, 10.0, 10.0, 20.0],
                                [0.15, 0.12, 0.10, 0.10, 0.08, 0.06])
        self.motion_values = {}
        self.objects.itemChanged.connect(self._object_changed)
        self.objects.model().rowsInserted.connect(self._schedule_object_sync)
        self.objects.model().rowsRemoved.connect(self._schedule_object_sync)

        path_layout = self.config_path.parentWidget().layout()
        self.background_path.setReadOnly(False)
        self.background_path.setText(str(DEFAULT_BACKGROUND))
        self.background_path.setPlaceholderText("Ground-side PNG stereo background")
        self.background_disparity = QtWidgets.QSpinBox()
        self.background_disparity.setRange(1, 128)
        self.background_disparity.setValue(8)
        background_controls = QtWidgets.QHBoxLayout()
        background_controls.addWidget(QtWidgets.QLabel("Background disparity (px)"))
        background_controls.addWidget(self.background_disparity)
        background_controls.addStretch()
        path_layout.addLayout(background_controls, 2, 1)
        speed_controls = QtWidgets.QHBoxLayout()
        speed_controls.addWidget(QtWidgets.QLabel("Target FPS"))
        self.fps_input = QtWidgets.QDoubleSpinBox()
        self.fps_input.setRange(0.1, 60.0)
        self.fps_input.setValue(8.0)
        speed_controls.addWidget(self.fps_input)
        speed_controls.addWidget(QtWidgets.QLabel("Link MB/s"))
        self.link_input = QtWidgets.QDoubleSpinBox()
        self.link_input.setRange(0.1, 1000.0)
        self.link_input.setValue(10.0)
        speed_controls.addWidget(self.link_input)
        speed_controls.addStretch()
        motion_layout = self.start_button.parentWidget().layout()
        for _ in range(3):
            clear_layout(motion_layout.takeAt(0).layout())
        self.motion_table = QtWidgets.QTableWidget(0, 13)
        self.motion_table.setHorizontalHeaderLabels((
            "Target", "Amp X m", "Amp Y m", "Amp Z m", "Amp Roll °", "Amp Pitch °", "Amp Yaw °",
            "Freq X Hz", "Freq Y Hz", "Freq Z Hz", "Freq Roll Hz", "Freq Pitch Hz", "Freq Yaw Hz"))
        self.motion_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.motion_table.setMinimumHeight(175)
        motion_layout.insertWidget(0, self.motion_table)
        motion_layout.insertLayout(1, speed_controls)
        satellite_path = QtWidgets.QHBoxLayout()
        satellite_path.addWidget(QtWidgets.QLabel("Satellite calibration"))
        self.satellite_calibration = QtWidgets.QLineEdit()
        satellite_path.addWidget(self.satellite_calibration)
        motion_layout.insertLayout(2, satellite_path)

        controls = self.takeCentralWidget()
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(controls_scroll, "Parameters & control")
        views = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(views)
        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Target"))
        selector_row.addWidget(self.mesh_selector, 1)
        root.addLayout(selector_row)
        root.addWidget(QtWidgets.QLabel("Satellite left image (available before the tracker starts)"))
        self.preview_labels["left"] = self._image_label("Waiting for satellite left image")
        root.addWidget(self.preview_labels["left"], 2)
        root.addWidget(QtWidgets.QLabel("Tracker output: left image | pose overlay | depth (0.3–5.0 m)"))
        self.preview_labels["combined"] = self._image_label("Waiting for tracker composite")
        root.addWidget(self.preview_labels["combined"], 2)
        bottom = QtWidgets.QHBoxLayout()
        root.addLayout(bottom, 2)
        mask_column = QtWidgets.QVBoxLayout()
        mask_column.addWidget(QtWidgets.QLabel("First-frame masks (all targets)"))
        self.preview_labels["mask"] = self._image_label("Waiting for first-frame masks")
        mask_column.addWidget(self.preview_labels["mask"], 1)
        self.mask_legend = QtWidgets.QLabel("Waiting for first-frame masks")
        self.mask_legend.setWordWrap(True)
        mask_column.addWidget(self.mask_legend)
        bottom.addLayout(mask_column, 1)
        model_column = QtWidgets.QVBoxLayout()
        model_column.addWidget(QtWidgets.QLabel("Textured model previews (OBJ origin, fixed camera)"))
        self.models_layout = QtWidgets.QHBoxLayout()
        model_column.addLayout(self.models_layout, 1)
        bottom.addLayout(model_column, 2)
        self.pose_table = QtWidgets.QTableWidget(0, 7)
        self.pose_table.setHorizontalHeaderLabels((
            "Target", "Type", "XYZ (m)", "RPY ° / 误差÷10°\n固定轴 XYZ / Rz·Ry·Rx", "q (x y z w)",
            "|Δt| (m)", "ΔR÷10°"))
        self.pose_table.horizontalHeaderItem(3).setToolTip(
            "估计和真值行单位为度；误差行是原始角度差除以 10° 后的无量纲数值")
        self.pose_table.horizontalHeaderItem(6).setToolTip(
            "ΔR 误差除以 10° 后的无量纲数值；10° 对应 1")
        self.pose_table.verticalHeader().setVisible(False)
        self.pose_table.setMinimumHeight(210)
        for column, width in enumerate((95, 60, 165, 155, 205, 90, 85)):
            self.pose_table.setColumnWidth(column, width)
        self.pose_table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.pose_table)
        self.tabs.addTab(views, "Tracking monitor")

    @staticmethod
    def _image_label(text):
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumSize(160, 100)
        label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        label.setStyleSheet("background:#20242a;color:#d9e0e5")
        label.setWordWrap(True)
        return label

    def _load_config(self):
        try:
            objects, calibration = load_objects(self.config_path.text())
            with open(self.config_path.text(), encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            combined_topic = self.ros_node.configure_topics(
                config["foundationpose_stereo_tracker_multi_node"]["ros__parameters"],
                config.get("grounded_sam_multi_first_mask_node", {}).get("ros__parameters", {}))
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, InvalidTopicNameException) as error:
            QtWidgets.QMessageBox.critical(self, "Configuration error", str(error))
            return
        self.calibration = calibration
        self.satellite_calibration.setText(str(calibration))
        self._shown_frames.clear()
        self.preview_labels["combined"].setText("Waiting for tracker composite")
        self.preview_labels["combined"].setToolTip(combined_topic)
        self._loading_config = True
        self.objects.setRowCount(0)
        for index, (name, mesh) in enumerate(objects):
            pose = ((index - (len(objects) - 1) / 2) * 0.3, 0.0, 0.8, 0.0, 0.0, 0.0)
            self._add_object(name, mesh, pose)
        self._loading_config = False
        self._objects_signature = None
        self._sync_objects()
        self.statusBar().showMessage(f"Loaded {len(objects)} objects; composite topic: {combined_topic}", 5000)

    def _object_changed(self, item):
        if item.column() < 2:
            self._schedule_object_sync()

    def _schedule_object_sync(self, *_args):
        if not self._loading_config:
            QtCore.QTimer.singleShot(0, self._sync_objects)

    def _sync_objects(self):
        for row in range(self.motion_table.rowCount()):
            item = self.motion_table.item(row, 0)
            if item is not None:
                self.motion_values[item.text()] = [self.motion_table.cellWidget(row, column).value()
                                                   for column in range(1, 13)]
        objects = []
        for row in range(self.objects.rowCount()):
            name, mesh = (self.objects.item(row, column) for column in (0, 1))
            if name is not None and mesh is not None and name.text().strip() and mesh.text().strip():
                objects.append((name.text().strip(), mesh.text().strip()))
        if tuple(objects) == self._objects_signature:
            return
        self._objects_signature = tuple(objects)
        selected = self.mesh_selector.currentText()
        self.ros_node.set_mask_objects(objects)
        self.mesh_selector.blockSignals(True)
        self.mesh_selector.clear()
        for name, mesh in objects:
            self.mesh_selector.addItem(name, mesh)
        self.mesh_selector.setCurrentIndex(max(0, self.mesh_selector.findText(selected)))
        self.mesh_selector.blockSignals(False)
        while self.models_layout.count():
            cell = self.models_layout.takeAt(0)
            if cell.widget():
                cell.widget().deleteLater()
        self.model_labels = {}
        self.pose_table.setRowCount(len(objects) * 3)
        self.motion_table.setRowCount(len(objects))
        for row, (name, mesh) in enumerate(objects):
            self.motion_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            values = self.motion_values.get(name, [*self.motion_defaults[0], *self.motion_defaults[1]])
            for column, value in enumerate(values, 1):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(0.0, 360.0 if 4 <= column <= 6 else 100.0 if column <= 3 else 20.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)
                spin.setValue(value)
                self.motion_table.setCellWidget(row, column, spin)
            label = self._image_label(f"Loading {name}...")
            label.setToolTip(mesh)
            self.models_layout.addWidget(label, 1)
            self.model_labels[name] = label
            for offset, kind in enumerate(("Estimate", "Truth", "Error")):
                table_row = row * 3 + offset
                self.pose_table.setRowHeight(table_row, 25)
                self.pose_table.setItem(table_row, 0, QtWidgets.QTableWidgetItem(name))
                self.pose_table.setItem(table_row, 1, QtWidgets.QTableWidgetItem(kind))
                for column in range(2, 7):
                    self.pose_table.setItem(table_row, column, QtWidgets.QTableWidgetItem("—"))
            if mesh not in self.mesh_jobs:
                self.mesh_jobs[mesh] = self.mesh_pool.submit(render_mesh_preview, mesh)
        self.preview_labels["mask"].setText("Waiting for first-frame masks")
        self.mask_legend.setText("Waiting for first-frame masks")
        self._shown_frames.pop("mask", None)
        self.preview_labels["mask"].setToolTip(
            f"{self.ros_node.mask_prefix}/<target>; white means overlapping masks")

    def _show_pose(self):
        for name, label in self.model_labels.items():
            index = self.mesh_selector.findText(name)
            job = self.mesh_jobs.get(self.mesh_selector.itemData(index))
            if job is not None and job.done():
                if not hasattr(label, "source_pixmap"):
                    try:
                        label.source_pixmap = image_pixmap(job.result())
                    except Exception as error:
                        label.setText(f"{name}: {error}")
                        continue
                if getattr(label, "shown_size", None) != label.size():
                    label.setPixmap(label.source_pixmap.scaled(
                        label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                    label.shown_size = label.size()
        with self.ros_node.frame_lock:
            frames = dict(self.ros_node.frames)
            poses = dict(self.ros_node.poses)
            truths = dict(self.ros_node.truth_by_stamp)
            latest_truth = dict(self.ros_node.latest_truth)
        for row in range(self.pose_table.rowCount() // 3):
            table_row = row * 3
            name = self.pose_table.item(table_row, 0).text()
            estimate = poses.get(name)
            truth = truths.get(estimate[0], {}).get(name) if estimate else None
            shown_truth = truth or latest_truth.get(name)
            self.pose_table.item(table_row + 1, 1).setText("Truth" if truth else "Latest")
            if estimate:
                for column, value in enumerate(format_pose(estimate[1]), 2):
                    self.pose_table.item(table_row, column).setText(value)
            if shown_truth:
                for column, value in enumerate(format_pose(shown_truth), 2):
                    self.pose_table.item(table_row + 1, column).setText(value)
            if estimate and truth:
                raw_error = pose_error(estimate[1], truth)
                delta_xyz, delta_rpy, delta_t, delta_r = format_error(raw_error)
                self.pose_table.item(table_row + 2, 2).setText(delta_xyz)
                rpy_item = self.pose_table.item(table_row + 2, 3)
                rpy_item.setText(delta_rpy)
                rpy_item.setToolTip("原始 RPY 误差 (°): " +
                                    "  ".join(f"{value:+.2f}" for value in raw_error[1]))
                self.pose_table.item(table_row + 2, 5).setText(delta_t)
                rotation_item = self.pose_table.item(table_row + 2, 6)
                rotation_item.setText(delta_r)
                rotation_item.setToolTip(f"原始 ΔR 误差: {raw_error[3]:.2f}°")
            else:
                for column in (2, 3, 5, 6):
                    item = self.pose_table.item(table_row + 2, column)
                    item.setText("—")
                    item.setToolTip("")
        names = [self.mesh_selector.itemText(index) for index in range(self.mesh_selector.count())]
        masks = [frames.get(f"mask:{name}") for name in names]
        self.mask_legend.setText("   ".join(
            f'<span style="color:rgb({color[2]},{color[1]},{color[0]})">■</span> '
            f'{escape(name)} ({"ready" if mask is not None else "waiting"})'
            for index, (name, mask) in enumerate(zip(names, masks))
            for color in (MASK_COLORS[index % len(MASK_COLORS)],)) +
            "   <span style=\"color:white;background:#555\">■</span> overlap")
        for key in ("left", "combined", "mask"):
            frame = frames.get(key) if key != "mask" else None
            source = frame if key != "mask" else tuple(masks)
            label = self.preview_labels[key]
            previous, size = self._shown_frames.get(key, (None, None))
            unchanged = (frame is previous) if key != "mask" else (
                previous is not None and len(previous) == len(source) and
                all(old is new for old, new in zip(previous, source)))
            if unchanged and size == label.size():
                continue
            if key == "mask" and any(mask is not None for mask in masks):
                try:
                    frame = combine_masks(names, masks)
                except ValueError as error:
                    self.preview_labels["mask"].setText(str(error))
                    self._shown_frames.pop("mask", None)
                    continue
            if frame is None:
                continue
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
            label.setPixmap(image_pixmap(display).scaled(
                label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self._shown_frames[key] = (source, label.size())

    def _browse_background(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ground stereo background", self.background_path.text(), "PNG image (*.png)")
        if path:
            self.background_path.setText(path)

    def _use_default_background(self):
        self.background_path.setText(str(DEFAULT_BACKGROUND))

    def _start_process(self, start_motion):
        if self.pending is not None:
            return
        self._sync_objects()
        if self.objects.rowCount() < 2 or self.calibration is None:
            raise ValueError("Load at least two objects and a tracker calibration")
        objects = [(self.objects.item(r, 0).text().strip(),
                    self.objects.item(r, 1).text().strip(),
                    [self.objects.cellWidget(r, c).value() for c in range(2, 8)],
                    [self.motion_table.cellWidget(r, c).value() for c in range(1, 7)],
                    [self.motion_table.cellWidget(r, c).value() for c in range(7, 13)])
                   for r in range(self.objects.rowCount())]
        if len({item[0] for item in objects}) != len(objects):
            raise ValueError("Object names must be unique")
        self.pending_names = [item[0] for item in objects]
        ros2 = shutil.which("ros2")
        if not ros2:
            raise RuntimeError("Source the ROS 2 and satellite_stereo workspaces first")
        if not self.calibration.is_file():
            raise FileNotFoundError(str(self.calibration))
        if self.ros_node.control_publisher.get_subscription_count() == 0:
            raise RuntimeError("Satellite render node is not online; start it on the satellite first")
        background = Path(self.background_path.text()).expanduser().resolve()
        if not background.is_file() or background.suffix.lower() != ".png":
            raise ValueError("Select an existing PNG stereo background on the ground host")
        read_background(background)
        receiver_settings = (str(background), self.background_disparity.value(), str(self.calibration))
        self.pending_config = configuration(
            640, 480, self.fps_input.value(), int(self.link_input.value() * 1000000),
            self.satellite_calibration.text().strip(), objects)
        if self.receiver is not None and self.receiver.poll() is None and receiver_settings != getattr(self, "receiver_settings", None):
            os.killpg(self.receiver.pid, signal.SIGINT)
            self.receiver.wait(timeout=3)
        if self.receiver is None or self.receiver.poll() is not None:
            self.receiver = subprocess.Popen(
                [ros2, "run", "satellite_stereo", "stereo_bytes_to_images_node", "--ros-args",
                 "-p", f"calibration_path:={self.calibration}",
                 "-p", f"background_path:={background}",
                 "-p", f"background_disparity_px:={self.background_disparity.value()}"],
                cwd=REPO_ROOT, start_new_session=True)
            self.receiver_settings = receiver_settings
        self._request(3, "reconfigure")

    def _request(self, kind, stage, payload=b""):
        self.request_id = (self.request_id + 1) & 0xFFFFFFFF
        request_id = self.request_id
        self.pending = (request_id, stage)
        self.ros_node.send_control(kind, request_id, payload)
        self.motion_status.setText(f"Satellite: {stage}...")
        QtCore.QTimer.singleShot(60000, lambda: self._request_timeout(request_id))

    def _request_timeout(self, request_id):
        if self.pending is not None and self.pending[0] == request_id:
            self.pending = None
            self.statusBar().showMessage("Satellite control response timed out", 10000)
            QtWidgets.QMessageBox.warning(self, "Satellite timeout", "No SCT1 acknowledgement received")

    def _on_control_response(self, request_id, status, detail):
        if self.pending is None or self.pending[0] != request_id:
            return
        _, stage = self.pending
        self.pending = None
        if status != 0:
            self.statusBar().showMessage(f"Satellite rejected {stage}: {detail}", 10000)
            QtWidgets.QMessageBox.critical(self, "Satellite control failed", detail)
            return
        if stage == "reconfigure":
            self._set_motion_state(False, "Stopped")
            self._request(1, "configure", self.pending_config)
        elif stage == "configure":
            self.ros_node.set_truth_objects(self.pending_names)
            self._request(2, "start")
        elif stage == "start":
            self._set_motion_state(True, "Rendering")
        elif stage == "stop":
            self._set_motion_state(False, "Stopped")

    def _toggle_motion(self):
        try:
            if self.pending is not None:
                return
            if self.motion_running:
                self._request(3, "stop")
            else:
                self._start_process(True)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Renderer failed", str(error))

    def _stop_process(self):
        if self.ros_node.control_publisher.get_subscription_count():
            self.ros_node.send_control(3, (self.request_id + 1) & 0xFFFFFFFF)

    def closeEvent(self, event):
        self.pose_timer.stop()
        self.ros_node.on_ack = None
        self.mesh_pool.shutdown(wait=False, cancel_futures=True)
        self._stop_process()
        if self.receiver is not None and self.receiver.poll() is None:
            os.killpg(self.receiver.pid, signal.SIGINT)
            try:
                self.receiver.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.receiver.pid, signal.SIGKILL)
                self.receiver.wait()
        event.accept()


def sanitize_topic_name(name):
    return "".join(chr(byte).lower() if byte < 128 and chr(byte).isalnum()
                   else chr(byte) if byte in (45, 95) else "_"
                   for byte in name.encode("utf-8")) or "object"


def stamp_key(stamp):
    return stamp.sec, stamp.nanosec


def pose_values(pose):
    p, q = pose.position, pose.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm == 0:
        raise ValueError("zero-length pose quaternion")
    quaternion = tuple(value / norm for value in (q.x, q.y, q.z, q.w))
    return ((p.x, p.y, p.z), quaternion_to_rpy_degrees(*quaternion), quaternion)


def format_pose(values):
    position, rpy, quaternion = values
    return ("  ".join(f"{value:.3f}" for value in position),
            "  ".join(f"{value:.2f}" for value in rpy),
            "  ".join(f"{value:.3f}" for value in quaternion))


def pose_error(estimate, truth):
    delta = tuple(a - b for a, b in zip(estimate[0], truth[0]))
    delta_rpy = tuple((a - b + 180.0) % 360.0 - 180.0
                      for a, b in zip(estimate[1], truth[1]))
    dot = abs(sum(a * b for a, b in zip(estimate[2], truth[2])))
    angle = math.degrees(2 * math.acos(min(1.0, max(0.0, dot))))
    return delta, delta_rpy, math.sqrt(sum(value * value for value in delta)), angle


def scale_angle_error(degrees):
    """Convert an angular error in degrees to a dimensionless 10-degree scale."""
    return degrees / 10.0


def format_error(error):
    delta, delta_rpy, translation_error, rotation_error = error
    return ("  ".join(f"{value:+.3f}" for value in delta),
            "  ".join(f"{scale_angle_error(value):+.3f}" for value in delta_rpy),
            f"{translation_error:.3f}", f"{scale_angle_error(rotation_error):.3f}")


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


def image_pixmap(image):
    image = np.ascontiguousarray(image)
    return QtGui.QPixmap.fromImage(QtGui.QImage(
        image.data, image.shape[1], image.shape[0], image.strides[0],
        QtGui.QImage.Format_BGR888).copy())


MASK_COLORS = ((48, 88, 245), (50, 205, 65), (235, 160, 35),
               (205, 65, 200), (45, 205, 220), (170, 180, 80))


def combine_masks(names, masks):
    available = [(index, mask) for index, mask in enumerate(masks) if mask is not None]
    if not available:
        raise ValueError("No masks received")
    shape = available[0][1].shape
    if any(mask.ndim != 2 or mask.shape != shape for _, mask in available):
        raise ValueError("Mask dimensions do not match; check first-frame topics")
    image = np.zeros((*shape, 3), np.uint8)
    coverage = np.zeros(shape, np.uint8)
    for index, mask in available:
        foreground = mask > 0
        image[foreground] = MASK_COLORS[index % len(MASK_COLORS)]
        coverage += foreground.astype(np.uint8)
    image[coverage > 1] = (255, 255, 255)
    return image


def render_mesh_preview(mesh_path, size=(320, 240)):
    mesh_path = resolve_mesh_path(mesh_path)
    vertices = []
    uv = []
    faces = []
    face_uv = []
    material_file = None
    with mesh_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if fields[:1] == ["mtllib"] and len(fields) >= 2:
                material_file = mesh_path.parent / " ".join(fields[1:])
            if fields[:1] == ["v"] and len(fields) >= 4:
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[:1] == ["vt"] and len(fields) >= 3:
                uv.append([float(value) for value in fields[1:3]])
            elif fields[:1] == ["f"] and len(fields) >= 4:
                indices = [int(value.split("/")[0]) for value in fields[1:]]
                uv_indices = [int(value.split("/")[1]) if "/" in value and value.split("/")[1] else 0
                              for value in fields[1:]]
                if 0 in indices:
                    raise ValueError("OBJ face indices must be nonzero")
                indices = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                if any(index < 0 for index in indices):
                    raise ValueError("OBJ face index out of range")
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))
                    face_uv.append((uv_indices[0], uv_indices[i], uv_indices[i + 1]))
    if not vertices or not faces:
        raise ValueError("OBJ has no renderable vertices or faces")
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int32)
    if not np.all(np.isfinite(points)) or triangles.max() >= len(points):
        raise ValueError("OBJ contains invalid vertices or face indices")
    texture = None
    if material_file and material_file.is_file():
        for line in material_file.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if fields[:1] == ["map_Kd"] and len(fields) >= 2:
                texture_path = material_file.parent / " ".join(fields[1:])
                texture = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
                break
    points -= points.mean(axis=0)
    camera = np.asarray([[0.707, -0.707, 0.0], [0.408, 0.408, -0.816],
                         [0.577, 0.577, 0.577]], dtype=np.float32)
    projected = points @ camera[:2].T
    extent = np.ptp(projected, axis=0)
    scale = 0.78 * min(size[0] / max(extent[0], 1e-6), size[1] / max(extent[1], 1e-6))
    pixels = np.empty((len(points), 2), np.float32)
    pixels[:, 0] = np.rint(size[0] / 2 + projected[:, 0] * scale)
    pixels[:, 1] = np.rint(size[1] / 2 - projected[:, 1] * scale)
    depth = points @ camera[2]
    image = np.full((size[1], size[0], 3), (36, 42, 48), np.uint8)
    normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                       points[triangles[:, 2]] - points[triangles[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    shades = (65 + 150 * np.abs(normals @ camera[2])).astype(int)
    uv_points = np.asarray(uv, dtype=np.float32) if uv else np.empty((0, 2), np.float32)
    uv_triangles = np.asarray(face_uv, dtype=np.int32)
    textured = texture is not None and len(uv_points) and np.all(uv_triangles != 0)
    if textured:
        uv_triangles = np.where(uv_triangles > 0, uv_triangles - 1, len(uv_points) + uv_triangles)
        if uv_triangles.min() < 0 or uv_triangles.max() >= len(uv_points):
            raise ValueError("OBJ has invalid texture coordinates")
    for index in np.argsort(depth[triangles].mean(axis=1)):
        triangle = pixels[triangles[index]]
        if abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) < 0.4:
            continue
        x0 = max(0, int(np.floor(triangle[:, 0].min())))
        x1 = min(size[0], int(np.ceil(triangle[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(triangle[:, 1].min())))
        y1 = min(size[1], int(np.ceil(triangle[:, 1].max())) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        roi = image[y0:y1, x0:x1]
        polygon = np.rint(triangle - (x0, y0)).astype(np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        if textured:
            coordinates = uv_points[uv_triangles[index]].copy()
            coordinates[:, 0] *= texture.shape[1] - 1
            coordinates[:, 1] = (1 - coordinates[:, 1]) * (texture.shape[0] - 1)
            transform = cv2.getAffineTransform(triangle.astype(np.float32), coordinates.astype(np.float32))
            transform[:, 2] += transform[:, 0] * x0 + transform[:, 1] * y0
            patch = cv2.warpAffine(texture, transform, (x1 - x0, y1 - y0),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)
            roi[mask > 0] = patch[mask > 0]
            continue
        shade = int(shades[index])
        roi[mask > 0] = (shade // 2, shade, min(255, shade + 24))
    return image


def main():
    if "--check" in sys.argv:
        objects, calibration = load_objects(DEFAULT_CONFIG)
        if not calibration.is_file() or not shutil.which("ros2"):
            raise RuntimeError("Calibration or sourced ROS 2 environment missing")
        print(f"configuration valid: {len(objects)} objects")
        return 0
    app = QtWidgets.QApplication(sys.argv)
    rclpy.init()
    node = TrackingNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    window = SatelliteWindow(node)
    window.show()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(250)
    try:
        return app.exec_()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
