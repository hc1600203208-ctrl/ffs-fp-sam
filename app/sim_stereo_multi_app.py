#!/usr/bin/env python3
"""Desktop controls for rendering several meshes into one stereo pair."""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from PyQt5 import QtCore, QtWidgets
import rclpy
from rclpy.executors import SingleThreadedExecutor

from sim_stereo_app import (
    REPO_ROOT,
    SOURCE_LEFT_INFO_TOPIC,
    SOURCE_LEFT_TOPIC,
    SOURCE_RIGHT_INFO_TOPIC,
    SOURCE_RIGHT_TOPIC,
    SimBridgeNode,
    SimWindow,
    resolve_mesh_path,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "fp/src/foundationpose_cpp/config/"
    "foundationpose_stereo_tracker_multi_bluepink_example.yaml"
)
POSE_LABELS = ("X (m)", "Y (m)", "Z (m)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)")


def load_objects(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    parameters = config["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
    names = parameters["tracked_object_names"]
    meshes = parameters["tracked_mesh_paths"]
    if not isinstance(names, list) or not isinstance(meshes, list) or len(names) < 2 or len(names) != len(meshes):
        raise ValueError("Tracker YAML needs at least two matching object names and mesh paths")
    calibration = Path(parameters["caminfo_path"]).expanduser().resolve()
    return list(zip(names, meshes)), calibration


def renderer_command(ros2, objects, poses, calibration):
    if len(objects) < 2 or len(poses) != len(objects):
        raise ValueError("Select at least two meshes with an initial pose for each")
    names = [name.strip() for name, _ in objects]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Object names must be nonempty and unique")
    meshes = [str(resolve_mesh_path(path)) for _, path in objects]
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {calibration}")
    parameters = {
        "mesh_paths": meshes,
        "object_names": names,
        "initial_poses": [float(value) for pose in poses for value in pose],
        "calibration_path": str(calibration),
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "interactive": True,
        "left_image_topic": SOURCE_LEFT_TOPIC,
        "right_image_topic": SOURCE_RIGHT_TOPIC,
        "left_camera_info_topic": SOURCE_LEFT_INFO_TOPIC,
        "right_camera_info_topic": SOURCE_RIGHT_INFO_TOPIC,
    }
    command = [ros2, "run", "sim_stereo_cpp", "stereo_render_multi_cpp_node", "--ros-args"]
    for key, value in parameters.items():
        encoded = yaml.safe_dump(value, default_flow_style=True).strip() if isinstance(value, list) else str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(("-p", f"{key}:={encoded}"))
    return command


class MultiSimWindow(SimWindow):
    def __init__(self, ros_node):
        self.calibration = None
        super().__init__(ros_node)
        self.setWindowTitle("Multi-Mesh Stereo Simulation")
        self.resize(1300, 650)
        self._load_config()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(9)

        paths = QtWidgets.QFrame()
        paths.setObjectName("panel")
        path_layout = QtWidgets.QGridLayout(paths)
        path_layout.addWidget(QtWidgets.QLabel("Tracker YAML"), 0, 0)
        self.config_path = QtWidgets.QLineEdit(str(DEFAULT_CONFIG))
        path_layout.addWidget(self.config_path, 0, 1)
        browse_config = QtWidgets.QToolButton()
        browse_config.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        browse_config.setToolTip("Select tracker YAML")
        browse_config.clicked.connect(self._browse_config)
        path_layout.addWidget(browse_config, 0, 2)
        load_button = QtWidgets.QPushButton("Load objects")
        load_button.clicked.connect(self._load_config)
        path_layout.addWidget(load_button, 0, 3)

        path_layout.addWidget(QtWidgets.QLabel("Background image"), 1, 0)
        self.background_path = QtWidgets.QLineEdit()
        self.background_path.setPlaceholderText("Built-in studio background")
        self.background_path.setReadOnly(True)
        path_layout.addWidget(self.background_path, 1, 1)
        browse_background = QtWidgets.QToolButton()
        browse_background.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        browse_background.setToolTip("Select background image")
        browse_background.clicked.connect(self._browse_background)
        path_layout.addWidget(browse_background, 1, 2)
        reset_background = QtWidgets.QToolButton()
        reset_background.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        reset_background.setToolTip("Restore built-in background")
        reset_background.clicked.connect(self._use_default_background)
        path_layout.addWidget(reset_background, 1, 3)
        root.addWidget(paths)

        objects_panel = QtWidgets.QFrame()
        objects_panel.setObjectName("panel")
        objects_layout = QtWidgets.QVBoxLayout(objects_panel)
        title = QtWidgets.QHBoxLayout()
        title.addWidget(QtWidgets.QLabel("Objects (render order: top to bottom)"))
        title.addStretch()
        add_button = QtWidgets.QPushButton("Add object")
        add_button.clicked.connect(lambda: self._add_object("", "", self._default_pose(self.objects.rowCount())))
        title.addWidget(add_button)
        browse_mesh = QtWidgets.QPushButton("Browse selected mesh")
        browse_mesh.clicked.connect(self._browse_selected_mesh)
        title.addWidget(browse_mesh)
        remove_button = QtWidgets.QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_selected)
        title.addWidget(remove_button)
        objects_layout.addLayout(title)

        self.objects = QtWidgets.QTableWidget(0, 8)
        self.objects.setHorizontalHeaderLabels(("Name", "Mesh path", *POSE_LABELS))
        self.objects.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.objects.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.objects.verticalHeader().setVisible(False)
        self.objects.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.objects.setColumnWidth(0, 130)
        for column in range(2, 8):
            self.objects.setColumnWidth(column, 104)
        self.objects.setMinimumHeight(170)
        objects_layout.addWidget(self.objects)
        root.addWidget(objects_panel, 1)

        motion_panel = QtWidgets.QFrame()
        motion_panel.setObjectName("panel")
        motion_layout = QtWidgets.QVBoxLayout(motion_panel)
        self.translation_amplitude_inputs = self._add_three_inputs(
            motion_layout, "Shared position amp.", ("X (m)", "Y (m)", "Z (m)"),
            (0.05, 0.05, 0.02), (0.0, 100.0),
        )
        self.rotation_amplitude_inputs = self._add_three_inputs(
            motion_layout, "Shared rotation amp.", ("Roll (deg)", "Pitch (deg)", "Yaw (deg)"),
            (10.0, 10.0, 20.0), (0.0, 360.0),
        )
        self.frequency_inputs = self._add_six_inputs(
            motion_layout, "Shared frequency",
            ("X (Hz)", "Y (Hz)", "Z (Hz)", "Roll (Hz)", "Pitch (Hz)", "Yaw (Hz)"),
            (0.15, 0.12, 0.10, 0.10, 0.08, 0.06), (0.0, 20.0),
        )
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start motion")
        self.start_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self._toggle_motion)
        buttons.addWidget(self.start_button)
        self.reset_button = QtWidgets.QPushButton("Apply parameters and reset")
        self.reset_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.reset_button.clicked.connect(self._reset_renderer)
        buttons.addWidget(self.reset_button)
        buttons.addStretch()
        self.motion_status = QtWidgets.QLabel("Not started")
        self.motion_status.setObjectName("motionStatus")
        buttons.addWidget(self.motion_status)
        motion_layout.addLayout(buttons)
        root.addWidget(motion_panel)
        self.statusBar().showMessage("Publishing /left/image_raw and /right/image_raw")

    @staticmethod
    def _default_pose(index):
        return ((index - 0.5) * 0.3, 0.0, 0.8, 0.0, 0.0, 0.0)

    def _add_object(self, name, mesh, pose):
        row = self.objects.rowCount()
        self.objects.insertRow(row)
        self.objects.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        self.objects.setItem(row, 1, QtWidgets.QTableWidgetItem(str(mesh)))
        for index, value in enumerate(pose):
            spin = self._spin(value, (-100.0, 100.0))
            self.objects.setCellWidget(row, index + 2, spin)
        self.objects.setRowHeight(row, 40)

    def _remove_selected(self):
        row = self.objects.currentRow()
        if row >= 0:
            self.objects.removeRow(row)

    def _browse_selected_mesh(self):
        row = self.objects.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Select object", "Select an object row first")
            return
        current = self.objects.item(row, 1).text()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select OBJ mesh", current or str(Path.home()), "OBJ mesh (*.obj)"
        )
        if path:
            self.objects.item(row, 1).setText(path)

    def _browse_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select tracker YAML", self.config_path.text(), "YAML (*.yaml *.yml)"
        )
        if path:
            self.config_path.setText(path)
            self._load_config()

    def _load_config(self):
        try:
            objects, calibration = load_objects(self.config_path.text())
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            QtWidgets.QMessageBox.critical(self, "Configuration error", str(error))
            return
        self.calibration = calibration
        self.objects.setRowCount(0)
        for index, (name, mesh) in enumerate(objects):
            pose = ((index - (len(objects) - 1) / 2) * 0.3, 0.0, 0.8, 0.0, 0.0, 0.0)
            self._add_object(name, mesh, pose)
        self.statusBar().showMessage(f"Loaded {len(objects)} objects", 5000)

    def _start_process(self, start_motion):
        objects = [
            (self.objects.item(row, 0).text(), self.objects.item(row, 1).text())
            for row in range(self.objects.rowCount())
        ]
        poses = [
            tuple(self.objects.cellWidget(row, column).value() for column in range(2, 8))
            for row in range(self.objects.rowCount())
        ]
        ros2 = shutil.which("ros2")
        script = shutil.which("script")
        if not ros2 or not script:
            raise RuntimeError("A sourced ROS 2 environment and util-linux script are required")
        if self.calibration is None:
            raise ValueError("Load a tracker YAML with stereo calibration first")
        command = renderer_command(ros2, objects, poses, self.calibration)
        amplitudes = tuple(widget.value() for widget in self.translation_amplitude_inputs + self.rotation_amplitude_inputs)
        frequencies = tuple(widget.value() for widget in self.frequency_inputs)
        for key, value in (("amplitudes", amplitudes), ("frequencies", frequencies)):
            command.extend(("-p", f"{key}:={yaml.safe_dump(list(value), default_flow_style=True).strip()}"))

        self._stop_process()
        self.process = subprocess.Popen(
            [script, "-qefc", shlex.join(command), os.devnull],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
        if start_motion:
            self._write_commands(["start"])
        self._set_motion_state(start_motion, "Running" if start_motion else "Reset; waiting to start")


def main():
    if "--check" in sys.argv:
        objects, calibration = load_objects(DEFAULT_CONFIG)
        renderer_command("ros2", objects, [MultiSimWindow._default_pose(i) for i in range(len(objects))], calibration)
        if not shutil.which("ros2") or not shutil.which("script"):
            raise RuntimeError("ros2 and script must be available")
        print(f"configuration valid: {len(objects)} objects")
        return 0

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Multi-Mesh Stereo Simulation")
    rclpy.init(args=None)
    ros_node = SimBridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    window = MultiSimWindow(ros_node)
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
