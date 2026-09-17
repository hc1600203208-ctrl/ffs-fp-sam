#!/usr/bin/env python3

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from PyQt5 import QtCore, QtWidgets
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MESH_DIRECTORY = Path("/home/hc/weizi/dataset/jrnew-blue/mesh")
DEFAULT_CALIBRATION = REPO_ROOT / "ffs/64091.txt"
SOURCE_LEFT_TOPIC = "/sim_app/rendered_left"
SOURCE_RIGHT_TOPIC = "/sim_app/rendered_right"
SOURCE_LEFT_INFO_TOPIC = "/sim_app/left/camera_info"
SOURCE_RIGHT_INFO_TOPIC = "/sim_app/right/camera_info"


@dataclass(frozen=True)
class MotionValues:
    pose: tuple
    translation_amplitude: tuple
    rotation_amplitude: tuple
    frequency: tuple


def resolve_mesh_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".obj":
            raise ValueError("Mesh file must be an .obj")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Mesh directory does not exist: {path}")
    candidates = sorted(path.glob("*.obj"))
    if not candidates:
        raise FileNotFoundError(f"No .obj file found in mesh directory: {path}")
    for preferred in ("textured_mesh.obj", "textured_simple.obj"):
        candidate = path / preferred
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_background(width=1280, height=720):
    image = np.empty((height, width, 3), np.uint8)
    horizon = int(height * 0.62)
    image[:horizon] = (196, 201, 204)
    image[horizon:] = (126, 132, 135)

    for y in range(horizon):
        light = int(28 * (1.0 - y / max(1, horizon)))
        image[y] = np.clip(image[y].astype(np.int16) + (light, light, light), 0, 255)
    for y in range(horizon, height):
        shade = int(34 * (y - horizon) / max(1, height - horizon))
        image[y] = np.clip(image[y].astype(np.int16) - (shade, shade, shade), 0, 255)

    cv2.line(image, (0, horizon), (width, horizon), (82, 88, 92), 3, cv2.LINE_AA)
    for x in range(0, width, max(1, width // 8)):
        cv2.line(image, (x, 0), (x, horizon), (181, 187, 191), 1, cv2.LINE_AA)
    for y in range(height // 7, horizon, height // 7):
        cv2.line(image, (0, y), (width, y), (181, 187, 191), 1, cv2.LINE_AA)

    center = (width // 2, int(height * 0.77))
    axes = (int(width * 0.25), int(height * 0.08))
    shadow = np.zeros_like(image)
    cv2.ellipse(shadow, center, axes, 0, 0, 360, (45, 48, 50), -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(8, width / 90))
    image = cv2.addWeighted(image, 1.0, shadow, 0.24, 0.0)
    return image


def read_background(path):
    data = np.fromfile(str(Path(path).expanduser()), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read background image: {path}")
    return image


def fit_background(background, width, height):
    source_height, source_width = background.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        background,
        (max(width, round(source_width * scale)), max(height, round(source_height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x = (resized.shape[1] - width) // 2
    y = (resized.shape[0] - height) // 2
    return resized[y : y + height, x : x + width].copy()


def foreground_mask(rendered_bgr):
    non_black = np.max(rendered_bgr, axis=2) > 2
    binary = non_black.astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(binary.shape, np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= 4.0:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask


def composite_background(rendered_bgr, background):
    fitted = fit_background(background, rendered_bgr.shape[1], rendered_bgr.shape[0])
    mask = foreground_mask(rendered_bgr)
    result = fitted
    result[mask > 0] = rendered_bgr[mask > 0]
    return result


def motion_commands(values):
    pose = " ".join(f"{value:g}" for value in values.pose)
    translation_amplitude = " ".join(
        f"{value:g}" for value in values.translation_amplitude
    )
    rotation_amplitude = " ".join(f"{value:g}" for value in values.rotation_amplitude)
    frequency = " ".join(f"{value:g}" for value in values.frequency)
    return [
        f"pose {pose}",
        f"amp {translation_amplitude} {rotation_amplitude}",
        f"freq {frequency}",
    ]


class SimBridgeNode(Node):
    def __init__(self):
        super().__init__("sim_stereo_visual_app")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.background = default_background()

        reliable_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.left_publisher = self.create_publisher(Image, "/left/image_raw", reliable_image_qos)
        self.right_publisher = self.create_publisher(Image, "/right/image_raw", reliable_image_qos)
        self.left_info_publisher = self.create_publisher(CameraInfo, "/left/camera_info", 10)
        self.right_info_publisher = self.create_publisher(CameraInfo, "/right/camera_info", 10)

        self.create_subscription(
            Image,
            SOURCE_LEFT_TOPIC,
            lambda message: self._render_callback(message, self.left_publisher),
            reliable_image_qos,
        )
        self.create_subscription(
            Image,
            SOURCE_RIGHT_TOPIC,
            lambda message: self._render_callback(message, self.right_publisher),
            reliable_image_qos,
        )
        self.create_subscription(
            CameraInfo,
            SOURCE_LEFT_INFO_TOPIC,
            self.left_info_publisher.publish,
            10,
        )
        self.create_subscription(
            CameraInfo,
            SOURCE_RIGHT_INFO_TOPIC,
            self.right_info_publisher.publish,
            10,
        )

    def set_background(self, image):
        with self.lock:
            self.background = image.copy()

    def _render_callback(self, message, publisher):
        try:
            rendered = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            with self.lock:
                background = self.background.copy()
            output = composite_background(rendered, background)
            output_message = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
            output_message.header = message.header
            publisher.publish(output_message)
        except Exception as error:
            self.get_logger().warning(f"cannot compose image: {error}")


class SimWindow(QtWidgets.QMainWindow):
    process_message = QtCore.pyqtSignal(str)

    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.process = None
        self.motion_running = False
        self.background_image = default_background()
        self.current_mesh = None

        self.setWindowTitle("Stereo Mesh Simulation")
        self.resize(1500, 340)
        self._build_ui()
        self._apply_style()
        self.process_message.connect(self._process_line)
        self.ros_node.set_background(self.background_image)

        self.process_timer = QtCore.QTimer(self)
        self.process_timer.timeout.connect(self._poll_process)
        self.process_timer.start(250)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(9)

        paths = QtWidgets.QFrame()
        paths.setObjectName("panel")
        path_layout = QtWidgets.QGridLayout(paths)
        path_layout.setContentsMargins(10, 8, 10, 8)
        path_layout.addWidget(QtWidgets.QLabel("Mesh directory"), 0, 0)
        self.mesh_path = QtWidgets.QLineEdit(str(DEFAULT_MESH_DIRECTORY))
        path_layout.addWidget(self.mesh_path, 0, 1)
        mesh_button = QtWidgets.QToolButton()
        mesh_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        mesh_button.setToolTip("Select mesh directory")
        mesh_button.clicked.connect(self._browse_mesh)
        path_layout.addWidget(mesh_button, 0, 2)

        path_layout.addWidget(QtWidgets.QLabel("Background image"), 1, 0)
        self.background_path = QtWidgets.QLineEdit()
        self.background_path.setPlaceholderText("Built-in studio background")
        self.background_path.setReadOnly(True)
        path_layout.addWidget(self.background_path, 1, 1)
        background_button = QtWidgets.QToolButton()
        background_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        background_button.setToolTip("Select background image")
        background_button.clicked.connect(self._browse_background)
        path_layout.addWidget(background_button, 1, 2)
        default_background_button = QtWidgets.QToolButton()
        default_background_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        default_background_button.setToolTip("Restore built-in background")
        default_background_button.clicked.connect(self._use_default_background)
        path_layout.addWidget(default_background_button, 1, 3)
        root.addWidget(paths)

        motion_panel = QtWidgets.QFrame()
        motion_panel.setObjectName("panel")
        motion_layout = QtWidgets.QVBoxLayout(motion_panel)
        motion_layout.setContentsMargins(10, 8, 10, 8)
        motion_layout.setSpacing(7)
        self.pose_inputs = self._add_six_inputs(
            motion_layout,
            "Initial pose",
            ("X (m)", "Y (m)", "Z (m)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"),
            (0.0, 0.0, 0.8, 0.0, 0.0, 0.0),
            (-100.0, 100.0),
        )
        self.translation_amplitude_inputs = self._add_three_inputs(
            motion_layout,
            "Translation amplitude",
            ("X (m)", "Y (m)", "Z (m)"),
            (0.05, 0.05, 0.02),
            (0.0, 100.0),
        )
        self.rotation_amplitude_inputs = self._add_three_inputs(
            motion_layout,
            "Rotation amplitude",
            ("Roll (deg)", "Pitch (deg)", "Yaw (deg)"),
            (10.0, 10.0, 20.0),
            (0.0, 360.0),
        )
        self.frequency_inputs = self._add_six_inputs(
            motion_layout,
            "Six-axis cycle speed / frequency",
            ("X (Hz)", "Y (Hz)", "Z (Hz)", "Roll (Hz)", "Pitch (Hz)", "Yaw (Hz)"),
            (0.15, 0.12, 0.10, 0.10, 0.08, 0.06),
            (0.0, 20.0),
        )

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start motion")
        self.start_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self._toggle_motion)
        buttons.addWidget(self.start_button)
        self.reset_button = QtWidgets.QPushButton("Apply parameters and reset")
        self.reset_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.reset_button.clicked.connect(self._reset_renderer)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        self.motion_status = QtWidgets.QLabel("Not started")
        self.motion_status.setObjectName("motionStatus")
        buttons.addWidget(self.motion_status)
        motion_layout.addLayout(buttons)
        root.addWidget(motion_panel)
        root.addStretch(1)

        self.statusBar().showMessage(
            "Publishing /left/image_raw and /right/image_raw"
        )

    def _add_six_inputs(self, parent, title, labels, defaults, limits):
        row = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_label.setMinimumWidth(130)
        row.addWidget(title_label)
        inputs = []
        for label, default in zip(labels, defaults):
            row.addWidget(QtWidgets.QLabel(label))
            spin = self._spin(default, limits)
            row.addWidget(spin, 1)
            inputs.append(spin)
        parent.addLayout(row)
        return inputs

    def _add_three_inputs(self, parent, title, labels, defaults, limits):
        row = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_label.setMinimumWidth(130)
        row.addWidget(title_label)
        inputs = []
        for label, default in zip(labels, defaults):
            row.addWidget(QtWidgets.QLabel(label))
            spin = self._spin(default, limits)
            row.addWidget(spin, 1)
            inputs.append(spin)
        row.addStretch(3)
        parent.addLayout(row)
        return inputs

    @staticmethod
    def _spin(default, limits):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(*limits)
        spin.setDecimals(4)
        spin.setSingleStep(0.01)
        spin.setValue(default)
        spin.setKeyboardTracking(False)
        return spin

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#eef1f3; color:#182026; font-size:13px; }
            QFrame#panel { background:#ffffff; border:1px solid #cbd1d6; border-radius:6px; }
            QLineEdit, QDoubleSpinBox { background:#ffffff; border:1px solid #adb5bd; border-radius:4px; padding:4px 6px; }
            QPushButton, QToolButton { background:#ffffff; border:1px solid #9da7af; border-radius:4px; padding:5px 9px; }
            QPushButton:hover, QToolButton:hover { background:#e7ecef; }
            QLabel#motionStatus { color:#3d4952; font-weight:600; padding:4px 8px; }
            QStatusBar { background:#ffffff; border-top:1px solid #cbd1d6; }
            """
        )

    def _browse_mesh(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select mesh directory", self.mesh_path.text()
        )
        if path:
            self.mesh_path.setText(path)

    def _browse_background(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select background image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            image = read_background(path)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Background image error", str(error))
            return
        self.background_image = image
        self.background_path.setText(path)
        self.ros_node.set_background(image)

    def _use_default_background(self):
        self.background_image = default_background()
        self.background_path.clear()
        self.ros_node.set_background(self.background_image)

    def _motion_values(self):
        return MotionValues(
            pose=tuple(widget.value() for widget in self.pose_inputs),
            translation_amplitude=tuple(
                widget.value() for widget in self.translation_amplitude_inputs
            ),
            rotation_amplitude=tuple(
                widget.value() for widget in self.rotation_amplitude_inputs
            ),
            frequency=tuple(widget.value() for widget in self.frequency_inputs),
        )

    def _start_process(self, start_motion):
        mesh = resolve_mesh_path(self.mesh_path.text())
        if not DEFAULT_CALIBRATION.is_file():
            raise FileNotFoundError(f"Calibration file does not exist: {DEFAULT_CALIBRATION}")
        ros2 = shutil.which("ros2")
        script = shutil.which("script")
        if not ros2 or not script:
            raise RuntimeError("A sourced ROS 2 environment and the util-linux script command are required")

        self._stop_process()
        command = [
            ros2,
            "run",
            "sim_stereo_cpp",
            "stereo_render_cpp_node",
            "--ros-args",
            "-p",
            f"mesh_path:={mesh}",
            "-p",
            f"calibration_path:={DEFAULT_CALIBRATION}",
            "-p",
            "width:=640",
            "-p",
            "height:=480",
            "-p",
            "fps:=30.0",
            "-p",
            "interactive:=true",
            "-p",
            f"left_image_topic:={SOURCE_LEFT_TOPIC}",
            "-p",
            f"right_image_topic:={SOURCE_RIGHT_TOPIC}",
            "-p",
            f"left_camera_info_topic:={SOURCE_LEFT_INFO_TOPIC}",
            "-p",
            f"right_camera_info_topic:={SOURCE_RIGHT_INFO_TOPIC}",
        ]
        self.process = subprocess.Popen(
            [script, "-qefc", shlex.join(command), "/dev/null"],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.current_mesh = mesh
        values = self._motion_values()
        prompt_lines = [
            " ".join(f"{value:g}" for value in values.pose[:3]),
            " ".join(f"{value:g}" for value in values.pose[3:]),
            " ".join(f"{value:g}" for value in values.translation_amplitude),
            " ".join(f"{value:g}" for value in values.frequency[:3]),
            " ".join(f"{value:g}" for value in values.rotation_amplitude),
            " ".join(f"{value:g}" for value in values.frequency[3:]),
        ]
        self._write_commands(prompt_lines + (["start"] if start_motion else []))
        threading.Thread(target=self._read_output, args=(self.process,), daemon=True).start()
        self._set_motion_state(start_motion, "Running" if start_motion else "Reset; waiting to start")

    def _write_commands(self, commands):
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("Renderer node is not running")
        self.process.stdin.write("\n".join(commands) + "\n")
        self.process.stdin.flush()

    def _read_output(self, process):
        if process.stdout is None:
            return
        for line in process.stdout:
            plain = line.replace("\x1b[0m", "").strip()
            if any(word in plain for word in ("ERROR", "FATAL", "failed", "running at")):
                self.process_message.emit(plain)

    def _process_line(self, line):
        self.statusBar().showMessage(line, 8000)

    def _toggle_motion(self):
        try:
            if self.process is None or self.process.poll() is not None:
                self._start_process(start_motion=True)
            elif self.motion_running:
                self._write_commands(["pause"])
                self._set_motion_state(False, "Motion stopped")
            else:
                self._write_commands(["start"])
                self._set_motion_state(True, "Running")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Control failed", str(error))

    def _reset_renderer(self):
        try:
            self._start_process(start_motion=False)
        except Exception as error:
            self._set_motion_state(False, "Reset failed")
            QtWidgets.QMessageBox.critical(self, "Reset failed", str(error))

    def _set_motion_state(self, running, text):
        self.motion_running = running
        self.start_button.setText("Stop motion" if running else "Start motion")
        self.start_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.SP_MediaPause if running else QtWidgets.QStyle.SP_MediaPlay
            )
        )
        self.motion_status.setText(text)
        self.motion_status.setStyleSheet(
            "color:#24735b;font-weight:600;padding:4px 8px;"
            if running
            else "color:#6c757d;font-weight:600;padding:4px 8px;"
        )

    def _poll_process(self):
        if self.process is not None and self.process.poll() is not None:
            self.process = None
            self._set_motion_state(False, "Renderer node exited")

    def _stop_process(self):
        process = self.process
        if process is None or process.poll() is not None:
            self.process = None
            return
        try:
            if process.stdin is not None:
                process.stdin.write("quit\n")
                process.stdin.flush()
            process.wait(timeout=4)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        self.process = None

    def closeEvent(self, event):
        self._stop_process()
        event.accept()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="C++ stereo mesh simulation desktop app")
    parser.add_argument("--check", action="store_true", help="validate local prerequisites")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if args.check:
        mesh = resolve_mesh_path(DEFAULT_MESH_DIRECTORY)
        if not DEFAULT_CALIBRATION.is_file():
            raise FileNotFoundError(DEFAULT_CALIBRATION)
        if not shutil.which("ros2") or not shutil.which("script"):
            raise RuntimeError("ros2 and script must be available")
        print(f"configuration valid: {mesh}")
        return 0

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Stereo Mesh Simulation")
    rclpy.init(args=None)
    ros_node = SimBridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    window = SimWindow(ros_node)
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
