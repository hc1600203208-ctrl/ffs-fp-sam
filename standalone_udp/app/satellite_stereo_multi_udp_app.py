#!/usr/bin/env python3
"""ROS-free ground control, UDP stereo receive, and multi-object pose monitor."""
import math
import socket
import struct
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from sct_protocol import command, configuration, parse_ack  # noqa: E402
from udp_wire import ACK, CONTROL, RESULT, STEREO_PNG, TRUTH, Reassembler, encode  # noqa: E402


def quaternion_to_rpy(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def errors(estimate, truth):
    xyz = tuple(a - b for a, b in zip(estimate[:3], truth[:3]))
    rpy_e, rpy_t = quaternion_to_rpy(estimate[3:]), quaternion_to_rpy(truth[3:])
    rpy = tuple((a - b + 180) % 360 - 180 for a, b in zip(rpy_e, rpy_t))
    dot = abs(sum(a * b for a, b in zip(estimate[3:], truth[3:])))
    rotation = math.degrees(2 * math.acos(min(1, dot)))
    return xyz, rpy, math.sqrt(sum(v * v for v in xyz)), rotation


def pixmap(image):
    image = np.ascontiguousarray(image)
    return QtGui.QPixmap.fromImage(QtGui.QImage(
        image.data, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_BGR888).copy())


class Link(QtCore.QObject):
    ack = QtCore.pyqtSignal(int, int, str)
    image = QtCore.pyqtSignal(object)
    truth = QtCore.pyqtSignal(int, object)
    result = QtCore.pyqtSignal(int, object, object, object)
    fault = QtCore.pyqtSignal(str)

    def __init__(self, listen_port, tracker_port, result_port, background, disparity):
        super().__init__()
        self.listen_port = listen_port
        self.tracker_port = tracker_port
        self.result_port = result_port
        self.background = cv2.imread(str(background), cv2.IMREAD_COLOR) if background else None
        self.disparity = disparity
        self.stop_event = threading.Event()
        self.satellite = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.satellite.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.satellite.bind(("0.0.0.0", listen_port))
        self.satellite.settimeout(0.2)
        self.tracker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tracker.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.tracker.bind(("127.0.0.1", result_port))
        self.tracker.settimeout(0.2)
        self.threads = [threading.Thread(target=self._satellite_loop, daemon=True),
                        threading.Thread(target=self._tracker_loop, daemon=True)]
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=1)
        self.satellite.close()
        self.tracker.close()

    def send_control(self, host, port, kind, request_id, payload=b""):
        for packet in encode(CONTROL, request_id, 0, command(kind, request_id, payload)):
            self.satellite.sendto(packet, (host, port))

    def _composite(self, stereo):
        if self.background is None:
            return stereo
        height, total_width = stereo.shape[:2]
        width = total_width // 2
        bg_h, bg_w = self.background.shape[:2]
        for eye in range(2):
            image = stereo[:, eye * width:(eye + 1) * width]
            scaled = cv2.resize(self.background, (width, height))
            shift = int(eye * self.disparity * bg_w / width)
            if shift:
                scaled = np.roll(scaled, -round(shift * width / bg_w), axis=1)
            mask = np.all(image == 0, axis=2)
            image[mask] = scaled[mask]
        return stereo

    def _satellite_loop(self):
        reassembler = Reassembler()
        while not self.stop_event.is_set():
            try:
                packet, peer = self.satellite.recvfrom(2048)
            except socket.timeout:
                reassembler.expire()
                continue
            except OSError:
                break
            message = reassembler.push(packet, peer)
            if not message:
                continue
            kind, frame_id, stamp_ns, payload = message
            try:
                if kind == ACK:
                    request_id, status, detail = parse_ack(payload)
                    self.ack.emit(request_id, status, detail)
                elif kind == TRUTH:
                    count = struct.unpack_from("<H", payload)[0]
                    if len(payload) != 2 + count * 56:
                        raise ValueError("invalid truth packet")
                    poses = [struct.unpack_from("<7d", payload, 2 + i * 56) for i in range(count)]
                    self.truth.emit(stamp_ns, poses)
                elif kind == STEREO_PNG:
                    stereo = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if stereo is None or stereo.shape[1] % 2:
                        raise ValueError("invalid stereo PNG")
                    stereo = self._composite(stereo)
                    self.image.emit(stereo[:, :stereo.shape[1] // 2].copy())
                    ok, png = cv2.imencode(".png", stereo)
                    if ok:
                        for outbound in encode(STEREO_PNG, frame_id, stamp_ns, png.tobytes()):
                            self.tracker.sendto(outbound, ("127.0.0.1", self.tracker_port))
            except (ValueError, struct.error, OSError) as error:
                self.fault.emit(str(error))

    def _tracker_loop(self):
        reassembler = Reassembler()
        while not self.stop_event.is_set():
            try:
                packet, peer = self.tracker.recvfrom(2048)
            except socket.timeout:
                reassembler.expire()
                continue
            except OSError:
                break
            message = reassembler.push(packet, peer)
            if not message or message[0] != RESULT:
                continue
            try:
                _, _, stamp, payload = message
                at = 0
                count = struct.unpack_from("<H", payload, at)[0]; at += 2
                poses = {}
                for _ in range(count):
                    length = struct.unpack_from("<H", payload, at)[0]; at += 2
                    name = payload[at:at + length].decode("utf-8"); at += length
                    valid = payload[at]; at += 1
                    pose = struct.unpack_from("<7d", payload, at); at += 56
                    if valid:
                        poses[name] = pose
                image_len = struct.unpack_from("<I", payload, at)[0]; at += 4
                image = cv2.imdecode(np.frombuffer(payload[at:at + image_len], np.uint8), cv2.IMREAD_COLOR)
                at += image_len
                mask_count = struct.unpack_from("<H", payload, at)[0]; at += 2
                masks = []
                for _ in range(mask_count):
                    length = struct.unpack_from("<I", payload, at)[0]; at += 4
                    masks.append(cv2.imdecode(np.frombuffer(payload[at:at + length], np.uint8),
                                              cv2.IMREAD_GRAYSCALE) if length else None)
                    at += length
                self.result.emit(stamp, poses, image, masks)
            except (ValueError, struct.error, UnicodeDecodeError) as error:
                self.fault.emit(str(error))


class Window(QtWidgets.QMainWindow):
    def __init__(self, yaml_path, satellite_ip, satellite_port, listen_port):
        super().__init__()
        self.yaml_path = Path(yaml_path)
        self.satellite_port = satellite_port
        self.truths = {}
        self.results = {}
        self.latest_truth = None
        self.latest_result_stamp = None
        self.latest_result_poses = {}
        self.pending = None
        self.request_id = 0
        self._build_ui(satellite_ip)
        self._load_yaml()
        self.link = Link(listen_port, 50100, 50101, ROOT / "app/assets/stereo_background.png", 8)
        self.link.ack.connect(self._ack)
        self.link.image.connect(lambda frame: self._show_image(self.left_label, frame))
        self.link.truth.connect(self._truth)
        self.link.result.connect(self._result)
        self.link.fault.connect(lambda reason: self.statusBar().showMessage(reason, 5000))

    def _build_ui(self, satellite_ip):
        self.setWindowTitle("Satellite stereo UDP control and tracking (no ROS)")
        self.resize(1450, 900)
        tabs = QtWidgets.QTabWidget(); self.setCentralWidget(tabs)
        controls = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(controls)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Satellite IP"))
        self.host = QtWidgets.QLineEdit(satellite_ip); row.addWidget(self.host)
        row.addWidget(QtWidgets.QLabel("Calibration in satellite bundle"))
        self.calibration = QtWidgets.QLineEdit("calibration.txt"); row.addWidget(self.calibration)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.fps = QtWidgets.QDoubleSpinBox(); self.fps.setRange(0.1, 60); self.fps.setValue(8)
        self.rate = QtWidgets.QDoubleSpinBox(); self.rate.setRange(0.1, 1000); self.rate.setValue(10)
        row.addWidget(QtWidgets.QLabel("FPS")); row.addWidget(self.fps)
        row.addWidget(QtWidgets.QLabel("Link MB/s")); row.addWidget(self.rate)
        self.start = QtWidgets.QPushButton("Configure and start satellite")
        self.start.clicked.connect(self._start)
        row.addWidget(self.start)
        self.stop = QtWidgets.QPushButton("Stop satellite")
        self.stop.clicked.connect(lambda: self._send(3, "stop"))
        row.addWidget(self.stop)
        layout.addLayout(row)
        self.initial = QtWidgets.QTableWidget(0, 8)
        self.initial.setHorizontalHeaderLabels(("Target", "Ground mesh", "X m", "Y m", "Z m", "Roll °", "Pitch °", "Yaw °"))
        self.initial.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(QtWidgets.QLabel("Initial poses; satellite mesh path is meshes/<row>/mesh file"))
        layout.addWidget(self.initial)
        self.motion = QtWidgets.QTableWidget(0, 13)
        self.motion.setHorizontalHeaderLabels(("Target", *(f"Amp {axis}" for axis in "XYZRPY"),
                                               *(f"Freq {axis}" for axis in "XYZRPY")))
        layout.addWidget(QtWidgets.QLabel("Independent motion: amplitude (m or degrees), frequency (Hz)"))
        layout.addWidget(self.motion)
        tabs.addTab(controls, "Parameters and control")
        monitor = QtWidgets.QWidget(); view = QtWidgets.QVBoxLayout(monitor)
        self.left_label = QtWidgets.QLabel("Waiting for satellite left image")
        self.combined_label = QtWidgets.QLabel("Waiting for tracker")
        self.mask_label = QtWidgets.QLabel("Waiting for first masks")
        for label in (self.left_label, self.combined_label, self.mask_label):
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumHeight(120)
            label.setStyleSheet("background:#20242a;color:#d9e0e5")
            label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        view.addWidget(QtWidgets.QLabel("Satellite left image")); view.addWidget(self.left_label, 2)
        view.addWidget(QtWidgets.QLabel("Tracker: left | pose | depth")); view.addWidget(self.combined_label, 2)
        view.addWidget(QtWidgets.QLabel("First-frame masks")); view.addWidget(self.mask_label, 1)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(("Target", "Type", "XYZ m", "RPY ° (ZYX) / error÷10",
                                               "q(x y z w)", "|Δt| m", "ΔR÷10°"))
        for i, width in enumerate((135, 90, 185, 255, 225, 90, 90)):
            self.table.setColumnWidth(i, width)
        self.table.setMinimumHeight(230)
        view.addWidget(QtWidgets.QLabel("Euler order: R = Rz(yaw) Ry(pitch) Rx(roll). Angular error columns show raw degrees ÷ 10."))
        view.addWidget(self.table)
        tabs.addTab(monitor, "Tracking monitor")

    def _load_yaml(self):
        config = yaml.safe_load(self.yaml_path.read_text())
        fp = config["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
        self.names = fp["tracked_object_names"]
        meshes = fp["tracked_mesh_paths"]
        self.initial.setRowCount(len(self.names)); self.motion.setRowCount(len(self.names))
        self.table.setRowCount(len(self.names) * 3)
        for i, (name, mesh) in enumerate(zip(self.names, meshes)):
            self.initial.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.initial.setItem(i, 1, QtWidgets.QTableWidgetItem(mesh))
            pose = ((i - (len(self.names) - 1) / 2) * .3, 0, .8, 0, 0, 0)
            for j, number in enumerate(pose, 2):
                spin = QtWidgets.QDoubleSpinBox(); spin.setRange(-100, 100); spin.setDecimals(3); spin.setValue(number)
                self.initial.setCellWidget(i, j, spin)
            self.motion.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            for j, number in enumerate((.05, .05, .02, 10, 10, 20, .15, .12, .10, .10, .08, .06), 1):
                spin = QtWidgets.QDoubleSpinBox(); spin.setRange(0, 360); spin.setDecimals(3); spin.setValue(number)
                self.motion.setCellWidget(i, j, spin)
            for offset, label in enumerate(("Estimate", "Truth", "Error")):
                row = i * 3 + offset
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(label))
                for column in range(2, 7):
                    self.table.setItem(row, column, QtWidgets.QTableWidgetItem("—"))

    def _send(self, kind, stage, payload=b""):
        self.request_id = (self.request_id + 1) & 0xffffffff
        self.pending = (self.request_id, stage)
        self.link.send_control(self.host.text().strip(), self.satellite_port, kind, self.request_id, payload)
        self.statusBar().showMessage(f"Satellite {stage}…")
        QtCore.QTimer.singleShot(3000, lambda request_id=self.request_id: self._retry(request_id, kind, payload))

    def _retry(self, request_id, kind, payload):
        if self.pending and self.pending[0] == request_id:
            self.link.send_control(self.host.text().strip(), self.satellite_port, kind, request_id, payload)
            QtCore.QTimer.singleShot(3000, lambda: self._timeout(request_id))

    def _timeout(self, request_id):
        if self.pending and self.pending[0] == request_id:
            self.pending = None
            QtWidgets.QMessageBox.warning(self, "Satellite timeout", "No UDP acknowledgement received")

    def _start(self):
        objects = []
        for i, name in enumerate(self.names):
            mesh = Path(self.initial.item(i, 1).text())
            pose = [self.initial.cellWidget(i, j).value() for j in range(2, 8)]
            amp = [self.motion.cellWidget(i, j).value() for j in range(1, 7)]
            freq = [self.motion.cellWidget(i, j).value() for j in range(7, 13)]
            objects.append((name, f"meshes/{i}/{mesh.name}", pose, amp, freq))
        self.config_payload = configuration(640, 480, self.fps.value(), int(self.rate.value() * 1e6),
                                            self.calibration.text().strip(), objects)
        self._send(3, "stop-before-configure")

    def _ack(self, request_id, status, detail):
        if not self.pending or request_id != self.pending[0]:
            return
        _, stage = self.pending; self.pending = None
        if status:
            QtWidgets.QMessageBox.critical(self, "Satellite error", detail)
            return
        if stage == "stop-before-configure": self._send(1, "configure", self.config_payload)
        elif stage == "configure": self._send(2, "start")
        else: self.statusBar().showMessage(f"Satellite {stage}: OK", 5000)

    def _truth(self, stamp, poses):
        if len(poses) != len(self.names):
            return
        self.truths[stamp] = dict(zip(self.names, poses))
        self.latest_truth = self.truths[stamp]
        if len(self.truths) > 256:
            del self.truths[next(iter(self.truths))]
        shown_stamp = self.latest_result_stamp if self.latest_result_stamp is not None else stamp
        self._update_table(shown_stamp, self.latest_result_poses)

    def _result(self, stamp, poses, image, masks):
        if self.latest_result_stamp is not None and stamp < self.latest_result_stamp:
            return
        self.latest_result_stamp = stamp
        self.latest_result_poses = poses
        self.results[stamp] = poses
        if len(self.results) > 256:
            del self.results[next(iter(self.results))]
        if image is not None: self._show_image(self.combined_label, image)
        valid_masks = [mask for mask in masks if mask is not None]
        if valid_masks:
            colored = np.zeros((*valid_masks[0].shape, 3), np.uint8)
            for i, mask in enumerate(masks):
                if mask is not None: colored[mask > 0] = ((i * 87 + 40) % 255, (i * 113 + 75) % 255, (i * 67 + 200) % 255)
            self._show_image(self.mask_label, colored)
        self._update_table(stamp, poses)

    def _update_table(self, stamp, poses):
        same_frame = self.truths.get(stamp, {})
        for i, name in enumerate(self.names):
            row = i * 3
            estimate = poses.get(name)
            truth = same_frame.get(name)
            shown_truth = truth or (self.latest_truth or {}).get(name)
            self.table.item(row + 1, 1).setText("Truth" if truth else "Latest")
            for row_offset, pose in ((0, estimate), (1, shown_truth)):
                if pose:
                    xyz = "  ".join(f"{v:.3f}" for v in pose[:3])
                    rpy = "  ".join(f"{v:.2f}" for v in quaternion_to_rpy(pose[3:]))
                    quat = "  ".join(f"{v:.3f}" for v in pose[3:])
                    for column, text in ((2, xyz), (3, rpy), (4, quat)):
                        self.table.item(row + row_offset, column).setText(text)
                else:
                    for column in (2, 3, 4):
                        self.table.item(row + row_offset, column).setText("—")
            if estimate and truth:
                xyz, rpy, distance, rotation = errors(estimate, truth)
                values = {2: "  ".join(f"{v:+.3f}" for v in xyz),
                          3: "  ".join(f"{v / 10:+.3f}" for v in rpy),
                          5: f"{distance:.3f}", 6: f"{rotation / 10:.3f}"}
                for column, text in values.items(): self.table.item(row + 2, column).setText(text)
                self.table.item(row + 2, 3).setToolTip("Raw RPY degrees: " + str(rpy))
                self.table.item(row + 2, 6).setToolTip(f"Raw ΔR: {rotation:.3f}°")
            else:
                for column in (2, 3, 5, 6): self.table.item(row + 2, column).setText("—")

    def _show_image(self, label, image):
        label.setPixmap(pixmap(image).scaled(label.size(), QtCore.Qt.KeepAspectRatio,
                                               QtCore.Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.link.close()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="tracker YAML from the existing project")
    parser.add_argument("--satellite-ip", required=True)
    parser.add_argument("--satellite-port", type=int, default=50000)
    parser.add_argument("--listen-port", type=int, default=50000)
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = Window(args.config, args.satellite_ip, args.satellite_port, args.listen_port)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
