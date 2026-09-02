from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .calibration import StereoCalibration, load_calibration
from .motion import MotionProfile
from .renderer import StereoRenderer


def _mat_to_quat(mat: np.ndarray) -> tuple[float, float, float, float]:
    m = mat[:3, :3]
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


class StereoRenderNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("stereo_render_node")

        self.declare_parameter("mesh_path", args.mesh_path)
        self.declare_parameter("calibration_path", args.calibration_path)
        self.declare_parameter("mesh_to_origin_path", args.mesh_to_origin_path)
        self.declare_parameter("width", args.width)
        self.declare_parameter("height", args.height)
        self.declare_parameter("fps", args.fps)
        self.declare_parameter("left_image_topic", args.left_image_topic)
        self.declare_parameter("right_image_topic", args.right_image_topic)
        self.declare_parameter("left_camera_info_topic", args.left_camera_info_topic)
        self.declare_parameter("right_camera_info_topic", args.right_camera_info_topic)
        self.declare_parameter("frame_id_left", args.frame_id_left)
        self.declare_parameter("frame_id_right", args.frame_id_right)
        self.declare_parameter("interactive", args.interactive)

        mesh_path = self.get_parameter("mesh_path").value
        calibration_path = self.get_parameter("calibration_path").value
        mesh_to_origin_path = self.get_parameter("mesh_to_origin_path").value
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)

        self.calibration: StereoCalibration = load_calibration(calibration_path)
        self.renderer = StereoRenderer(
            mesh_path=mesh_path,
            calibration=self.calibration,
            width=width,
            height=height,
            mesh_to_origin_path=mesh_to_origin_path,
        )

        self.left_image_pub = self.create_publisher(Image, self.get_parameter("left_image_topic").value, 10)
        self.right_image_pub = self.create_publisher(Image, self.get_parameter("right_image_topic").value, 10)
        self.left_info_pub = self.create_publisher(CameraInfo, self.get_parameter("left_camera_info_topic").value, 10)
        self.right_info_pub = self.create_publisher(CameraInfo, self.get_parameter("right_camera_info_topic").value, 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/sim/object_pose", 10)

        self.left_info_msg = self._make_camera_info(
            k=self.calibration.k_left,
            d=self.calibration.d_left,
            frame_id=self.get_parameter("frame_id_left").value,
        )
        self.right_info_msg = self._make_camera_info(
            k=self.calibration.k_right,
            d=self.calibration.d_right,
            frame_id=self.get_parameter("frame_id_right").value,
        )

        self.motion = MotionProfile.default()
        self._motion_lock = threading.Lock()
        self._motion_elapsed_sec = 0.0
        self._motion_running = False
        self._motion_run_started_at: float | None = None

        self.interactive = bool(self.get_parameter("interactive").value)
        if self.interactive:
            self.motion = MotionProfile.from_user_input()
            self._stdin_thread = threading.Thread(target=self._command_loop, daemon=True)
            self._stdin_thread.start()

        self.period = 1.0 / max(1e-6, float(self.get_parameter("fps").value))
        self.timer = self.create_timer(self.period, self._on_timer)

        self.get_logger().info(
            f"Publishing stereo images at {1.0 / self.period:.1f} Hz. Type 'help' for runtime commands."
        )

    def _make_camera_info(self, k: np.ndarray, d: np.ndarray, frame_id: str) -> CameraInfo:
        info = CameraInfo()
        info.width = int(self.get_parameter("width").value)
        info.height = int(self.get_parameter("height").value)
        info.k = k.reshape(-1).tolist()
        info.d = d.reshape(-1).tolist()
        info.r = np.eye(3, dtype=np.float64).reshape(-1).tolist()
        p = np.zeros((3, 4), dtype=np.float64)
        p[:3, :3] = k
        info.p = p.reshape(-1).tolist()
        info.distortion_model = "plumb_bob"
        info.header.frame_id = frame_id
        return info

    def _image_msg_from_bgr(self, image: np.ndarray, frame_id: str, stamp) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        return msg

    def _motion_now(self) -> float:
        with self._motion_lock:
            elapsed = self._motion_elapsed_sec
            running = self._motion_running
            run_started_at = self._motion_run_started_at
        if running and run_started_at is not None:
            elapsed += time.monotonic() - run_started_at
        return elapsed

    def _current_pose(self) -> np.ndarray:
        with self._motion_lock:
            motion = self.motion
        elapsed = self._motion_now()
        return motion.pose_at(elapsed)

    def _update_motion_profile(self, motion: MotionProfile) -> None:
        now = time.monotonic()
        with self._motion_lock:
            self.motion = motion
            self._motion_elapsed_sec = 0.0
            self._motion_run_started_at = now if self._motion_running else None

    def _start_motion(self) -> None:
        now = time.monotonic()
        with self._motion_lock:
            if self._motion_running:
                return
            self._motion_running = True
            self._motion_run_started_at = now

    def _pause_motion(self) -> None:
        now = time.monotonic()
        with self._motion_lock:
            if not self._motion_running:
                return
            if self._motion_run_started_at is not None:
                self._motion_elapsed_sec += now - self._motion_run_started_at
            self._motion_running = False
            self._motion_run_started_at = None

    def _toggle_motion(self) -> None:
        with self._motion_lock:
            running = self._motion_running
        if running:
            self._pause_motion()
        else:
            self._start_motion()

    def _on_timer(self) -> None:
        pose = self._current_pose()
        left_bgr, right_bgr = self.renderer.render(pose)
        stamp = self.get_clock().now().to_msg()
        left_msg = self._image_msg_from_bgr(left_bgr, self.left_info_msg.header.frame_id, stamp)
        right_msg = self._image_msg_from_bgr(right_bgr, self.right_info_msg.header.frame_id, stamp)
        self.left_image_pub.publish(left_msg)
        self.right_image_pub.publish(right_msg)

        self.left_info_msg.header.stamp = stamp
        self.right_info_msg.header.stamp = stamp
        self.left_info_pub.publish(self.left_info_msg)
        self.right_info_pub.publish(self.right_info_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.left_info_msg.header.frame_id
        pose_msg.pose.position.x = float(pose[0, 3])
        pose_msg.pose.position.y = float(pose[1, 3])
        pose_msg.pose.position.z = float(pose[2, 3])
        qx, qy, qz, qw = _mat_to_quat(pose)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg)

    def _command_loop(self) -> None:
        help_text = (
            "Commands: help | show | start | pause | toggle | pose x y z roll pitch yaw | "
            "amp tx ty tz roll pitch yaw | freq fx fy fz fr fp fyaw | "
            "reset | quit"
        )
        print(help_text)
        print("Motion is paused until you type 'start' or 'toggle'.")
        while rclpy.ok():
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            if cmd == "help":
                print(help_text)
                continue
            if cmd == "show":
                with self._motion_lock:
                    motion = self.motion
                    running = self._motion_running
                    elapsed = self._motion_elapsed_sec
                    run_started_at = self._motion_run_started_at
                if running and run_started_at is not None:
                    elapsed += time.monotonic() - run_started_at
                state = "running" if running else "paused"
                print(f"{state}, motion_time={elapsed:.3f}s")
                print(motion)
                continue
            if cmd == "start":
                self._start_motion()
                print("Motion started.")
                continue
            if cmd == "pause":
                self._pause_motion()
                print("Motion paused.")
                continue
            if cmd == "toggle":
                self._toggle_motion()
                print("Motion toggled.")
                continue
            if cmd == "reset":
                self._pause_motion()
                self._update_motion_profile(MotionProfile.default())
                print("Motion reset to defaults.")
                continue
            if cmd == "quit":
                rclpy.shutdown()
                break
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                print("Invalid numeric value.")
                continue
            with self._motion_lock:
                motion = self.motion
            if cmd == "pose" and len(values) == 6:
                self._update_motion_profile(
                    MotionProfile(
                        base_translation=np.asarray(values[:3], dtype=np.float64),
                        base_rpy_rad=np.deg2rad(np.asarray(values[3:], dtype=np.float64)),
                        trans_amp=motion.trans_amp,
                        trans_freq=motion.trans_freq,
                        rot_amp_rad=motion.rot_amp_rad,
                        rot_freq=motion.rot_freq,
                    )
                )
            elif cmd == "amp" and len(values) == 6:
                self._update_motion_profile(
                    MotionProfile(
                        base_translation=motion.base_translation,
                        base_rpy_rad=motion.base_rpy_rad,
                        trans_amp=np.asarray(values[:3], dtype=np.float64),
                        trans_freq=motion.trans_freq,
                        rot_amp_rad=np.deg2rad(np.asarray(values[3:], dtype=np.float64)),
                        rot_freq=motion.rot_freq,
                    )
                )
            elif cmd == "freq" and len(values) == 6:
                self._update_motion_profile(
                    MotionProfile(
                        base_translation=motion.base_translation,
                        base_rpy_rad=motion.base_rpy_rad,
                        trans_amp=motion.trans_amp,
                        trans_freq=np.asarray(values[:3], dtype=np.float64),
                        rot_amp_rad=motion.rot_amp_rad,
                        rot_freq=np.asarray(values[3:], dtype=np.float64),
                    )
                )
            else:
                print("Usage: pose x y z roll pitch yaw | amp tx ty tz roll pitch yaw | freq fx fy fz fr fp fyaw")
                continue
            print("Updated motion profile.")

    def destroy_node(self) -> bool:
        self.renderer.close()
        return super().destroy_node()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a textured mesh as stereo images and publish them on ROS2.")
    parser.add_argument("--mesh-path", default="/home/hc/weizi/dataset/jrnew-blue/mesh1/textured_mesh.obj")
    parser.add_argument("--calibration-path", default="/home/hc/weizi/ffs+fp+sam/ffs/640.txt")
    parser.add_argument("--mesh-to-origin-path", default="/home/hc/weizi/dataset/jrnew-blue/mesh1/to_origin.txt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--left-image-topic", default="/sim/left/image_raw")
    parser.add_argument("--right-image-topic", default="/sim/right/image_raw")
    parser.add_argument("--left-camera-info-topic", default="/sim/left/camera_info")
    parser.add_argument("--right-camera-info-topic", default="/sim/right/camera_info")
    parser.add_argument("--frame-id-left", default="stereo_left_optical_frame")
    parser.add_argument("--frame-id-right", default="stereo_right_optical_frame")
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = StereoRenderNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
