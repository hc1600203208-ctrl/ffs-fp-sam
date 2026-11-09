"""Run with QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s app."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import rclpy
import yaml
from PyQt5 import QtWidgets
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

import satellite_stereo_multi_app as app


OBJ = "v -1 -1 0\nv 1 -1 0\nv 0 1 0\nf -3 -2 -1 # triangle\n"


class PreviewTest(unittest.TestCase):
    def test_obj_preview_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            path.write_text(OBJ)
            image = app.render_mesh_preview(path, (160, 120))
            self.assertEqual(image.shape, (120, 160, 3))
            self.assertFalse(np.array_equal(image[60, 80], image[0, 0]))
            for face in ("f 1 2 99\n", "f 0 1 2\n", "f -9 -2 -1\n"):
                path.write_text(OBJ + face)
                with self.assertRaises(ValueError):
                    app.render_mesh_preview(path)

    def test_mask_topic_matches_sam_names(self):
        self.assertEqual(app.sanitize_topic_name("Blue-carton 1"), "blue-carton_1")
        self.assertEqual(app.sanitize_topic_name(""), "object")
        self.assertEqual(app.sanitize_topic_name("星"), "___")

    def test_multiple_masks_share_one_image(self):
        first = np.zeros((80, 100), np.uint8)
        second = np.zeros_like(first)
        first[30:60, 10:50] = 255
        second[45:70, 35:75] = 255
        image = app.combine_masks(["blue", "gold"], [first, second])
        self.assertTupleEqual(tuple(image[35, 20]), app.MASK_COLORS[0])
        self.assertTupleEqual(tuple(image[65, 65]), app.MASK_COLORS[1])
        self.assertTupleEqual(tuple(image[50, 40]), (255, 255, 255))
        self.assertTupleEqual(tuple(image[75, 90]), (0, 0, 0))
        partial = app.combine_masks(["blue", "gold"], [first, None])
        self.assertTupleEqual(tuple(partial[35, 20]), app.MASK_COLORS[0])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            app.combine_masks(["blue", "gold"], [first, second[:40]])

    def test_obj_texture_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            path.write_text("mtllib mesh.mtl\nv -1 -1 0\nv 1 -1 0\nv 0 1 0\n"
                            "vt 0 0\nvt 1 0\nvt 0.5 1\nf 1/1 2/2 3/3\n")
            (Path(directory) / "mesh.mtl").write_text("newmtl material\nmap_Kd texture.png\n")
            cv2.imwrite(str(Path(directory) / "texture.png"),
                        np.full((8, 8, 3), (7, 42, 230), np.uint8))
            image = app.render_mesh_preview(path, (160, 120))
            self.assertTrue(np.any(np.all(image == (7, 42, 230), axis=2)))


class MonitorIntegrationTest(unittest.TestCase):
    def test_ros_images_latched_masks_model_selection_and_resize(self):
        application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        rclpy.init()
        producer = Node("satellite_monitor_test_publisher")
        node = app.TrackingNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(producer)
        window = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                mesh = Path(directory) / "mesh.obj"
                mesh.write_text(OBJ)
                config_path = Path(directory) / "tracker.yaml"
                params = {"tracked_object_names": ["Blue carton", "Gold"],
                          "tracked_mesh_paths": [str(mesh), str(mesh)],
                          "caminfo_path": str(Path(directory) / "calibration.txt"),
                          "combined_visualization_topic": "/monitor_test/combined",
                          "pose_topic": "/monitor_test/pose"}
                config_path.write_text(yaml.safe_dump({
                    "foundationpose_stereo_tracker_multi_node": {"ros__parameters": params},
                    "grounded_sam_multi_first_mask_node": {"ros__parameters": {
                        "mask_topic_prefix": "/monitor_test/masks"}}}))
                bridge = CvBridge()
                mask_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                mask_pubs = [producer.create_publisher(Image, "/monitor_test/masks/" + name, mask_qos)
                             for name in ("blue_carton", "gold")]
                # Publish before the app subscribes: one-shot masks must still be received.
                for i, publisher in enumerate(mask_pubs):
                    mask = np.zeros((48, 64), np.uint8)
                    mask[28:42, 5 + 25 * i:25 + 25 * i] = 255
                    publisher.publish(bridge.cv2_to_imgmsg(mask, "mono8"))
                with patch.object(app, "DEFAULT_CONFIG", config_path):
                    window = app.SatelliteWindow(node)
                node.set_truth_objects(["Blue carton", "Gold"])
                window.tabs.setCurrentIndex(1)
                window.resize(1200, 800)
                window.show()
                left = producer.create_publisher(Image, "/left/image_raw",
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
                for _ in range(40):
                    left.publish(bridge.cv2_to_imgmsg(np.full((48, 64, 3), (0, 255, 0), np.uint8), "bgr8"))
                    executor.spin_once(timeout_sec=0.02)
                    application.processEvents()
                    window._show_pose()
                    if "left" in node.frames:
                        break
                self.assertIn("left", node.frames)
                self.assertIsNotNone(window.preview_labels["left"].pixmap())
                self.assertIsNone(window.preview_labels["combined"].pixmap())
                combined = producer.create_publisher(Image, params["combined_visualization_topic"],
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
                truth_pub = producer.create_publisher(PoseArray, "/satellite/ground_truth", 10)
                poses = [producer.create_publisher(PoseStamped, params["pose_topic"] + "/" + name, 1)
                         for name in ("blue_carton", "gold")]
                rgb = np.full((48, 192, 3), (255, 0, 0), np.uint8)
                stamp = producer.get_clock().now().to_msg()
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    combined.publish(bridge.cv2_to_imgmsg(rgb, "bgr8"))
                    truth = PoseArray()
                    truth.header.stamp = stamp
                    for _ in range(2):
                        pose = Pose()
                        pose.position.z = 1.0
                        pose.orientation.w = 1.0
                        truth.poses.append(pose)
                    truth_pub.publish(truth)
                    for index, publisher in enumerate(poses):
                        pose_message = PoseStamped()
                        pose_message.header.stamp = stamp
                        pose_message.pose.position.z = 1.25 + index
                        pose_message.pose.orientation.w = 1.0
                        publisher.publish(pose_message)
                    executor.spin_once(timeout_sec=0.02)
                    application.processEvents()
                    window._show_pose()
                    if (all(key in node.frames for key in ("combined", "mask:Blue carton", "mask:Gold"))
                            and len(node.poses) == 2 and
                            all(value[0] in node.truth_by_stamp for value in node.poses.values()) and
                            all(hasattr(label, "source_pixmap") for label in window.model_labels.values())):
                        break
                self.assertEqual(len(node.poses), 2)
                self.assertEqual(tuple(node.frames["combined"][0, 0]), (255, 0, 0))
                for key in ("combined", "mask"):
                    self.assertIsNotNone(window.preview_labels[key].pixmap(), key)
                self.assertEqual(window.pose_table.item(3, 2).text(), "0.000  0.000  2.250")
                self.assertEqual(window.pose_table.item(1, 2).text(), "0.000  0.000  1.000")
                self.assertEqual(window.pose_table.item(2, 5).text(), "0.250")
                self.assertEqual(window.pose_table.item(2, 6).text(), "0.00")
                self.assertEqual(window.pose_table.item(0, 4).text(), "0.000  0.000  0.000  1.000")
                self.assertNotEqual(window.motion_table.cellWidget(0, 1), window.motion_table.cellWidget(1, 1))
                window.motion_table.cellWidget(0, 1).setValue(0.7)
                self.assertNotEqual(window.motion_table.cellWidget(0, 1).value(),
                                    window.motion_table.cellWidget(1, 1).value())
                for label in window.model_labels.values():
                    self.assertIsNotNone(label.pixmap())
                window._show_pose()
                merged = app.combine_masks(
                    ["Blue carton", "Gold"],
                    [node.frames["mask:Blue carton"], node.frames["mask:Gold"]])
                self.assertTupleEqual(tuple(merged[35, 10]), app.MASK_COLORS[0])
                self.assertTupleEqual(tuple(merged[35, 35]), app.MASK_COLORS[1])
                window.resize(900, 650)
                application.processEvents()
                window._show_pose()
                self.assertEqual(window._shown_frames["combined"][1], window.preview_labels["combined"].size())
                screenshot = os.environ.get("SATELLITE_APP_SCREENSHOT")
                if screenshot:
                    window.grab().save(screenshot)
                with node.frame_lock:
                    node.frames.pop("mask:Gold")
                window._show_pose()
                self.assertIsNotNone(window.preview_labels["mask"].pixmap())
        finally:
            if window is not None:
                window.close()
            executor.shutdown()
            node.destroy_node()
            producer.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
