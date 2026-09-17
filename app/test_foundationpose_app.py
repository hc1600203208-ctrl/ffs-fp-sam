import math
import tempfile
import unittest
from pathlib import Path

import yaml

import foundationpose_app as app


class FoundationPoseAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.objects = app.load_pipeline_config(app.DEFAULT_CONFIG)

    def test_default_config_and_filtering(self):
        self.assertEqual([item.name for item in self.objects], ["blue carton", "pink carton"])
        filtered = app.build_session_config(self.data, [1])
        sam = filtered["grounded_sam_multi_first_mask_node"]["ros__parameters"]
        tracker = filtered["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
        self.assertEqual(sam["object_names"], ["pink carton"])
        self.assertEqual(tracker["tracked_object_names"], ["pink carton"])
        self.assertEqual(sam["mask_output_names"], tracker["tracked_mask_image_names"])
        self.assertTrue(tracker["publish_visualization"])
        self.assertFalse(tracker["show_visualization_window"])

    def test_names_and_pose_conversion(self):
        self.assertEqual(app.sanitize_name("Blue Carton"), "blue_carton")
        roll, pitch, yaw = app.quaternion_to_rpy_degrees(
            0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)
        )
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 90.0, places=6)

    def test_calibration_parser(self):
        tracker = self.data["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
        calibration = app.load_stereo_calibration(tracker["caminfo_path"])
        self.assertEqual(calibration[0].shape, (3, 3))
        self.assertEqual(calibration[4].shape, (3, 3))
        self.assertEqual(calibration[5].shape, (3,))

    def test_invalid_config_is_rejected(self):
        broken = {
            "grounded_sam_multi_first_mask_node": {
                "ros__parameters": {"object_names": ["one"]}
            },
            "foundationpose_stereo_tracker_multi_node": {
                "ros__parameters": {"tracked_object_names": ["two"]}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.yaml"
            path.write_text(yaml.safe_dump(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                app.load_pipeline_config(path)


if __name__ == "__main__":
    unittest.main()
