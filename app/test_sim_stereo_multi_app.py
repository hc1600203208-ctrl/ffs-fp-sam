import tempfile
import unittest
from pathlib import Path

import yaml

from sim_stereo_multi_app import load_objects, renderer_command


class MultiSimAppTest(unittest.TestCase):
    def test_example_config_and_renderer_parameters(self):
        from sim_stereo_multi_app import DEFAULT_CONFIG

        objects, calibration = load_objects(DEFAULT_CONFIG)
        self.assertGreaterEqual(len(objects), 2)
        poses = [(index * 0.3, 0.0, 0.8, 0.0, 0.0, 0.0) for index in range(len(objects))]
        command = renderer_command("ros2", objects, poses, calibration)
        self.assertIn("stereo_render_multi_cpp_node", command)
        params = dict(argument.split(":=", 1) for flag, argument in zip(command, command[1:]) if flag == "-p")
        self.assertEqual(yaml.safe_load(params["object_names"]), [name for name, _ in objects])
        self.assertEqual(len(yaml.safe_load(params["initial_poses"])), 6 * len(objects))
        self.assertEqual(params["left_image_topic"], "/sim_app/rendered_left")
        self.assertEqual(params["interactive"], "true")

    def test_rejects_too_few_objects(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            renderer_command("ros2", [("one", "mesh.obj")], [(0,) * 6], Path("calibration.txt"))

    def test_three_objects_keep_render_order(self):
        from sim_stereo_multi_app import DEFAULT_CONFIG

        objects, calibration = load_objects(DEFAULT_CONFIG)
        objects = [objects[0], objects[1], ("third", objects[0][1])]
        command = renderer_command("ros2", objects, [(0, 0, 0.8, 0, 0, 0)] * 3, calibration)
        params = dict(argument.split(":=", 1) for flag, argument in zip(command, command[1:]) if flag == "-p")
        self.assertEqual(yaml.safe_load(params["object_names"]), [objects[0][0], objects[1][0], "third"])
        self.assertEqual(len(yaml.safe_load(params["initial_poses"])), 18)

    def test_load_rejects_mismatched_meshes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "tracker.yaml"
            config.write_text(yaml.safe_dump({"foundationpose_stereo_tracker_multi_node": {
                "ros__parameters": {"tracked_object_names": ["a", "b"],
                                    "tracked_mesh_paths": ["one.obj"]}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matching"):
                load_objects(config)


if __name__ == "__main__":
    unittest.main()
