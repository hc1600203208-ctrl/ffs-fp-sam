import unittest

import numpy as np

import sim_stereo_app as app


class SimStereoAppTest(unittest.TestCase):
    def test_mesh_resolution(self):
        path = app.resolve_mesh_path(app.DEFAULT_MESH_DIRECTORY)
        self.assertEqual(path.name, "textured_mesh.obj")

    def test_background_composition_preserves_mesh(self):
        rendered = np.zeros((40, 60, 3), np.uint8)
        rendered[10:30, 20:40] = (10, 80, 230)
        background = np.full((20, 20, 3), (200, 120, 40), np.uint8)
        output = app.composite_background(rendered, background)
        self.assertTupleEqual(tuple(output[0, 0]), (200, 120, 40))
        self.assertTupleEqual(tuple(output[20, 30]), (10, 80, 230))

    def test_black_texture_hole_is_filled_by_silhouette(self):
        rendered = np.zeros((40, 60, 3), np.uint8)
        rendered[10:30, 20:40] = 200
        rendered[15:25, 25:35] = 0
        mask = app.foreground_mask(rendered)
        self.assertEqual(mask[20, 30], 255)

    def test_motion_commands(self):
        values = app.MotionValues(
            pose=(0.0, 0.0, 0.8, 1.0, 2.0, 3.0),
            translation_amplitude=(0.1, 0.2, 0.3),
            rotation_amplitude=(10.0, 20.0, 30.0),
            frequency=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )
        self.assertEqual(
            app.motion_commands(values),
            [
                "pose 0 0 0.8 1 2 3",
                "amp 0.1 0.2 0.3 10 20 30",
                "freq 0.4 0.5 0.6 0.7 0.8 0.9",
            ],
        )

    def test_resume_does_not_reset_motion_profile(self):
        class RunningProcess:
            @staticmethod
            def poll():
                return None

        class Window:
            process = RunningProcess()
            motion_running = False

            def __init__(self):
                self.commands = []

            def _write_commands(self, commands):
                self.commands.extend(commands)

            def _set_motion_state(self, running, _text):
                self.motion_running = running

        window = Window()
        app.SimWindow._toggle_motion(window)
        self.assertEqual(window.commands, ["start"])
        self.assertTrue(window.motion_running)


if __name__ == "__main__":
    unittest.main()
