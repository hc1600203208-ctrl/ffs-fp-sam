from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


DEFAULT_CONFIG = (
    Path("/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/"
         "foundationpose_stereo_tracker_multi_bluepink_example.yaml")
)


def _renderer(context):
    with Path(LaunchConfiguration("params_file").perform(context)).expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["foundationpose_stereo_tracker_multi_node"]["ros__parameters"]
    paths = config["tracked_mesh_paths"]
    names = config["tracked_object_names"]
    if len(paths) < 2 or len(names) != len(paths):
        raise ValueError("The tracker configuration needs at least two matching names and mesh paths")
    poses = [value for index in range(len(paths))
             for value in ((index - (len(paths) - 1) / 2) * 0.3, 0.0, 0.8, 0.0, 0.0, 0.0)]
    return [Node(
        package="sim_stereo_cpp",
        executable="stereo_render_multi_cpp_node",
        name="stereo_render_multi_cpp_node",
        output="screen",
        parameters=[{
            "mesh_paths": paths,
            "object_names": names,
            "initial_poses": poses,
            "calibration_path": config["caminfo_path"],
            "interactive": False,
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=str(DEFAULT_CONFIG)),
        OpaqueFunction(function=_renderer),
    ])
