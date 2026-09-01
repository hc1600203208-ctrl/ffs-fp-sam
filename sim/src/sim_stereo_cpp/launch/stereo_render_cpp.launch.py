import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    defaults = os.path.join(
        get_package_share_directory("sim_stereo_cpp"),
        "config",
        "defaults.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="sim_stereo_cpp",
                executable="stereo_render_cpp_node",
                name="stereo_render_cpp_node",
                output="screen",
                emulate_tty=True,
                parameters=[defaults],
            )
        ]
    )
