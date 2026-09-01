from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="sim",
                executable="stereo_render_node",
                name="stereo_render_node",
                output="screen",
                emulate_tty=True,
                parameters=["/home/hc/weizi/ffs+fp+sam/sim/config/defaults.yaml"],
            )
        ]
    )
