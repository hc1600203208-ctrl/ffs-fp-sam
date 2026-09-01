from pathlib import Path

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


_MASK_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _is_true(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _clean_first_mask_output_dir(context, *args, **kwargs):
    if not _is_true(kwargs["run_first_mask"].perform(context)):
        return []

    output_dir = Path(kwargs["first_mask_output_dir"].perform(context)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_count = 0
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in _MASK_IMAGE_SUFFIXES:
            path.unlink()
            removed_count += 1

    logger = launch.logging.get_logger("foundationpose_stereo_tracker_multi.launch")
    logger.info(f"Cleaned {removed_count} old first-mask image(s) from {output_dir}")
    return []


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    run_first_mask = LaunchConfiguration("run_first_mask")
    left_image_topic = LaunchConfiguration("left_image_topic")
    first_mask_output_dir = LaunchConfiguration("first_mask_output_dir")
    input_width = LaunchConfiguration("input_width")
    input_height = LaunchConfiguration("input_height")
    box_threshold = LaunchConfiguration("box_threshold")

    launch_args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=(
                "/home/bit/ffs+fp+sam/fp/src/foundationpose_cpp/config/"
                "foundationpose_stereo_tracker_multi_bluepink_example.yaml"
            ),
            description="Shared parameters for multi-object masks and FoundationPose tracking.",
        ),
        DeclareLaunchArgument("run_first_mask", default_value="true"),
        DeclareLaunchArgument("left_image_topic", default_value="/left/image_raw"),
        DeclareLaunchArgument(
            "first_mask_output_dir",
            default_value=(
                "/home/bit/ffs+fp+sam/Grounded-Segment-Anything/ros2_outputs/"
                "first_mask_multi"
            ),
        ),
        DeclareLaunchArgument("input_width", default_value="640"),
        DeclareLaunchArgument("input_height", default_value="480"),
        DeclareLaunchArgument("box_threshold", default_value="0.3"),
    ]

    clean_first_mask_output_dir = OpaqueFunction(
        function=_clean_first_mask_output_dir,
        kwargs={
            "run_first_mask": run_first_mask,
            "first_mask_output_dir": first_mask_output_dir,
        },
    )

    first_mask_node = Node(
        package="grounded_sam_trt",
        executable="grounded_sam_multi_first_mask_node",
        name="grounded_sam_multi_first_mask_node",
        output="screen",
        condition=IfCondition(run_first_mask),
        parameters=[
            params_file,
            {
                "image_topic": left_image_topic,
                "output_dir": first_mask_output_dir,
                "input_width": ParameterValue(input_width, value_type=int),
                "input_height": ParameterValue(input_height, value_type=int),
                "box_threshold": ParameterValue(box_threshold, value_type=float),
            },
        ],
    )

    tracker_node = Node(
        package="foundationpose_cpp",
        executable="foundationpose_stereo_tracker_multi_node",
        name="foundationpose_stereo_tracker_multi_node",
        output="screen",
        parameters=[
            params_file,
            {"mask_image_directory": first_mask_output_dir},
        ],
    )

    return launch.LaunchDescription(
        launch_args + [clean_first_mask_output_dir, first_mask_node, tracker_node]
    )
