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

    logger = launch.logging.get_logger("foundationpose_stereo_tracker_fast.launch")
    logger.info(f"Cleaned {removed_count} old first-mask image(s) from {output_dir}")
    return []


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    run_first_mask = LaunchConfiguration("run_first_mask")
    left_image_topic = LaunchConfiguration("left_image_topic")
    first_mask_output_dir = LaunchConfiguration("first_mask_output_dir")
    first_mask_output_name = LaunchConfiguration("first_mask_output_name")
    mask_topic = LaunchConfiguration("mask_topic")
    publish_mask_topic = LaunchConfiguration("publish_mask_topic")
    dino_engine = LaunchConfiguration("dino_engine")
    sam_encoder_engine = LaunchConfiguration("sam_encoder_engine")
    sam_decoder_engine = LaunchConfiguration("sam_decoder_engine")
    input_width = LaunchConfiguration("input_width")
    input_height = LaunchConfiguration("input_height")
    box_threshold = LaunchConfiguration("box_threshold")
    max_detections = LaunchConfiguration("max_detections")
    save_frame_outputs = LaunchConfiguration("save_frame_outputs")
    frame_output_dir = LaunchConfiguration("frame_output_dir")

    launch_args = [
        DeclareLaunchArgument(
            "params_file",
            default_value="/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_fast_example.yaml",
            description="Parameter file for the fast integrated stereo FoundationPose tracker.",
        ),
        DeclareLaunchArgument("run_first_mask", default_value="true"),
        DeclareLaunchArgument("left_image_topic", default_value="/left/image_raw"),
        DeclareLaunchArgument(
            "first_mask_output_dir",
            default_value="/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_outputs/first_mask",
        ),
        DeclareLaunchArgument("first_mask_output_name", default_value="first_mask.png"),
        DeclareLaunchArgument("mask_topic", default_value="/fisrt_mask"),
        DeclareLaunchArgument("publish_mask_topic", default_value="true"),
        DeclareLaunchArgument(
            "dino_engine",
            default_value="/home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_fixed_prompt.engine",
        ),
        DeclareLaunchArgument(
            "sam_encoder_engine",
            default_value="/home/hc/weizi/ffs+fp+sam/sam/engines/sam_image_encoder.engine",
        ),
        DeclareLaunchArgument(
            "sam_decoder_engine",
            default_value="/home/hc/weizi/ffs+fp+sam/sam/engines/sam_mask_decoder.engine",
        ),
        DeclareLaunchArgument("input_width", default_value="640"),
        DeclareLaunchArgument("input_height", default_value="480"),
        DeclareLaunchArgument("box_threshold", default_value="0.3"),
        DeclareLaunchArgument("max_detections", default_value="16"),
        DeclareLaunchArgument(
            "save_frame_outputs",
            default_value="false",
            description="Save each successful pose output and its pose visualization frame.",
        ),
        DeclareLaunchArgument(
            "frame_output_dir",
            default_value="/home/hc/weizi/ffs+fp+sam/fp/stereo_tracker_fast_outputs",
            description="Directory for per-frame fast tracker visualization images and poses.csv.",
        ),
    ]

    first_mask_node = Node(
        package="grounded_sam_trt",
        executable="grounded_sam_first_mask_node",
        name="grounded_sam_first_mask_node",
        output="screen",
        condition=IfCondition(run_first_mask),
        parameters=[
            {
                "image_topic": left_image_topic,
                "mask_topic": mask_topic,
                "publish_mask_topic": ParameterValue(
                    publish_mask_topic, value_type=bool
                ),
                "output_dir": first_mask_output_dir,
                "output_name": first_mask_output_name,
                "dino_engine": dino_engine,
                "sam_encoder_engine": sam_encoder_engine,
                "sam_decoder_engine": sam_decoder_engine,
                "input_width": ParameterValue(input_width, value_type=int),
                "input_height": ParameterValue(input_height, value_type=int),
                "box_threshold": ParameterValue(box_threshold, value_type=float),
                "max_detections": ParameterValue(max_detections, value_type=int),
            }
        ],
    )

    clean_first_mask_output_dir = OpaqueFunction(
        function=_clean_first_mask_output_dir,
        kwargs={
            "run_first_mask": run_first_mask,
            "first_mask_output_dir": first_mask_output_dir,
        },
    )

    tracker_node = Node(
        package="foundationpose_cpp",
        executable="foundationpose_stereo_tracker_fast_node",
        name="foundationpose_stereo_tracker_fast_node",
        output="screen",
        parameters=[
            params_file,
            {
                "save_frame_outputs": ParameterValue(save_frame_outputs, value_type=bool),
                "frame_output_dir": frame_output_dir,
                "mask_image_path": "",
                "mask_image_directory": first_mask_output_dir,
                "mask_image_name": first_mask_output_name,
            },
        ],
    )

    return launch.LaunchDescription(
        launch_args
        + [
            clean_first_mask_output_dir,
            first_mask_node,
            tracker_node,
        ]
    )
