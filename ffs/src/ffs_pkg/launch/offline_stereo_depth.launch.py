import launch

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "engine_file_path",
            # default_value='["/home/hc/weizi/ffs/model/feature_runner_fp16_5060.engine", "/home/hc/weizi/ffs/model/post_runner_fp16_5060.engine"]',
            default_value='["/home/hc/model/ffs/20-30-48/feature_runner.engine", "/home/hc/model/ffs/20-30-48/post_runner.engine"]',
            description="The absolute file path to the TensorRT engine files",
        ),
        DeclareLaunchArgument(
            "model_type",
            default_value="FAST_FOUNDATION_STEREO",
            choices=["FAST_FOUNDATION_STEREO"],
            description="Model type",
        ),
        DeclareLaunchArgument(
            "dataset_root",
            default_value="/home/hc/dataset/blue-fast0.25",
            description="Root directory of the offline stereo dataset",
        ),
        DeclareLaunchArgument(
            "left_subdir",
            default_value="camera1",
            description="Left image subdirectory under dataset_root",
        ),
        DeclareLaunchArgument(
            "right_subdir",
            default_value="camera2",
            description="Right image subdirectory under dataset_root",
        ),
        DeclareLaunchArgument(
            "output_subdir",
            default_value="depth",
            description="Output depth subdirectory under dataset_root",
        ),
        DeclareLaunchArgument(
            "raw_depth_subdir",
            default_value="depth_raw",
            description="Output directory for raw depth maps before confidence filtering; relative paths are resolved under dataset_root",
        ),
        DeclareLaunchArgument(
            "vis_subdir",
            default_value="vis",
            description="Output directory for colorized depth visualization images; relative paths are resolved under dataset_root",
        ),
        DeclareLaunchArgument(
            "caminfo_path",
            default_value="/home/hc/weizi/dataset/jrnew-blue/caminfo.txt",
            description="Path to stereo calibration txt file",
        ),
        DeclareLaunchArgument(
            "input_image_width",
            default_value="1920",
            description="Expected input image width",
        ),
        DeclareLaunchArgument(
            "input_image_height",
            default_value="1080",
            description="Expected input image height",
        ),
        DeclareLaunchArgument(
            "model_input_width",
            default_value="640",
            description="The model input width",
        ),
        DeclareLaunchArgument(
            "model_input_height",
            default_value="448",
            description="The model input height",
        ),
        DeclareLaunchArgument(
            "min_depth_meters",
            default_value="0.1",
            description="Minimum saved depth in meters",
        ),
        DeclareLaunchArgument(
            "max_depth_meters",
            default_value="100.0",
            description="Maximum saved depth in meters",
        ),
        DeclareLaunchArgument(
            "depth_scale",
            default_value="1000.0",
            description="Scale factor for saved 16-bit PNG depth values",
        ),
        DeclareLaunchArgument(
            "save_input_resolution",
            default_value="true",
            description="Whether to restore depth maps to the original input resolution before saving",
        ),
    ]

    offline_node = Node(
        package="fast_foundation_stereo",
        executable="offline_stereo_depth_node",
        name="offline_stereo_depth_node",
        output="screen",
        parameters=[
            {"model_type": LaunchConfiguration("model_type")},
            {"engine_file_path": LaunchConfiguration("engine_file_path")},
            {"dataset_root": LaunchConfiguration("dataset_root")},
            {"left_subdir": LaunchConfiguration("left_subdir")},
            {"right_subdir": LaunchConfiguration("right_subdir")},
            {"output_subdir": LaunchConfiguration("output_subdir")},
            {"raw_depth_subdir": LaunchConfiguration("raw_depth_subdir")},
            {"vis_subdir": LaunchConfiguration("vis_subdir")},
            {"caminfo_path": LaunchConfiguration("caminfo_path")},
            {"input_image_width": LaunchConfiguration("input_image_width")},
            {"input_image_height": LaunchConfiguration("input_image_height")},
            {"model_input_width": LaunchConfiguration("model_input_width")},
            {"model_input_height": LaunchConfiguration("model_input_height")},
            {"min_depth_meters": LaunchConfiguration("min_depth_meters")},
            {"max_depth_meters": LaunchConfiguration("max_depth_meters")},
            {"depth_scale": LaunchConfiguration("depth_scale")},
            {"save_input_resolution": LaunchConfiguration("save_input_resolution")},
        ],
    )

    return launch.LaunchDescription(launch_args + [offline_node])
