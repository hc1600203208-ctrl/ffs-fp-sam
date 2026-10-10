import launch

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "engine_file_path",
           default_value='["/home/hc/model/ffs/23-36-37/feature_runner_fp16_5060.engine", "/home/hc/model/ffs/23-36-37/post_runner_with_prob_fp16_5060.engine"]',
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
            default_value="/home/bit/dataset/blue720_640",
            description="Root directory of the offline stereo dataset",
        ),
        DeclareLaunchArgument(
            "left_subdir",
            default_value="rgb",
            description="Left image subdirectory under dataset_root",
        ),
        DeclareLaunchArgument(
            "right_subdir",
            default_value="camera2",
            description="Right image subdirectory under dataset_root",
        ),
        DeclareLaunchArgument(
            "output_subdir",
            default_value="depth_filter",
            description="Output depth subdirectory under dataset_root",
        ),
        DeclareLaunchArgument(
            "raw_depth_subdir",
            default_value="depth",
            description="Output directory for raw depth maps before confidence filtering; relative paths are resolved under dataset_root",
        ),
        DeclareLaunchArgument(
            "vis_subdir",
            default_value="vis",
            description="Output directory for colorized depth visualization images; relative paths are resolved under dataset_root",
        ),
        DeclareLaunchArgument("disparity_subdir", default_value="disparity",
                              description="Output 16-bit disparity PNG directory (pixels x 256, at model resolution)"),
        DeclareLaunchArgument("topk_cost_subdir", default_value="topk_cost",
                              description="Output float32 top-7 matching-cost NPY directory"),
        DeclareLaunchArgument("laf_rgb_subdir", default_value="laf_rgb",
                              description="Output rectified left RGB directory at model resolution"),
        DeclareLaunchArgument("model_confidence_subdir", default_value="ffs_confidence",
                              description="Output FFS internal matching confidence PNG directory (uint16 / 65535)"),
        DeclareLaunchArgument("model_filtered_depth_subdir", default_value="depth_ffs_filter",
                              description="Output depth filtered only by FFS internal confidence"),
        DeclareLaunchArgument("model_conf_threshold", default_value="0.3",
                              description="FFS internal matching confidence threshold in [0, 1]"),
        DeclareLaunchArgument(
            "caminfo_path",
            default_value="jr714.txt",
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
            {"disparity_subdir": LaunchConfiguration("disparity_subdir")},
            {"topk_cost_subdir": LaunchConfiguration("topk_cost_subdir")},
            {"laf_rgb_subdir": LaunchConfiguration("laf_rgb_subdir")},
            {"model_confidence_subdir": LaunchConfiguration("model_confidence_subdir")},
            {"model_filtered_depth_subdir": LaunchConfiguration("model_filtered_depth_subdir")},
            {"model_conf_threshold": LaunchConfiguration("model_conf_threshold")},
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
