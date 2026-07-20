import launch

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            "engine_file_path",
            default_value='["/home/hc/model/ffs/23-36-37/feature_runner_fp16_5060.engine", "/home/hc/model/ffs/23-36-37/post_runner_fp16_5060.engine"]',
            description="Absolute file paths to the TensorRT engine files",
        ),
        DeclareLaunchArgument(
            "model_type",
            default_value="FAST_FOUNDATION_STEREO",
            choices=["FAST_FOUNDATION_STEREO"],
            description="Stereo model type",
        ),
        DeclareLaunchArgument(
            "dataset_root",
            default_value="/home/hc/weizi/dataset/jrnew-blue",
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
            "caminfo_path",
            default_value="/home/hc/weizi/dataset/jrnew-blue/caminfo.txt",
            description="Path to stereo calibration txt file",
        ),
        DeclareLaunchArgument(
            "output_root",
            default_value="/home/hc/weizi/dataset/jrnew-blue/depth_benchmark",
            description="Root directory for benchmark result groups",
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
            description="Model input width",
        ),
        DeclareLaunchArgument(
            "model_input_height",
            default_value="448",
            description="Model input height",
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
            description="Whether to align depth maps back to the original left image before saving",
        ),
        DeclareLaunchArgument(
            "max_pairs",
            default_value="1000",
            description="Maximum number of stereo pairs to process; 0 means all pairs",
        ),
        DeclareLaunchArgument(
            "median_kernel_size",
            default_value="5",
            description="Median filter kernel size for the baseline group",
        ),
        DeclareLaunchArgument(
            "bilateral_d",
            default_value="5",
            description="Bilateral filter diameter for the baseline group",
        ),
        DeclareLaunchArgument(
            "bilateral_sigma_color",
            default_value="0.05",
            description="Bilateral filter sigmaColor for metric depth",
        ),
        DeclareLaunchArgument(
            "bilateral_sigma_space",
            default_value="5.0",
            description="Bilateral filter sigmaSpace for metric depth",
        ),
        DeclareLaunchArgument(
            "use_temporal_confidence",
            default_value="true",
            description="Whether confidence_full uses C_tmp when a previous frame is available",
        ),
        DeclareLaunchArgument(
            "conf_threshold",
            default_value="0.35",
            description="Default confidence threshold for groups D-G",
        ),
        DeclareLaunchArgument(
            "threshold_sweep",
            default_value="0.20,0.30,0.35,0.40,0.50",
            description="Comma-separated threshold values for confidence_full_threshold_sweep",
        ),
    ]

    benchmark_node = Node(
        package="fast_foundation_stereo",
        executable="offline_depth_benchmark_node",
        name="offline_depth_benchmark_node",
        output="screen",
        parameters=[
            {"model_type": LaunchConfiguration("model_type")},
            {"engine_file_path": LaunchConfiguration("engine_file_path")},
            {"dataset_root": LaunchConfiguration("dataset_root")},
            {"left_subdir": LaunchConfiguration("left_subdir")},
            {"right_subdir": LaunchConfiguration("right_subdir")},
            {"caminfo_path": LaunchConfiguration("caminfo_path")},
            {"output_root": LaunchConfiguration("output_root")},
            {"input_image_width": LaunchConfiguration("input_image_width")},
            {"input_image_height": LaunchConfiguration("input_image_height")},
            {"model_input_width": LaunchConfiguration("model_input_width")},
            {"model_input_height": LaunchConfiguration("model_input_height")},
            {"min_depth_meters": LaunchConfiguration("min_depth_meters")},
            {"max_depth_meters": LaunchConfiguration("max_depth_meters")},
            {"depth_scale": LaunchConfiguration("depth_scale")},
            {"save_input_resolution": LaunchConfiguration("save_input_resolution")},
            {"max_pairs": LaunchConfiguration("max_pairs")},
            {"median_kernel_size": LaunchConfiguration("median_kernel_size")},
            {"bilateral_d": LaunchConfiguration("bilateral_d")},
            {"bilateral_sigma_color": LaunchConfiguration("bilateral_sigma_color")},
            {"bilateral_sigma_space": LaunchConfiguration("bilateral_sigma_space")},
            {"use_temporal_confidence": LaunchConfiguration("use_temporal_confidence")},
            {"conf_threshold": LaunchConfiguration("conf_threshold")},
            {"threshold_sweep": LaunchConfiguration("threshold_sweep")},
        ],
    )

    return launch.LaunchDescription(launch_args + [benchmark_node])
