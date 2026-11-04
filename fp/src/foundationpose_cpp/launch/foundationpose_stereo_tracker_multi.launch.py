from pathlib import Path
import os

import launch
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# 阅读主线：本文件 -> sam/src/grounded_sam_multi_first_mask_node.cpp (首帧掩码)
# -> fp/src/foundationpose_cpp/src/foundationpose_stereo_tracker_multi_node.cpp (深度与位姿)。
# 两个节点共享 YAML；掩码 PNG 是它们之间的实际数据接口。

_MASK_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _is_true(value):
    # OpaqueFunction 收到的是 launch 替换后的字符串，需自行解析布尔值。
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _clean_first_mask_output_dir(context, *args, **kwargs):
    # launch 执行期解析 LaunchConfiguration；仅清除旧图像，避免 tracker 读到上次运行的掩码。
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
    # 声明参数发生在生成描述时；LaunchConfiguration 的值到实际启动时才求值。
    params_file = LaunchConfiguration("params_file")
    run_first_mask = LaunchConfiguration("run_first_mask")
    left_image_topic = LaunchConfiguration("left_image_topic")
    first_mask_output_dir = LaunchConfiguration("first_mask_output_dir")
    input_width = LaunchConfiguration("input_width")
    input_height = LaunchConfiguration("input_height")
    box_threshold = LaunchConfiguration("box_threshold")
    enable_profiling = LaunchConfiguration("enable_profiling")
    profiling_log_interval_frames = LaunchConfiguration("profiling_log_interval_frames")
    enable_depth_confidence_filter = LaunchConfiguration("enable_depth_confidence_filter")
    publish_visualization = LaunchConfiguration("publish_visualization")
    publish_depth_visualization = LaunchConfiguration("publish_depth_visualization")
    depth_visualization_fps = LaunchConfiguration("depth_visualization_fps")

    # The tracker links against both its package-local shared libraries and
    # the package-level detection library.  Keep these directories ahead of
    # any older overlay (for example foundationpose_cpp_inline_mask), which
    # otherwise can make the multi tracker load an ABI-incompatible library.
    tracker_library_path = os.pathsep.join(
        [
            "/home/hc/weizi/ffs+fp+sam/fp/install/foundationpose_cpp/lib/foundationpose_cpp",
            "/home/hc/weizi/ffs+fp+sam/fp/install/foundationpose_cpp/lib",
            os.environ.get("LD_LIBRARY_PATH", ""),
        ]
    )

    launch_args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=(
                "/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/"
                "foundationpose_stereo_tracker_multi_bluegold_example.yaml"
            ),
            description="Shared parameters for multi-object masks and FoundationPose tracking.",
        ),
        DeclareLaunchArgument("run_first_mask", default_value="true"),
        DeclareLaunchArgument("left_image_topic", default_value="/left/image_raw"),
        DeclareLaunchArgument(
            "first_mask_output_dir",
            default_value=(
                "/home/hc/weizi/ffs+fp+sam/fp/masks/"
                "first_mask_multi"
            ),
        ),
        DeclareLaunchArgument("input_width", default_value="640"),
        DeclareLaunchArgument("input_height", default_value="480"),
        DeclareLaunchArgument("box_threshold", default_value="0.3"),
        DeclareLaunchArgument("publish_visualization", default_value="false"),
        DeclareLaunchArgument("publish_depth_visualization", default_value="false"),
        DeclareLaunchArgument("depth_visualization_fps", default_value="5.0"),
        DeclareLaunchArgument(
            "enable_profiling",
            default_value="false",
            description="Enable per-frame timing statistics for the multi-object tracker.",
        ),
        DeclareLaunchArgument(
            "profiling_log_interval_frames",
            default_value="30",
            description="How often to print cumulative timing averages, in processed tracking frames.",
        ),
        DeclareLaunchArgument(
            "enable_depth_confidence_filter",
            default_value="false",
            description="Use full FFS photometric/edge/temporal confidence filtering before FoundationPose.",
        ),
    ]

    clean_first_mask_output_dir = OpaqueFunction(
        # OpaqueFunction 在 Node 动作前执行，返回空动作列表即可。
        function=_clean_first_mask_output_dir,
        kwargs={
            "run_first_mask": run_first_mask,
            "first_mask_output_dir": first_mask_output_dir,
        },
    )

    first_mask_node = Node(
        # launch_ros 根据 package/executable 查找已安装的 ROS 2 可执行文件。
        package="grounded_sam_trt",
        executable="grounded_sam_multi_first_mask_node",
        name="grounded_sam_multi_first_mask_node",
        output="screen",
        condition=IfCondition(run_first_mask),
        # 后面的字典覆盖同名 YAML 参数；ParameterValue 防止数字被当作字符串。
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
        # tracker 与 SAM 同时启动；tracker 持续轮询所有 PNG，齐备后才注册目标。
        package="foundationpose_cpp",
        executable="foundationpose_stereo_tracker_multi_node",
        name="foundationpose_stereo_tracker_multi_node",
        output="screen",
        additional_env={"LD_LIBRARY_PATH": tracker_library_path},
        parameters=[
            params_file,
            {"mask_image_directory": first_mask_output_dir},
            {
                "publish_visualization": ParameterValue(publish_visualization, value_type=bool),
                "publish_depth_visualization": ParameterValue(publish_depth_visualization, value_type=bool),
                "depth_visualization_fps": ParameterValue(depth_visualization_fps, value_type=float),
                "enable_profiling": ParameterValue(enable_profiling, value_type=bool),
                "profiling_log_interval_frames": ParameterValue(
                    profiling_log_interval_frames, value_type=int
                ),
                "enable_depth_confidence_filter": ParameterValue(
                    enable_depth_confidence_filter, value_type=bool
                ),
            },
        ],
    )

    return launch.LaunchDescription(
        # 顺序保证清理动作先执行，但不保证两个节点的首帧处理顺序。
        launch_args + [clean_first_mask_output_dir, first_mask_node, tracker_node]
    )
