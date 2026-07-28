import launch

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    run_first_mask = LaunchConfiguration("run_first_mask")
    use_conda_for_first_mask = LaunchConfiguration("use_conda_for_first_mask")
    conda_executable = LaunchConfiguration("conda_executable")
    conda_env_name = LaunchConfiguration("conda_env_name")
    left_image_topic = LaunchConfiguration("left_image_topic")
    python_executable = LaunchConfiguration("python_executable")
    first_mask_script = LaunchConfiguration("first_mask_script")
    first_mask_output_dir = LaunchConfiguration("first_mask_output_dir")
    first_mask_output_name = LaunchConfiguration("first_mask_output_name")
    text_prompt = LaunchConfiguration("text_prompt")
    box_threshold = LaunchConfiguration("box_threshold")
    text_threshold = LaunchConfiguration("text_threshold")
    bert_base_uncased_path = LaunchConfiguration("bert_base_uncased_path")
    run_first_mask_with_conda = PythonExpression(
        ["'", run_first_mask, "' == 'true' and '", use_conda_for_first_mask, "' == 'true'"]
    )
    run_first_mask_without_conda = PythonExpression(
        ["'", run_first_mask, "' == 'true' and '", use_conda_for_first_mask, "' == 'false'"]
    )

    default_params_file = PathJoinSubstitution(
        [
            get_package_share_directory("foundationpose_cpp"),
            "config",
            "foundationpose_stereo_tracker_example.yaml",
        ]
    )

    launch_args = [
        DeclareLaunchArgument(
            "params_file",
            default_value="/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_example.yaml",
            description="Parameter file for the integrated stereo FoundationPose tracker.",
        ),
        DeclareLaunchArgument(
            "run_first_mask",
            default_value="true",
            description="Run Grounded-SAM once to generate the first-frame mask file.",
        ),
        DeclareLaunchArgument(
            "use_conda_for_first_mask",
            default_value="true",
            description="Run Grounded-SAM in the configured conda environment.",
        ),
        DeclareLaunchArgument(
            "conda_executable",
            default_value="/home/hc/anaconda3/bin/conda",
            description="Conda executable used for optional Grounded-SAM first-mask generation.",
        ),
        DeclareLaunchArgument(
            "conda_env_name",
            default_value="sam",
            description="Conda environment name for optional Grounded-SAM first-mask generation.",
        ),
        DeclareLaunchArgument(
            "left_image_topic",
            default_value="/left/image_raw",
            description="Left raw stereo image topic used by optional first-mask generation.",
        ),
        DeclareLaunchArgument(
            "python_executable",
            default_value="/home/hc/anaconda3/envs/sam/bin/python",
            description="Python executable for optional Grounded-SAM when use_conda_for_first_mask=false.",
        ),
        DeclareLaunchArgument(
            "first_mask_script",
            default_value="/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_first_mask_node.py",
            description="Path to ros2_first_mask_node.py.",
        ),
        DeclareLaunchArgument(
            "first_mask_output_dir",
            default_value="/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_outputs/first_mask",
            description="Directory where Grounded-SAM saves first_mask.png.",
        ),
        DeclareLaunchArgument(
            "first_mask_output_name",
            default_value="first_mask.png",
            description="First mask file name.",
        ),
        DeclareLaunchArgument(
            "text_prompt",
            default_value="blue object",
            description="Text prompt for Grounded-SAM first-mask generation.",
        ),
        DeclareLaunchArgument(
            "box_threshold",
            default_value="0.3",
            description="GroundingDINO box threshold.",
        ),
        DeclareLaunchArgument(
            "text_threshold",
            default_value="0.25",
            description="GroundingDINO text threshold.",
        ),
        DeclareLaunchArgument(
            "bert_base_uncased_path",
            default_value="/home/hc/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
            description="Local bert-base-uncased path used by GroundingDINO to avoid HuggingFace downloads.",
        ),
    ]

    first_mask_ros_args = [
        first_mask_script,
        "--ros-args",
        "-p",
        ["image_topic:=", left_image_topic],
        "-p",
        "publish_mask_topic:=false",
        "-p",
        ["output_dir:=", first_mask_output_dir],
        "-p",
        ["output_name:=", first_mask_output_name],
        "-p",
        ["text_prompt:=", text_prompt],
        "-p",
        ["box_threshold:=", box_threshold],
        "-p",
        ["text_threshold:=", text_threshold],
        "-p",
        ["bert_base_uncased_path:=", bert_base_uncased_path],
    ]

    first_mask_conda_process = ExecuteProcess(
        cmd=[
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            conda_env_name,
            "python",
            *first_mask_ros_args,
        ],
        output="screen",
        condition=IfCondition(run_first_mask_with_conda),
    )

    first_mask_python_process = ExecuteProcess(
        cmd=[
            python_executable,
            *first_mask_ros_args,
        ],
        output="screen",
        condition=IfCondition(run_first_mask_without_conda),
    )

    tracker_node = Node(
        package="foundationpose_cpp",
        executable="foundationpose_stereo_tracker_node",
        name="foundationpose_stereo_tracker_node",
        output="screen",
        parameters=[params_file],
    )

    return launch.LaunchDescription(launch_args + [first_mask_conda_process, first_mask_python_process, tracker_node])
