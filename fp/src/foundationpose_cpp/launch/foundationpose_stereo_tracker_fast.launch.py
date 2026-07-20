import launch

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    save_frame_outputs = LaunchConfiguration("save_frame_outputs")
    frame_output_dir = LaunchConfiguration("frame_output_dir")
    run_first_mask_with_conda = PythonExpression(
        ["'", run_first_mask, "' == 'true' and '", use_conda_for_first_mask, "' == 'true'"]
    )
    run_first_mask_without_conda = PythonExpression(
        ["'", run_first_mask, "' == 'true' and '", use_conda_for_first_mask, "' == 'false'"]
    )

    launch_args = [
        DeclareLaunchArgument(
            "params_file",
            default_value="/home/bit/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_fast_example.yaml",
            description="Parameter file for the fast integrated stereo FoundationPose tracker.",
        ),
        DeclareLaunchArgument("run_first_mask", default_value="true"),
        DeclareLaunchArgument("use_conda_for_first_mask", default_value="true"),
        DeclareLaunchArgument("conda_executable", default_value="/home/bit/anaconda3/bin/conda"),
        DeclareLaunchArgument("conda_env_name", default_value="sam"),
        DeclareLaunchArgument("left_image_topic", default_value="/left/image_raw"),
        DeclareLaunchArgument("python_executable", default_value="/home/bit/anaconda3/envs/sam/bin/python"),
        DeclareLaunchArgument(
            "first_mask_script",
            default_value="/home/bit/ffs+fp+sam/Grounded-Segment-Anything/ros2_first_mask_node.py",
        ),
        DeclareLaunchArgument(
            "first_mask_output_dir",
            default_value="/home/bit/ffs+fp+sam/Grounded-Segment-Anything/ros2_outputs/first_mask",
        ),
        DeclareLaunchArgument("first_mask_output_name", default_value="first_mask.png"),
        DeclareLaunchArgument("text_prompt", default_value="blue object"),
        DeclareLaunchArgument("box_threshold", default_value="0.7"),
        DeclareLaunchArgument("text_threshold", default_value="0.4"),
        DeclareLaunchArgument(
            "save_frame_outputs",
            default_value="true",
            description="Save each successful pose output and its pose visualization frame.",
        ),
        DeclareLaunchArgument(
            "frame_output_dir",
            default_value="/home/bit/ffs+fp+sam/fp/stereo_tracker_fast_outputs",
            description="Directory for per-frame fast tracker visualization images and poses.csv.",
        ),
        DeclareLaunchArgument(
            "bert_base_uncased_path",
            default_value="/home/bit/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
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
        executable="foundationpose_stereo_tracker_fast_node",
        name="foundationpose_stereo_tracker_fast_node",
        output="screen",
        parameters=[
            params_file,
            {
                "save_frame_outputs": ParameterValue(save_frame_outputs, value_type=bool),
                "frame_output_dir": frame_output_dir,
            },
        ],
    )

    return launch.LaunchDescription(launch_args + [first_mask_conda_process, first_mask_python_process, tracker_node])
