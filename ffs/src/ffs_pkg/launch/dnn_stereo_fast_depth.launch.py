import launch

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='["/home/hc/model/ffs/20-30-48/320-512/feature_runner_fp16_5060.engine", "/home/hc/model/ffs/20-30-48/320-512/post_runner_fp16_5060.engine"]',
            #  default_value='["/home/bit/model/ffs/20-30-48/320-512/feature_runner_fp16_thor.engine", "/home/bit/model/ffs/20-30-48/320-512/post_runner_fp16_thor.engine"]',
            description='Absolute paths to the TensorRT engine files'),
        DeclareLaunchArgument(
            'model_type',
            default_value='FAST_FOUNDATION_STEREO',
            choices=['FAST_FOUNDATION_STEREO'],
            description='Stereo model type'),
        DeclareLaunchArgument(
            'model_input_width',
            default_value='512',
            description='Model input width'),
        DeclareLaunchArgument(
            'model_input_height',
            default_value='320',
            description='Model input height'),
        DeclareLaunchArgument(
            'min_depth_meters',
            default_value='0.1',
            description='Minimum valid depth in meters'),
        DeclareLaunchArgument(
            'max_depth_meters',
            default_value='100.0',
            description='Maximum valid depth in meters'),
        DeclareLaunchArgument(
            'max_timestamp_delta_seconds',
            default_value='0.07',
            description='Maximum allowed timestamp delta between stereo images'),
        DeclareLaunchArgument(
            'sync_queue_size',
            default_value='10',
            description='Approximate stereo synchronizer queue size'),
        DeclareLaunchArgument(
            'caminfo_path',
            default_value='640.txt',
            description='Path to stereo calibration txt file'),
        DeclareLaunchArgument(
            'left_image_topic',
            default_value='/left/image_raw',
            description='Left raw stereo image topic'),
        DeclareLaunchArgument(
            'right_image_topic',
            default_value='/right/image_raw',
            description='Right raw stereo image topic'),
        DeclareLaunchArgument(
            'depth_image_topic',
            default_value='/depth_image',
            description='Output metric depth image topic'),
        DeclareLaunchArgument(
            'save_frame_outputs',
            default_value='false',
            description='Save every stereo pair and generated depth image to disk'),
        DeclareLaunchArgument(
            'frame_output_dir',
            default_value='/home/bit/ffs+fp+sam/ffs/fast_depth_outputs',
            description='Output directory for saved left/right/depth images'),
    ]

    fast_depth_node = Node(
        package='fast_foundation_stereo',
        executable='dnn_stereo_fast_depth_node',
        name='dnn_stereo_fast_depth_node',
        output='screen',
        parameters=[
            {'image_reliability': 1},
            {'model_type': LaunchConfiguration('model_type')},
            {'model_input_height': LaunchConfiguration('model_input_height')},
            {'model_input_width': LaunchConfiguration('model_input_width')},
            {'min_depth_meters': LaunchConfiguration('min_depth_meters')},
            {'max_depth_meters': LaunchConfiguration('max_depth_meters')},
            {'max_timestamp_delta_seconds': LaunchConfiguration('max_timestamp_delta_seconds')},
            {'sync_queue_size': LaunchConfiguration('sync_queue_size')},
            {'engine_file_path': LaunchConfiguration('engine_file_path')},
            {'caminfo_path': LaunchConfiguration('caminfo_path')},
            {'save_frame_outputs': ParameterValue(LaunchConfiguration('save_frame_outputs'), value_type=bool)},
            {'frame_output_dir': LaunchConfiguration('frame_output_dir')},
        ],
        remappings=[
            ('left_ir_image', LaunchConfiguration('left_image_topic')),
            ('right_ir_image', LaunchConfiguration('right_image_topic')),
            ('depth_image', LaunchConfiguration('depth_image_topic')),
        ],
    )

    return launch.LaunchDescription(launch_args + [fast_depth_node])
