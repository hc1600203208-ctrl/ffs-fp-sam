# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import launch

from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    launch_args = [
        DeclareLaunchArgument(
            'engine_file_path',
            # default_value='["/home/hc/weizi/ffs/model/feature_runner_fp16_5060.engine", "/home/hc/weizi/ffs/model/post_runner_fp16_5060.engine"]',
            default_value='["/home/hc/model/ffs/20-30-48/feature_runner.engine", "/home/hc/model/ffs/20-30-48/post_runner.engine"]',
            description='The absolute file path to the TensorRT engine file'),
        DeclareLaunchArgument(
            'model_type',
            default_value='FAST_FOUNDATION_STEREO',
            choices=['FAST_FOUNDATION_STEREO'],
            description='Model type'),
        DeclareLaunchArgument(
            'input_image_width',
            default_value='1920',
            description='The input image width'),
        DeclareLaunchArgument(
            'input_image_height',
            default_value='1080',
            description='The input image height'),
        DeclareLaunchArgument(
            'model_input_width',
            default_value='640',
            description='The model input width'),
        DeclareLaunchArgument(
            'model_input_height',
            default_value='448',
            description='The model input height'),
        DeclareLaunchArgument(
            'trigger_on_demands',
            default_value='false',
            description='trigger_on_demands'),
        DeclareLaunchArgument(
            'publish_filtered_depth',
            default_value='false',
            description='Whether /depth_image publishes confidence-filtered depth instead of raw depth'),
        DeclareLaunchArgument(
            'publish_disparity',
            default_value='false',
            description='Whether to publish /disparity'),
        DeclareLaunchArgument(
            'publish_disparity_vis',
            default_value='false',
            description='Whether to publish /disparity_vis'),
        DeclareLaunchArgument(
            'publish_depth_image',
            default_value='true',
            description='Whether to publish /depth_image'),
        DeclareLaunchArgument(
            'publish_depth_image_raw',
            default_value='false',
            description='Whether to publish /depth_image_raw'),
        DeclareLaunchArgument(
            'publish_depth_image_filtered',
            default_value='false',
            description='Whether to publish /depth_image_filtered'),
        DeclareLaunchArgument(
            'publish_confidence_map',
            default_value='false',
            description='Whether to publish /confidence_map'),
        DeclareLaunchArgument(
            'publish_weight_map',
            default_value='false',
            description='Whether to publish /weight_map'),
        DeclareLaunchArgument(
            'save_results',
            default_value='true',
            description='Whether to save each aligned depth frame and corresponding left RGB image'),
        DeclareLaunchArgument(
            'save_output_dir',
            default_value='/home/hc/weizi/ffs+fp+sam/ffs/results',
            description='Output root directory for saved rgb/depth frames'),
        DeclareLaunchArgument(
            'save_depth_scale',
            default_value='1000.0',
            description='Scale factor for saving metric depth to 16-bit PNG'),
        DeclareLaunchArgument(
            'caminfo_path',
            default_value='jrnew.txt',
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
            'play_rosbag',
            default_value='false',
            description='Whether to auto-play rosbag from launch'),
        DeclareLaunchArgument(
            'rosbag_path',
            default_value='/home/hc/bag/jrnew-blue',
            description='The rosbag directory to play when play_rosbag=true'),
        DeclareLaunchArgument(
            'rosbag_start_delay',
            default_value='3.0',
            description='Delay in seconds before starting rosbag playback, giving subscribers time to initialize'),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Whether to launch RViz'),
    ]

    # Image preprocessing parameters
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    model_input_width = LaunchConfiguration('model_input_width')
    model_input_height = LaunchConfiguration('model_input_height')

    model_type = LaunchConfiguration('model_type')

    trigger_on_demands = LaunchConfiguration('trigger_on_demands')
    publish_filtered_depth = LaunchConfiguration('publish_filtered_depth')
    publish_disparity = LaunchConfiguration('publish_disparity')
    publish_disparity_vis = LaunchConfiguration('publish_disparity_vis')
    publish_depth_image = LaunchConfiguration('publish_depth_image')
    publish_depth_image_raw = LaunchConfiguration('publish_depth_image_raw')
    publish_depth_image_filtered = LaunchConfiguration('publish_depth_image_filtered')
    publish_confidence_map = LaunchConfiguration('publish_confidence_map')
    publish_weight_map = LaunchConfiguration('publish_weight_map')
    save_results = LaunchConfiguration('save_results')
    save_output_dir = LaunchConfiguration('save_output_dir')
    save_depth_scale = LaunchConfiguration('save_depth_scale')
    caminfo_path = LaunchConfiguration('caminfo_path')
    left_image_topic = LaunchConfiguration('left_image_topic')
    right_image_topic = LaunchConfiguration('right_image_topic')
    play_rosbag = LaunchConfiguration('play_rosbag')
    rosbag_path = LaunchConfiguration('rosbag_path')
    rosbag_start_delay = LaunchConfiguration('rosbag_start_delay')
    use_rviz = LaunchConfiguration('use_rviz')

    # TensorRT parameters
    engine_file_path = LaunchConfiguration('engine_file_path')

    dnn_stereo_depth_node = Node(
        package="fast_foundation_stereo",
        executable="dnn_stereo_rectified_depth_node",
        name="dnn_stereo_rectified_depth_node",
        parameters=[
            {"image_reliability": 1},
            {"cam_info_reliability": 1},

            {"model_type": model_type},

            {"model_input_height": model_input_height},
            {"model_input_width": model_input_width},
            {"input_image_height": input_image_height},
            {"input_image_width": input_image_width},

            {"engine_file_path": engine_file_path},
            {"caminfo_path": caminfo_path},

            {"trigger_on_demands": trigger_on_demands},
            {"publish_filtered_depth": publish_filtered_depth},
            {"publish_disparity": publish_disparity},
            {"publish_disparity_vis": publish_disparity_vis},
            {"publish_depth_image": publish_depth_image},
            {"publish_depth_image_raw": publish_depth_image_raw},
            {"publish_depth_image_filtered": publish_depth_image_filtered},
            {"publish_confidence_map": publish_confidence_map},
            {"publish_weight_map": publish_weight_map},
            {"save_results": save_results},
            {"save_output_dir": save_output_dir},
            {"save_depth_scale": save_depth_scale},
        ],
        remappings=[
            ("left_ir_image", left_image_topic),
            ("right_ir_image", right_image_topic),

            ("disparity", "/disparity/image_raw"),
            ("disparity_vis", "/disparity/image_vis"),
        ],
    )

    # Ros2 bag
    bag_play = TimerAction(
    period=rosbag_start_delay,
    actions=[
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('rosbag_path'), '-l'],
            shell=False,
            output='screen',
        )
    ],
    condition=IfCondition(play_rosbag)
   )


    rviz_config_path = os.path.join(get_package_share_directory(
        'fast_foundation_stereo'), 'rviz', 'dnn_stereo_depth.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(use_rviz))


    final_launch_description = launch_args + [
        dnn_stereo_depth_node,
        bag_play,
        rviz_node,
    ]

    # final_launch_description = launch_args + [dnn_stereo_depth_node, disparity2depth_node, rviz_node]
    # final_launch_description = launch_args + [dnn_stereo_depth_node, disparity2depth_node]
    return launch.LaunchDescription(final_launch_description)
