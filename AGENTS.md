# AGENTS.md

## 项目概述
该项目包含三个子项目，第一个项目基于c++部署的foundationstereo双目深度估计项目，该项目通过ros2订阅双目图像话题，进行深度估计和置信度后处理发布深度图等话题；第二个项目是基于GINO+SAM的物体实例分割项目，该项目基于conda环境对任意未知物体进行分割，输入prompt提示词，输出物体分割mask图；第三个项目是基于c++部署的foundationpose六自由度位姿跟踪项目，该项目订阅左rgb图像和上一个项目发布的深度图话题，并用第二个项目提前分割的mask图作为第一帧mask；现在需要将这三个项目整合为一个总项目，免去中间复杂的ros2通信，只保留对原始双目图像的订阅，以及最终位姿跟踪结果的发布。

目前已经可以跑通的子项目的启动脚本：
第一个项目的启动脚本：/home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/launch/dnn_stereo_depth.launch.py
第二个项目的启动脚本：/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_first_mask_node.py、
第三个项目的启动脚本：ros2 run foundationpose_cpp foundationpose_file_mask_tracker_node \
  --ros-args --params-file /home/hc/weizi/fp/src/foundationpose_cpp/config/foundationpose_file_mask_ros2_example.yaml



