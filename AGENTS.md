# AGENTS.md

## 项目概述
该项目包含三个子项目，第一个项目基于c++部署的foundationstereo双目深度估计项目，该项目通过ros2订阅双目图像话题，进行深度估计和置信度后处理发布深度图等话题；第二个项目是基于GINO+SAM的物体实例分割项目，该项目基于conda环境对任意未知物体进行分割，输入prompt提示词，输出物体分割mask图；第三个项目是基于c++部署的foundationpose六自由度位姿跟踪项目，该项目订阅左rgb图像和上一个项目发布的深度图话题，并用第二个项目提前分割的mask图作为第一帧mask；现在需要将这三个项目整合为一个总项目，免去中间复杂的ros2通信，只保留对原始双目图像的订阅，以及最终位姿跟踪结果的发布。

目前已经可以跑通的子项目的启动脚本：
第一个项目的启动脚本：/home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/launch/dnn_stereo_depth.launch.py
第二个项目的启动脚本：/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_first_mask_node.py、
第三个项目的启动脚本：ros2 run foundationpose_cpp foundationpose_file_mask_tracker_node \
  --ros-args --params-file /home/hc/weizi/fp/src/foundationpose_cpp/config/foundationpose_file_mask_ros2_example.yaml


查看当前的/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/launch/foundationpose_stereo_tracker_fast.launch.py的启动文件，先理解该启动文件的启动流程，然后将当前负责第一帧mask图生成的python脚本，替换为/home/hc/weizi/ffs+fp+sam/sam的c++部署版本，重新在sam中集成ros2，替换掉原来/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything/ros2_first_mask_node.py的任务

/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/launch/foundationpose_stereo_tracker_fast.launch.py的处理逻辑当前只能对一个物体进行位姿跟踪，当要修改物体时，必须提前准备好新的grounding dino的engine文件，并修改/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_fast_example.yaml配置文件中的mesh_path和object_name，现在我想同时跟踪多个不同的物体，请修改配置文件和代码逻辑，在第一帧进行多次mask提取来获取不同物体的mask区域，之后执行foundationpose推理时要对多个物体同时执行多个并行的跟踪，不同物体可视化时用不同颜色的包围框，三个旋转坐标轴仍然共用红蓝绿，且把物体名字打在物体的左上角，原先imshow显示框左上角打印的位姿输出只打印第一个物体的输出。增加一条要求，尽量不要修改当前工作区的其他代码，给新任务编写新的源文件、yaml配置文件和launch启动脚本。同时先用/home/hc/bag/bluepink bag包做测试bag，该bag中图像包含有blue carton和pink carton两个物体；mesh路径分别为blue carton: /home/hc/weizi/dataset/jrnew-blue/mesh/textured_mesh.obj
pink carton: /home/hc/dataset/myobject/lgj/bundlesdf_540/textured_mesh.obj ;engine文件路径为/home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_fixed_blue_carton.engine和/home/hc/weizi/ffs+fp+sam/sam/engines/grounding_dino_fixed_pink_carton.engine



