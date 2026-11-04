# 双目深度与多目标 6D 位姿跟踪部署

本仓库把 Fast-FoundationStereo（FFS）、GroundingDINO + SAM、FoundationPose 的 C++/TensorRT 推理接成 ROS 2 双目跟踪流程，并提供仿真双目渲染及应用程序。目标是使用左右相机图像、相机标定、目标网格和首帧掩码，连续输出每个物体相对于左相机的 6D 位姿。发布到 GitHub 前，请按下文的“开源准备”检查模型、数据和路径。

## 功能与代码流程

1. `fp/.../foundationpose_stereo_tracker_*_node.cpp` 同步左右 ROS 图像，检查时间戳，对双目图像做极线校正并按 FFS 输入尺寸缩放补边。
2. `ffs/src/ffs_pkg` 的 TensorRT FFS 模型输出视差；节点按标定参数还原米制深度，将其映射回原始左图坐标。多目标节点还可启用深度置信度过滤。
3. `sam/src/grounded_sam_multi_first_mask_node.cpp` 用固定文本提示的 GroundingDINO 找目标框，再用 SAM 生成每个目标的二值 PNG 首帧掩码。跟踪节点轮询这些 PNG。也可以手工提供同尺寸掩码。
4. 每个目标用网格、RGB、深度和掩码执行一次 FoundationPose `Register`。随后每帧用上帧滤波预测位姿作为 `Track` 的初值，再通过 refiner 得到观测位姿。多目标节点为每个对象创建独立跟踪 worker，同一帧统一收集结果。
5. 共用的自适应 SE(3) 滤波器根据右不变运动模型预测位姿，以观测残差估算质量，调整观测协方差，并用马氏距离门控异常值。最终位姿同时用于下一帧跟踪初值、`PoseStamped` 发布、日志和可视化；跟踪调用失败时输出预测位姿。
6. `sim/` 提供 C++/Python 双目渲染和 `satellite_stereo` 字节流到 ROS 图像的接口，`app/` 提供多目标演示和星地双目应用界面。

单目标入口是 `foundationpose_stereo_tracker_fast_node`，多目标入口是 `foundationpose_stereo_tracker_multi_node`。多目标 ROS 位姿话题为 `pose_topic/<目标名规范化后缀>`，第一个目标另发到 `pose_topic`。坐标系默认继承左图 `header.frame_id`，也可用 `pose_frame_id` 覆盖。输出平移单位为米，旋转以四元数表示。

## 所依赖论文与项目改动

- [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130) 用知识蒸馏压缩骨干网络、分块神经架构搜索优化代价滤波、结构化剪枝减少迭代细化开销，并用自动伪标注扩充野外双目训练数据，以兼顾零样本泛化和实时速度。本仓库将它以 TensorRT 双阶段引擎接入双目深度链路。
- [FoundationPose](https://github.com/NVlabs/FoundationPose) 利用已知物体网格和首帧掩码做 6D 位姿注册，以渲染与网络细化进行后续跟踪。本仓库的 C++ 版本把位姿转回原始 mesh 坐标系，集成多目标工作线程和预测反馈。
- [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) 组合开放词汇检测与 SAM 分割，解决初始目标掩码获取问题；本仓库使用固定提示词导出的 TensorRT 引擎，换提示词时需重新导出对应 DINO 模型。
- 自适应滤波来自本地 FoundationPose Python 脚本 `run_demo_adaptive_filter.py` 及其 `tracking/adaptive_se3_pose_filter.py`，属于本工程新增的部署逻辑，并非上述论文的原创贡献。C++ 实现在 `fp/src/foundationpose_cpp/src/adaptive_se3_pose_filter.hpp`。

## 环境要求

当前工程按 Linux、ROS 2 Jazzy、C++17、NVIDIA GPU、CUDA、TensorRT 10、OpenCV 4、Eigen3 构建；`fp` 的 CMake 目前将 TensorRT 头文件和库路径指向 `/usr/src/tensorrt` 的 x86_64 布局，迁移到其他平台须调整该路径。还需要 `colcon`、`ament_cmake`、`rclcpp`、`sensor_msgs`、`geometry_msgs`、`std_msgs`、`cv_bridge`、`message_filters`、`tf2`，以及工程 CMake 所需的 Assimp、CUDA 开发包等依赖。模型导出阶段另需相应 Python 上游工程及 PyTorch；在线 C++ 推理只加载已生成的 TensorRT engine。

TensorRT engine 往往与 GPU 架构、TensorRT/CUDA 版本相关；需在实际部署设备上生成或重新构建。默认示例 YAML 含开发机绝对路径，不能直接用于另一台机器。

## 依赖和模型准备

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install python3-colcon-common-extensions libopencv-dev libeigen3-dev libassimp-dev
```

安装匹配设备的 NVIDIA 驱动、CUDA Toolkit、TensorRT 开发与运行库，并确认 `nvcc` 和 TensorRT 头文件/库可由 CMake 找到。FFS 需要 feature/post 两个 engine，FoundationPose 需要 refiner/scorer 两个 engine；自动首帧掩码还需固定提示词 DINO engine 和 SAM encoder/decoder engine。生成方法及导出限制见 `ffs/src/ffs_pkg/USAGE.md`、`fp/src/foundationpose_cpp/README_cn.md` 和 `sam/README.md`。网格文件需包含可加载的几何体及对应纹理；左右相机标定须与输入图像尺寸及实际相机一致。

## 编译和实际部署

以下命令在仓库根目录执行。各子工程独立使用自己的 `build/` 和 `install/`。

```bash
source /opt/ros/jazzy/setup.bash
cd ffs && colcon build && cd ..
cd fp && colcon build --packages-select foundationpose_cpp --cmake-args -DENABLE_TENSORRT=ON -DBUILD_SIMPLE_TESTS=OFF && cd ..
cd sam && ./scripts/build_cpp.sh && ./scripts/build_ros.sh && cd ..
```

按对象数量复制 `fp/src/foundationpose_cpp/config/` 中的 fast 或 multi 示例 YAML，修改左右图话题、`caminfo_path`、FFS/FP engine 路径、网格、掩码目录及名称、相机内参、输出话题。自动掩码还须修改 YAML 中 `grounded_sam_multi_first_mask_node` 的对象名称、DINO/SAM engine 和输出目录，并使其输出 PNG 名称与跟踪器配置完全一致。确保模型输入与实际图像尺寸、FFS engine profile、标定一致。单目标若使用手工 PNG，可不启动自动掩码节点。

```bash
source /opt/ros/jazzy/setup.bash
source "$PWD/fp/install/setup.bash"
source "$PWD/sam/ros2_install/setup.bash"
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py \
  params_file:="$PWD/fp/src/foundationpose_cpp/config/你的配置.yaml"
```

另一个终端启动双目相机驱动，发布与 YAML 相同的左右图话题；离线验证可用 `ros2 bag play <bag目录>`。单目标改用 `foundationpose_stereo_tracker_fast.launch.py`。启动后可用 `ros2 topic list` 和 `ros2 topic echo /foundationpose/pose` 检查位姿。默认 launch 的部分路径仍与本仓库开发机相关，部署时检查 launch/YAML，尤其是 mask 输出目录与动态库路径。没有真实相机时，可先运行 `sim/src/sim_stereo_cpp` 的渲染节点，流程见 `sim/README.md`；星地数据流场景见 `sim/src/satellite_stereo/README.md` 与 `app/satellite_stereo_multi_app.py`。

## 自适应滤波参数与行为

两个跟踪节点使用同一套 YAML 参数，默认值对齐 `run_demo_adaptive_filter.py` 的当前参数。`filter_fps` 为时间戳不可用时的回退帧率；正常运行时按相邻左图时间戳求 `dt`。`filter_q_pos/q_rot/q_vel/q_omega` 为 12 维状态预测噪声；`filter_sigma_pos_obs/sigma_rot_obs` 为观测标准差；`filter_sigma_pos_motion/sigma_rot_motion` 控制残差到质量分数的衰减；`filter_min_quality` 限制质量下界；`filter_gate_threshold` 是马氏距离平方门限；`filter_reject_cov_increase` 是拒绝观测时增加的协方差。`filter_enable_gating` 与 `filter_enable_adaptive_R` 可分别关闭门控和质量驱动的观测噪声调整。

首帧直接以 `Register` 位姿初始化滤波器、零速度和 `1e-3 I` 协方差。每帧先预测 `T_pred = T · Exp(xi·dt)`；`Track` 使用 `T_pred` 作初值。右不变残差 `Log(T_pred⁻¹ · T_obs)` 同时用于质量评估、卡尔曼更新与门控。观测被拒绝时保留预测位姿并增加协方差；跟踪失败时也使用预测位姿。日志中的 `quality`、`mahalanobis`、`accepted` 可用于调参。相机或物体运动速度与 Python 演示不同，门限与噪声仍应结合实际数据标定。

## 开源准备

本仓库代码可整理后发布到 GitHub，但不能把忽略列表当作完整的授权检查。发布前确认上游 FFS、FoundationPose、Grounded SAM、EasyDeploy 等代码、模型与数据各自的许可证和再分发条件；补齐本仓库许可证、作者及论文引用；检查 Git 历史和示例配置中的本机路径、私人数据、权重、engine 与密钥。当前 `.gitignore` 排除了常见模型和输出，仍需逐项人工检查已有提交历史。
