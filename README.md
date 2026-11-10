# 双目深度与多目标 6D 位姿跟踪部署

本仓库把 Fast-FoundationStereo（FFS）、GroundingDINO + Segment Anything（SAM）、FoundationPose 的 C++/TensorRT 推理接成 ROS 2 双目跟踪流程，并提供仿真双目渲染及应用程序。输入左右相机图像、相机标定、目标网格和首帧掩码，连续输出每个物体相对于左相机的 6D 位姿。另有[无 ROS 双机 UDP 版本](#无-ros-双机-udp-版本接口打包与操作)，用于把轻依赖渲染器部署到独立卫星 Linux 主机。

## 实验视频

点击预览图可在 GitHub 打开或下载仓库中的原始 MP4。它们展示跟踪叠加效果，并非精度基准或性能测量。

| 场景 | 视频 |
| --- | --- |
| 三目标快速运动 | [![三目标快速运动](media/three_fast.jpg)](media/three_fast.mp4) |
| 三目标慢速运动 | [![三目标慢速运动](media/three_slow.jpg)](media/three_slow.mp4) |
| 室内实验 | [![室内实验](media/indoor.jpg)](media/indoor.mp4) |
| 室外实验 | [![室外实验](media/outdoor.jpg)](media/outdoor.mp4) |

## 自采数据集

项目使用的自采数据集已存放在公共网盘，涵盖五个物体，分为重建与慢速运动数据、快速运动数据两份。数据集通过网盘单独提供，不包含在本 Git 仓库中。

| 压缩包 | 内容 | 下载链接 |
| --- | --- | --- |
| `reconstruct.zip` | 五个物体的重建与慢速运动数据 | [百度网盘下载（提取码：tiaq）](https://pan.baidu.com/s/1HcqH1y95vTCZS0oTlzi8PA?pwd=tiaq) |
| `fast-move.zip` | 五个物体的快速运动数据 | [百度网盘下载（提取码：yhfr）](https://pan.baidu.com/s/1aLCFS98yQqDvIIp9RXmsWA?pwd=yhfr) |

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
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 与 [Segment Anything](https://github.com/facebookresearch/segment-anything) 分别提供开放词汇目标框和掩码。本仓库使用固定提示词导出的 TensorRT 引擎，换提示词时需重新导出对应 DINO 模型。当前 `sam/scripts/export_onnx.py` 导入的是原版 `segment_anything`，**不是 SAM 2**。
- 自适应滤波来自本地 FoundationPose Python 脚本 `run_demo_adaptive_filter.py` 及其 `tracking/adaptive_se3_pose_filter.py`，属于本工程新增的部署逻辑，并非上述论文的原创贡献。C++ 实现在 `fp/src/foundationpose_cpp/src/adaptive_se3_pose_filter.hpp`。

## 环境要求

当前工程按 Linux、ROS 2 Jazzy、C++17、NVIDIA GPU、CUDA、TensorRT 10、OpenCV 4、Eigen3 构建；`fp` 的 CMake 目前将 TensorRT 头文件和库路径指向 `/usr/src/tensorrt` 的 x86_64 布局，迁移到其他平台须调整该路径。还需要 `colcon`、`ament_cmake`、`rclcpp`、`sensor_msgs`、`geometry_msgs`、`std_msgs`、`cv_bridge`、`message_filters`、`tf2`，以及工程 CMake 所需的 Assimp、CUDA 开发包等依赖。模型导出阶段另需相应 Python 上游工程及 PyTorch；在线 C++ 推理只加载已生成的 TensorRT engine。

以下是 **2026-11-05 在本项目开发主机上实测** 的版本，仅作复现参考：

| 项目 | 本机版本 |
| --- | --- |
| 系统 / 内核 / 架构 | Ubuntu 24.04.4 LTS / Linux 7.0.0-28-generic / x86_64 |
| GPU | NVIDIA GeForce RTX 5060 Ti，显存 16311 MiB，计算能力 12.0 |
| NVIDIA 驱动 / CUDA Toolkit | 580.178.04 / 12.8.93 |
| TensorRT | 10.8.0.43（本机 `NvInferVersion.h`；`trtexec` 显示 v100800） |
| ROS 2 | Jazzy；`rclcpp` 28.1.18，`ament_cmake` 2.5.6，`cv_bridge` 4.1.0 |
| 编译工具 | GCC 13.3.0，CMake 3.28.3，colcon-core 0.20.1，Python 3.12.3 |
| 图像和几何库 | OpenCV 4.11.0（当前 `pkg-config`/Python），系统 `libopencv-dev` 4.6.0；Eigen3 3.4.0，Assimp 5.3.1 |

本机同时装有两套 OpenCV，构建时链接器报告 ROS `cv_bridge` 的 4.6 库与工程发现的 4.11 库可能冲突。该工程在本机编译通过，但复现时应尽量统一 OpenCV 运行时，特别是在不同设备或容器中。示例 YAML 含开发机绝对路径，需要改为本机路径。

项目作者另报告已在 NVIDIA Jetson Thor 开发套件、JetPack 7.2 上跑通。上述版本表来自 x86_64 开发主机，不代表 Thor 的实测库版本。Thor 上必须重新生成 engine，并把 `ffs/src/ffs_pkg/CMakeLists.txt`、`fp/src/foundationpose_cpp/CMakeLists.txt` 中写死的 `x86_64-linux-gnu` TensorRT 库路径改为设备上的 `aarch64-linux-gnu` 或实际安装路径；`sam/CMakeLists.txt` 可通过 `TENSORRT_ROOT` 搜索两种架构。还需修改配置 YAML 与 launch 文件中的模型、标定、掩码和动态库路径。

## 依赖和模型准备

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install python3-colcon-common-extensions libopencv-dev libeigen3-dev libassimp-dev
```

安装匹配设备的 NVIDIA 驱动、CUDA Toolkit、TensorRT 开发与运行库，并确认 `nvcc` 和 TensorRT 头文件/库可由 CMake 找到。FFS 需要 feature/post 两个 engine，FoundationPose 需要 refiner/scorer 两个 engine；自动首帧掩码还需固定提示词 DINO engine 和 SAM encoder/decoder engine。网格文件需包含可加载的几何体及对应纹理；左右相机标定须与输入图像尺寸及实际相机一致。

**本仓库不提供 ONNX、原始权重或 TensorRT engine。** 请先从各官方项目或其授权的发布渠道下载相应模型文件，阅读各自模型许可；不要把一个设备上生成的 engine 直接复制到另一台 GPU 使用。下面命令保留了项目使用的网络输入尺寸和 batch profile，输出文件名可按自己的 YAML 修改。`TRTEXEC` 可设为当前设备上 `trtexec` 的路径。

```bash
TRTEXEC=/usr/src/tensorrt/bin/trtexec
FFS_ONNX=/path/to/official/ffs/onnx
FP_ONNX=/path/to/authorized/foundationpose/onnx
ENGINE_DIR=/path/to/local/engines
mkdir -p "$ENGINE_DIR"

"$TRTEXEC" --onnx="$FFS_ONNX/feature_runner.onnx" \
  --saveEngine="$ENGINE_DIR/feature_runner_fp16.engine" --fp16 --useCudaGraph
"$TRTEXEC" --onnx="$FFS_ONNX/post_runner.onnx" \
  --saveEngine="$ENGINE_DIR/post_runner_fp16.engine" --fp16 --useCudaGraph

"$TRTEXEC" --onnx="$FP_ONNX/scorer_hwc.onnx" \
  --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
  --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  --saveEngine="$ENGINE_DIR/scorer_hwc_dynamic_fp32.engine"
"$TRTEXEC" --onnx="$FP_ONNX/refiner_hwc.onnx" \
  --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
  --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  --saveEngine="$ENGINE_DIR/refiner_hwc_dynamic_fp32.engine"
```

上面的 FP 命令按提供的示例生成 FP32 engine；如果打算用 FP16，请在目标设备上重新验证精度与性能，并在 YAML 中填写实际文件名。GroundingDINO 和 SAM 当前流程先从各自的 PTH 权重导出 ONNX，再运行 `sam/scripts/build_engines.sh` 转 engine；`sam/scripts/export_all.sh` 和 `sam/scripts/export_dino.sh` 提供导出入口。固定提示词 DINO 的 ONNX 必须与推理时目标词一致；当前脚本使用原版 SAM 的 `segment_anything` API，不能直接用 SAM 2 的权重替换。导出所需 Python 依赖由相应上游工程提供。

## 编译和实际部署

以下命令在仓库根目录执行。各子工程独立使用自己的 `build/` 和 `install/`。

```bash
source /opt/ros/jazzy/setup.bash
cd ffs && colcon build && cd ..
cd fp && colcon build --packages-select foundationpose_cpp --cmake-args -DENABLE_TENSORRT=ON -DBUILD_SIMPLE_TESTS=OFF && cd ..
cd sam && ./scripts/build_cpp.sh && ./scripts/build_ros2.sh && cd ..
```

按对象数量复制 `fp/src/foundationpose_cpp/config/` 中的 fast 或 multi 示例 YAML，修改左右图话题、`caminfo_path`、FFS/FP engine 路径、网格、掩码目录及名称、相机内参、输出话题。自动掩码还须修改 YAML 中 `grounded_sam_multi_first_mask_node` 的对象名称、DINO/SAM engine 和输出目录，并使其输出 PNG 名称与跟踪器配置完全一致。确保模型输入与实际图像尺寸、FFS engine profile、标定一致。单目标若使用手工 PNG，可不启动自动掩码节点。

```bash
source /opt/ros/jazzy/setup.bash
source "$PWD/fp/install/setup.bash"
source "$PWD/sam/ros2_install/setup.bash"
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py \
  params_file:="$PWD/fp/src/foundationpose_cpp/config/你的配置.yaml"
```

另一个终端启动双目相机驱动，发布与 YAML 相同的左右图话题；离线验证可用 `ros2 bag play <bag目录>`。单目标改用 `foundationpose_stereo_tracker_fast.launch.py`。启动后可用 `ros2 topic list` 和 `ros2 topic echo /foundationpose/pose` 检查位姿。默认 launch 的部分路径仍与本仓库开发机相关，部署时检查 launch/YAML，尤其是 mask 输出目录与动态库路径。没有真实相机时，可先编译 `sim/src/sim_stereo_cpp` 并启动 `stereo_render_cpp.launch.py`，或者在相应 Python 环境运行 `app/sim_stereo_multi_app.py`；星地字节流与 ROS 图像转换入口在 `sim/src/satellite_stereo`，应用程序入口是 `app/satellite_stereo_multi_app.py`。

### 星地多目标 App

完成 `satellite_stereo`、`foundationpose_cpp` 和 SAM 的编译后，分别在三个终端运行。先启动卫星渲染节点和 App，在 App 的“Parameters & control”页加载目标配置，给每个物体设置独立的初始位姿、平移/旋转振幅和频率，再点击“Start satellite renderer”。App 会启动地面 `stereo_bytes_to_images_node`，还原双目图像；此时“Tracking monitor”页已能显示原始左图，不依赖 FoundationPose。随后启动跟踪器，合成图、首帧掩码和位姿才会出现。

```bash
# 终端 1：卫星渲染端
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
ros2 run satellite_stereo satellite_render_node

# 终端 2：地面 App；启动后在界面中下发配置并开始渲染
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
python3 /home/hc/weizi/ffs+fp+sam/app/satellite_stereo_multi_app.py

# 终端 3：需要跟踪结果时再启动
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py
```

卫星端以配置中的物体顺序向 `/satellite/ground_truth` 发布 `geometry_msgs/PoseArray`，时间戳与 `/satellite/stereo_bytes` 中的双目帧相同。真值为原始 mesh 原点在左相机光学坐标系中的位姿；跟踪器的 `pose_frame_id` 若在 YAML 中自定义，比较误差前须确认它仍表示同一左相机坐标系。App 按图像时间戳匹配跟踪位姿和对应帧的真值；匹配成功时真值行标记 `Truth`，否则显示最新真值并标记 `Latest`。同帧误差为：`ΔXYZ = 跟踪平移 − 真值平移`；原始 RPY 误差逐轴计算 `wrap₍₋₁₈₀°,₁₈₀°₎(估计角 − 真值角)`，避免跨越 ±180° 时产生约 360° 的假差值；`|Δt|` 为三维平移误差范数；原始 `ΔR = 2·acos(|q估计·q真值|)`（弧度转度，四元数均归一化）是完整相对旋转的最短角度，范围为 0–180°，不等于三个 RPY 差值的和。**界面误差行的 RPY 和 ΔR 均显示无量纲缩放值 `原始角度误差(°) ÷ 10°`**：0–10° 对应 0–1，负的 RPY 误差保留符号，大于 10° 时显示大于 1，不截断；鼠标悬停可查看原始角度。估计和真值行的 RPY 仍以度显示，位置误差仍以米显示。未匹配到真值的跟踪帧，误差显示为 `—`。旋转同时显示四元数 `(x,y,z,w)` 和固定轴 **XYZ** 欧拉角，即 `R = Rz(yaw) · Ry(pitch) · Rx(roll)`。RPY 分量差在接近万向节锁时会受欧拉角表示影响，整体旋转误差以原始 `ΔR` 为准。

## 无 ROS 双机 UDP 版本：接口、打包与操作

这一版本的代码集中在 `standalone_udp/`，现有 ROS 节点和启动文件仍可照常使用。卫星 Linux 主机运行 `satellite_renderer_udp`，负责按物体网格与运动参数生成左右图、回传真值；地面 Linux 主机运行 `satellite_stereo_multi_udp_app.py` 和 `ground_tracker_udp`。地面跟踪器继续使用 FFS 深度、GroundingDINO + SAM 首帧掩码、FoundationPose 和自适应 SE(3) 位姿滤波，运行时不需要 ROS。星上程序不运行神经网络，只使用 POSIX UDP、C++、libpng、zlib；Eigen 仅在编译时使用。

### 星地进程与端口

以下以网线直连、地面 IP `192.168.50.1`、卫星 IP `192.168.50.2` 为例。两台主机可以同时使用端口 50000，因为它们的 IP 不同。

| 发送端 → 接收端 | 地址和端口 | 数据 | 说明 |
| --- | --- | --- | --- |
| 地面 App → 卫星渲染器 | `192.168.50.2:50000/UDP` | 配置、启动、停止命令 | App 的发送套接字同时绑定地面 `0.0.0.0:50000`；卫星通过该命令的来源地址确定回传目的地。 |
| 卫星渲染器 → 地面 App | `192.168.50.1:50000/UDP` | 命令 ACK、双目 PNG、同帧渲染真值 | 卫星监听 `0.0.0.0:50000`。启动渲染前，App 必须至少发送过一次成功的控制命令。 |
| 地面 App → 地面跟踪器 | `127.0.0.1:50100/UDP` | 填充背景后重新编码的双目 PNG | 只在地面主机内部传输；跟踪器未启动时，App 仍可显示卫星左图和真值。 |
| 地面跟踪器 → 地面 App | `127.0.0.1:50101/UDP` | 滤波后的位姿、JPEG 合成图、PNG 掩码 | 本机环回接口，不需要在卫星侧开放 50100/50101。 |

默认端口由 [App](standalone_udp/app/satellite_stereo_multi_udp_app.py) 和两个启动脚本确定。卫星端口可同时修改 `./run_satellite.sh` 与 `start_ground_app.sh` 的最后一个参数；地面 50100/50101 目前在 App 中固定，按下文默认值运行。卫星和 App 的 IP 不必与示例一致，但必须处于可互通的网络中。帧时间戳由卫星生成，并原样传递给跟踪器与 App；同帧匹配不依赖两台主机预先校时。

### UDP 数据接口

星地消息与地面本机消息均使用 `SUD1` 外层格式。所有多字节整数为**小端序**，位姿浮点数为 IEEE 754 `float64`。每个 UDP 数据报由 32 字节头部加最多 1152 字节数据组成，单个完整消息上限为 8 MiB。

| 头部偏移（字节） | 长度 | 字段 | 含义 |
| --- | ---: | --- | --- |
| 0–3 | 4 | `magic` | ASCII `SUD1` |
| 4 | 1 | `version` | 当前为 `1` |
| 5 | 1 | `kind` | 下表中的消息类型 |
| 6–7 | 2 | `reserved` | 当前为 `0` |
| 8–11 | 4 | `message_id` | 控制/ACK 为请求号，图像/真值/结果为帧号；每次重新启动渲染，帧号从 0 开始 |
| 12–13 | 2 | `part_index` | 分片编号，从 0 开始 |
| 14–15 | 2 | `part_count` | 本消息的分片总数 |
| 16–19 | 4 | `total_size` | 重组后的数据字节数 |
| 20–27 | 8 | `stamp_ns` | 图像、真值及跟踪结果共用的 Unix 时间戳，单位纳秒；控制与 ACK 为 0 |
| 28–31 | 4 | `chunk_crc32` | 本分片数据的 CRC32，不包含头部 |

| `kind` | 方向 | 重组后的数据 |
| ---: | --- | --- |
| 1 `CONTROL` | App → 卫星 | 下述 `SCT1` 控制报文 |
| 2 `ACK` | 卫星 → App | `SCT1` 应答报文 |
| 3 `STEREO_PNG` | 卫星 → App；App → 地面跟踪器 | 单张 PNG：左图在左、右图在右。当前 App 下发的渲染尺寸固定为每眼 **640×480**，因此整张 PNG 解码后为 1280×480。卫星传来的是黑底渲染图；App 将黑色背景像素替换为仓库的 `app/assets/stereo_background.png` 后，再转发给跟踪器。 |
| 4 `TRUTH` | 卫星 → App | `uint16` 物体数，随后按配置顺序为每个物体发送 7 个 `float64`：`x,y,z,qx,qy,qz,qw`。平移单位为米，位姿表示原始 mesh 原点相对于左相机光学坐标系。 |
| 5 `RESULT` | 地面跟踪器 → App | `uint16` 物体数；每物体依次为 `uint16` 名称字节数、UTF-8 名称、`uint8` 有效标志、7 个 `float64` 位姿；随后为 `uint32` 合成图字节数、JPEG 数据、`uint16` 掩码数；每个掩码为 `uint32` 字节数加 PNG 数据，长度 0 表示尚无掩码。合成图从左到右为左图、位姿框、伪彩深度。 |

`CONTROL` 内层 `SCT1` 头部依次为：4 字节 `SCT1`、1 字节命令类型、1 字节标志（0）、2 字节保留值（0）、4 字节请求号、4 字节数据长度。命令类型 `1` 为配置、`2` 为启动、`3` 为停止。配置数据从 `width:uint32, height:uint32, fps:float64, bytes_per_second:uint32, object_count:uint16, config_version:uint16` 开始；当前 App 发送 `config_version=1`。其后是以 `uint16` 长度加 UTF-8 字节表示的包内相对标定路径，再按物体顺序发送名称、包内相对 OBJ 路径，以及各 6 个 `float64` 的初始 `XYZRPY`、运动振幅、频率。位置与振幅的 XYZ 单位为米，RPY 单位为度，频率单位为 Hz。卫星逐轴按 `初值 + 振幅 × sin(2π × 频率 × 运行时间)` 生成运动，欧拉角按 `Rz(yaw) Ry(pitch) Rx(roll)` 组成旋转。

`ACK` 的内层命令类型为 `0x80`，随后是 `status:uint8`（0 成功、1 失败）、`text_length:uint16` 和 UTF-8 提示文本。App 点击“Configure and start satellite”后依次发送**停止 → 配置 → 启动**，每一步收到成功 ACK 才继续；3 秒未应答会重发一次，之后再超时 3 秒则弹窗。卫星按来源地址、端口和请求号缓存 ACK，重复命令不会再次重启渲染。图像分片不重传：接收端校验 CRC32 并在 3 秒内重组，缺片时丢弃该帧，继续接收后续帧。可用“Link MB/s”限制卫星发包速率，但它不是可靠传输或自动拥塞控制。

### 准备配置、模型与网格

下面的示例 YAML 含开发机绝对路径，部署前先复制并修改。**同一个配置文件**用于星上包生成、地面跟踪器和 App；后两者每次启动时读取它。当前打包脚本将构建机的 `/home` 只读挂载进 Docker 容器，所以仓库、配置文件、网格和标定都应位于 `/home` 下。以下命令在仓库根目录执行：

```bash
CONFIG="$HOME/tracker_udp.yaml"
cp fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml "$CONFIG"
nano "$CONFIG"
```

修改时核对以下字段；两个列表中的物体**顺序必须一致**：

| YAML 位置 | 需要填写的内容 |
| --- | --- |
| `foundationpose_stereo_tracker_multi_node.ros__parameters.stereo_engine_file_path` | 地面 GPU 上的 FFS feature/post engine，按此顺序填写 |
| 同节 `refiner_engine_path`、`scorer_engine_path` | 地面 GPU 上的 FoundationPose engine |
| 同节 `caminfo_path`、`fx/fy/cx/cy` | 与双目图对应的标定文件和左相机内参；打包器把 `caminfo_path` 复制为卫星包内 `assets/calibration.txt` |
| 同节 `tracked_object_names`、`tracked_mesh_paths`、`tracked_mask_image_names` | 每个目标的名称、地面 OBJ 绝对路径和首帧掩码文件名；`mask_image_directory` 要指向可写目录 |
| `grounded_sam_multi_first_mask_node.ros__parameters.dino_engines` | 按目标顺序填写每个目标的 DINO engine |
| 同节 `sam_encoder_engine`、`sam_decoder_engine` | 地面 SAM engine 路径 |

打包器按 `tracked_mesh_paths` 的顺序复制**每个 OBJ 所在的整个目录**：第一项放到 `assets/meshes/0/`，第二项放到 `assets/meshes/1/`，以此类推，包括该目录内的 MTL、纹理与辅助文件。App 会用 `meshes/<从0开始的序号>/<OBJ文件名>` 配置卫星端。例如两目标示例的首项在卫星上是 `assets/meshes/0/textured_mesh.obj`。改动目标数量、顺序、OBJ、纹理或标定后，重新打包并重启相关进程。卫星包只含渲染所需的网格与标定；FFS、FP、DINO、SAM engine 在地面主机使用，须按前文说明在该 GPU 上由官方模型转换。

卫星渲染器读取 OBJ 的顶点、纹理坐标和面，并从 MTL 的 `map_Kd` 加载 PNG 纹理；复杂材质、其他网格格式及 JPG 纹理应先转成这一格式。网格尺寸应以米为单位，并与地面 FoundationPose 使用的同一目标网格保持一致。标定文本沿用 [`ffs/64091.txt`](ffs/64091.txt) 的布局，包含左右相机内参、各 5 个畸变系数、双目旋转和平移及米制基线；地面深度还会读取这份标定，因此不能只修改其中一侧。

### 制作并传输 Ubuntu 20.04 卫星离线包

在**可联网的 x86_64 构建机**上安装 Docker 并确认 `docker` 命令可运行，然后执行。构建镜像时需要网络安装 Ubuntu 20.04 的编译依赖；目标卫星主机不需要联网。`CONFIG` 必须是 `/home` 下的**绝对路径**，不要把相对路径传给打包脚本。

```bash
./standalone_udp/scripts/build_satellite_ubuntu20.sh "$CONFIG"
tar -tzf standalone_udp/dist/satellite_udp_ubuntu20_x86_64.tar.gz | head -n 30
sha256sum standalone_udp/dist/satellite_udp_ubuntu20_x86_64.tar.gz
```

归档目录 `satellite_udp_ubuntu20_x86_64/` 含 `bin/satellite_renderer_udp`、`lib/` 内的动态加载器及 `ldd` 查到的运行库、`assets/calibration.txt`、`assets/meshes/<序号>/`、`licenses/`、`run_satellite.sh`、`MANIFEST.json` 和 `SHA256SUMS`。这是面向 **Ubuntu 20.04、x86_64** 的包，不要把当前 x86_64 包直接拿到 ARM/Jetson 上运行。归档文件被 Git 忽略，不会随源码推送。若卫星已开启 SSH，先按下节配置网口，再通过直连网线复制；也可使用 U 盘。以下 `satellite` 应替换为卫星主机的实际用户名：

```bash
scp standalone_udp/dist/satellite_udp_ubuntu20_x86_64.tar.gz satellite@192.168.50.2:~/
```

在卫星主机上，以接收归档的用户登录，执行：

```bash
tar -xzf ~/satellite_udp_ubuntu20_x86_64.tar.gz -C ~/
cd ~/satellite_udp_ubuntu20_x86_64
sha256sum -c SHA256SUMS
./run_satellite.sh 50000
```

`sha256sum -c` 应全部显示 `OK`。启动脚本显式使用包内的 `ld-linux-x86-64.so.2` 和 `lib/`，所以不要只复制 `bin/` 下的单个可执行文件。程序输出 `Satellite UDP renderer listening on 50000` 后等待 App 控制；这时还不会自行发送图像。卫星端不需要 ROS、CUDA、TensorRT 或 Python 运行环境。首次在真实星上主机部署时，检查其系统与架构确为 Ubuntu 20.04 x86_64，并在目标机实际执行上述校验和启动命令。

### 两台主机的网线配置

先用 `ip -br link` 查出实际网卡名。以下 `enp1s0` 只是示例，命令需分别在对应主机执行并换成各自的真实网卡；`ip addr add` 设置在重启后不会自动保留，长期部署可改用各自系统的 Netplan 配置。

```bash
# 地面主机
GROUND_IF=enp1s0
sudo ip link set "$GROUND_IF" up
sudo ip addr add 192.168.50.1/24 dev "$GROUND_IF"

# 卫星主机
SAT_IF=enp1s0
sudo ip link set "$SAT_IF" up
sudo ip addr add 192.168.50.2/24 dev "$SAT_IF"
```

地面执行 `ping 192.168.50.2`，卫星执行 `ping 192.168.50.1`。防火墙需允许卫星主机收到来自地面的 UDP 50000，也需允许地面主机收到来自卫星的 UDP 50000；本机的 50100/50101 只在地面环回接口使用。若用 `ufw` 且策略阻断入站流量，可分别加入针对对端 IP 的规则：

```bash
# 地面主机
sudo ufw allow from 192.168.50.2 to any port 50000 proto udp
# 卫星主机
sudo ufw allow from 192.168.50.1 to any port 50000 proto udp
```

### 构建地面跟踪器与启动整套系统

地面主机需要与其 GPU 匹配的驱动、CUDA、TensorRT、OpenCV、Eigen3、Assimp、yaml-cpp、libpng、zlib；App 需要系统 Python 3 的 PyQt5、NumPy、OpenCV 和 PyYAML。常见 Ubuntu 开发包可先安装如下，CUDA/TensorRT 仍需按对应设备单独安装。**这一套启动流程不需要 `source /opt/ros/...`，也不需要启动 ROS daemon。**

```bash
sudo apt install build-essential cmake libopencv-dev libeigen3-dev libassimp-dev \
  libyaml-cpp-dev libpng-dev zlib1g-dev python3-pyqt5 python3-numpy \
  python3-opencv python3-yaml
cd '/home/hc/weizi/ffs+fp+sam'  # 换成地面主机上的仓库路径
./standalone_udp/scripts/build_ground.sh
```

构建结果是 `standalone_udp/build/ground_tracker_udp`；脚本会检查其动态依赖中没有 ROS 库。部署时依次进行：

1. **卫星终端：** 保持前述 `./run_satellite.sh 50000` 运行。
2. **地面 App 终端：** 在仓库根目录执行下列命令，随后打开“Parameters and control”页设置参数，点击 **Configure and start satellite**。此时“Tracking monitor”页应开始显示左图和渲染真值。

   ```bash
   ./standalone_udp/scripts/start_ground_app.sh "$CONFIG" 192.168.50.2 50000
   ```

3. **地面跟踪终端：** 需要位姿结果时再运行跟踪器。它会在首帧生成掩码并执行注册；之后 App 显示估计位姿、合成图、掩码和同帧误差。

   ```bash
   ./standalone_udp/scripts/start_ground_tracker.sh "$CONFIG"
   ```

这里的 `CONFIG` 与打包时使用的文件一致；若在新终端中执行，先再次设置 `CONFIG="$HOME/tracker_udp.yaml"`。星上端口如改为其他值，应同时修改卫星启动命令和 App 启动命令的最后一个参数。App 关闭时释放地面 UDP 端口；停止渲染可先点 **Stop satellite**，再在各终端按 `Ctrl+C`。只改运动参数时，可再次点击 **Configure and start satellite**；地面跟踪器看到帧号从 0 重新开始后会重置目标注册。修改 YAML 中的模型、目标或标定时，请停止相关进程后重启；卫星资源变化还需重新打包。

### App 界面与参数设置

App 有两个页签，启动时通过 `--config` 读取目标名称与网格列表。默认示例的参数会显示在控件中，可逐目标修改。

| “Parameters and control”区域 | 设置方式与作用 |
| --- | --- |
| `Satellite IP` | 卫星主机网口 IP，初始值来自启动命令；必须能从地面主机访问。 |
| `Calibration in satellite bundle` | **相对于卫星包 `assets/` 的路径**，默认 `calibration.txt`；不要填地面 YAML 的绝对路径。 |
| `FPS` | 卫星渲染目标帧率，界面允许 0.1–60，默认 8。首次联调可先设为 1，再按链路和计算能力提高。 |
| `Link MB/s` | 卫星 UDP 发包速率上限，按十进制 `1 MB = 1,000,000 字节` 换算，界面允许 0.1–1000，默认 10。该值要结合实际网线速率和帧大小调整。 |
| `Initial poses` 表 | 每行一个物体；`X/Y/Z` 为米、`Roll/Pitch/Yaw` 为度，表示物体原始 mesh 原点在左相机光学坐标系中的初始位姿。行顺序来自 YAML 的 `tracked_object_names`。 |
| `Ground mesh` 列 | 从 YAML 的 `tracked_mesh_paths` 读入。App 取该列的**文件名**来拼接卫星路径 `meshes/<行号从0开始>/<文件名>`；地面跟踪器仍从启动时的 YAML 绝对路径加载网格。要换目标，应同步修改 YAML 并重新打包，不能只改此单元格。 |
| `Independent motion` 表 | 各目标分别设置 `Amp XYZRPY` 和 `Freq XYZRPY`。XYZ 振幅单位为米，RPY 振幅单位为度，六个频率均为 Hz；振幅或频率设为 0 可使对应轴保持初值。 |
| `Configure and start satellite`、`Stop satellite` | 前者依次停止、配置、启动卫星并显示 ACK 状态；后者只停止渲染，卫星进程继续监听控制命令。 |

| “Tracking monitor”区域 | 显示内容 |
| --- | --- |
| `Satellite left image` | 收到卫星双目 PNG 后显示左眼图；App 会把渲染图的黑色背景替换为本地背景图片。无需启动地面跟踪器。 |
| `Tracker: left \| pose \| depth` | 跟踪器回传的三联图：填充背景后的左图、跟踪位姿框、伪彩深度。 |
| `First-frame masks` | 按目标着色的首帧掩码；目标尚未检测到时对应区域为空。 |
| 位姿表 | 每个目标有 `Estimate`、`Truth/Latest`、`Error` 三行。估计值是滤波后的 XYZ、ZYX 欧拉角和 `(qx,qy,qz,qw)`；真值行按同帧时间戳匹配，找不到同帧真值时显示最新值并标 `Latest`。只有同帧匹配成功才填写误差行；`Δt` 列是米制平移误差范数，`ΔR` 是两四元数的最短相对旋转角。 |

角度误差的**显示值**沿用现有演示约定：RPY 误差与 `ΔR` 都是原始角度误差除以 10；估计/真值行的角度仍为原始度数，鼠标悬停误差单元格可看原始角度。这种缩放只改变界面数字，**不代表实际角度精度提高**。欧拉角转序明确为 `R = Rz(yaw) Ry(pitch) Rx(roll)`。改变网格、标定或目标顺序后，应同时更新包、地面 YAML 与 App 使用的 YAML，并重启跟踪器和 App。

### 联调检查

先确认卫星输出 `Satellite UDP renderer listening on 50000`，地面 `ping` 成功，App 点击启动后状态栏显示 `Satellite start: OK`。如果 App 提示 `Satellite timeout`，依次核对星地 IP、网卡、防火墙、两端 UDP 50000 和卫星进程日志。可用 `ss -lun` 查看监听端口。若左图已显示而位姿未出现，查看地面跟踪器终端的 engine/网格/标定加载错误、DINO/SAM 首帧检测结果，以及是否监听 `127.0.0.1:50100`；确认 `mask_image_directory` 可写。丢帧明显时先将 App 的 FPS 降至 1，并核对 `Link MB/s` 与实际链路容量；SUD1 对缺片帧不会补传。

本工程已完成本机和 Ubuntu 20.04 容器桥接网络的两目标联调、卫星包 SHA256 校验及无开发库基础镜像启动。**尚未在两台真实 Linux 主机之间实测**；最终部署仍需在目标卫星系统上核对 Ubuntu/架构、网卡、防火墙、带宽与实际帧率。

## 自适应滤波参数与行为

两个跟踪节点使用同一套 YAML 参数，默认值对齐 `run_demo_adaptive_filter.py` 的当前参数。`filter_fps` 为时间戳不可用时的回退帧率；正常运行时按相邻左图时间戳求 `dt`。`filter_q_pos/q_rot/q_vel/q_omega` 为 12 维状态预测噪声；`filter_sigma_pos_obs/sigma_rot_obs` 为观测标准差；`filter_sigma_pos_motion/sigma_rot_motion` 控制残差到质量分数的衰减；`filter_min_quality` 限制质量下界；`filter_gate_threshold` 是马氏距离平方门限；`filter_reject_cov_increase` 是拒绝观测时增加的协方差。`filter_enable_gating` 与 `filter_enable_adaptive_R` 可分别关闭门控和质量驱动的观测噪声调整。

首帧直接以 `Register` 位姿初始化滤波器、零速度和 `1e-3 I` 协方差。每帧先预测 `T_pred = T · Exp(xi·dt)`；`Track` 使用 `T_pred` 作初值。右不变残差 `Log(T_pred⁻¹ · T_obs)` 同时用于质量评估、卡尔曼更新与门控。观测被拒绝时保留预测位姿并增加协方差；跟踪失败时也使用预测位姿。日志中的 `quality`、`mahalanobis`、`accepted` 可用于调参。相机或物体运动速度与 Python 演示不同，门限与噪声仍应结合实际数据标定。

## 作者与论文引用

本项目的集成和部署代码由 **hc**（GitHub：`hc1600203208-ctrl`）维护；其他组件的原作者见根目录 `AUTHORS`、`NOTICE` 和各源文件版权声明。引用本部署项目时可使用 `CITATION.cff`，并按实际使用的模型引用其原论文：

```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
@inproceedings{foundationposewen2024,
  title={{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
  author={Bowen Wen and Wei Yang and Jan Kautz and Stan Birchfield},
  booktitle={CVPR},
  year={2024}
}
@article{liu2023grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
@article{kirillov2023segany,
  title={Segment Anything},
  author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

若使用 BundleSDF 重建目标网格，还应引用 [BundleSDF（CVPR 2023）](https://arxiv.org/abs/2303.14158)。上述论文描述模型方法；自适应 SE(3) 滤波和本仓库多目标部署属于本项目集成工作。
