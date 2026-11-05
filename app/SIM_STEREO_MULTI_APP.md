# 本地多目标双目渲染 App

对应程序：[`sim_stereo_multi_app.py`](sim_stereo_multi_app.py)。本文描述当前实现的本地虚拟双目数据源；FoundationPose 多目标跟踪由独立的 launch 启动。

## 1. 功能、作用与输入输出

这个 App 用于在一台地面主机上搭建可重复的多目标双目场景。用户选择两个或更多 OBJ 模型，为每个目标指定相对左相机的初始 6 自由度位姿，再为所有目标设置共用的周期运动参数。点击界面按钮后，App 启动 `sim_stereo_cpp` 的 C++ 多目标渲染节点，并把渲染图像叠到背景上，供 SAM 首帧分割和 FoundationPose 跟踪使用。

| 类别 | 当前实现 |
| --- | --- |
| 配置输入 | Tracker YAML 中的目标名称、mesh 路径和双目标定文件；界面中的初始位姿、运动幅值、频率和背景图。 |
| 模型文件 | OBJ；贴图和材质应与 OBJ 一起提供，通常包括 MTL 和纹理图片。 |
| 渲染节点原始输出 | `/sim_app/rendered_left`、`/sim_app/rendered_right`，以及相应的 `/sim_app/left/camera_info`、`/sim_app/right/camera_info`。 |
| App 向跟踪链路发布 | 背景合成后的 `/left/image_raw`、`/right/image_raw`（BGR8）和 `/left/camera_info`、`/right/camera_info`。 |
| App 自身显示 | 目标配置、运动控制和节点状态；**本地 App 当前没有内嵌跟踪结果画面**。跟踪结果由 tracker 的窗口或 ROS 话题查看。 |

App 默认读取 [`foundationpose_stereo_tracker_multi_bluepink_example.yaml`](../fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluepink_example.yaml)。示例 YAML 里的模型、标定和 TensorRT engine 路径是当前机器的绝对路径，换环境时要先逐项检查。

## 2. 基于 mesh 生成双目图像的简要原理

渲染节点先读取 OBJ 的几何与纹理，并读取双目相机的内参和左右相机相对位姿。对于同一个目标，它用当前时刻的位姿分别计算其在左、右相机中的投影，把可见表面及纹理绘制到两张 640×480 图像。多个目标按表格顺序绘制在同一双目帧中。位置和角度随时间围绕初始值作正弦变化，例如 `x(t) = x₀ + Aₓ sin(2πfₓt)`；各轴使用界面给出的共用幅值与频率。

App 收到渲染节点输出后，把模型前景与选定的静态背景合成，再发布标准双目图像话题。背景主要方便观察场景；左右图中的背景是同一张图，不能视作具有真实立体视差的背景。最终深度和跟踪效果仍要以实际输出检验。

## 3. 界面组成与参数设置

| 区域 | 如何设置 |
| --- | --- |
| `Tracker YAML` | 选择与待启动多目标 tracker **同一份**配置，点击 `Load objects`。这会重建目标表和标定路径。 |
| `Background image` | 可选一张本地图片；留空时用程序内置背景。该背景仅供图像合成，不是 3D mesh。 |
| `Objects` 表 | 每行一个目标：`Name`、`Mesh path`、`X/Y/Z (m)`、`Roll/Pitch/Yaw (deg)`。`Add object`、`Browse selected mesh` 和 `Remove selected` 管理行；至少保留两个目标，名称不能重复。 |
| `Shared position amp.` | 全部目标在 X/Y/Z 方向的共用平移幅值，单位米。输入 0 表示该轴不往复运动。 |
| `Shared rotation amp.` | 全部目标的 Roll/Pitch/Yaw 共用角幅值，单位度。 |
| `Shared frequency` | 平移三轴与旋转三轴各自的往复频率，单位 Hz。 |
| 运行按钮 | `Start motion` 创建渲染进程并开始运动；运行中点击 `Stop motion` 暂停运动；`Apply parameters and reset` 用当前参数重新创建渲染进程并等待再次启动。 |

当前 App 固定将渲染器设为 **640×480、目标 30 FPS**；实际发布帧率还受渲染用时和主机负载限制。初始位姿的参考是左相机坐标系，模型中心的定义仍取决于模型及渲染器的加载方式。

## 4. 环境与构建

以下命令基于此仓库当前的 Linux + ROS 2 Jazzy 布局。图形界面需要 PyQt5；App Python 环境还需 `rclpy`、`cv_bridge`、PyYAML、NumPy 和 OpenCV。`sim_stereo_cpp` 使用 CUDA、OpenCV、Eigen，并链接本工程已安装的 FoundationPose 渲染库；tracker 和 SAM 使用其各自的 GPU/TensorRT engine。

先按仓库的 FoundationPose、SAM 构建说明准备 `fp/install` 和 `sam/ros2_install`，再构建本地渲染包：

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
colcon build --packages-select sim_stereo_cpp
```

## 5. 与 `foundationpose_stereo_tracker_multi.launch.py` 搭配运行

先在 App 的 `Tracker YAML` 中选定同一配置；下面命令以 App 默认的 bluepink YAML 为例。不同终端分别执行：

**终端 A：启动地面多目标 SAM + FoundationPose。** launch 会从左图生成各目标的首帧 mask，tracker 等全部 mask 准备好后进行注册和连续跟踪。

```bash
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py \
  params_file:=/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluepink_example.yaml
```

**终端 B：启动 App，加载模型和参数，然后点击 `Start motion`。** 渲染节点由 App 自动启动，无须再单独运行 `stereo_render_multi_cpp_node`。

```bash
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
python3 /home/hc/weizi/ffs+fp+sam/app/sim_stereo_multi_app.py
```

当前 tracker 配置可发布 `/foundationpose/combined_visualization`，也可通过配置开启独立可视化话题或 OpenCV 窗口；每个目标的位姿在 `/foundationpose/pose/<目标名经过话题字符规范化>`。可先检查双目输入和总图是否产生：

```bash
ros2 topic hz /left/image_raw
ros2 topic hz /right/image_raw
ros2 topic info /foundationpose/combined_visualization --verbose
```

在 RViz 的 Image 显示中选择总图话题即可查看跟踪叠加与深度。首次启动时 SAM 会先初始化模型并计算各目标 mask，因此首帧位姿可能晚于双目图像。更换目标模型或重做初始注册时，重启多目标 launch，以重新生成 mask 并清除旧注册状态。

## 6. 使用边界

- App 修改目标表只改变本地渲染输入；tracker/SAM 的目标名、mesh 路径、mask 文件名仍由其启动时的 YAML 决定。要让跟踪目标同步改变，先更新 YAML，再重启 launch。
- 本地渲染器需要 GPU/CUDA 和 FoundationPose 渲染库；它与星地方案的 CPU 渲染器是两个不同的程序。
- `--check` 可验证默认示例的模型和标定等基本路径：`python3 /home/hc/weizi/ffs+fp+sam/app/sim_stereo_multi_app.py --check`。
