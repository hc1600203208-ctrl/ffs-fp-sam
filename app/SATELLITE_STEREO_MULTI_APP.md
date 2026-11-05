# 星地多目标双目渲染 App 与部署

对应程序：[`satellite_stereo_multi_app.py`](satellite_stereo_multi_app.py)。这是在[本地多目标双目渲染 App](SIM_STEREO_MULTI_APP.md)场景上的星地扩展：目标配置和观察留在地面，mesh 的双目光栅化放在星上。以下首先描述**当前已实现的 ROS 2 双主机版本**，最后说明脱离 ROS 2 时需要替换什么。

## 1. 职责划分和 App 功能

| 位置 | 当前组件 | 职责 |
| --- | --- | --- |
| 星上 CPU 主机 | `satellite_render_node` | 收到参数后读取 OBJ/MTL/PNG 与双目标定，渲染多个目标的左右视图，按设定链路速率发送字节包。星上渲染代码只使用 C++、Eigen、libpng 及 ROS 2 基础库，不调用 CUDA、CV-CUDA 或 OpenCV。 |
| 地面主机 | `satellite_stereo_multi_app.py` | 设置目标、初始位姿、共用运动、链路速率、背景和标定；向星上发送控制字节包并等待确认；启动地面的字节转换节点；显示结果。 |
| 地面主机 | `stereo_bytes_to_images_node` | 从字节包恢复完整双目帧，在空背景像素处加入静态背景和固定左右视差，再发布标准 ROS 双目图像及 CameraInfo。 |
| 地面 GPU 主机 | `grounded_sam_multi_first_mask_node` 与 `foundationpose_stereo_tracker_multi_node` | 用第一帧左图分割各目标，使用双目图估计深度并注册/跟踪位姿，发布总图、mask 和每目标位姿。两者由 `foundationpose_stereo_tracker_multi.launch.py` 启动。 |

当前 App 有两个页面：

- `Parameters & control`：选择 tracker YAML；编辑每个目标的名称、星上 mesh 路径、XYZ（米）与 Roll/Pitch/Yaw（度）；设置全部目标共用的六轴运动幅值和频率；设置 `Target FPS`、`Link MB/s`、卫星标定文件路径、地面背景 PNG 与背景视差；用按钮启动/停止或重新配置星上渲染。
- `Tracking monitor`：显示 tracker 发布的 `/foundationpose/combined_visualization` 总图（左图、跟踪叠加、深度可视化），将各目标首帧 mask 合成一张彩色图，并展示每个模型的单帧纹理预览与各目标 XYZ/欧拉角位姿表格。App **不**单独订阅原左图、独立深度图和独立跟踪图。

App 默认加载 [`foundationpose_stereo_tracker_multi_bluegold_example.yaml`](../fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml)，初始目标是 blue carton 与 golden object。App 的 YAML 必须和地面 tracker launch 使用的配置一致；目标名称应与 SAM 的 `object_names`、tracker 的 `tracked_object_names` 对应。改变 App 表格中的模型或目标后，地面 tracker/SAM 不会自动重新加载，需同步修改 YAML 并重启 launch。

## 2. 星地通信流程

```text
地面 App ── SCT1 控制字节 ────────────> 星上 CPU 渲染节点
地面 App <── SCT1 应答字节 ──────────── 星上 CPU 渲染节点
地面接收节点 <── STR1 双目 RGB565 字节包 ─ 星上 CPU 渲染节点
地面接收节点 ── 左右 BGR8 + CameraInfo ─> 地面 SAM / FFS / FoundationPose
地面 App <── 总图 + 各目标 mask/位姿 ─── 地面跟踪链路
```

当前传输载体是三个 ROS 2 字节话题（消息类型均为 `std_msgs/msg/UInt8MultiArray`）：

| 方向 | 话题 | 内容 |
| --- | --- | --- |
| 地面 → 星上 | `/satellite/control_bytes` | SCT1：STOP（3）、CONFIG（1）、START（2）命令。 |
| 星上 → 地面 | `/satellite/control_ack_bytes` | SCT1：每条命令的请求 ID、状态和原因文本。 |
| 星上 → 地面 | `/satellite/stereo_bytes` | STR1：有序分片的左右图原始 RGB565 像素。 |

点击 App 的 `Start satellite renderer` 后，App 会先确保地面转换节点已启动，然后依次发 STOP → CONFIG → START，每步都等待星上应答。SCT1 使用小端序，控制包以 16 字节头开始，包含命令类型、请求 ID 和负载长度；CONFIG 负载包含画面尺寸（当前 App 固定 640×480）、目标 FPS、字节每秒上限、六轴幅值/频率、**星上可访问**的标定路径，以及每个目标的名称、星上 mesh 路径和初始 6 自由度位姿。发送的是路径字符串，不是 OBJ、纹理或标定文件本身；文件须提前部署到星上。协议允许配置 1～32 个目标，但这个多目标 App 要求至少两个。正在渲染时点击按钮发送 STOP；`Apply and restart` 使用当前界面设置重新配置并启动。SCT1 应答超时时 App 会提示失败。

STR1 每包以 32 字节头开始：`STR1` 标识、帧号、分片序号、总分片数、宽、高、秒和纳秒时间戳；后接不超过 32 KiB 的 RGB565 数据。完整帧依次存放左图和右图，每像素 2 字节，没有 JPEG/PNG 压缩；各像素的两个字节按小端序排列，位布局是红色 5 位、绿色 6 位、蓝色 5 位。地面节点按帧号和分片顺序重组；帧不完整时丢弃，不把残缺图送给 tracker。640×480 一对双目图约为 **1,228,800 字节**（另有分片头和 ROS/DDS 开销），8 FPS 仅像素数据就需约 **9.83 MB/s**。App 默认 `Target FPS = 8`、`Link MB/s = 10`，这里的 MB/s 指每秒 **百万字节**，不是 Mbit/s；速率上限、渲染耗时和 DDS/网络开销都会影响实际帧率。各字段的字节偏移和应答状态码见[星地接口规范](../sim/src/satellite_stereo/INTERFACE.md)。

地面转换节点发布 `/left/image_raw`、`/right/image_raw`（BGR8）以及 `/left/camera_info`、`/right/camera_info`。它把星上 RGB565 值为 0 的空像素填成背景：默认使用 [`stereo_background.png`](assets/stereo_background.png)，右目背景相对左目有 8 像素视差。背景仅驻留地面、不随图像字节包传输。当前方法把值恰好为 0 的模型纹理像素也视作空背景；严格区分前景需升级传输格式，加入显式前景掩码。背景能减少纯黑区域中的匹配混乱，但不是对任意场景深度精度的保证。

地面 launch 从左图得到各目标首帧 mask PNG，tracker 在 mask 就绪后注册模型并持续处理双目帧。App 订阅总图 `/foundationpose/combined_visualization`、每目标 `/first_masks/<目标名>` 和 `/foundationpose/pose/<目标名>`；mask 在界面中按颜色合成，重叠处显示白色。总图默认最多 5 Hz，由 `depth_visualization_fps` 控制，实际速度受 tracker 计算时间限制。位姿表将 PoseStamped 的四元数转换为 ZYX 约定的 Roll/Pitch/Yaw（度）。

## 3. 环境与依赖

| 星上 x86-64 Linux | 地面 Linux GPU 主机 |
| --- | --- |
| 与构建机兼容的 x86-64 Linux/glibc；当前发布节点需要 ROS 2 Jazzy 运行时、DDS、`libpng16`、`libstdc++`。Eigen 为头文件库，编译后无需单独部署 `.so`。 | ROS 2 Jazzy、`satellite_stereo` 包、Python 3 的 PyQt5/rclpy/cv_bridge/PyYAML/NumPy/OpenCV、FoundationPose/FFS/SAM 编译产物和相应 CUDA/TensorRT engine。 |
| OBJ + MTL + PNG 纹理及星上可访问的双目标定文件。无需 CUDA/OpenCV。 | Tracker YAML、mesh 与纹理（供跟踪与 App 预览）、双目标定、地面背景 PNG。 |

当前 `satellite_render_node` 是**动态链接的 ROS 2 可执行程序**，不是此前 musl 文档设想的独立静态 ELF。只复制二进制而不提供 ROS 2/DDS 运行库，无法直接在空文件系统启动。用 `ldd` 检查目标机缺失的动态库；部署不同 libc/架构时，需要在对应目标环境重新构建。下面给出当前 ROS 2 版本可实际执行的打包流程。

## 4. 在地面构建并打包星上部分

在地面构建机先安装 ROS 2 Jazzy、colcon、Eigen3 和 libpng 开发包，再构建星地公共包；地面跟踪链路的 FP、SAM 另按各自工程构建。

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
colcon build --packages-select satellite_stereo
```

以下示例把两个默认模型、纹理、星上标定和星上二进制放到 `/home/hc/satellite_bundle`。这样地面 App 也能预览同一路径的 mesh；若卫星用户名或路径不同，须把 App 中的 `Mesh path` 和 `Satellite calibration` 改成**实际星上绝对路径**，并确保地面同一路径也有对应 mesh 供预览。

```bash
mkdir -p /home/hc/satellite_bundle/bin /home/hc/satellite_bundle/models/blue /home/hc/satellite_bundle/models/gold /home/hc/satellite_bundle/calibration
cp /home/hc/weizi/ffs+fp+sam/sim/install/satellite_stereo/lib/satellite_stereo/satellite_render_node /home/hc/satellite_bundle/bin/
cp /home/hc/weizi/dataset/jrnew-blue/mesh/textured_mesh.obj /home/hc/weizi/dataset/jrnew-blue/mesh/material.mtl /home/hc/weizi/dataset/jrnew-blue/mesh/material_0.png /home/hc/satellite_bundle/models/blue/
cp /home/hc/dataset/myobject/weixing1/weixing_mesh/textured_mesh.obj /home/hc/dataset/myobject/weixing1/weixing_mesh/material.mtl /home/hc/dataset/myobject/weixing1/weixing_mesh/material_0.png /home/hc/satellite_bundle/models/gold/
cp /home/hc/weizi/ffs+fp+sam/ffs/64091.txt /home/hc/satellite_bundle/calibration/
tar -C /home/hc -czf /home/hc/satellite_bundle.tar.gz satellite_bundle
tar -tzf /home/hc/satellite_bundle.tar.gz | head
```

上传并在星上展开。下面的 `192.0.2.10` 是文档示例地址，执行前改成卫星主机的实际 IP；这一步需要星上已安装与程序兼容的 ROS 2 Jazzy 和 libpng 运行库：

```bash
SAT_HOST=hc@192.0.2.10
scp /home/hc/satellite_bundle.tar.gz "$SAT_HOST:/home/hc/"
ssh "$SAT_HOST" 'tar -C /home/hc -xzf /home/hc/satellite_bundle.tar.gz'
ssh "$SAT_HOST" 'bash -lc "source /opt/ros/jazzy/setup.bash && ldd /home/hc/satellite_bundle/bin/satellite_render_node"'
```

检查 `ldd` 输出中没有 `not found`。这里的包只含星上渲染所需文件；**地面**还要保留 App、`stereo_bytes_to_images_node`、SAM、FFS 和 FoundationPose。上传前可确认 OBJ 引用的 MTL 和 `map_Kd` 纹理文件确实在同一模型目录。实际专用通信链路或 musl 静态部署需要单独适配传输层与构建方式，不能直接用这份 ROS 2 包代替。

## 5. 星上和地面的完整启动顺序

两台主机需能通过 ROS 2 DDS 互相发现，使用同一 `ROS_DOMAIN_ID`，并允许对应网络的发现与数据 UDP 流量。示例取 `ROS_DOMAIN_ID=42`；避免与现场其他 ROS 网络撞域。

**终端 1，星上：启动空闲渲染节点。** 无须在命令行给模型；App 会发送 SCT1 配置。

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
/home/hc/satellite_bundle/bin/satellite_render_node
```

**终端 2，地面 GPU 主机：启动 SAM + FoundationPose。** 与 App 默认 bluegold YAML 保持一致。launch 会在启动时清理其指定 mask 目录中的旧图片；不要与另一个仍在使用该目录的跟踪任务并行。

```bash
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash
export ROS_DOMAIN_ID=42
ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py \
  params_file:=/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluegold_example.yaml
```

**终端 3，地面图形会话：启动 App。** 地面字节转换节点由 App 自动启动；勿重复手动启动同名转换节点。

```bash
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sim/install/setup.bash
export ROS_DOMAIN_ID=42
python3 /home/hc/weizi/ffs+fp+sam/app/satellite_stereo_multi_app.py
```

在 App 的 `Parameters & control` 中检查或填写：

1. `Tracker YAML` 为终端 2 所用 bluegold 文件，点击 `Load objects`。
2. 两个模型名称与 tracker/SAM 配置一致；把表格里的 `Mesh path` 分别设为 `/home/hc/satellite_bundle/models/blue/textured_mesh.obj` 和 `/home/hc/satellite_bundle/models/gold/textured_mesh.obj`。这些文件在**地面和星上**都应能读取。
3. `Satellite calibration` 设为 `/home/hc/satellite_bundle/calibration/64091.txt`；地面 converter 和 tracker 仍读取 YAML 中的地面 `caminfo_path`。
4. 按需设置初始位姿、共用运动幅值/频率、`Target FPS`、`Link MB/s`、地面 `Background image`（默认图可直接使用）及 `Background disparity (px)`（默认 8）。
5. 点击 `Start satellite renderer`，待界面显示 `Rendering` 后切到 `Tracking monitor`。首帧 mask 的模型初始化与推理可能比第一帧双目图慢；mask 没齐时 tracker 继续等待，不代表链路一定失败。

可用下面命令分别检查三个阶段：

```bash
ros2 topic info /satellite/control_bytes --verbose
ros2 topic hz /satellite/stereo_bytes
ros2 topic hz /left/image_raw
ros2 topic info /foundationpose/combined_visualization --verbose
```

停止时在 App 中点击停止按钮，结束地面的 tracker launch，最后停止星上节点。更换目标或希望重新生成首帧 mask 时，应重启 tracker launch；App 的 `Apply and restart` 仅重新配置卫星渲染，**不会**清除 SAM 的一次性分割状态或 tracker 的已注册位姿。

## 6. 从 ROS 2 模拟走向真实星地链路

SCT1 与 STR1 的字节布局本身不依赖 ROS 2；当前 ROS 2 话题提供了进程发现、数据传输与订阅。换成实际通信设备时，应在地面控制端、星上发送端和地面接收端实现对相同字节包的收发，另补链路分帧、完整性校验、丢包处理/重传、超时与带宽控制。星上程序也需要脱离 `rclcpp` 和 ROS 消息重新构建；目前仓库中的 `satellite_render_node` 还不是这种可直接接入通信机的独立版本。
