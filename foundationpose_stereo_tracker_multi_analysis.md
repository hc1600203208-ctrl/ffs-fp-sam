# FoundationPose 多目标双目跟踪工程分析

## 1. 总体架构

当前 `foundationpose_stereo_tracker_multi.launch.py` 已经将 FFS 双目深度估计、C++ Grounded-SAM 和多目标 FoundationPose 串成一个 ROS 2 运行单元。ROS 2 层面只需要订阅左右原始图像，最终发布各物体位姿及可选的可视化图像。

```text
/left/image_raw ─┬─> Grounded-SAM C++ 首帧节点
                 │      每个物体：DINO -> bbox -> SAM -> mask PNG
                 │
                 └─> Multi Tracker ─┐
/right/image_raw ────────────────────┤
                                     ├─ 双目矫正
                                     ├─ FFS TensorRT -> disparity -> depth
mask PNG ────────────────────────────┤
                                     ├─ 首次 Register（逐物体）
                                     ├─ 后续 Track（每物体一个线程）
                                     ├─ PoseStamped 发布
                                     └─ 包围框、坐标轴与名称可视化
```

这里没有独立启动原来的 FFS ROS 节点，也不订阅深度话题。FFS 源码被直接编译为 `fast_foundation_stereo_embedded`，链接进 tracker。

## 2. 主 Launch 流程

入口文件：

```text
fp/src/foundationpose_cpp/launch/foundationpose_stereo_tracker_multi.launch.py
```

启动顺序如下：

1. 声明公共参数文件、图像话题、mask 目录、阈值、profiling 和深度置信过滤开关。
2. 默认加载 `foundationpose_stereo_tracker_multi_bluepink_example.yaml`。
3. 当 `run_first_mask=true` 时，删除 mask 目录下的旧图像，避免 tracker 误用上一次结果。
4. 启动 `grounded_sam_multi_first_mask_node`。
5. 同时启动 `foundationpose_stereo_tracker_multi_node`。
6. tracker 每隔 `mask_poll_interval_ms` 检查一次 mask 文件，等待首帧分割完成，并非由 launch 显式等待。
7. launch 为 tracker 调整 `LD_LIBRARY_PATH`，优先加载当前 `foundationpose_cpp` 安装目录，避免旧 overlay 中的 ABI 不兼容库。

该 launch 不负责播放 `/home/hc/bag/bluepink`，bag 需要另行播放。两个节点启动后都会等待图像输入。

## 3. 配置结构

默认配置文件：

```text
fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluepink_example.yaml
```

同一份 YAML 按 ROS 2 节点名分为两段。

### 3.1 Grounded-SAM 参数

- `object_names`：物体提示词。
- `dino_engines`：每个提示词对应一个固定 prompt 的 DINO TensorRT engine。
- `mask_output_names`：和 tracker 约定的 mask 文件名。
- `sam_encoder_engine`、`sam_decoder_engine`：所有物体使用的 SAM engine。
- `publish_mask_topics`：是否额外发布 transient-local mask 话题。
- `mask_topic_prefix`：mask 话题前缀。
- `box_threshold`、`max_detections`：检测后处理参数。

### 3.2 FoundationPose 参数

- 左右图话题与近似同步参数。
- 两个 FFS TensorRT engine。
- 双目标定文件、模型输入尺寸和深度范围。
- 公共 FoundationPose refiner/scorer engine。
- 每个物体的名称、mesh 和 mask 文件。
- 位姿发布、可视化和显示平滑参数。
- 注册和跟踪迭代次数。
- profiling 和深度置信度过滤参数。

三个多目标数组依靠相同下标建立关联：

```text
tracked_object_names[i]
tracked_mesh_paths[i]
tracked_mask_image_names[i]
```

节点启动时会验证数组非空、长度一致、名称转换后的 ROS topic 后缀唯一，并检查 mesh、engine 和标定文件存在。

## 4. 首帧 Mask 调用链

入口文件：

```text
sam/src/grounded_sam_multi_first_mask_node.cpp
```

调用链如下：

```text
ImageCallback
  -> RosImageToRgb
  -> 对 object_pipelines_ 逐个执行
     -> GroundedSamPipeline::run
        -> GroundingDinoRunner::detect
        -> SamRunner::segment
     -> 原子写入 PNG
     -> 可选发布 /first_masks/<object>
  -> 释放图像订阅，节点保持 idle
```

主要行为：

- 仅处理第一张成功接收的左图。
- 所有物体的分割当前是串行执行。
- 每个物体单独持有一个 DINO 和一套 SAM pipeline，因此当前实现会重复加载 SAM encoder/decoder。
- DINO engine 已固化 prompt，运行时的 `object_name` 主要用于标识，不会动态改变检测文本。
- DINO 输出满足阈值的 bbox。
- SAM 对所有 bbox 分别解码，然后按位 OR 合并成该物体的一张二值 mask。
- mask 先写临时文件再 rename，避免 tracker 读到尚未写完的 PNG。
- 分割失败时会恢复 `processed_`，允许下一帧重试。
- mask topic 没有被 tracker 使用，实际握手仍然通过文件系统完成。

核心组合位于：

```text
sam/src/pipeline.cpp
```

```cpp
const auto detections = dino_.detect(rgb, options_.box_threshold, options_.max_detections);
return sam_.segment(rgb, detections);
```

## 5. Tracker 初始化链

入口文件：

```text
fp/src/foundationpose_cpp/src/foundationpose_stereo_tracker_multi_node.cpp
```

节点构造顺序：

```text
DeclareParameters
-> LoadParameters
-> BuildStereoEstimator
-> BuildObjectTrackers
-> InitializeProfiler
-> SetupRosInterfaces
```

`BuildObjectTrackers()` 为每个物体创建：

- Assimp mesh loader。
- mask 文件路径。
- 独立 Pose publisher。
- 循环颜色表中的包围框颜色。
- 原始位姿和显示平滑状态。

FoundationPose 模型不会在这里立即创建，而是在注册阶段按物体创建。

## 6. 每帧数据流

左右图通过 `message_filters::ApproximateTime` 同步，回调流程为：

```text
StereoCallback
  -> try_lock，上一帧未完成则丢弃当前帧
  -> 再次检查左右时间差
  -> 左图转换为 FoundationPose RGB
  -> BuildDepthFromStereo
  -> 尚未全部注册：
       TryInitialRegistration
     已全部注册：
       RunTracking
```

该策略优先保证低延迟。当算力不足时直接丢弃新到达的帧，不积压旧帧。

### 6.1 深度生成链路

`BuildDepthFromStereo()` 的处理流程：

1. 根据输入尺寸生成或复用双目矫正映射。
2. 使用 `cv::remap` 矫正左右图。
3. 等比例缩放到模型输入尺寸并补边。
4. 调用 `FastFoundationStereoEstimator::inference()` 得到视差。
5. 使用 `fx * baseline / disparity` 转为米制深度。
6. 过滤非法、过近和过远深度。
7. 可选执行光度、纹理、边缘和时序置信度过滤。
8. 去除模型输入补边并恢复到矫正图分辨率。
9. 将矫正坐标系深度重新对齐到原始左图坐标。

复用的 FFS 实现包括：

- `fast_foundation_stereo_estimator.cpp`
- `gwc_volume_kernel.cu`
- `depth_confidence.cpp`
- `stereo_calibration_utils.hpp`

这些文件来自 `ffs/src/ffs_pkg`，由 FoundationPose 包直接编译和引用。

## 7. 首次注册与并行跟踪

### 7.1 首次注册

注册过程当前按物体串行执行：

```text
逐物体读取 mask
-> mask 通道和类型标准化并二值化
-> FoundationPose::Register
   -> 根据 mask 和 depth 生成初始平移
   -> 生成 252 个旋转假设
   -> renderer/refiner
   -> scorer 排序
   -> 输出最佳 mesh pose
-> 为该物体重新创建 tracking model
-> 启动常驻 tracking worker
```

只有所有物体都注册成功后，主流程才进入常规 tracking。某个 mask 一直为空或某个物体持续注册失败，会阻止所有对象进入跟踪阶段。

### 7.2 后续跟踪

- 每个物体拥有一个常驻 `std::thread`。
- 主回调向所有 worker 投递相同的 RGB、depth 和各自上一帧 pose。
- 每个 worker 调用独立的 `FoundationPose::Track()`。
- 主线程等待全部 worker 完成，然后统一更新和发布结果。
- 这是帧内多物体并行；下一帧仍然需要等待当前帧最慢的物体完成。
- 每个 worker 拥有独立 FoundationPose、TensorRT refiner/scorer context 和 renderer，避免不同线程共享同一个推理上下文。

## 8. 输出和可视化

默认发布话题：

- `/foundationpose/pose`：只发布第一个物体，用于兼容原单目标接口。
- `/foundationpose/pose/blue_carton`：blue carton 位姿。
- `/foundationpose/pose/pink_carton`：pink carton 位姿。
- `/foundationpose/multi_visualization`：仅在 `publish_visualization=true` 时发布。

位姿 topic 发布原始跟踪 pose；平滑 pose 主要用于日志和窗口左上角显示。

可视化规则：

- 不同物体使用循环颜色表绘制 3D 包围框。
- 物体名称绘制在投影包围框左上方。
- 所有物体坐标轴固定使用红、蓝、绿。
- 窗口左上角位姿文本只显示第一个物体。
- 绘框前调用 `ConvertPoseMesh2BBox()`，补偿 mesh 坐标系与包围框坐标系差异。
- 平移使用线性低通，旋转使用四元数 SLERP。

## 9. FoundationPose 内部结构

```text
FoundationPose
├─ Base6DofDetectionModel
├─ AssimpMeshLoader
├─ FoundationPoseSampler
├─ FoundationPoseRenderer
│  └─ nvdiffrast / CUDA rasterizer
├─ refiner TensorRT core
├─ scorer TensorRT core
└─ CUDA preprocess/postprocess/decoder
```

公共接口定义于：

```text
fp/src/foundationpose_cpp/detection_6d_foundationpose/include/
  detection_6d_foundationpose/foundationpose.hpp
```

- `Register()`：根据 RGB、depth 和 mask 生成、细化并评分大量初始姿态。
- `Track()`：以上一帧 pose 为单一初始假设，只运行 refinement。
- `easy_deploy_tool/inference_core/trt_core`：TensorRT engine、binding 和显存管理。
- `nvdiffrast`：根据 mesh 和假设位姿渲染模型 RGB/depth。
- Assimp：读取 OBJ mesh、顶点和物体尺寸。
- Eigen：SE(3)、旋转矩阵和四元数。
- OpenCV：图像转换、双目矫正、mask 处理和可视化。

## 10. 构建和运行依赖

### 10.1 ROS 2 依赖

- `ament_cmake`
- `rclcpp`
- `sensor_msgs`
- `geometry_msgs`
- `std_msgs`
- `cv_bridge`
- `message_filters`
- `tf2`

### 10.2 原生和 CUDA 依赖

- OpenCV
- Eigen
- Assimp
- glog
- CUDA Runtime
- TensorRT：`nvinfer`、`nvonnxparser`、`nvinfer_plugin`
- pthread
- nvdiffrast 内嵌源码
- ONNX Runtime：SAM 工程中的可选 DINO 后端；当前 YAML 使用 TensorRT，不进入该路径

主要构建文件：

```text
fp/src/foundationpose_cpp/CMakeLists.txt
fp/src/foundationpose_cpp/package.xml
fp/src/foundationpose_cpp/detection_6d_foundationpose/CMakeLists.txt
sam/CMakeLists.txt
sam/package.xml
```

## 11. 编码规范与复用方式

当前多目标代码的主要风格和复用方式如下：

- 使用 C++ RAII，通过 `unique_ptr`、`shared_ptr` 管理模型、推理核心和线程资源。
- 类名和函数名使用 `PascalCase`，局部变量及成员使用 `snake_case`，成员变量以 `_` 结尾。
- ROS 参数集中在 `DeclareParameters()` 和 `LoadParameters()` 中处理。
- 启动阶段进行路径、数组长度、输入尺寸和模型类型校验。
- 将回调之外的纯处理函数放在匿名命名空间中。
- 使用 OpenCV `cv::Mat` 作为三个推理模块之间的公共图像数据结构。
- 通过 `BaseInferCore`、`BaseMeshLoader`、`Base6DofDetectionModel` 复用已有接口。
- 直接编译 FFS 既有源码复用深度能力，没有复制一份深度算法实现。
- 通过每物体状态结构 `TrackedObject` 扩展单目标逻辑，没有引入新的 ROS 消息类型。
- 通过对象数组的相同下标维持跨节点配置映射。
- 对高频图像处理采用丢帧而不是排队策略，控制端到端延迟。

## 12. 当前架构边界

ROS 通信已经只承担原始双目图像输入和最终结果输出，但 Grounded-SAM 与 FoundationPose tracker 之间仍通过一次性 PNG 文件交换 mask，而不是进程内调用。

当前设计的主要边界包括：

- 首帧多个物体的 Grounded-SAM 推理是串行的。
- 每个物体 pipeline 会重复加载 SAM encoder/decoder engine。
- 多物体 FoundationPose 注册是串行的。
- 只有全部物体注册成功后才会进入 tracking。
- 跟踪阶段虽然按物体并行，但每帧整体速度由最慢的 worker 决定。
- 参数数组依靠下标关联，配置时必须保持顺序严格一致。
- launch 使用若干绝对路径，运行环境迁移时需要同步修改配置。

