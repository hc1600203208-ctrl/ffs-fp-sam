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

## 13. 模型预处理、TensorRT 推理和后处理细节

本节描述源码能够直接验证的张量操作、Host/Device 数据流和 TensorRT 调用。`.engine` 是已经序列化的优化网络，仓库没有保留 engine 的逐层网络定义；因此不能仅由 C++ 推断卷积层、注意力层或具体算子融合细节。

### 13.1 共通 TensorRT 生命周期

三个模型族均遵循相同的基本生命周期：

```text
.engine 二进制文件
  -> 读入 Host 内存
  -> nvinfer1::createInferRuntime(logger)
  -> runtime->deserializeCudaEngine(...)
  -> engine->createExecutionContext()
  -> 为每个 I/O tensor 分配 CUDA device buffer
  -> setInputShape（动态 shape 时）
  -> setTensorAddress(name, device_ptr)
  -> enqueueV3(cuda_stream)
  -> 输出 Device-to-Host 拷贝或直接在 GPU 后处理
```

FFS 和 Grounded-SAM 都使用 TensorRT 10 命名 I/O tensor API，即 `getNbIOTensors()`、`getIOTensorName()`、`setTensorAddress()` 与 `enqueueV3()`。FoundationPose 的 `trt_core` 也使用同一 API，但再封装了一层可复用 blob buffer。

### 13.2 FFS 双目深度估计

实现文件：

```text
ffs/src/ffs_pkg/src/estimator/fast_foundation_stereo_estimator.cpp
ffs/src/ffs_pkg/src/estimator/gwc_volume_kernel.cu
```

#### 输入准备和图像张量

tracker 在调用 FFS 前完成如下 CPU 图像操作：

```text
原始左右图
-> 双目校正 cv::remap
-> 按比例 resize 到模型输入范围
-> cv::copyMakeBorder 复制边缘像素补齐
-> 左右 cv::Mat，H=448、W=640、通常为 CV_8UC3/BGR HWC
```

模型构造函数要求输入高宽均能被 32 整除。当前 bluepink 配置将输入设为 `640x448`。

FFS 不会先在 CPU 上构造 `float CHW` 输入。它先将 `H*W*3` 字节 BGR HWC 图像直接异步上传：

```text
cv::Mat.data (Host uint8 BGR HWC)
  -- cudaMemcpyAsync HostToDevice --> dRawRGBLeft / dRawRGBRight
  -- LaunchPreprocessKernel --> dLeft / dRight
```

CUDA 预处理 kernel 将 BGR HWC `uint8` 变换为 RGB CHW `float`：

```text
[H, W, 3] uint8 BGR -> [1, 3, H, W] float RGB
```

该输入对应 feature engine 的命名输入 tensor：

```text
left:  [1, 3, 448, 640]
right: [1, 3, 448, 640]
```

源码中的 GPU 预处理没有除以 `255` 或 ImageNet 均值方差归一化；是否由 engine 内部承担该缩放取决于 engine 导出方式。

#### 两阶段 TensorRT 推理

FFS 被拆为两个 engine：`feature_runner` 和 `post_runner`。

1. engine 文件被读入 `std::vector<char>`，再调用 `deserializeCudaEngine`；两个 engine 共用一个 TensorRT runtime，各自创建 execution context。
2. 为 feature context 固定 `left` 和 `right` 的输入 shape，并读取输出 shape。
3. feature engine 绑定输入与输出地址后执行 `feature_context->enqueueV3(stream)`。
4. feature engine 产生如下 GPU float feature tensor：

```text
features_left_04
features_left_08
features_left_16
features_left_32
features_right_04
stem_2x
```

5. 为降低 group-wise correlation volume 的显存占用，`features_left_04` 和 `features_right_04` 在 GPU 上由 float 转为 FP16。
6. `LaunchGwcVolumeKernel` 根据左右 1/4 特征构造 GPU FP16 `gwc_volume`，其形状由 post engine 解析，通常为：

```text
[1, groups, max_disparity_at_quarter_scale, H/4, W/4]
```

7. post engine 的输入地址直接指向 feature engine 输出和自定义 CUDA kernel 输出，没有中间的 GPU-to-CPU-to-GPU 拷贝。
8. post engine 使用 `post_context->enqueueV3(stream)`，输出 `disp`；若 engine 包含可选 `prob`，还会产生聚合视差概率。

#### FFS Host/Device 数据流和后处理

```text
Host CV_8UC3 左右图
  -> H2D 原始图像
  -> GPU BGR-HWC 到 RGB-CHW float
  -> feature TensorRT engine
  -> GPU float-to-half + GWC volume CUDA kernel
  -> post TensorRT engine
  -> D2H float disp
  -> Host CV_32FC1 disparity
```

所有上述 H2D、kernel、两个 engine 和 D2H 操作都进入同一个 CUDA stream，最后由 `cudaStreamSynchronize()` 完成同步。`disp` 被复制到 `std::vector<float> m_hDisp`，再复制到 `CV_32FC1` 的 OpenCV 矩阵。

tracker 随后在 CPU 上执行：

```text
depth_m = rectified_fx * rectified_baseline_m / disparity
-> 深度范围过滤
-> 可选置信度过滤
-> 去 padding、恢复分辨率并对齐到原始左图
```

可选 `prob` 输出不会直接给 FoundationPose 使用。若调用方请求，它会 D2H 拷贝后在 CPU 上沿 disparity 维度取最大值，生成 1/4 分辨率置信度图。

### 13.3 Grounding DINO + SAM 首帧分割

实现文件：

```text
sam/src/image_utils.cpp
sam/src/grounding_dino_runner.cpp
sam/src/sam_runner.cpp
sam/src/trt_utils.cpp
```

#### Grounded-SAM 的通用 TensorRT 封装

`grounded_sam::TrtEngine` 为每个 engine 创建独立的 runtime、engine、execution context 和 CUDA stream。

初始化时它遍历 `getNbIOTensors()`，为每个 I/O tensor 保存：名称、shape、数据类型、输入/输出模式、Host 字节向量与 Device 指针。固定 shape 或调用 `setInputShape()` 后，封装会：

```text
Host std::vector<uint8_t> buffer
Device cudaMalloc buffer
context->setTensorAddress(name, device_buffer)
```

每次 `infer()` 都执行：

```text
所有输入 Host buffer -> cudaMemcpyAsync HostToDevice
-> context->enqueueV3(stream)
-> 所有输出 cudaMemcpyAsync DeviceToHost
-> cudaStreamSynchronize(stream)
```

`copyFloatToTensor()` 会按 engine tensor 类型写入 Host buffer：支持 FP32、FP16 与 INT32；`tensorToFloatVector()` 会将 FP16、UINT8、BOOL 或 INT32 输出转为 CPU FP32 vector 供后处理。

#### Grounding DINO 预处理、推理和检测后处理

输入 ROS 图像被转换为 RGB `CV_8UC3`。每个固定-prompt DINO engine 独立执行以下过程：

1. 按 Grounding DINO 规则缩放：短边目标为 800，长边不超过 1333，再受 engine 输入 `target_h/target_w` 限制。
2. 线性插值 resize 到 `resized_h x resized_w`。
3. 右侧和下侧零填充到固定画布。
4. RGB `uint8 HWC` 转为 `float CHW`，范围缩放到 `[0, 1]`，再使用 ImageNet 参数归一化：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

images = (rgb / 255 - mean) / std
shape: [1, 3, target_h, target_w]
```

5. padding mask 的有效区域写 `0`，padding 区域保留 `1`；若 engine 暴露名为 `masks` 的 tensor，则其 shape 被设置为 `[1, target_h, target_w]` 并作为输入绑定。
6. 输入 `images` 和可选 `masks` 经 Host buffer H2D、`enqueueV3` 执行。输出 `pred_logits` 与 `pred_boxes` D2H 回到 Host。
7. 仅支持 batch 1。代码将每个 query 的所有 token logit 取最大值，执行 sigmoid 获得 score，按 `box_threshold` 过滤。
8. `pred_boxes` 按 `[cx, cy, w, h]` 解释为相对原图尺寸坐标，转换为原图 `xyxy` 后裁剪到图像边界；按 score 降序排序并截断到 `max_detections`。

这里没有执行 NMS；若多个 query 高分且重叠，会一起交给 SAM。

#### SAM 预处理、推理和 mask 后处理

当某物体至少有一个 DINO box 时，SAM image encoder 每张图只执行一次：

1. RGB 图按照最长边缩放到 `1024`，短边按比例计算；右侧和下侧零填充至 `1024x1024`。
2. RGB `uint8 HWC` 转为 float CHW，并按 SAM 像素均值方差归一化：

```text
pixel_mean = [123.675, 116.28, 103.53]
pixel_std  = [58.395, 57.12, 57.375]

image = (rgb - pixel_mean) / pixel_std
shape: [1, 3, 1024, 1024]
```

3. encoder 输出 `image_embeddings`，固定输入给 decoder：

```text
image_embeddings: [1, 256, 64, 64]
point_coords:     [1, 2, 2]
point_labels:     [1, 2]
mask_input:       [1, 1, 256, 256]
has_mask_input:   [1]
```

4. 对每个 DINO box，box 四个坐标按相同 resize scale 映射到 SAM 输入空间，作为两个 prompt point：`[x1, y1]` 和 `[x2, y2]`；标签分别为 `2.0`、`3.0`，表示 box 的左上角和右下角。
5. `mask_input` 与 `has_mask_input` 清零，因此当前实现不使用前一轮 mask 作为 decoder 的迭代提示。
6. 每个 box 调用一次 decoder engine。输出 `low_res_masks` 被读取为 `256x256` FP32 logits。
7. logits 线性插值至 `1024x1024`，裁剪掉 resize 后图像外的 padding，再线性插值回原图大小。像素 `logit > 0` 设为 `255`，否则为 `0`。
8. 同一物体的多个 box mask 通过 `cv::bitwise_or` 合并，生成 `CV_8UC1` 二值 PNG。

该 node 在首帧依次为每个物体运行独立的 DINO+SAM pipeline；不同物体目前不会共享 encoder embedding，因而同一帧会重复执行 SAM encoder。

### 13.4 FoundationPose 注册与跟踪

实现文件：

```text
fp/src/foundationpose_cpp/detection_6d_foundationpose/src/foundationpose.cpp
fp/src/foundationpose_cpp/detection_6d_foundationpose/src/foundationpose_render.cpp
fp/src/foundationpose_cpp/detection_6d_foundationpose/src/foundationpose_utils.cu
fp/src/foundationpose_cpp/easy_deploy_tool/inference_core/trt_core/src/trt_core.cpp
```

#### 输入与 GPU 预处理

FoundationPose 接收：

```text
rgb:   CV_8UC3, H x W x 3，来自左图 RGB
depth: CV_32FC1, H x W，单位为米，来自 FFS
mask:  CV_8UC1, H x W，仅 Register 使用
```

`UploadDataToDevice()` 在每次 Register 或 Track 调用时创建临时 GPU buffer：

```text
rgb_on_host   -> cudaMalloc + cudaMemcpy HostToDevice -> uint8 RGB HWC
depth_on_host -> cudaMalloc + cudaMemcpy HostToDevice -> float H x W
depth device tensor -> convert_depth_to_xyz_map CUDA kernel -> float XYZ H x W x 3
```

XYZ kernel 使用相机内参计算：

```text
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
z = depth
```

深度小于 `0.001 m` 的像素不写入有效 XYZ。RGB、depth 和 XYZ GPU buffer 由 pipeline package 的 `shared_ptr` 生命周期管理，调用结束后释放。

#### 注册：采样、渲染、refiner 和 scorer

首次注册调用 `FoundationPose::Register()`：

1. `FoundationPoseSampler::GetHypPoses()` 使用 Device depth 与 Host mask 生成初始假设。当前注册 batch 配置为 `252`。
2. GPU renderer 使用 mesh 顶点、法线、三角面、UV 和纹理；这些 mesh 资源在 renderer 初始化时一次性 H2D 上传。
3. `RenderAndTransform()` 对每个假设姿态将观测 RGB/depth/XYZ 和渲染结果裁剪、变换为两个 GPU FP32 输入 tensor：

```text
render_input: [N, 160, 160, 6]
transf_input: [N, 160, 160, 6]
```

其中 `N` 为假设数量；当前注册阶段为 252，跟踪阶段通常为 1。该代码使用 NHWC 布局，而不是常见 NCHW。

4. refiner 的输入 tensor 标记为 `DEVICE`，所以 `trt_core::PreProcess()` 不会做 H2D 拷贝；TensorRT 直接绑定 renderer 输出的 GPU 地址。
5. refiner engine 输出：

```text
trans: [N, 3]
rot:   [N, 3]
```

默认输出 buffer 保持 Host 位置，因此 `trt_core::PostProcess()` 会 D2H 拷贝。CPU 端将 translation 按 `mesh_diameter / 2` 缩放，将 rotation 经过 `tanh * 0.34906585` 变为 angle-axis，再将增量合成到每个假设姿态。
6. 注册按 `register_refine_iterations` 重复 refiner。
7. scorer 阶段再次渲染所有 refinement 后的假设，输入仍直接保留在 GPU；其 `scores` 输出也标记为 `DEVICE`，不做 D2H。
8. `getMaxScoreIndex()` 用 Thrust 在 GPU `scores` 上取最大值索引，CPU 只取得整数 index；对应 pose 成为注册结果。

#### 跟踪

`Track()` 不使用当前帧 mask，也不执行 sampler 或 scorer：

```text
上一帧 mesh pose
-> 转到 centered-mesh pose
-> RenderAndTransform，N=1
-> refiner engine，重复 track_refine_iterations 次
-> CPU 合成 delta pose
-> 转回 mesh pose
```

这解释了 tracker 为什么仅需要首帧 mask。后续各物体的线程各持有一套 FoundationPose/refiner/scorer 对象；这既保证模型状态独立，也为多物体并行提供了独立 TensorRT execution context。

#### `trt_core` 的 TensorRT 和内存策略

FoundationPose 通过 `CreateTrtInferCore()` 载入 `.engine`：读取 engine 二进制、调用 `initLibNvInferPlugins`、创建 runtime、反序列化 engine，再解析 I/O tensor 名称、类型和最大 shape。

每个 blob buffer 同时分配：

```text
buffer_on_host:   new u_char[]
buffer_on_device: cudaMalloc
```

每次 `SyncInfer()` 可分为三阶段：

```text
PreProcess:
  仅当输入 tensor 当前位于 HOST 时，Host -> Device

Inference:
  绑定所有 device pointer
  设置当前 input shape
  context->enqueueV3(inference_stream)

PostProcess:
  仅当输出 tensor 当前位于 HOST 时，Device -> Host
```

每个 `TrtInferCore` 按线程 ID 缓存一个 execution context。多目标 tracker 的 worker 各使用独立 model/core，因此每个 worker 也有独立 engine context；同一 core 若被多线程调用，也会按线程创建 context。

## 14. 三类模型的数据移动对比

| 模型 | 主要输入 Host 形式 | H2D | GPU 推理和自定义 CUDA | D2H | 主要 CPU 后处理 |
| --- | --- | --- | --- | --- | --- |
| FFS | 左右 `CV_8UC3` BGR HWC | 原始 uint8 左右图 | BGR-to-RGB/CHW、feature engine、GWC volume、post engine | disparity，及可选 probability | disparity 转米制 depth、过滤和对齐 |
| Grounding DINO | RGB `CV_8UC3`，CPU float CHW | `images`，可选 `masks` | DINO engine | logits、boxes | sigmoid、阈值、坐标反变换、排序 |
| SAM | RGB `CV_8UC3`，CPU float CHW；box prompt | encoder image；decoder embedding 和 prompt | encoder engine；每个 box 一次 decoder | embedding、low-res mask logits | resize、裁剪、阈值、mask OR |
| FoundationPose refiner | RGB uint8、depth float、render 输入 | RGB/depth；renderer 输出不复制 | XYZ、采样、nvdiffrast renderer、refiner | `trans`、`rot` | delta pose 合成 |
| FoundationPose scorer | renderer GPU tensor | 无额外 H2D | renderer、scorer、GPU max score | 无 | 使用 GPU 返回的最大索引选择 pose |

## 15. 软件和硬件运行环境安装（交付文档第 6 章补充稿）

本章说明基于智能视觉的位姿确定 APP 在训练开发、嵌入式推理验证和地面半物理仿真平台上的安装与检查方法。以下版本和路径来自 2026-10-10 对当前工程及主机的检查；目标设备序列号、机械安装参数、接线图和最终验收结果仍须在现场配置记录中填写。

### 15.1 软件运行环境安装

#### 15.1.1 Ubuntu 开发环境

当前开发计算机的实测软件基线如下。

| 项目 | 当前环境 | 用途或说明 |
| --- | --- | --- |
| CPU 架构 | `x86_64` | 开发、模型转换、离线验证 |
| 操作系统 | Ubuntu 24.04.4 LTS | 当前开发主机；内核 `7.0.0-28-generic` |
| ROS 2 | Jazzy | C++ 节点、话题通信和 launch 管理 |
| Python | 系统 Python 3.12.3；Conda Python 3.12.13 | 模型导出、离线评估和 BundleSDF 流程 |
| C/C++ 编译器 | GCC/G++ 13.3.0 | 工程统一使用 C++17 |
| CMake | 3.28.3 | FFS/FoundationPose/SAM CMake 构建 |
| OpenCV | 4.11.0（`/usr/local`）；系统包 4.6.0 | 当前二进制存在两个版本混用，交付前须统一 |

开发目录按本工程现状设置为：

```text
/home/hc/weizi/ffs+fp+sam/
├── ffs/                         # Fast Foundation Stereo
├── fp/                          # FoundationPose C++/ROS 2
├── sam/                         # Grounding DINO + SAM TensorRT C++
├── Grounded-Segment-Anything/   # Python 模型及 ONNX 导出源码
├── sim/                         # 双目渲染仿真节点
└── app/                         # APP 启动界面和配置入口
```

基础安装应至少包括 ROS 2 Jazzy、`colcon`、CMake、GCC/G++、CUDA Toolkit、TensorRT 开发包、OpenCV、Eigen3、PCL、Assimp、glog、CVCUDA/NVCV，以及 `package.xml` 中声明的 ROS 2 包。可使用 `rosdep` 从各工作空间根目录解析 ROS 依赖；CUDA、TensorRT、CVCUDA 和相机驱动应使用 NVIDIA/设备厂商与目标系统匹配的软件源安装，不应由 `rosdep` 替代。

开发环境初始化：

```bash
source /opt/ros/jazzy/setup.bash
export CUDA_HOME=/usr/local/cuda
export TENSORRT_ROOT=/usr/src/tensorrt
```

模型训练和导出不属于 C++ APP 的运行时依赖。当前已有独立 Conda 环境 `ffs`、`sam`、`foundationpose` 和 `bundlesdf`；其中实测 `ffs`、`sam` 均使用 Python 3.12.13、PyTorch 2.9.1+cu128、CUDA 12.8 和 ONNX 1.22.0，`sam` 另含 torchvision 0.24.1+cu128。Grounded-SAM 的 Python 依赖以 `Grounded-Segment-Anything/requirements.txt`、GroundingDINO 和 Segment Anything 的 `setup.py` 为准。BundleSDF 位于 `/home/hc/weizi/BundleSDF`，使用其独立 `environment.yml` 安装，不与实时 C++ 运行环境合并。

#### 15.1.2 NVIDIA 训练与推理依赖

当前开发主机已检测到 CUDA 12.8（`nvcc 12.8.93`）和 TensorRT 10.8（`trtexec v100800`）。工程运行时直接使用 TensorRT 10 命名 I/O API，链接 `nvinfer`、`nvonnxparser` 和 `nvinfer_plugin`；FoundationPose 还依赖 CVCUDA/NVCV，FFS 和 nvdiffrast 使用自定义 CUDA kernel。

当前主机的 NVIDIA 内核模块为 `580.173.02`，用户态 NVML 库为 `580.178`，执行 `nvidia-smi` 报告 driver/library version mismatch。因此当前只能确认工具链文件存在，不能确认 GPU 推理环境已通过。交付前应先统一驱动与用户态库并重启，再执行：

```bash
nvidia-smi
nvcc --version
/usr/src/tensorrt/bin/trtexec --version
```

三条命令均成功后，再使用 `trtexec --loadEngine=<engine>` 或对应 C++ 最小样例检查 engine 能否反序列化和推理。TensorRT `.engine` 与 CPU 架构、TensorRT/CUDA 版本、GPU 能力及构建参数相关，不能把 x86_64/RTX 平台生成的 engine 直接复制到 Jetson Thor 使用；FFS、DINO、SAM、refiner 和 scorer engine 均应在目标平台使用对应 ONNX 重新构建并留存构建日志。

当前多目标配置使用的运行文件为：

| 类型 | 配置路径或数量 |
| --- | --- |
| FFS | `feature_runner_fp16_5060.engine`、`post_runner_fp16_5060.engine` |
| Grounding DINO | 每个固定文本提示词一个 engine；当前为 blue/pink 两个 |
| SAM | `sam_image_encoder.engine`、`sam_mask_decoder.engine` |
| FoundationPose | FP16 refiner、FP16 scorer 各一个 |
| 标定文件 | `/home/hc/weizi/ffs+fp+sam/ffs/640.txt` |
| MESH | 每个跟踪目标一个 OBJ/纹理目录 |
| 运行参数 | `fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluepink_example.yaml` |

当前 blue/pink 配置引用的八个 engine 合计约 898.14 MB，已安装的核心可执行文件和共享库合计约 73.87 MiB。仅“编译程序”小于 1 GB；若把全部候选 engine、ONNX、ROS 2/CUDA/TensorRT 动态库和日志计入安装包，则当前工程不能直接证明总包小于 1 GB，应在确定交付口径后用最终文件清单重新统计。

#### 15.1.3 C++ 编译环境

工程采用 CMake、ament/colcon 和 C++17。主要依赖为：

- ROS 2：`rclcpp`、`sensor_msgs`、`geometry_msgs`、`std_msgs`、`cv_bridge`、`image_transport`、`message_filters`、`tf2`、`tf2_eigen`、`pcl_conversions` 和 `depth_image_proc`。
- 原生库：OpenCV、Eigen3、PCL、Assimp、glog 和 pthread。
- NVIDIA：CUDA Runtime、TensorRT、CVCUDA/NVCV；FoundationPose 内嵌 nvdiffrast CUDA rasterizer。
- 可选项：Grounding DINO ONNX Runtime 后端默认关闭，当前主链路使用 TensorRT。

按依赖顺序执行最小构建流程：

```bash
cd /home/hc/weizi/ffs+fp+sam/ffs
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fast_foundation_stereo --cmake-args -DCMAKE_BUILD_TYPE=Release

cd /home/hc/weizi/ffs+fp+sam/fp
source /opt/ros/jazzy/setup.bash
colcon build --packages-select foundationpose_cpp \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DENABLE_TENSORRT=ON

cd /home/hc/weizi/ffs+fp+sam/sam
source /opt/ros/jazzy/setup.bash
./scripts/build_ros2.sh

cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
colcon build --packages-select sim_stereo_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
```

主程序运行前按顺序加载 overlay：

```bash
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash
```

当前源码尚未完全满足“与硬件和软件系统无关”：FFS 和 FoundationPose CMake 将 TensorRT 库目录写为 `/usr/src/tensorrt/targets/x86_64-linux-gnu/lib`，主 launch 还包含 `/home/hc/...` 绝对路径。Grounded-SAM CMake 已同时搜索 x86_64 和 aarch64。部署 Thor 前须将 FFS/FoundationPose 的 TensorRT 路径改为目标机 aarch64 路径或 CMake 参数，并通过 YAML/launch 参数替换主机绝对路径。当前二进制同时链接 OpenCV 4.11 和 4.6，也须统一 OpenCV 来源后全量重编译。

### 15.2 硬件运行环境安装

#### 15.2.1 双目相机安装

可使用 D435i 或自组合双目相机，但本工程不直接绑定相机 SDK，只订阅 ROS 2 图像话题。设备驱动或采集节点必须同步发布：

```text
/left/image_raw
/right/image_raw
```

节点支持 `rgb8`、`bgr8`、`rgba8`、`bgra8`、`mono8` 和 `8UC1`，内部统一转为三通道图像。安装与配置要求如下：

1. 将左右相机刚性固定在支架或二维转台上；标定完成后不得改变焦距、分辨率、左右相机相对姿态或基线。
2. 左右图必须具有相同宽高、时间戳处于同步容差内且帧序一致。当前配置分辨率为 `640x480`、同步队列为 10、最大同步间隔为 0.07 s。
3. 导入双目标定文本，内容包括左右 `K`、左右畸变、右目到左目的旋转和平移，平移单位为米。当前 `ffs/640.txt` 记录基线约 0.0797661 m；该文件只能用于对应的实际相机和分辨率。
4. 检查图像坐标方向、左右目顺序、曝光和运动模糊。左右目颠倒、标定单位错误或图像已被二次矫正都会直接导致 FFS 深度错误。
5. D435i 设备序列号、USB 端口、实际使用的左右光学流、曝光参数、安装位姿和线缆固定方式：待补充。

连接后至少执行以下检查：

```bash
ros2 topic list | grep -E '/left/image_raw|/right/image_raw'
ros2 topic hz /left/image_raw
ros2 topic hz /right/image_raw
ros2 topic echo --once /left/image_raw
```

#### 15.2.2 NVIDIA Jetson Thor Developer Kit 安装

Jetson Thor 为 ARM+aarch64 目标平台，应使用 Thor 官方支持的 JetPack/Jetson Linux、CUDA、TensorRT、CVCUDA 和容器运行时组合，不能照搬当前 x86_64 开发主机的软件包。工程 `sam/README.md` 记录过 Thor + TensorRT 10.16.2 的 Grounding DINO 静态 batch 验证，但这只是代码库记录，不等同于本次目标板已验收；最终 JetPack、L4T、驱动、CUDA 和 TensorRT 版本须以设备实测记录为准。

安装步骤：

1. 固定开发套件并接好额定电源、主动散热、网络、相机和调试显示/串口；记录板卡型号、序列号、存储容量和电源模式。
2. 刷写 NVIDIA 官方支持的 Thor 系统镜像，安装 CUDA、TensorRT、CVCUDA、ROS 2 与编译依赖，确认 `uname -m` 返回 `aarch64`。
3. 修改 FFS/FoundationPose 的 x86_64 TensorRT 搜索路径并在板端重新构建所有 C++ ROS 2 包。
4. 在板端从 ONNX 重建所有 TensorRT engine。Grounding DINO 保持静态 batch 1、固定 prompt 和固定位置编码；代码库明确记录动态 batch engine 在该平台可能产生错误 logits。
5. 依次运行 `nvidia-smi` 或 Jetson 对应状态工具、`nvcc --version`、`trtexec --version`、`tegrastats`，再进行单模型和整链路测试。
6. 在持续运行条件下记录功耗模式、GPU/CPU 频率、温度、内存峰值、注册耗时、跟踪帧率及是否发生降频。

供电规格、散热器/风扇型号、安装结构、JetPack/Jetson Linux 版本、板卡序列号和性能检查结果：待补充。

#### 15.2.3 地面仿真设备安装与连接

软件仓库提供 `sim_stereo_cpp`，可加载 MESH、标定和目标运动参数，渲染左右图并发布 ROS 2 双目话题，用于在真实六自由度平台接入前验证视觉链路。启动方式为：

```bash
cd /home/hc/weizi/ffs+fp+sam/sim
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch sim_stereo_cpp stereo_render_multi_cpp.launch.py
```

半物理平台安装时，将卫星模型刚性安装在六自由度运动平台，将双目相机和二维转台安装在全向无人车上，并分别完成机械限位、急停、供电、保护接地、网络和通信检查。建议按“六自由度平台单机点动 → 转台单机点动 → 车辆单机运动 → 相机静态采集 → 联动低速测试”的顺序验收，并确认全部运动包络无机械干涉。

当前仓库只实现仿真双目图像发布和视觉位姿 `geometry_msgs/msg/PoseStamped` 输出，未发现二维转台驱动、控制律、控制消息或控制话题实现。因此运动平台、转台和车辆的通信协议、接线图、坐标变换、角度/速度限幅、急停逻辑以及由位姿生成转台指令的节点均为待补充项，不能在现阶段标记为“控制指令输出检查通过”。

#### 15.2.4 APP 部署与运行检查

部署包至少应包含以下内容，并在清单中记录 SHA-256 或版本号：

```text
install/ 或 ros2_install/    # APP 可执行文件、共享库、launch 和 package metadata
config/*.yaml                # 多目标、话题、阈值和模型路径
*.engine                     # FFS、DINO、SAM、refiner、scorer
mesh/                        # OBJ、纹理和材质文件
caminfo.txt                  # 双目标定
launch/*.launch.py           # 启动入口
logs/                        # 运行和验收日志目录
```

当前多目标整链路启动命令为：

```bash
source /opt/ros/jazzy/setup.bash
source /home/hc/weizi/ffs+fp+sam/fp/install/setup.bash
source /home/hc/weizi/ffs+fp+sam/sam/ros2_install/setup.bash

ros2 launch foundationpose_cpp foundationpose_stereo_tracker_multi.launch.py \
  params_file:=/home/hc/weizi/ffs+fp+sam/fp/src/foundationpose_cpp/config/foundationpose_stereo_tracker_multi_bluepink_example.yaml
```

必须显式传入 `params_file`：主 launch 当前默认值是 blue/gold 配置，而本章引用的是 blue/pink 配置。启动后按以下顺序检查并记录结果：

1. **静态文件检查**：所有 engine、MESH、纹理、标定和输出目录存在且具有读取/写入权限；YAML 数组中目标名称、DINO engine、mask 名称和 MESH 数量一一对应。
2. **双目输入检查**：`/left/image_raw` 和 `/right/image_raw` 持续发布，相同分辨率，时间差不超过配置值；保存一组原始左右图用于追溯。
3. **深度检查**：FFS engine 成功反序列化，视差非空；由 `fx * baseline / disparity` 得到米制深度，并满足当前 `0.1~100.0 m` 范围。用已知距离平面检查尺度误差。
4. **首帧 mask 检查**：Grounded-SAM 在首帧为每个目标生成非空 PNG，并可发布 `/first_masks/<object>`；核对 mask 与左图尺寸、目标轮廓和文件名。
5. **MESH/注册检查**：每个 OBJ、材质和纹理成功加载，FoundationPose 完成 252 个初始假设、refiner 和 scorer 注册，无 CUDA/TensorRT 错误。
6. **跟踪输出检查**：主目标发布 `/foundationpose/pose`，每个目标发布 `/foundationpose/pose/<sanitized_name>`；检查 `frame_id=camera_color_optical_frame`、时间戳、位置单位为米、四元数归一化且连续。
7. **可视化和性能检查**：检查 `/foundationpose/multi_visualization` 或显示窗口，记录注册耗时、稳定跟踪帧率、GPU/CPU/内存/温度、连续运行时长和丢帧数。
8. **转台控制检查**：待转台控制节点、消息类型和通信接口实现后，验证像素/位姿到方位俯仰指令的坐标变换、方向、零位、限幅、频率、超时归零和急停。当前版本不执行此项。

异常日志至少应包括时间、节点名、配置文件、模型文件哈希、输入帧号、CUDA/TensorRT 错误、是否自动恢复和处置结论。禁止在 GPU 检查失败、engine 版本不匹配、标定不对应或 mask/MESH 对象映射错误时继续进行系统级验收。

### 15.3 环境配置记录

安装完成后按下表归档；“当前检查值”仅用于建立基线，目标交付平台必须重新填写。

| 配置项 | 当前检查值 | 最终记录/结论 |
| --- | --- | --- |
| 检查日期 | 2026-10-10 | 待补充 |
| 开发主机 OS/架构 | Ubuntu 24.04.4 LTS / x86_64 | 待确认是否作为正式开发基线 |
| 目标板 OS/架构 | Jetson Thor / aarch64 | JetPack、L4T 版本待补充 |
| NVIDIA 驱动 | 内核 580.173.02；NVML 580.178 | 当前不一致，修复后填写通过结论 |
| CUDA | 12.8.93 | 目标板待补充 |
| TensorRT | 当前开发主机 10.8；代码库记录 Thor 10.16.2 | 目标板实测待补充 |
| ROS 2 | Jazzy | 目标板待补充 |
| GCC/G++、CMake | 13.3.0、3.28.3 | 目标板待补充 |
| OpenCV | 4.11 与 4.6 混用 | 统一版本并重编译后填写 |
| Python/PyTorch | Python 3.12.13、PyTorch 2.9.1+cu128 | 仅训练/导出环境 |
| APP 源码版本 | 当前工作区存在未提交修改 | 提交 ID、分支和修改状态待补充 |
| 模型版本 | YAML 指向的八个 engine | ONNX/engine 哈希和构建命令待补充 |
| 相机 | D435i 或自组合双目 | 型号、序列号、固件和连接端口待补充 |
| 双目标定 | `ffs/640.txt`，640x480，基线约 0.0797661 m | 标定日期、设备对应关系和误差待补充 |
| MESH | YAML 中 blue/pink OBJ 路径 | 目标编号、尺度、坐标系和文件哈希待补充 |
| Thor 开发套件 | 未在本机读取 | 板卡型号、序列号、存储和功耗模式待补充 |
| 六自由度平台 | 未接入代码检查 | 型号、通信接口、地址和限位待补充 |
| 二维转台 | 当前无控制节点 | 型号、协议、零位、限幅和验收结果待补充 |
| 全向无人车 | 当前无控制节点 | 型号、协议、地址和验收结果待补充 |
| APP 大小 | 核心二进制约 73.87 MiB；所选 engine 约 856.53 MiB | 按最终交付口径重新统计 |
| 安装检查结论 | 工程文件和工具链已检查，GPU运行未通过 | 修复驱动、统一 OpenCV、完成 ARM 构建后复验 |

建议随配置记录一并保存 `uname -a`、`nvidia-smi`、`nvcc --version`、`trtexec --version`、`ros2 doctor --report`、`ldd`、最终 YAML、engine 构建日志、相机标定报告、线缆/安装照片和整链路测试日志，以保证环境可复现、设备可追溯。
