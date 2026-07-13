当前是一个基于c++和ros2部署的fast foundation stereo的工程代码，我当前运行ros2 launch fast_foundation_stereo dnn_stereo_fast_depth.launch.py脚本，能正常发布深度图话题，但深度值完全不准确；同时在另一个工程下，我使用官方提供的推理脚本进行单帧推理则效果很好，另一个工程目录在/home/hc/weizi/Fast-FoundationStereo下，conda环境名为ffs，conda安装位置为~/anaconda3,我运行python scripts/run_demo_tensorrt.py --onnx_dir /home/hc/model/ffs/20-30-48/ --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir output/ --remove_invisible 0 --denoise_cloud 1  --get_pc 1 --zfar 100。二者采用同样的engine模型文件，请分析导致二者原因，先使用官方的测试双目图像进行测试，测试图像在/home/hc/weizi/ffs+fp+sam/ffs/demo_data，避免其他干扰

查看当前工程下的ffs目录，这是一个基于c++并集成ros2部署的fast foundation stereo的工程代码，在原有模型基础上我加入了深度图置信度估计来对深度图结果进行后处理，现在需要对我的改进进行实验分析以验证有效性，当前的ros2 launch fast_foundation_stereo dnn_stereo_depth.launch.py脚本已经实现了对ros2图像话题进行实时深度估计和深度置信度后处理，现在需要你单独编写一个离线节点对离线双目图像数据作相同的处理，将源码放在/home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/src/benchmark下，launch启动文件放在/home/hc/weizi/ffs+fp+sam/ffs/src/ffs_pkg/launch，并且按以下分组生成不同的深度结果保存下来
A. raw_depth
   - 原始 Fast-FoundationStereo 深度图
   - 不做过滤

B. median_filter
   - 中值滤波 baseline

C. bilateral_filter
   - 双边滤波 baseline

D. confidence_photo_only
   - 只使用光度重投影置信度 C_photo

E. confidence_tex_grad
   - 使用 C_tex + C_grad

F. confidence_photo_tex_grad
   - 使用 C_photo + C_tex + C_grad

G. confidence_full
   - 使用完整置信度：
     C_photo + C_tex + C_grad + C_tmp
   - 如果当前没有 C_tmp，则退化为 C_photo + C_tex + C_grad

H. confidence_full_threshold_sweep
   - threshold = [0.20, 0.30, 0.35, 0.40, 0.50]
我的离线数据存放在：左相机图像在/home/hc/weizi/dataset/jrnew-blue/rgb文件夹中，右相机图像在/home/hc/weizi/dataset/jrnew-blue/camera2文件夹中，相机标定参数在/home/hc/weizi/dataset/jrnew-blue/caminfo.txt文件中