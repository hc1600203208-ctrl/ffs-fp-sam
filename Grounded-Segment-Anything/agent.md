查看/home/hc/weizi/ffs+fp+sam/Grounded-Segment-Anything工程目录，当前我已经在本地conda环境中配置好了该项目的python运行环境，我的conda安装位置~/anaconda3,我的虚拟环境名为sam，并跑通了运行命令：python patrick_sam.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --input_folder /home/hc/weizi/dataset/jrnew-blue/rgb \
  --output_dir /home/hc/weizi/dataset/jrnew-blue/mask \
  --text_prompt "blue carton" \
  --device cuda
  实现了对rgb图像进行grounding dino+sam的物体mask图获取，但是我现在想通过c++和tensorrt进行对该项目进行部署，实现和上面命令同样的功能，请在新目录/home/hc/weizi/ffs+fp+sam/sam下完成c++版本的部署，应该会涉及到onnx模型的导出和engine文件的生成，请先进行计划，然后再开始写代码

  注意事项：不要修改原项目的任何代码