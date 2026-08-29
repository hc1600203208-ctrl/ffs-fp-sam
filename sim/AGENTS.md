在/home/hc/weizi/ffs+fp+sam/sim路径下完成以下任务：对指定的mesh三维模型，以/home/hc/weizi/dataset/jrnew-blue/mesh1为例，构建一个实时渲染双目图像，并将图像通过ros2发布出来的工程；
具体完成以下功能：
以相机光心为主坐标系，
每次通过键盘输入给定物体相对相机的初始位姿，和六自由度的各六个的循环速度，比如x：0.1m,y：0.2m z：0.3m 代表相对初始位置做振幅x方向为0.1m，y方向为0.2m，z方向为0.3m的往复运动，并实时发布图像。
双目相机参数参考/home/hc/weizi/ffs+fp+sam/ffs/640.txt
使用python语言，使用conda现有的虚拟环境，conda安装位置~/anaconda3，环境名为sam，缺包直接帮我安装，但要整理好一共安装了哪些新包
