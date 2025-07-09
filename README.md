# baoan_motorcycle_management

代码文件在ultralytics-main\bzd_code这个位置

其中all是主要程序，半自动标注等功能都在其中

datasets文件夹是数据集的保存位置

output_result文件夹是输出的预测视频的位置

pretrain_model文件夹是预训练模型的位置

加入了违停预测的功能（region_count.py）

reID功能还在适配中

# 主要流程是：

1、使用预训练模型预测：判断是否需要半自动标注进一步优化模型

2、开始半自动标注：首先手动标注3-5帧，而后流程会自动标注其余帧的物体，可以依据需要修改半自动标注的数据集结果

3、使用训练得到的模型，完成违停检测、摩的载人检测等功能

注：

join_data.py文件用于整合数据集，以实现模型的自我迭代优化（还未整合到流程中）

提示框弹出输入绝对路径的提示时需要输入绝对路径

# 环境依赖

env_file.yaml是环境依赖文件，使用Anaconda Navigator可以一键导入对应的依赖（适用于windows）
