import os
import subprocess
import sys
from pathlib import Path

def should_skip():
    """检查用户是否想跳过当前阶段"""
    print("\n按 'q' 跳过当前阶段，其他键继续...")
    choice = input().strip().lower()
    return choice == 'q'

def run_track_ds(video_path):
    """预训练模型效果预览"""
    print("预训练模型效果预览")
    if should_skip():
        print("已跳过预训练模型效果预览")
        return True  # 返回True表示需要改进
    result = subprocess.run([sys.executable, "ultralytics-main/bzd_code/track--ds.py", video_path])

    if result.returncode != 0:
        print("预训练模型效果预览失败")
        return False
    print("\n 预训练模型效果是否满足预期？")
    print("1. 满足 (退出流程)")
    print("2. 不满足 (继续流程)")
    choice = input("请输入选择 (1/2): ").strip().lower()
    if choice == 'q':
        return True  # 跳过，进入下一阶段
    return choice == "2"

def run_labeling(video_path, max_frames=10):
    """手动标注"""
    print("请手动标注5帧...")
    if should_skip():
        print("已跳过手动标注")
        return True  # 返回True表示成功
    result = subprocess.run([sys.executable, "ultralytics-main/bzd_code/label.py", video_path])
    if result.returncode != 0:
        print("手动标注出错")
        return False
    return True

def run_training():
    """运行训练脚本"""
    print("开始模型训练...")
    if should_skip():
        print("已跳过训练阶段")
        return True  # 返回True表示成功
    result = subprocess.run([sys.executable, "ultralytics-main/bzd_code/train.py"])
    if result.returncode != 0:
        print("模型训练失败")
        return False
    return True

def run_pretrack_ds(video_path):
    """开始半自动的标注"""
    print("开始半自动的标注...")
    if should_skip():
        print("已跳过半自动的标注")
        return (False, None)  # 修改返回值格式
    
    result = subprocess.run([sys.executable, "ultralytics-main/bzd_code/pretrack_ds_bzd.py", video_path])
    if result.returncode != 0:
        print("优化跟踪脚本运行失败")
        return (False, None)  # 修改返回值格式
    
    
    print("\n pretrack是否满足预期？")

    print("1. 满足 (进入展示阶段)")
    print("2. 不满足 (选择返回标注或训练阶段)")
    choice = input("请输入选择 (1/2): ").strip().lower()
    if choice == 'q':
        return (False, None)  # 修改返回值格式
    

    if choice == '2':
        print("\n请选择返回的阶段：")
        print("a. 返回标注阶段")
        print("b. 返回训练阶段")
        stage_choice = input("请输入选择 (a/b): ").strip().lower()
        while stage_choice not in ['a', 'b']:
            print("无效选择，请重新输入。")
            stage_choice = input("请输入选择 (a/b): ").strip().lower()
        return (True, stage_choice)
    
    return (False, None)  # 满足预期，进入展示阶段

def main():
    video_path = "./"  # 默认视频路径
    #兴华一路海滨中学对面_2025-07-03_10-49-25_300s.mp4
    # 第一阶段：初始跟踪
    need_improvement = run_track_ds(video_path)
    
    # 如果是第一次不满足预期，进入标注+训练流程
    if need_improvement:
        # 第二阶段：标注
        if not run_labeling(video_path):
            return
        # 第三阶段：训练
        if not run_training():
            return
    
    # 第四阶段：优化后跟踪
    pretrack_result = run_pretrack_ds(video_path)
    
    # 处理返回结果
    if pretrack_result[0]:  # 需要返回标注或训练阶段
        stage_choice = pretrack_result[1]
        if stage_choice == 'a':
            print("\n返回标注阶段...")
            if not run_labeling(video_path):
                return
            # 标注后继续训练
            if not run_training():
                return
            # 重新运行优化跟踪
            pretrack_result = run_pretrack_ds(video_path)
        elif stage_choice == 'b':
            print("\n返回训练阶段...")
            if not run_training():
                return
            # 训练后重新运行优化跟踪
            pretrack_result = run_pretrack_ds(video_path)
    
    # 无论是否跳过第四阶段，都进入第五阶段展示
    if not run_show(video_path):
        return
    
    print("流程结束")

def run_show(video_path):
    """运行展示脚本"""
    print("\n请选择展示模式：")
    print("1. 预测模式 (show.py)")
    print("2. 追踪模式 (show_track.py)")
    print("3. 违停车辆预警(region_count.py)")
    choice = input("请输入选择 : ").strip().lower()
    
    if choice == '1':
        script = "ultralytics-main/bzd_code/show.py"
    elif choice == '2':
        script = "ultralytics-main/bzd_code/show_track.py"
    elif choice == '3':
        script = "ultralytics-main/bzd_code/region_count.py"
    else:
        print("无效选择，默认使用预测模式")
        script = "ultralytics-main/bzd_code/show.py"
    
    print(f"运行{script}...")
    result = subprocess.run([sys.executable, script, video_path])
    if result.returncode != 0: 
        print("展示脚本运行失败")
        return False
    print("展示阶段完成")
    return True

if __name__ == "__main__":
    main()