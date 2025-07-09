import cv2
import os
from datetime import timedelta

def time_to_seconds(time_str):
    """将HH:MM:SS格式的时间转换为秒数"""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def cut_video(input_path, output_path, start_time, end_time):
    """裁剪视频并保存指定片段"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算开始和结束帧
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 定位到开始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 读取并写入指定范围内的帧
    current_frame = start_frame
    while current_frame <= end_frame and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
    
    cap.release()
    out.release()
    return True

def main():
    # 输入视频路径
    input_path = input("请输入视频文件路径: ").strip()
    if not os.path.exists(input_path):
        print("文件不存在")
        return
    
    # 输入开始和结束时间
    start_time = input("请输入开始时间(格式 HH:MM:SS): ").strip()
    end_time = input("请输入结束时间(格式 HH:MM:SS): ").strip()
    
    try:
        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time)
    except:
        print("时间格式错误，请使用HH:MM:SS格式")
        return
    
    if start_sec >= end_sec:
        print("结束时间必须大于开始时间")
        return
    
    # 设置输出路径
    output_dir = "cut_results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"cut_{filename}")
    
    # 执行裁剪
    print(f"正在裁剪视频: {start_time} 到 {end_time}...")
    if cut_video(input_path, output_path, start_sec, end_sec):
        print(f"视频裁剪完成，保存为: {output_path}")
    else:
        print("视频裁剪失败")

if __name__ == "__main__":
    main()
