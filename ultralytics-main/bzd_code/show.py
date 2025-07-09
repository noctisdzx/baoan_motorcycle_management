import cv2
import pygame
import os
from datetime import datetime
from ultralytics import YOLO

# 初始化模型
default_path = 'runs/detect/train6/weights/best.pt'
model = YOLO(input(f"请输入模型路径 (默认: {default_path}): ").strip() or default_path)

# 视频路径
default_pathA = 'cut_results/cut_新安二-前进一路_2025-06-18_18-04-20_300s.mp4'
video_path = input(f"请输入video路径 (默认: {default_pathA}): ").strip() or default_pathA
cap = cv2.VideoCapture(video_path)

# 初始化pygame
pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("output")


# 创建输出目录
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# 获取当前时间作为文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_video = os.path.join(output_dir, f"results_{current_time}.mp4")

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (screen_width, screen_height))

# 添加控制变量
show_labels = True  # 默认显示标签

# 主循环
running = True
while cap.isOpened() and running:
    # 处理pygame事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:  # 按T键切换标签显示
                show_labels = not show_labels
            elif event.key == pygame.K_q:  # 新增：按Q键退出
                running = False
    
    # 读取视频帧
    success, frame = cap.read()
    if not success:
        break
    
    # YOLO推理
    results = model.track(frame,imgsz=1280,conf=0.1,iou=0.4,show_conf=False,device='0')
    
    # 可视化结果
    annotated_frame = results[0].plot(labels=show_labels,conf=False)  # 根据show_labels控制标签显示
    
    # 计算物体数量
    motorcycle_count = 0
    bicycle_count = 0
    for box in results[0].boxes:
        if results[0].names[int(box.cls)] == "motorcycle":
            motorcycle_count += 1
        elif results[0].names[int(box.cls)] == "bicycle":
            bicycle_count += 1
    
    # 将OpenCV图像转换为Pygame表面
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    annotated_frame = cv2.resize(annotated_frame, (screen_width, screen_height))
    
    # 写入视频帧
    out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    
    pygame_frame = pygame.surfarray.make_surface(annotated_frame.swapaxes(0, 1))
    
    # 显示图像
    screen.blit(pygame_frame, (0, 0))
    
    # 在屏幕最上方中央显示物体数量（红色字体）
    font = pygame.font.SysFont('Arial', 30)
    count_text = font.render(f"Motorcycles: {motorcycle_count}, Bicycles: {bicycle_count}", True, (255, 0, 0))
    text_rect = count_text.get_rect(center=(screen_width//2, 20))
    screen.blit(count_text, text_rect)
    
    pygame.display.flip()

# 释放资源
cap.release()
out.release()  # 关闭视频写入器
pygame.quit()