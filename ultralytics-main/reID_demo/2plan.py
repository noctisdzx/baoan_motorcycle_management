import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pygame
from pygame.locals import *
from pathlib import Path

# 全局变量
ref_points = []  # 存储参考矩形的四个点
perspective_matrix = None  # 透视变换矩阵
model = YOLO(input(f"请输入训练集保存路径 (默认: {'runs/detect/train17/weights/best.pt'}): ").strip() or 'runs/detect/train17/weights/best.pt')  # 加载YOLO模型
tracker = DeepSort(max_age=20)  # DeepSORT追踪器
track_history = {}  # 存储轨迹历史

def get_perspective_transform(frame):
    """获取透视变换矩阵"""
    global ref_points, perspective_matrix
    
    # 假设参考矩形在实际地面上的尺寸是2m x 2m
    dst_points = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
    
    if len(ref_points) == 4:
        src_points = np.float32(ref_points)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print("透视变换矩阵计算完成")
        return True
    return False

def create_plan_view(frame, detections=None):
    """创建抽象平面图 - 显示追踪ID"""
    global track_history
    
    if perspective_matrix is None:
        return None
    
    # 创建黑色背景画布
    plan_view = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 添加淡灰色网格线
    grid_color = (50, 50, 50)
    for i in range(0, 200, 10):
        cv2.line(plan_view, (i, 0), (i, 200), grid_color, 1)
        cv2.line(plan_view, (0, i), (200, i), grid_color, 1)
    
    current_points = {}  # 存储当前帧的红点位置
    
    if detections:
        # 准备DeepSORT需要的检测结果格式
        bbs = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bbs.append(([x1, y1, x2-x1, y2-y1], conf, str(cls)))
        
        # 使用DeepSORT更新追踪
        tracks = tracker.update_tracks(bbs, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # 计算底边中点并映射
            bottom_center = ((x1 + x2) // 2, y2)
            point = np.array([[[bottom_center[0], bottom_center[1]]]], dtype=np.float32)
            mapped_point = cv2.perspectiveTransform(point, perspective_matrix)[0][0]
            mapped_point = tuple(map(int, mapped_point))
            
            # 存储轨迹历史
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(mapped_point)
            
            # 绘制红点
            cv2.circle(plan_view, mapped_point, 5, (0, 0, 255), -1)
            cv2.putText(plan_view, str(track_id), 
                       (mapped_point[0]+10, mapped_point[1]+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 绘制轨迹箭头(至少需要2个点)
            if len(track_history[track_id]) > 1:
                prev_point = track_history[track_id][-2]
                dx = mapped_point[0] - prev_point[0]
                dy = mapped_point[1] - prev_point[1]
                
                if (dx**2 + dy**2) > 4:  # 移动距离阈值
                    end_point = (mapped_point[0] + dx*3, mapped_point[1] + dy*3)  # 延长向量
                    cv2.arrowedLine(plan_view, mapped_point, end_point, 
                                  (0, 255, 255), 2, tipLength=0.3)
    
    # 更新上一帧点位置
    prev_points = current_points
    
    # 不再在此处缩放图像
    return plan_view

def cv2_to_pygame(img):
    """将OpenCV图像转换为Pygame图像"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")

def main():
    global perspective_matrix, ref_points
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('simhei', 24)

    cap = cv2.VideoCapture(input(f"请输入训练集保存路径 (默认: {'视频/try.mp4'}): ").strip() or '视频/try.mp4')
    if not cap.read()[0]:
        print("无法读取视频")
        return

    # 选择参考矩形
    ret, first_frame = cap.read()
    selecting_rect = True
    while selecting_rect:
        screen.fill((0, 0, 0))
        frame_surface = cv2_to_pygame(first_frame)
        screen.blit(frame_surface, (0, 0))
        screen.blit(font.render("请点击四个点选择参考矩形，按空格确认", True, (255, 255, 255)), (20, 20))

        # 绘制已选点
        for i, pt in enumerate(ref_points):
            pygame.draw.circle(screen, (255, 0, 0), pt, 5)
            text = font.render(str(i + 1), True, (255, 0, 0))
            screen.blit(text, (pt[0] + 10, pt[1] + 10))
            if i > 0:
                pygame.draw.line(screen, (0, 255, 0), ref_points[i - 1], pt, 2)
        if len(ref_points) == 4:
            pygame.draw.line(screen, (0, 255, 0), ref_points[3], ref_points[0], 2)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN and len(ref_points) < 4:
                ref_points.append(event.pos)
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if get_perspective_transform(first_frame):
                        selecting_rect = False
                elif event.key == K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    return

    # 处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 先进行目标检测与追踪
        results = model(frame)
        detections = [[int(box.xyxy[0][i]) for i in range(4)] + [float(box.conf[0]), int(box.cls[0])] 
                      for result in results for box in result.boxes]
        for x1, y1, x2, y2, conf, cls in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[cls]}:{conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracks = tracker.update_tracks([([x1, y1, x2-x1, y2-y1], conf, str(cls)) 
                                       for x1, y1, x2, y2, conf, cls in detections], frame=frame)
        for track in tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.putText(frame, f'ID:{track.track_id}', (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 在检测完成后绘制参考矩形
        if len(ref_points) == 4:
            # 绘制半透明边框
            for i in range(4):
                cv2.line(frame, ref_points[i], ref_points[(i+1)%4], 
                        (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str(i+1), 
                           (ref_points[i][0]+10, ref_points[i][1]+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        plan_view = create_plan_view(frame, detections)

        if plan_view is not None:
            # 计算平面视图的合适高度(与原视频高度一致)
            frame_height = frame.shape[0]
            plan_view = cv2.resize(plan_view, 
                                 (int(plan_view.shape[1] * frame_height / plan_view.shape[0]), 
                                  frame_height),
                                 interpolation=cv2.INTER_LINEAR)
            
            # 水平拼接
            combined = cv2.hconcat([frame, plan_view])
            
            # 计算缩放比例(最长边1280像素)
            scale = 1280 / max(combined.shape[1], combined.shape[0])
            combined = cv2.resize(combined, 
                                (int(combined.shape[1] * scale), 
                                 int(combined.shape[0] * scale)),
                                interpolation=cv2.INTER_LINEAR)
        else:
            combined = frame

        # 转换为Pygame格式并显示
        screen.fill((0, 0, 0))
        screen.blit(cv2_to_pygame(combined), 
                   ((1280 - combined.shape[1]) // 2,  # 水平居中
                    (720 - combined.shape[0]) // 2))  # 垂直居中
        
        # 显示标签(调整位置)
        screen.blit(font.render("原视频视图", True, (255, 255, 255)), 
                   (20, 20))
        if plan_view is not None:
            screen.blit(font.render("平面投影视图", True, (255, 255, 255)), 
                       (frame.shape[1] * scale + 20, 20))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type in [QUIT, KEYDOWN] and (event.type == KEYDOWN and event.key == K_q):
                cap.release()
                pygame.quit()
                return

        clock.tick(30)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()