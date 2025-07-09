import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import ReIDExtractor, cosine_similarity
import pygame
from pygame.locals import *
from pathlib import Path
from collections import deque
import time

# 全局变量
ref_points_cam1 = []  # 存储摄像头1参考矩形的四个点
ref_points_cam2 = []  # 存储摄像头2参考矩形的四个点
perspective_matrix_cam1 = None  # 摄像头1透视变换矩阵
perspective_matrix_cam2 = None  # 摄像头2透视变换矩阵
model = YOLO(input(f"请输入训练集保存路径 (默认: {'runs/detect/train17/weights/best.pt'}): ").strip() or 'runs/detect/train17/weights/best.pt')  # 加载YOLO模型
tracker1 = DeepSort(max_age=20)  # 摄像头1 DeepSORT追踪器
tracker2 = DeepSort(max_age=20)  # 摄像头2 DeepSORT追踪器
track_history_cam1 = {}  # 存储摄像头1轨迹历史
track_history_cam2 = {}  # 存储摄像头2轨迹历史
reid = ReIDExtractor("reID_demo/reID_model/osnet_x0_25.pth")  # 特征提取器

class FeatureCache:
    def __init__(self, max_size=5):
        self.cache = {}  # {track_id: deque([{'feature':..., 'position':..., 'time':...}])}
        self.max_size = max_size

    def add(self, track_id, feature, position):
        if track_id not in self.cache:
            self.cache[track_id] = deque(maxlen=self.max_size)

        entry = {
            'feature': feature,
            'position': position,
            'time': time.time()
        }
        self.cache[track_id].append(entry)

    def get_velocity(self, track_id):
        """计算平均速度(m/s)"""
        entries = list(self.cache.get(track_id, []))
        if len(entries) < 2:
            return (0, 0)

        velocities = []
        for i in range(1, len(entries)):
            dt = entries[i]['time'] - entries[i-1]['time']
            if dt > 0:
                dx = entries[i]['position'][0] - entries[i-1]['position'][0]
                dy = entries[i]['position'][1] - entries[i-1]['position'][1]
                velocities.append((dx/dt, dy/dt))

        return np.mean(velocities, axis=0) if velocities else (0, 0)

feature_cache_cam1 = FeatureCache()
feature_cache_cam2 = FeatureCache()

def get_perspective_transform(frame, ref_points):
    """获取透视变换矩阵"""
    # 假设参考矩形在实际地面上的尺寸是2m x 2m
    dst_points = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])

    if len(ref_points) == 4:
        src_points = np.float32(ref_points)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print("透视变换矩阵计算完成")
        return perspective_matrix
    return None

def create_plan_view(frame, detections, perspective_matrix, track_history):
    """创建抽象平面图 - 显示追踪ID"""
    if perspective_matrix is None:
        return None

    # 创建黑色背景画布
    plan_view = np.zeros((200, 200, 3), dtype=np.uint8)

    # 添加淡灰色网格线
    grid_color = (50, 50, 50)
    for i in range(0, 200, 10):
        cv2.line(plan_view, (i, 0), (i, 200), grid_color, 1)
        cv2.line(plan_view, (0, i), (200, i), grid_color, 1)

    if detections:
        # 准备DeepSORT需要的检测结果格式
        bbs = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bbs.append(([x1, y1, x2-x1, y2-y1], conf, str(cls)))

        # 使用DeepSORT更新追踪
        tracks = tracker1.update_tracks(bbs, frame=frame)

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

    return plan_view

def multi_dim_match(feat2, pos2, vel2, cache, perspective_matrix1, perspective_matrix2):
    best_match = None
    max_score = 0

    for track_id, entries in cache.items():
        if not entries:
            continue

        # 1. 外观特征相似度 (40%权重)
        feat_sims = [cosine_similarity(e['feature'], feat2) for e in entries]
        feat_score = np.mean(feat_sims) * 0.4

        # 2. 位置相似度 (30%权重)
        # 将cam1的位置映射到cam2的坐标系
        last_pos = entries[-1]['position']
        pos_dist = np.sqrt((last_pos[0]-pos2[0])**2 + (last_pos[1]-pos2[1])**2)
        pos_score = np.exp(-pos_dist/5.0) * 0.3  # 5米衰减系数

        # 3. 速度相似度 (30%权重)
        vel1 = cache.get_velocity(track_id)
        vel_diff = np.sqrt((vel1[0]-vel2[0])**2 + (vel1[1]-vel2[1])**2)
        vel_score = np.exp(-vel_diff/2.0) * 0.3  # 2m/s衰减系数

        total_score = feat_score + pos_score + vel_score
        if total_score > max_score:
            max_score = total_score
            best_match = track_id

    return best_match if max_score > 0.5 else None

def select_reference_rect(frame, screen, font):
    """选择参考矩形"""
    ref_points = []
    selecting_rect = True
    while selecting_rect:
        screen.fill((0, 0, 0))
        frame_surface = cv2_to_pygame(frame)
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
                return None
            elif event.type == MOUSEBUTTONDOWN and len(ref_points) < 4:
                ref_points.append(event.pos)
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if len(ref_points) == 4:
                        selecting_rect = False
                elif event.key == K_ESCAPE:
                    return None
    return ref_points

def cv2_to_pygame(img):
    """将OpenCV图像转换为Pygame图像"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")

def main():
    global perspective_matrix_cam1, perspective_matrix_cam2
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('simhei', 24)

    cap1 = cv2.VideoCapture(input(f"请输入第一个视频路径 (默认: {'视频/try1.mp4'}): ").strip() or '视频/try1.mp4')
    cap2 = cv2.VideoCapture(input(f"请输入第二个视频路径 (默认: {'视频/try2.mp4'}): ").strip() or '视频/try2.mp4')

    if not cap1.read()[0] or not cap2.read()[0]:
        print("无法读取视频")
        return

    # 选择摄像头1参考矩形
    ret1, first_frame1 = cap1.read()
    print("选择第一个摄像头的参考矩形")
    ref_points_cam1 = select_reference_rect(first_frame1, screen, font)
    if ref_points_cam1 is None:
        return
    perspective_matrix_cam1 = get_perspective_transform(first_frame1, ref_points_cam1)

    # 选择摄像头2参考矩形
    ret2, first_frame2 = cap2.read()
    print("选择第二个摄像头的参考矩形")
    ref_points_cam2 = select_reference_rect(first_frame2, screen, font)
    if ref_points_cam2 is None:
        return
    perspective_matrix_cam2 = get_perspective_transform(first_frame2, ref_points_cam2)

    global_id_dict = {}  # 格式: { (cam_id, local_id): global_id }
    next_global_id = 1

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # 摄像头1目标检测与追踪
        results1 = model(frame1)
        detections1 = [[int(box.xyxy[0][i]) for i in range(4)] + [float(box.conf[0]), int(box.cls[0])]
                      for result in results1 for box in result.boxes]
        for x1, y1, x2, y2, conf, cls in detections1:
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, f'{model.names[cls]}:{conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracks1 = tracker1.update_tracks([([x1, y1, x2-x1, y2-y1], conf, str(cls))
                                       for x1, y1, x2, y2, conf, cls in detections1], frame=frame1)

        # 摄像头1特征提取与位置映射
        cam1_features = {}
        for track in tracks1:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # 计算底边中点并映射
            bottom_center = ((x1 + x2) // 2, y2)
            point = np.array([[[bottom_center[0], bottom_center[1]]]], dtype=np.float32)
            mapped_point = cv2.perspectiveTransform(point, perspective_matrix_cam1)[0][0]
            mapped_point = tuple(map(int, mapped_point))

            # 提取特征
            crop = frame1[y1:y2, x1:x2]
            if crop.size > 0:
                feat = reid(crop)
                feature_cache_cam1.add(track_id, feat, mapped_point)
                cam1_features[track_id] = feat

            cv2.putText(frame1, f'ID:{track_id}', (x1, y1-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 摄像头2目标检测与追踪
        results2 = model(frame2)
        detections2 = [[int(box.xyxy[0][i]) for i in range(4)] + [float(box.conf[0]), int(box.cls[0])]
                      for result in results2 for box in result.boxes]
        for x1, y1, x2, y2, conf, cls in detections2:
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame2, f'{model.names[cls]}:{conf:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracks2 = tracker2.update_tracks([([x1, y1, x2-x1, y2-y1], conf, str(cls))
                                       for x1, y1, x2, y2, conf, cls in detections2], frame=frame2)

        # 摄像头2特征提取与位置映射
        for track in tracks2:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # 计算底边中点并映射
            bottom_center = ((x1 + x2) // 2, y2)
            point = np.array([[[bottom_center[0], bottom_center[1]]]], dtype=np.float32)
            mapped_point = cv2.perspectiveTransform(point, perspective_matrix_cam2)[0][0]
            mapped_point = tuple(map(int, mapped_point))

            # 提取特征
            crop = frame2[y1:y2, x1:x2]
            if crop.size > 0:
                feat2 = reid(crop)

                # 计算速度
                vel2 = feature_cache_cam2.get_velocity(track_id)

                # 多维度匹配
                best_match = multi_dim_match(feat2, mapped_point, vel2, feature_cache_cam1, perspective_matrix_cam1, perspective_matrix_cam2)

                if best_match is not None:
                    if (1, best_match) in global_id_dict:
                        global_id_dict[(2, track_id)] = global_id_dict[(1, best_match)]
                    else:
                        global_id_dict[(1, best_match)] = next_global_id
                        global_id_dict[(2, track_id)] = next_global_id
                        next_global_id += 1

                    cv2.putText(frame2, f"GID:{global_id_dict[(2, track_id)]}",
                               (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                feature_cache_cam2.add(track_id, feat2, mapped_point)

            cv2.putText(frame2, f'ID:{track_id}', (x1, y1-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 创建平面图
        plan_view1 = create_plan_view(frame1, detections1, perspective_matrix_cam1, track_history_cam1)
        plan_view2 = create_plan_view(frame2, detections2, perspective_matrix_cam2, track_history_cam2)

        # 显示结果
        if plan_view1 is not None and plan_view2 is not None:
            frame_height = frame1.shape[0]
            plan_view1 = cv2.resize(plan_view1,
                                 (int(plan_view1.shape[1] * frame_height / plan_view1.shape[0]),
                                  frame_height),
                                 interpolation=cv2.INTER_LINEAR)
            plan_view2 = cv2.resize(plan_view2,
                                 (int(plan_view2.shape[1] * frame_height / plan_view2.shape[0]),
                                  frame_height),
                                 interpolation=cv2.INTER_LINEAR)

            combined1 = cv2.hconcat([frame1, plan_view1])
            combined2 = cv2.hconcat([frame2, plan_view2])

            combined = cv2.vconcat([combined1, combined2])

            scale = 1280 / max(combined.shape[1], combined.shape[0])
            combined = cv2.resize(combined,
                                (int(combined.shape[1] * scale),
                                 int(combined.shape[0] * scale)),
                                interpolation=cv2.INTER_LINEAR)
        else:
            combined = cv2.vconcat([frame1, frame2])

        screen.fill((0, 0, 0))
        screen.blit(cv2_to_pygame(combined),
                   ((1280 - combined.shape[1]) // 2,
                    (720 - combined.shape[0]) // 2))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type in [QUIT, KEYDOWN] and (event.type == KEYDOWN and event.key == K_q):
                cap1.release()
                cap2.release()
                pygame.quit()
                return

        clock.tick(30)

    cap1.release()
    cap2.release()
    pygame.quit()

if __name__ == "__main__":
    main()
