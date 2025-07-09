import pygame
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from datetime import datetime
import sys

def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

def non_max_suppression(boxes, scores, iou_threshold):
    """非极大值抑制"""
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if compute_iou(boxes[current], boxes[i]) < iou_threshold]
    return keep

# ------------ 参数配置 ------------
default_pathA = 'cut_results/cut_新安二-前进一路_2025-06-18_18-04-20_300s.mp4'
video_path = input(f"请输入video路径 (默认: {default_pathA}): ").strip() or default_pathA

CLASS_NAMES = ['bicycle', 'motorcycle', 'Shared_bicycle', 'Hire_motorcycle']  # 新增两种类型
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

default_model_path = 'runs/detect/train6/weights/best.pt'
model_path = input(f"请输入模型路径 (默认: {default_model_path}): ").strip() or default_model_path

# 检查模型文件是否存在
if not Path(model_path).exists():
    print(f"模型文件 {model_path} 不存在，请检查路径。")
    sys.exit()

# 检查视频文件是否存在
if not Path(video_path).exists():
    print(f"视频文件 {video_path} 不存在，请检查路径。")
    sys.exit()

# 获取输出视频名称
output_name = input(f"请输入输出视频名称: ").strip() or "output_tracking"
VIDEO_NAME = Path(output_name + ".mp4")

# 跟踪参数
conf_thres = 0.1
iou_thres = 0.2
deepsort_max_age = 10
deepsort_max_cosine_distance = 0.1
deepsort_n_init = 1
SCALE = 1280  # 显示窗口的最大尺寸

# 颜色定义
COLORS = {
    'bicycle': (0, 255, 0),          # 绿色
    'motorcycle': (0, 0, 255),       # 红色
    'Shared_bicycle': (255, 255, 0), # 黄色
    'Hire_motorcycle': (255, 0, 255),# 紫色
    'text': (255, 255, 255),         # 白色
}

class TrackingApp:
    def __init__(self):
        self.drawing = False
        self.start_pos = None
        self.current_box = None
        self.scale_factor = 1.0
        self.screen = None
        self.clock = pygame.time.Clock()
        self.show_labels = True  # 控制是否显示标签名称、ID和数量

        # 初始化Pygame
        pygame.init()

        # 模型初始化
        print("加载模型...")
        self.detector = YOLO(model_path).to('cuda')  # 添加GPU支持
        self.tracker = DeepSort(
            max_age=deepsort_max_age,
            max_cosine_distance=deepsort_max_cosine_distance,
            n_init=deepsort_n_init,
            nms_max_overlap=1.0,  # 添加NMS参数
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,  # 启用半精度以加速GPU计算
            bgr=True,
            embedder_gpu=True  # 启用embedder的GPU支持
        )

        # 视频初始化
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("无法打开视频文件")
            sys.exit()

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算缩放因子
        self.scale_factor = min(SCALE / max(self.original_w, self.original_h), 1.0)
        self.scaled_w = int(self.original_w * self.scale_factor)
        self.scaled_h = int(self.original_h * self.scale_factor)

        # 初始化Pygame窗口
        self.screen = pygame.display.set_mode((self.scaled_w, self.scaled_h + 60))
        pygame.display.set_caption("目标跟踪工具")

        # 视频输出
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPG编码
        self.output_path = VIDEO_NAME
        self.out = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.scaled_w, self.scaled_h))

    def cv2_to_pygame(self, cv2_img):
        """OpenCV图像转Pygame表面"""
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))

    def run_tracking(self):
        """跟踪阶段"""
        print("开始追踪...按'Q'退出")
        tracking = True
        try:
            while tracking and self.cap.isOpened():
                ret, original_frame = self.cap.read()
                if not ret:
                    break

                # 缩放帧
                display_frame = cv2.resize(original_frame, (self.scaled_w, self.scaled_h))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                # 目标检测
                results = self.detector(frame_rgb, verbose=False)[0]
                boxes, scores, classes = [], [], []
                for box in results.boxes.data:
                    x1, y1, x2, y2, score, cls = box.cpu().numpy()
                    cls = int(cls)
                    if cls in range(len(CLASS_NAMES)) and score >= conf_thres:
                        boxes.append([x1, y1, x2, y2])
                        scores.append(score)
                        classes.append(CLASS_NAMES[cls])

                # 非极大值抑制
                keep_indices = non_max_suppression(boxes, scores, iou_thres)
                detections = []
                for i in keep_indices:
                    x1, y1, x2, y2 = boxes[i]
                    w_box = x2 - x1
                    h_box = y2 - y1
                    detections.append(([x1, y1, w_box, h_box], scores[i], classes[i]))

                # 目标跟踪
                current_tracks = self.tracker.update_tracks(detections, frame=frame_rgb)

                # 绘制跟踪结果
                for track in current_tracks:
                    if not track.is_confirmed():
                        continue
                    x1, y1, x2, y2 = track.to_ltrb()
                    track_id = track.track_id
                    class_name = track.get_det_class()
                    color = COLORS[class_name]
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    if self.show_labels:
                        cv2.putText(display_frame, f"{class_name} ID {track_id}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 显示结果
                self.screen.fill((0, 0, 0))
                frame_surface = self.cv2_to_pygame(display_frame)
                self.screen.blit(frame_surface, (0, 60))

                # 绘制物体数量
                if self.show_labels:
                    bicycle_count = sum(1 for track in current_tracks if track.get_det_class() == 'bicycle' and track.is_confirmed())
                    motorcycle_count = sum(1 for track in current_tracks if track.get_det_class() == 'motorcycle' and track.is_confirmed())
                    shared_bicycle_count = sum(1 for track in current_tracks if track.get_det_class() == 'Shared_bicycle' and track.is_confirmed())
                    hire_motorcycle_count = sum(1 for track in current_tracks if track.get_det_class() == 'Hire_motorcycle' and track.is_confirmed())
                    total_count = bicycle_count + motorcycle_count + shared_bicycle_count + hire_motorcycle_count
                    count_text = f"(bicycle: {bicycle_count}, motorcycle: {motorcycle_count}, Shared_bicycle: {shared_bicycle_count}, Hire_motorcycle: {hire_motorcycle_count})"
                    self.screen.blit(pygame.font.SysFont('Arial', 20).render(count_text, True, COLORS['text']), (self.scaled_w - 700, 15))  # 调整了显示位置

                pygame.display.flip()

                # 事件处理
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        tracking = False
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            tracking = False
                        elif event.key == pygame.K_t:
                            self.show_labels = not self.show_labels
                            print(f"已切换显示标签状态: {'显示' if self.show_labels else '隐藏'}")

                # 写入输出视频
                self.out.write(display_frame)
                self.clock.tick(self.fps)

        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 清理资源
            self.cap.release()
            self.out.release()
            pygame.quit()
            print(f"追踪结束，视频已保存到 {self.output_path}")

if __name__ == "__main__":
    app = TrackingApp()
    app.run_tracking()