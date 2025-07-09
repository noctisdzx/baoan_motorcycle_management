from tkinter import messagebox
import pygame
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from datetime import datetime
import math
import json
import threading
# 修改导入部分
from PyQt5.QtWidgets import (QApplication, QMainWindow, QListWidget, QLabel, 
                            QLineEdit, QPushButton, QVBoxLayout, QWidget,
                            QMessageBox, QScrollArea, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QSize
import sys
import os
# 在文件开头添加
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/user/anaconda3/envs/yolov11/lib/python3.8/site-packages/PyQt5/Qt5/plugins'
def compute_iou(box1, box2):
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
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if compute_iou(boxes[current], boxes[i]) < iou_threshold]
    return keep
# 参数配置
default_pathA = 'baoan_video/noon/train/上川路-前进一路2_2025-06-09_11-31-14_60s.mp4'
video_path = input(f"请输入video路径 (默认: {default_pathA}): ").strip() or default_pathA
# 参数配置
CLASS_NAMES = ['bicycle', 'motorcycle', 'Shared_bicycle', 'Hire_motorcycle']  # 增加两个新类别
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 颜色定义
COLORS = {
    'bicycle': (0, 255, 0),
    'motorcycle': (0, 0, 255),
    'Shared_bicycle': (255, 255, 0),  # 黄色
    'Hire_motorcycle': (255, 0, 255),  # 紫色
    'text': (255, 255, 255),
    'button': (100, 100, 100),
    'button_text': (255, 255, 255),
    'save_hint': (255, 255, 0),
    'delete_hint': (255, 0, 0)
}
default_path = 'runs/detect/train6/weights/bestC.pt'
trained_model_path = input(f"请输入模型路径 (默认: {default_path}): ").strip() or default_path
VIDEO_NAME = Path(input(f"请输入video name: ").strip() or "output_tracking.mp4")
# 训练数据保存路径
default_dataset_path = "datasets/def"
output_base = Path(('datasets/' + input(f"请输入数据集路径 (默认: {'def'}): ")).strip() or default_dataset_path)
image_dirs = {'train': output_base / 'images/train', 'val': output_base / 'images/val'}
label_dirs = {'train': output_base / 'labels/train', 'val': output_base / 'labels/val'}
auto_boxes_dir = output_base / 'auto_boxes_data'
# 创建目录结构
for d in list(image_dirs.values()) + list(label_dirs.values()) + [auto_boxes_dir]:
    d.mkdir(parents=True, exist_ok=True)
(output_base / 'labels/classes.txt').write_text('\n'.join(CLASS_NAMES))
# 跟踪参数
conf_thres = 0.25
iou_thres = 0.5
deepsort_max_age = 2
deepsort_max_cosine_distance = 0.1
deepsort_n_init = 1
SCALE = 1600  # 显示窗口的最大尺寸
# Pygame初始化
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 20)
small_font = pygame.font.SysFont('Arial', 20)
# 颜色定义
COLORS = {
    'bicycle': (0, 255, 0),      # 绿色
    'motorcycle': (0, 0, 255),    # 蓝色
    'Shared_bicycle': (255, 255, 0),  # 黄色
    'Hire_motorcycle': (255, 0, 255),  # 紫色
    'text': (255, 255, 255),
    'button': (100, 100, 100),
    'button_text': (255, 255, 255),
    'save_hint': (255, 255, 0),
    'delete_hint': (255, 0, 0)
}
class DataModificationWindow(QMainWindow):
    def __init__(self, data_lock):
        super().__init__()
        self.data_lock = data_lock
        self.current_file = None
        self.setWindowTitle("数据修改")
        self.setFixedSize(600, 500)
        
        # 主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 数据列表
        self.data_list = QListWidget()
        self.data_list.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.data_list)
        
        # ID输入
        self.id_label = QLabel("输入要删除的框ID:")
        self.id_label.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.id_label)
        
        self.id_entry = QLineEdit()
        self.id_entry.setFont(QFont('Arial', 12))
        self.layout.addWidget(self.id_entry)
        
        # 按钮布局
        self.button_layout = QHBoxLayout()
        
        self.delete_button = QPushButton("删除")
        self.delete_button.setFont(QFont('Arial', 12))
        self.delete_button.clicked.connect(self.delete_box)
        self.button_layout.addWidget(self.delete_button)
        
        self.view_image_button = QPushButton("查看图像")
        self.view_image_button.setFont(QFont('Arial', 12))
        self.view_image_button.clicked.connect(self.view_image_for_current_file)
        self.button_layout.addWidget(self.view_image_button)
        
        self.layout.addLayout(self.button_layout)
        
        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setFont(QFont('Arial', 12))
        self.status_label.setStyleSheet("color: blue;")
        self.layout.addWidget(self.status_label)
        
        # 图像查看窗口
        self.image_window = None
    
    def update_data_display(self, auto_boxes_data):
        self.data_list.clear()
        for box in auto_boxes_data:
            self.data_list.addItem(f"ID: {box['id']}, 类别: {box['class']}")
    
    def set_current_file(self, file_path):
        self.current_file = file_path
    
    def convert_json_to_txt(self):
        """将JSON文件转换为YOLO格式的TXT文件"""
        if not self.current_file:
            return
            
        try:
            # 读取JSON文件
            with open(self.current_file, 'r') as f:
                data = json.load(f)
                
            # 确定对应的图像路径
            image_base_dir = image_dirs['train'] if 'train' in str(self.current_file) else image_dirs['val']
            image_name = Path(self.current_file).stem + '.jpg'
            image_path = os.path.join(image_base_dir, image_name)
            
            if not os.path.exists(image_path):
                QMessageBox.critical(self, "错误", f"未找到对应的图像文件: {image_path}")
                return
                
            # 读取图像尺寸
            img = cv2.imread(image_path)
            img_h, img_w = img.shape[:2]
            
            # 确定TXT文件路径
            txt_path = os.path.join(label_dirs['train' if 'train' in str(self.current_file) else 'val'], 
                                  Path(self.current_file).stem + '.txt')
            
            # 写入TXT文件
            with open(txt_path, 'w') as f:
                for box in data['frames'][0]['auto_boxes']:
                    x_min, y_min, x_max, y_max = box['box']
                    class_name = box['class']
                    
                    if class_name not in CLASS_TO_IDX:
                        continue
                        
                    # 转换为YOLO格式
                    x_center = ((x_min + x_max) / 2) / img_w
                    y_center = ((y_min + y_max) / 2) / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h
                    
                    f.write(f"{CLASS_TO_IDX[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"转换JSON到TXT时出错: {str(e)}")

    def delete_box(self):
        user_input = self.id_entry.text().strip()
        if not user_input.isdigit():
            QMessageBox.critical(self, "错误", "请输入有效的数字ID！")
            return
            
        delete_id = int(user_input)
        success = False
        
        with self.data_lock:
            try:
                with open(self.current_file, 'r') as f:
                    data = json.load(f)
                    
                modified = False
                for frame in data['frames']:
                    new_boxes = [box for box in frame['auto_boxes'] if box['id'] != delete_id]
                    if len(new_boxes) != len(frame['auto_boxes']):
                        modified = True
                        frame['auto_boxes'] = new_boxes
                        
                if modified:
                    with open(self.current_file, 'w') as f:
                        json.dump(data, f, indent=4)
                    success = True
                    self.update_data_display(data['frames'][0]['auto_boxes'])
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除过程中出现错误: {str(e)}")
                
        if success:
            self.status_label.setText(f"成功删除ID为{delete_id}的框。")
            self.status_label.setStyleSheet("color: green;")
            self.convert_json_to_txt()  # 现在这个方法已定义
        else:
            self.status_label.setText(f"未找到ID为{delete_id}的框。请检查输入是否正确。")
            self.status_label.setStyleSheet("color: red;")
    
    def view_image_for_current_file(self):
        """查看当前JSON文件对应的图像及其标注框"""
        if not self.current_file:
            QMessageBox.critical(self, "错误", "未选择文件！")
            return
            
        # 读取图像路径
        image_base_dir = image_dirs['train'] if 'train' in str(self.current_file) else image_dirs['val']
        image_name = Path(self.current_file).stem + '.jpg'
        image_path_train = os.path.join(image_dirs['train'], image_name)
        image_path_val = os.path.join(image_dirs['val'], image_name)
        
        if os.path.exists(image_path_train):
            image_path = image_path_train
        elif os.path.exists(image_path_val):
            image_path = image_path_val
        else:
            QMessageBox.critical(self, "错误", f"未找到图像 {image_name}")
            return
            
        # 创建图像查看窗口
        if not self.image_window:
            self.image_window = QMainWindow()
            self.image_window.setWindowTitle("图像查看器")
            self.image_window.resize(800, 600)
            
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            
            scroll = QScrollArea()
            scroll.setWidget(self.image_label)
            scroll.setWidgetResizable(True)
            
            self.image_window.setCentralWidget(scroll)
        
        try:
            # 加载图像
            original_image = QImage(image_path)
            if original_image.isNull():
                raise ValueError("无法加载图像")
                
            # 读取标注框信息
            with open(self.current_file, 'r') as f:
                data = json.load(f)
                auto_boxes = data['frames'][0]['auto_boxes']
                
            # 绘制标注框
            painter = QPainter(original_image)
            # 使用更粗的字体
            font = QFont('Arial', 12)
            font.setBold(True)  # 设置字体加粗
            painter.setFont(font)
            
            for box in auto_boxes:
                x_min, y_min, x_max, y_max = box['box']
                
                # 根据类别设置颜色
                if box['class'] == 'motorcycle':
                    color = QColor(255, 0, 0)  # 红色
                else:
                    color = QColor(*COLORS[box['class']])  # 保持原有颜色
                
                # 绘制矩形框
                painter.setPen(color)
                painter.drawRect(int(x_min), int(y_min), 
                               int(x_max - x_min), int(y_max - y_min))
                
                # 绘制文本
                painter.drawText(int(x_min), int(y_min - 5), 
                                f"{box['class']} ID {box['id']}")
            
            painter.end()
            
            # 显示图像
            pixmap = QPixmap.fromImage(original_image)
            self.image_label.setPixmap(pixmap)
            self.image_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示图像时出错: {str(e)}")
            if self.image_window:
                self.image_window.close()
class TrackingApp:
    def __init__(self):
        self.boxes_first_frame = []
        self.drawing = False
        self.deleting = False
        self.start_pos = None
        self.current_box = None
        self.scale_factor = 1.0
        self.screen = None
        self.clock = pygame.time.Clock()
        self.saved_frames_count = 0
        self.save_confirmation = False
        self.current_boxes = []  # 用于存储当前帧的手动标注框
        self.current_auto_boxes = []  # 用于存储当前帧的自动识别标注框
        self.show_labels = True  # 控��是否显示标签名称、ID和数量
        
        # 模型初始化
        print("加载模型...")
        self.detector = YOLO(trained_model_path)
        self.tracker = DeepSort(
            max_age=deepsort_max_age,
            max_cosine_distance=deepsort_max_cosine_distance,
            n_init=deepsort_n_init
        )
        
        # 视频初始化
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("无法打开视频文件")
            exit()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算缩放因子
        self.scale_factor = min(SCALE / max(self.original_w, self.original_h), 1.0)
        self.scaled_w = int(self.original_w * self.scale_factor)
        self.scaled_h = int(self.original_h * self.scale_factor)
        
        # 读取第一帧
        ret, self.frame_first = self.cap.read()
        if not ret:
            print("无法读取视频第一帧")
            exit()
        self.frame_first_display = cv2.resize(self.frame_first, (self.scaled_w, self.scaled_h))
        
        # 初始化Pygame窗口
        self.screen = pygame.display.set_mode((self.scaled_w, self.scaled_h + 60))
        pygame.display.set_caption("目标跟踪工具")
        
        # 视频输出
        self.output_path = VIDEO_NAME
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.scaled_w, self.scaled_h))
        
        # 当前原始帧缓存
        self.current_original_frame = None
        
        # 数据锁
        self.data_lock = threading.Lock()
        
    def draw_buttons(self):
        """绘制界面按钮"""
        pygame.draw.rect(self.screen, COLORS['button'], (10, 10, 100, 40))
        pygame.draw.rect(self.screen, COLORS['button'], (120, 10, 100, 40))
        self.screen.blit(font.render('start(S)', True, COLORS['button_text']), (15, 20))
        self.screen.blit(font.render('quit(Q)', True, COLORS['button_text']), (140, 20))
        # 更新类别选择提示
        self.screen.blit(font.render('1:bicycle 2:motorcycle', True, COLORS['text']), (250, 20))
        self.screen.blit(font.render('3:Shared_bicycle 4:Hire_motorcycle', True, COLORS['text']), (250, 45))
        # 绘制保存提示
        if self.save_confirmation:
            self.screen.blit(font.render('数据已保存', True, COLORS['save_hint']), (400, 20))
        # 绘制显示标签状态
        status_text = 'show' if self.show_labels else 'hide'
        self.screen.blit(font.render(status_text, True, COLORS['text']), (550, 20))
    def cv2_to_pygame(self, cv2_img):
        """OpenCV图像转Pygame表面"""
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
    def save_as_training_data(self, original_frame, frame_number):
        """保存训练数据"""
        if not os.path.exists(auto_boxes_dir):
            os.makedirs(auto_boxes_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{timestamp}"
        dataset_type = 'train' if (self.saved_frames_count % 5 != 0) else 'val'
        img_path = image_dirs[dataset_type] / f"{filename}.jpg"
        cv2.imwrite(str(img_path), original_frame)
        label_path = label_dirs[dataset_type] / f"{filename}.txt"
        auto_boxes_file = auto_boxes_dir / f"{filename}.json"
        
        with open(label_path, 'w') as f:
            # 只保存当前帧的手动标注框和自动跟踪的框，不保存第一帧的标注框
            for item in self.current_boxes:
                x_min, y_min, x_max, y_max = item['box']
                x_min = int(x_min / self.scale_factor)
                y_min = int(y_min / self.scale_factor)
                x_max = int(x_max / self.scale_factor)
                y_max = int(y_max / self.scale_factor)
                class_name = item['class']
                if class_name not in CLASS_TO_IDX:
                    continue
                img_h, img_w = original_frame.shape[:2]
                x_center = ((x_min + x_max) / 2) / img_w
                y_center = ((y_min + y_max) / 2) / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                f.write(f"{CLASS_TO_IDX[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
            for track in self.current_tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = track.to_ltrb()
                x1 = int(x1 / self.scale_factor)
                y1 = int(y1 / self.scale_factor)
                x2 = int(x2 / self.scale_factor)
                y2 = int(y2 / self.scale_factor)
                class_name = track.get_det_class()
                if class_name not in CLASS_TO_IDX:
                    continue
                img_h, img_w = original_frame.shape[:2]
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                f.write(f"{CLASS_TO_IDX[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # 读取并解析label_path中的数据
        auto_boxes_data = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_idx, x_center, y_center, width, height = map(float, parts)
                class_name = CLASS_NAMES[int(class_idx)]
                
                # 转换为绝对坐标
                img_h, img_w = original_frame.shape[:2]
                x_center_abs = x_center * img_w
                y_center_abs = y_center * img_h
                width_abs = width * img_w
                height_abs = height * img_h
                
                x_min = x_center_abs - width_abs / 2
                y_min = y_center_abs - height_abs / 2
                x_max = x_center_abs + width_abs / 2
                y_max = y_center_abs + height_abs / 2
                
                # 添加到auto_boxes_data中
                auto_boxes_data.append({
                    'id': idx,  # 用行号作为ID
                    'class': class_name,
                    'box': [x_min, y_min, x_max, y_max]
                })
        
        # 保存JSON文件
        with open(auto_boxes_file, 'w') as f:
            json.dump({
                'frames': [{
                    'frame_number': frame_number,
                    'auto_boxes': auto_boxes_data
                }]
            }, f, indent=4)
        
        self.saved_frames_count += 1
        print(f"训练数据已保存: {img_path}, {label_path}")
        self.save_confirmation = True
        self.current_boxes = []  # 清空当前帧的手动标注框
        
        # 直接在主线程中创建和显示PyQt窗口
        if not QApplication.instance():
            qt_app = QApplication(sys.argv)
        else:
            qt_app = QApplication.instance()
        
        data_window = DataModificationWindow(self.data_lock)
        data_window.set_current_file(auto_boxes_file)
        
        # 加载数据并显示
        try:
            with open(auto_boxes_file, 'r') as f:
                data = json.load(f)
                auto_boxes = data['frames'][0]['auto_boxes']
                data_window.update_data_display(auto_boxes)
        except Exception as e:
            QMessageBox.critical(data_window, "错误", f"加载数据时出错: {str(e)}")
        
        data_window.show()
        qt_app.exec_()
    def run_annotation(self):
        """标注阶段"""
        print("请用鼠标拖拽框选目标，完成后按'S'开始追踪，按'Q'退出")
        running = True
        while running:
            self.screen.fill((0, 0, 0))
            frame_surface = self.cv2_to_pygame(self.frame_first_display)
            self.screen.blit(frame_surface, (0, 60))
            self.draw_buttons()
            # 绘制已标注的框
            for item in self.boxes_first_frame:
                x_min, y_min, x_max, y_max = item['box']
                color = COLORS[item['class']]
                pygame.draw.rect(self.screen, color, (x_min, y_min + 60, x_max - x_min, y_max - y_min), 2)
                if self.show_labels:
                    self.screen.blit(font.render(item['class'], True, COLORS['text']), (x_min, y_min + 50))
            # 绘制当前正在绘制的框
            if self.drawing and self.start_pos:
                mouse_pos = pygame.mouse.get_pos()
                pygame.draw.rect(self.screen, (0, 255, 0),
                               (self.start_pos[0], self.start_pos[1] + 60,
                                mouse_pos[0] - self.start_pos[0], mouse_pos[1] - 60 - self.start_pos[1]), 1)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    self.cap.release()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        x, y = event.pos
                        if y > 60:  # 图像区域点击
                            self.drawing = True
                            self.start_pos = (x, y - 60)
                        elif 10 <= x <= 110 and 10 <= y <= 50:  # 开始按钮
                            if len(self.boxes_first_frame) == 0:
                                print("请至少框选一个目标!")
                            else:
                                return True
                        elif 120 <= x <= 220 and 10 <= y <= 50:  # 退出按钮
                            running = False
                            pygame.quit()
                            self.cap.release()
                            exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False
                        x, y = event.pos
                        end_pos = (x, y - 60)
                        x_min, x_max = sorted([self.start_pos[0], end_pos[0]])
                        y_min, y_max = sorted([self.start_pos[1], end_pos[1]])
                        self.current_box = [x_min, y_min, x_max, y_max]
                elif event.type == pygame.KEYDOWN and self.current_box:
                    if event.key == pygame.K_1:
                        self.boxes_first_frame.append({'box': self.current_box, 'class': 'bicycle'})
                        self.current_box = None
                    elif event.key == pygame.K_2:
                        self.boxes_first_frame.append({'box': self.current_box, 'class': 'motorcycle'})
                        self.current_box = None
                    elif event.key == pygame.K_3:  # 新增Shared_bicycle
                        self.boxes_first_frame.append({'box': self.current_box, 'class': 'Shared_bicycle'})
                        self.current_box = None
                    elif event.key == pygame.K_4:  # 新增Hire_motorcycle
                        self.boxes_first_frame.append({'box': self.current_box, 'class': 'Hire_motorcycle'})
                        self.current_box = None
            self.clock.tick(30)
        return False
    def run_manual_annotation(self, display_frame, original_frame, frame_number):
        """手动标注当前帧"""
        print("按下空格键暂停并开始手动标注，按下S键保存当前帧")
        manual_annotation_running = True
        while manual_annotation_running:
            self.screen.fill((0, 0, 0))
            frame_surface = self.cv2_to_pygame(display_frame)
            self.screen.blit(frame_surface, (0, 60))
            self.draw_buttons()
            
            # 绘制自动检测的框
            for track in self.current_tracks:
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
                self.current_auto_boxes.append({'box': [x1, y1, x2, y2], 'id': track_id, 'class': class_name})
            
            # 绘制手动标注的框
            for item in self.current_boxes:
                x_min, y_min, x_max, y_max = item['box']
                color = COLORS[item['class']]
                pygame.draw.rect(self.screen, color, (x_min, y_min + 60, x_max - x_min, y_max - y_min), 2)
                if self.show_labels:
                    self.screen.blit(font.render(f"{item['class']} ID", True, COLORS['text']), (x_min, y_min + 50))
            
            # 绘制物体数量
            if self.show_labels:
                bicycle_count = sum(1 for track in self.current_tracks if track.get_det_class() == 'bicycle' and track.is_confirmed())
                motorcycle_count = sum(1 for track in self.current_tracks if track.get_det_class() == 'motorcycle' and track.is_confirmed())
                total_count = bicycle_count + motorcycle_count
                count_text = f" (bicycle: {bicycle_count}, motorcycle: {motorcycle_count})"
                self.screen.blit(small_font.render(count_text, True, COLORS['text']), (self.scaled_w - small_font.size(count_text)[0], 10))
            
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        manual_annotation_running = False
                        print("退出手动标注模式")
                    elif event.key == pygame.K_s and self.current_original_frame is not None:
                        self.save_as_training_data(self.current_original_frame, frame_number)
                    elif event.key == pygame.K_t:
                        self.show_labels = not self.show_labels
                        print(f"已切换显示标签状态: {'显示' if self.show_labels else '隐藏'}")
                    elif event.key == pygame.K_d:  # 按下D键进入删除模式
                        print("请输入要删除的自动识别框ID:")
                        input_id = input().strip()  # 通过控制台输入ID
                        if input_id.isdigit():
                            delete_id = int(input_id)
                            # 查找并删除对应的自动识别框
                            self.current_auto_boxes = [box for box in self.current_auto_boxes if box['id'] != delete_id]
                            print(f"已删除ID为{delete_id}的自动识别框")
                        else:
                            print("请输入有效的数字ID")
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4] and self.current_box:
                        if event.key == pygame.K_1:
                            class_name = 'bicycle'
                        elif event.key == pygame.K_2:
                            class_name = 'motorcycle'
                        elif event.key == pygame.K_3:
                            class_name = 'Shared_bicycle'
                        elif event.key == pygame.K_4:
                            class_name = 'Hire_motorcycle'
                        self.current_boxes.append({'box': self.current_box, 'class': class_name})
                        self.current_box = None
                        print(f"已标注{class_name}")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if y > 60:
                        if event.button == 1:
                            self.drawing = True
                            self.start_pos = (x, y - 60)
                elif event.type == pygame.MOUSEBUTTONUP and self.drawing:
                    self.drawing = False
                    x, y = event.pos
                    end_pos = (x, y - 60)
                    x_min, x_max = sorted([self.start_pos[0], end_pos[0]])
                    y_min, y_max = sorted([self.start_pos[1], end_pos[1]])
                    self.current_box = [x_min, y_min, x_max, y_max]
    def run_tracking(self):
        """跟踪阶段"""
        # 初始化第一帧检测
        ret, first_frame = self.cap.read()
        if not ret:
            print("无法读取第一帧")
            exit()
        detections_first = []
        for item in self.boxes_first_frame:
            x_min, y_min, x_max, y_max = item['box']
            cls_name = item['class']
            w_box = x_max - x_min
            h_box = y_max - y_min
            detections_first.append(([x_min, y_min, w_box, h_box], 0.99, cls_name))
        self.tracker.update_tracks(detections_first, frame=first_frame)
        
        print("开始追踪...按空格键暂停并手动标注，按'S'保存训练数据，按'Q'退出")
        tracking = True
        manual_annotation_mode = False
        frame_number = 0
        while tracking and self.cap.isOpened():
            ret, original_frame = self.cap.read()
            if not ret:
                break
            frame_number += 1
            self.current_original_frame = original_frame.copy()
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
            self.current_tracks = self.tracker.update_tracks(detections, frame=frame_rgb)
            # 绘制跟踪结果
            for track in self.current_tracks:
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
            # 处理手动标注
            if manual_annotation_mode:
                self.run_manual_annotation(display_frame, original_frame, frame_number)
                manual_annotation_mode = False
            # 显示结果
            self.screen.fill((0, 0, 0))
            frame_surface = self.cv2_to_pygame(display_frame)
            self.screen.blit(frame_surface, (0, 60))
            self.draw_buttons()
            # 绘制物体数量
            if self.show_labels:
                bicycle_count = sum(1 for track in self.current_tracks if track.get_det_class() == 'bicycle' and track.is_confirmed())
                motorcycle_count = sum(1 for track in self.current_tracks if track.get_det_class() == 'motorcycle' and track.is_confirmed())
                total_count = bicycle_count + motorcycle_count
                count_text = f"(bicycle: {bicycle_count}, motorcycle: {motorcycle_count})"
                self.screen.blit(small_font.render(count_text, True, COLORS['text']), (self.scaled_w - small_font.size(count_text)[0], 10))
            pygame.display.flip()
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    tracking = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        tracking = False
                    elif event.key == pygame.K_s and self.current_original_frame is not None:
                        self.save_as_training_data(self.current_original_frame, frame_number)
                    elif event.key == pygame.K_SPACE:
                        manual_annotation_mode = True
                        print("进入手动标注模式")
                    elif event.key == pygame.K_t:
                        self.show_labels = not self.show_labels
                        print(f"已切换显示标签状态: {'显示' if self.show_labels else '隐藏'}")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if 10 <= x <= 110 and 10 <= y <= 50 and self.current_original_frame is not None:
                        self.save_as_training_data(self.current_original_frame, frame_number)
                    elif 120 <= x <= 220 and 10 <= y <= 50:
                        tracking = False
            self.clock.tick(self.fps)
        # 清理资源
        self.cap.release()
        self.out.release()
        pygame.quit()
        print(f"追踪结束，视频已保存到 {self.output_path}")
        print(f"共保存了 {self.saved_frames_count} 帧训练数据")
    def run(self):
        """运行应用程序"""
        if self.run_annotation():
            self.run_tracking()
if __name__ == "__main__":
    app = TrackingApp()
    app.run()