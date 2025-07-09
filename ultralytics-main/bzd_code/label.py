import pygame
import cv2
import os
import numpy as np
from pathlib import Path


# 参数设置
default_video = 'baoan_video/2025-06-07_21-55-52_60s.mp4'
video_path = input(f"请输入video路径 (默认: {default_video}): ").strip() or default_video
CLASS_NAMES = ['bicycle', 'motorcycle', 'Shared_bicycle', 'Hire_motorcycle']  # 增加两个新类别
default_output = 'datasets/def0'
output_base = Path(('datasets/' + input(f"请输入训练集保存路径 (默认: {'def0'}): ").strip()) or default_output)
image_dirs = {'train': output_base / 'images/train', 'val': output_base / 'images/val'}
label_dirs = {'train': output_base / 'labels/train', 'val': output_base / 'labels/val'}

for d in list(image_dirs.values()) + list(label_dirs.values()):
    d.mkdir(parents=True, exist_ok=True)
(output_base / 'labels/classes.txt').write_text('\n'.join(CLASS_NAMES))

MAX_EDGE = 1280
MAX_ANNOTATED_FRAMES = 5 # 总共标注5帧
TRAIN_COUNT = MAX_ANNOTATED_FRAMES  # 所有帧用于训练
# 如果需要所有帧用于测试，将 TRAIN_COUNT 设置为 0

pygame.init()
font = pygame.font.SysFont('Arial', 20)
screen = None
pygame.display.set_caption('标注工具')
clock = pygame.time.Clock()

def draw_buttons():
    pygame.draw.rect(screen, (0, 200, 0), (10, 10, 100, 40))
    pygame.draw.rect(screen, (200, 0, 0), (120, 10, 100, 40))
    screen.blit(font.render('Next', True, (0, 0, 0)), (30, 20))
    screen.blit(font.render('Quit', True, (0, 0, 0)), (140, 20))
    # 添加类别选择提示
    screen.blit(font.render('1:bicycle 2:motorcycle', True, (0, 0, 0)), (250, 20))
    screen.blit(font.render('3:Shared_bicycle 4:Hire_motorcycle', True, (0, 0, 0)), (250, 45))

# 定义不同类别的颜色
CLASS_COLORS = {
    0: (0, 255, 0),    # bicycle - 绿色
    1: (0, 0, 255),    # motorcycle - 蓝色
    2: (255, 255, 0),  # Shared_bicycle - 黄色
    3: (255, 0, 255)   # Hire_motorcycle - 紫色
}

def resize_frame(frame):
    h, w = frame.shape[:2]
    scale = MAX_EDGE / max(w, h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    else:
        return frame, 1.0

def save_label_and_image(frame, boxes, frame_id, is_train=True):
    name = f"frame_{frame_id:04d}.jpg"
    subdir = 'train' if is_train else 'val'
    img_path = image_dirs[subdir] / name
    label_path = label_dirs[subdir] / name.replace('.jpg', '.txt')
    cv2.imwrite(str(img_path), frame)
    h, w = frame.shape[:2]
    with open(label_path, 'w') as f:
        for cls, x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

def main():
    global screen
    cap = cv2.VideoCapture(video_path)
    frame_id, annotated = 0, 0
    if not cap.isOpened():
        print("无法打开视频")
        return
    paused = False
    drawing, start_pos, boxes, last_box = False, None, [], None
    ret, frame = cap.read()
    if not ret:
        print("视频为空")
        return
    resized_frame, scale = resize_frame(frame)
    h, w = resized_frame.shape[:2]
    screen = pygame.display.set_mode((w, h + 60))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened() and annotated < MAX_ANNOTATED_FRAMES:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame, scale = resize_frame(frame)
            boxes.clear()
            last_box = None
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        pygame_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pygame_frame = pygame.surfarray.make_surface(pygame_frame.swapaxes(0, 1))
        screen.fill((100, 100, 100))
        screen.blit(pygame_frame, (0, 60))
        draw_buttons()
        for cls_id, x1, y1, x2, y2 in boxes:
            color = CLASS_COLORS[cls_id]  # 根据类别ID获取对应颜色
            pygame.draw.rect(screen, color, pygame.Rect(x1, y1 + 60, x2 - x1, y2 - y1), 2)
        if drawing and start_pos:
            mx, my = pygame.mouse.get_pos()
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(start_pos[0], start_pos[1] + 60, mx - start_pos[0], my - 60 - start_pos[1]), 1)
        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = True
                    print("暂停，开始标注")
                elif event.key == pygame.K_1 and last_box:
                    boxes.append((0, *last_box))  # bicycle
                    last_box = None
                    print("标注：bicycle")
                elif event.key == pygame.K_2 and last_box:
                    boxes.append((1, *last_box))  # motorcycle
                    last_box = None
                    print("标注：motorcycle")
                elif event.key == pygame.K_3 and last_box:  # 新增Shared_bicycle
                    boxes.append((2, *last_box))
                    last_box = None
                    print("标注：Shared_bicycle")
                elif event.key == pygame.K_4 and last_box:  # 新增Hire_motorcycle
                    boxes.append((3, *last_box))
                    last_box = None
                    print("标注：Hire_motorcycle")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if 10 <= x <= 110 and 10 <= y <= 50:
                    # 保存标注
                    orig_boxes = []
                    for cls_id, x1, y1, x2, y2 in boxes:
                        ox1, oy1 = int(x1 / scale), int(y1 / scale)
                        ox2, oy2 = int(x2 / scale), int(y2 / scale)
                        orig_boxes.append((cls_id, ox1, oy1, ox2, oy2))
                    # 将所有数据同时保存到训练集和测试集
                    save_label_and_image(frame, orig_boxes, frame_id, is_train=True)
                    save_label_and_image(frame, orig_boxes, frame_id, is_train=False)
                    print(f"保存 frame {frame_id}")
                    annotated += 1
                    paused = False  # 继续播放
                elif 120 <= x <= 220 and 10 <= y <= 50:
                    cap.release()
                    pygame.quit()
                    return
                elif y > 60:
                    drawing = True
                    start_pos = (x, y - 60)
            elif event.type == pygame.MOUSEBUTTONUP and drawing:
                x, y = event.pos
                end_pos = (x, y - 60)
                x1, y1 = min(start_pos[0], end_pos[0]), min(start_pos[1], end_pos[1])
                x2, y2 = max(start_pos[0], end_pos[0]), max(start_pos[1], end_pos[1])
                last_box = [x1, y1, x2, y2]
                drawing = False
                print("请按 1（bicycle） 或 2（motorcycle）键标注类别")

    cap.release()
    pygame.quit()
    print(f"标注完成，共标注帧数: {annotated}")

if __name__ == '__main__':
    main()