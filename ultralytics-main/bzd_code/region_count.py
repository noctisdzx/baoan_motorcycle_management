import cv2
import pygame
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 定义全局字体大小变量
FONT_SIZE = 18

default_path = 'F:/digital archtecture/graduate study/code/baoan/smart_motorcyclecycle_management/pretrain_model/yolo11n.pt'
# 修改为只获取模型路径字符串
MODEL = input(f"请输入模型路径: ").strip() or default_path
default_path = "F:/digital archtecture/graduate study/code/baoan/smart_motorcyclecycle_management/video/try.mp4"
VIDEO_PATH = input(f"请输入视频路径: ").strip() or default_path
RESULT_PATH = "output_results/区域计数.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(RESULT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 读取第一帧用于绘制区域
success, first_frame = cap.read()
if not success:
    print("无法读取视频帧")
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    exit()

# 初始化pygame
pygame.init()
screen = pygame.display.set_mode((w, h))
drawing = True
all_regions = []
current_region = []
editing = False
input_text = ""
# 新增播放状态标志
is_playing = True

# 转换OpenCV图像到Pygame格式
def cv2_to_pygame(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")

frame = cv2_to_pygame(first_frame)

# 指定支持中文的字体文件路径，Windows 系统使用黑体
font_path = "C:/Windows/Fonts/simhei.ttf"
# 使用全局字体大小变量创建字体对象
font = pygame.font.Font(font_path, FONT_SIZE)

while drawing:
    # 显示原始图像
    screen.blit(frame, (0, 0))
    
    # 绘制已完成的区域（绿色）
    for i, region in enumerate(all_regions):
        pygame.draw.polygon(screen, (0, 255, 0), region, 2)
        # 计算区域的中心点
        region_array = np.array(region)
        M = cv2.moments(region_array)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            text = font.render(f"区域{i + 1}", True, (255, 255, 255))
            screen.blit(text, (cX, cY))
    
    # 绘制当前区域的点（红色圆点）
    for point in current_region:
        pygame.draw.circle(screen, (255, 0, 0), point, 5)
    
    # 只在有3个或更多点时绘制多边形预览
    if len(current_region) >= 3:
        pygame.draw.polygon(screen, (255, 0, 0), current_region, 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            current_region.append((x, y))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                if len(current_region) >= 3:
                    all_regions.append(current_region)
                    current_region = []
            elif event.key == pygame.K_SPACE:
                if len(current_region) >= 3:
                    all_regions.append(current_region)
                drawing = False

    pygame.display.flip()

model = YOLO(MODEL)

# 初始化区域统计信息
region_stats = []
for i, region in enumerate(all_regions):
    region_stats.append({
        'in_count': 0,
        'out_count': 0,
        'color': (0, 255, i*50)
    })

# 重置视频捕获到开头
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 定义 DeepSORT 跟踪器
tracker = DeepSort(max_age=5, n_init=2, nms_max_overlap=1.0)

# 主循环
running = True
while running and cap.isOpened():
    if is_playing:
        success, im0 = cap.read()
        if not success:
            break

    if not editing and is_playing:
        # 重置当前帧统计
        for stats in region_stats:
            stats['in_count'] = 0
            stats['out_count'] = 0

        # 进行目标检测
        results = model(im0)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])

        # 使用 DeepSORT 进行跟踪
        tracks = tracker.update_tracks(detections, frame=im0)

        plotted_img = im0.copy()

        # 在 OpenCV 图像上绘制区域及 ID
        for i, region in enumerate(all_regions):
            cv2.polylines(plotted_img, [np.array(region)], True, (0, 255, 0), 2)
            # 计算区域的中心点
            region_array = np.array(region)
            M = cv2.moments(region_array)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(plotted_img, f"region{i + 1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 处理跟踪结果
        overlay = im0.copy()  # 创建叠加层
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb().astype(int)
            bottom_center = (int((x1 + x2) / 2), int(y2))
            in_any_region = False

            # 检查每个区域
            for i, region in enumerate(all_regions):
                dist = cv2.pointPolygonTest(np.array(region), bottom_center, False)
                if dist >= 0:  # 在区域内
                    region_stats[i]['in_count'] += 1
                    in_any_region = True
                    # 在区域内的物体用绿色框标注
                    cv2.rectangle(plotted_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(plotted_img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

            # 如果不在任何区域内，则添加半透明填充色
            if not in_any_region:
                for stats in region_stats:
                    stats['out_count'] += 1
                # 绘制半透明红色填充矩形
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # -1 表示填充矩形
                cv2.putText(plotted_img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 合并叠加层和原始图像
        alpha = 0.3  # 透明度，取值范围 0-1
        cv2.addWeighted(overlay, alpha, plotted_img, 1 - alpha, 0, plotted_img)

        # 转换为 Pygame 格式
        pygame_img = cv2_to_pygame(plotted_img)
        screen.blit(pygame_img, (0, 0))

        # 显示每个区域的统计结果
        y_offset = 30
        for i, stats in enumerate(region_stats):
            text = font.render(f"停车区{i + 1}: 内{stats['in_count']} 外{stats['out_count']}", True, stats['color'])
            screen.blit(text, (10, 10 + i*30))
    else:
        screen.blit(cv2_to_pygame(im0), (0, 0))
        # 绘制已完成的区域（绿色）
        for i, region in enumerate(all_regions):
            pygame.draw.polygon(screen, (0, 255, 0), region, 2)
            # 计算区域的中心点
            region_array = np.array(region)
            M = cv2.moments(region_array)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                text = font.render(f"region{i + 1}", True, (255, 255, 255))
                screen.blit(text, (cX, cY))
        
        # 绘制当前区域的点（红色圆点）
        for point in current_region:
            pygame.draw.circle(screen, (255, 0, 0), point, 5)
        
        # 只在有3个或更多点时绘制多边形预览
        if len(current_region) >= 3:
            pygame.draw.polygon(screen, (255, 0, 0), current_region, 2)

        input_text_surface = font.render(f"要删的区域ID ( O-退出 N-下一区域): {input_text}", True, (255, 255, 255))
        screen.blit(input_text_surface, (10, h - 40))

    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if not editing and event.key == pygame.K_e:  # 按下 E 键进入编辑模式并暂停播放
                editing = True
                current_region = []
                input_text = ""
                is_playing = False
            elif event.key == pygame.K_o:  # 按下 O 键继续播放和计数
                is_playing = True
                editing = False
            elif editing:
                if event.key == pygame.K_RETURN:
                    if input_text.isdigit():
                        region_id = int(input_text) - 1
                        if region_id == -1:
                            editing = False
                        elif 0 <= region_id < len(all_regions):
                            del all_regions[region_id]
                            del region_stats[region_id]
                    input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.key == pygame.K_n:
                    if len(current_region) >= 3:
                        all_regions.append(current_region)
                        region_stats.append({
                            'in_count': 0,
                            'out_count': 0,
                            'color': (0, 255, len(all_regions)*50)
                        })
                        current_region = []
                elif event.key == pygame.K_SPACE:
                    if len(current_region) >= 3:
                        all_regions.append(current_region)
                        region_stats.append({
                            'in_count': 0,
                            'out_count': 0,
                            'color': (0, 255, len(all_regions)*50)
                        })
                    editing = False
                else:
                    input_text += event.unicode
            elif event.key == pygame.K_SPACE:  # 按下空格键退出主循环
                running = False
        elif editing and event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            current_region.append((x, y))

    pygame.display.flip()

    if not editing and is_playing:
        video_writer.write(plotted_img)

cap.release()
video_writer.release()
pygame.quit()