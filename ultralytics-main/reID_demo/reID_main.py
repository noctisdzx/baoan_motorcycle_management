import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import ReIDExtractor, cosine_similarity
import os
import numpy as np
import torch.nn.functional as F

# 初始化
model = YOLO("runs/detect/train20/weights/best.pt")
reid = ReIDExtractor("reID_demo/reID_model/osnet_x0_25.pth")
tracker1 = DeepSort(max_age=10)
tracker2 = DeepSort(max_age=10)

# 全局ID管理
global_id_dict = {}  # 格式: { (cam_id, local_id): global_id }
next_global_id = 1

# 读取两个摄像头视频
cap1 = cv2.VideoCapture("reID_demo/view-GL1.mp4")
cap2 = cv2.VideoCapture("reID_demo/view-GL2.mp4")

# 每个摄像头的ID特征字典
id_features_cam1 = {}
id_features_cam2 = {}

frame_idx = 0
def resize_frame(frame):
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# 在类定义前添加
class FeatureCache:
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
    
    def add(self, track_id, feature):
        if track_id not in self.cache:
            self.cache[track_id] = []
        self.cache[track_id].append(feature)
        if len(self.cache[track_id]) > self.max_size:
            self.cache[track_id].pop(0)
    
    def get(self, track_id):
        return self.cache.get(track_id, [])

# 初始化时添加
feature_cache_cam1 = FeatureCache()
feature_cache_cam2 = FeatureCache()

def get_object_detections(dets):
    result = []
    for det in dets.boxes.data:
        cls = int(det[5])
        conf = float(det[4])
        if conf > 0.3:  # 保留所有置信度>0.3的物体
            x1, y1, x2, y2 = map(int, det[:4])
            result.append(([x1, y1, x2-x1, y2-y1], conf, cls))
    return result

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # 缩放帧
    frame1 = resize_frame(frame1)
    frame2 = resize_frame(frame2)

    # YOLO检测
    detections1 = model(frame1)[0]
    detections2 = model(frame2)[0]

    # 使用DeepSort进行跟踪


    track1 = tracker1.update_tracks(get_object_detections(detections1), frame=frame1)
    track2 = tracker2.update_tracks(get_object_detections(detections2), frame=frame2)

    # 先处理摄像头1的所有跟踪结果并存储特征
    cam1_features = {}
    # 修改特征提取部分
    for t in track1:
        if not t.is_confirmed():
            continue
            
        track_id = t.track_id
        x, y, w, h = map(int, t.to_ltrb())
        
        # 确保裁剪区域有效
        x = max(0, x)
        y = max(0, y)
        w = min(frame1.shape[1], w)
        h = min(frame1.shape[0], h)
        
        crop = frame1[y:h, x:w]
        if crop.size > 0 and (h-y) > 50 and (w-x) > 20:  # 添加最小尺寸限制
            try:
                # 使用更稳定的特征提取
                feat = reid.extract(crop)
                feat = F.normalize(feat, p=2, dim=0)
                feature_cache_cam1.add(track_id, feat)  # 缓存特征
            except Exception as e:
                print(f"特征提取失败: {e}")

        # 绘制DeepSort跟踪框（绿色）
        cv2.rectangle(frame1, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame1, f"TID:{track_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # 提取并存储ReID特征
        crop = frame1[y:h, x:w]
        if crop.size > 0:
            try:
                feat = reid.extract(crop)
                feat = F.normalize(feat, p=2, dim=0)
                cam1_features[track_id] = feat
            except:
                continue

    # 然后处理摄像头2，寻找与摄像头1匹配的物体
    # 修改特征提取和匹配部分
    for t in track2:
        if not t.is_confirmed():
            continue
            
        track_id = t.track_id
        x, y, w, h = map(int, t.to_ltrb())
        
        # 绘制DeepSort跟踪框（蓝色）
        cv2.rectangle(frame2, (x, y), (w, h), (255, 0, 0), 2)
        cv2.putText(frame2, f"TID:{track_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        
        # 提取ReID特征
        crop = frame2[y:h, x:w]
        if crop.size > 0:
            try:
                feat2 = reid.extract(crop)
                feat2 = F.normalize(feat2, p=2, dim=0)
                
                # 改进匹配策略
                matched_id = None
                max_sim = 0.1  # 提高匹配阈值
                best_match = None
                
                # 使用多帧特征平均
                for tid1, feat1_list in id_features_cam1.items():
                    # 计算与cam1中该目标所有历史特征的相似度
                    similarities = []
                    for feat1 in feat1_list[-5:]:  # 取最近5帧特征
                        sim = cosine_similarity(feat1, feat2)
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                if avg_sim > max_sim:
                    max_sim = avg_sim
                    matched_id = tid1
            
                # 在摄像头1中寻找匹配
                matched_id = None
                max_sim = 0.1
                for tid1, feat1 in cam1_features.items():
                    sim = cosine_similarity(feat1, feat2)
                    if sim > max_sim:
                        max_sim = sim
                        matched_id = tid1
                
                # 如果找到匹配，更新全局ID
                if matched_id is not None:
                    if (1, matched_id) in global_id_dict:
                        global_id_dict[(2, track_id)] = global_id_dict[(1, matched_id)]
                    else:
                        global_id_dict[(1, matched_id)] = next_global_id
                        global_id_dict[(2, track_id)] = next_global_id
                        next_global_id += 1
                    
                    # 显示全局ID
                    cv2.putText(frame2, f"GID:{global_id_dict[(2, track_id)]}", 
                               (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    
            except:
                continue

    # 显示缩放后的帧
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()