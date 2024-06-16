import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from sklearn.neighbors import KNeighborsClassifier
import joblib

'''
handtype:
0   🤘
1   👍
2   ✌️
3   👌
4   🤙
5   🤌
'''
# 初始化gesture_code
gesture_code = 10
# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 创建Hand对象
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 保存路径
save_path = "my_gestures/data_"+f'{gesture_code}'+"/"  # 替换为你想保存的路径

# 创建保存路径（如果不存在）
os.makedirs(save_path, exist_ok=True)

# 初始化计时器
start_time = time.time()

def normalization(landmarks):
    # 将关键点转换为相对于手部中心点的坐标系
    landmarks = np.array(landmarks)
    center = np.mean(landmarks, axis=0)
    normalized_landmarks = landmarks - center
    max_value = np.max(np.linalg.norm(normalized_landmarks, axis=1))
    normalized_landmarks /= max_value
    return normalized_landmarks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 翻转图像
    frame = cv2.flip(frame, 1)
    # 将图像转换为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 检测手部
    results = hands.process(rgb_frame)
    
    # 获取当前时间
    current_time = time.time()
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取手的类型（左手或右手）
            hand_type = results.multi_handedness[idx].classification[0].label
            
            # 自定义绘制手部关键点和连接线
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # 在图像上显示手的类型
            cv2.putText(frame, hand_type, 
                        (int(hand_landmarks.landmark[0].x * frame.shape[1]), 
                         int(hand_landmarks.landmark[0].y * frame.shape[0])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 每秒保存一次
            if current_time - start_time >= 0.5:
                start_time = current_time  # 重置计时器
                
                # 保存当前帧图像
                image_filename = os.path.join(save_path, f"frame_{int(current_time)}.png")
                cv2.imwrite(image_filename, frame)
                
                # 保存关键点信息
                landmarks = [{
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                } for landmark in hand_landmarks.landmark]
                normalized_landmarks = normalization([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
                print('normalized')
                print(normalized_landmarks)
                # 重新按照原格式保存归一化后的关键点
                json_data = {
                    "hand_type": gesture_code,
                    "landmarks": [{"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2])} for lm in normalized_landmarks]
                }
                print('jsonfied')
                json_filename = os.path.join(save_path, f"frame_{int(current_time)}.json")
                with open(json_filename, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)
    
    # 显示图像
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
