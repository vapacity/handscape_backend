from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from io import BytesIO
from PIL import Image
import logging
import json
import os
import datetime
import time

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 加载训练好的 KNN 模型和标签编码器
knn = joblib.load('knn_gesture_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 设定距离阈值
distance_threshold = 0.5  # 根据需要调整此值

# 设定时间
get_new_folder_signal = 0 # 如果为1，则说明在record部分需要添加新文件夹
last_folder = None

def normalize_landmarks(landmarks):
    # 将关键点转换为相对于手部中心点的坐标系
    landmarks = np.array(landmarks)
    center = np.mean(landmarks, axis=0)
    normalized_landmarks = landmarks - center
    max_value = np.max(np.linalg.norm(normalized_landmarks, axis=1))
    normalized_landmarks /= max_value
    return normalized_landmarks
def get_last_num(base_path):
    max_num=-1
    for folder_name in os.listdir(base_path):
        if folder_name.startswith("data_"):
            try:
                num = int(folder_name.split("_")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
    return max_num

def get_new_folder_path(base_path):
    max_num = -1
    for folder_name in os.listdir(base_path):
        if folder_name.startswith("data_"):
            try:
                num = int(folder_name.split("_")[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
    new_folder_name = f"data_{max_num + 1}"
    new_folder_path = os.path.join(base_path, new_folder_name)
    return new_folder_path
@socketio.on('connect')
def handle_connect():
    global get_new_folder_signal
    print("客户端已连接")
    get_new_folder_signal = 1
    
@socketio.on('image')
def handle_image(data):
    start_time = time.time()  # 记录开始时间
    try:
        print('接收到的图像数据:', data['image'][:100])  # 只打印前100个字符以检查数据是否接收
        image_data = data['image'].split(',')[1]  # 去掉 base64 头部
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        frame = np.array(image)

        # 转换颜色通道 RGB 到 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 翻转图像
        frame = cv2.flip(frame, 1)

        # 检测手部
        results = hands.process(frame)
        gesture_label = '-1'

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 自定义绘制手部关键点和连接线
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # 提取并标准化关键点
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                normalized_landmarks = normalize_landmarks(landmarks)

                # 转换为特征向量
                feature_vector = np.array([coord for lm in normalized_landmarks for coord in lm]).reshape(1, -1)

                # 使用 KNN 模型进行预测
                distances, indices = knn.kneighbors(feature_vector, n_neighbors=1)
                nearest_distance = distances[0][0]
                gesture_code = knn.predict(feature_vector)[0]

                if nearest_distance < distance_threshold:
                    gesture_label = label_encoder.inverse_transform([gesture_code])[0]
                    cv2.putText(frame, f'{gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        gesture_label = int(gesture_label)

        data = {
            'gesture': gesture_label,
            'image': frame_data
        }

        json_data = json.dumps(data)
        emit('response_back', json_data)
        print('数据发送回客户端')
    except Exception as e:
        print(f'处理图像时出错: {e}')
    end_time = time.time()  # 记录结束时间
    processing_time = (end_time - start_time) * 1000  # 计算处理时间并转换为毫秒
    print(f'处理一帧图像所需时间: {processing_time:.2f} 毫秒')

@socketio.on('record')
def handle_record(data):
    global get_new_folder_signal, last_folder
    start_time = time.time()  # 记录开始时间
    try:
        print('接收到的图像数据:', data['image'][:100])  # 只打印前100个字符以检查数据是否接收
        image_data = data['image'].split(',')[1]  # 去掉 base64 头部
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        frame = np.array(image)
        timestamp =  time.time()
        # 转换颜色通道 RGB 到 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 翻转图像
        frame = cv2.flip(frame, 1)

        # 检测手部
        results = hands.process(frame)
        gesture_label = '-1'
        landmarks_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 自定义绘制手部关键点和连接线
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # 提取并标准化关键点
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                normalized_landmarks = normalize_landmarks(landmarks)

                # 转换为特征向量
                feature_vector = np.array([coord for lm in normalized_landmarks for coord in lm]).reshape(1, -1)

                # 保存特征点数据
                landmarks_data.append({
                    'landmarks': [{"x": lm[0], "y": lm[1], "z": lm[2]} for lm in normalized_landmarks],
                    'handtype': 6#需更改
                })

        # 获取新的存储路径
        # base_path = "my_gestures"
        # new_folder_path = get_new_folder_path(base_path)
        # os.makedirs(new_folder_path, exist_ok=True)
        folder_path = None
        if get_new_folder_signal ==1:
            folder_path = get_new_folder_path('my_gestures')
            last_folder = folder_path
            os.makedirs(folder_path, exist_ok=True)
            get_new_folder_signal = 0
        else:
            folder_path = last_folder
        print('数据保存到',folder_path)
        # 保存原始图像
        image_filename = os.path.join(folder_path, f"frame_{timestamp}.png")
        image.save(image_filename, "PNG")

        # 保存特征点数据
        landmarks_filename = os.path.join(folder_path, f"frame_{timestamp}.json")
        with open(landmarks_filename, 'w') as f:
            json.dump(landmarks_data, f)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        gesture_label = int(gesture_label)

        data = {
            'gesture': gesture_label,
            'image': frame_data,
        }

        json_data = json.dumps(data)
        emit('response_back', json_data)
        print('数据发送回客户端')
    except Exception as e:
        print(f'处理图像时出错: {e}')
    end_time = time.time()  # 记录结束时间
    processing_time = (end_time - start_time) * 1000  # 计算处理时间并转换为毫秒
    print(f'处理一帧图像所需时间: {processing_time:.2f} 毫秒')


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
@socketio.on('train_and_get_meaning')
def handel_meaning(data):
    # 数据路径
    max_num = get_last_num(base_path='my_gestures')
    data_paths =[]
    for i in range(max_num+1):
        data_paths.append(f"my_gestures\data_{i}")
    
    # 加载数据
    X = []
    y = []
    for data_path in data_paths:
        for filename in os.listdir(data_path):
            if filename.endswith(".json"):
                with open(os.path.join(data_path, filename), 'r') as f:
                    data = json.load(f)
                    # 提取关键点信息
                    landmarks = data["landmarks"]
                    if landmarks 
                    feature_vector = []
                    for lm in landmarks:
                        feature_vector.extend([lm["x"], lm["y"], lm["z"]])
                    X.append(feature_vector)
                    y.append(data["hand_type"])  # 使用手势类别标签
    X = np.array(X)
    y = np.array(y)
    pass

@socketio.on_error()
def error_handler(e):
    print(f'An error occurred: {e}')

@socketio.on_error_default
def default_error_handler(e):
    print(f'Default error handler: {e}')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
