import cv2
import mediapipe as mp
import numpy as np
import joblib

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 创建Hand对象
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# 加载训练好的KNN模型和标签编码器
knn = joblib.load('knn_gesture_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 设定距离阈值
distance_threshold = 0.5  # 根据需要调整此值

# 打开摄像头
cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
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
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 自定义绘制手部关键点和连接线
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # 提取并标准化关键点
            landmarks = [{
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            } for landmark in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
            
            # 转换为特征向量
            feature_vector = []
            for lm in normalized_landmarks:
                feature_vector.extend([lm[0], lm[1], lm[2]])
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # 使用KNN模型进行预测
            distances, indices = knn.kneighbors(feature_vector, n_neighbors=1)
            nearest_distance = distances[0][0]
            gesture_code = knn.predict(feature_vector)[0]
            
            # 检查预测结果是否在已知类别中
            if nearest_distance < distance_threshold:
                gesture_label = label_encoder.inverse_transform([gesture_code])[0]
                print('存在手势：', gesture_label)
                # 在图像上显示手势类别
                cv2.putText(frame, f'{gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                gesture_label = '-1'
            
            # 在图像上显示手势类别
            cv2.putText(frame, f'{gesture_label}', 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 显示图像
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
