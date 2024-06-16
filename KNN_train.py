import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 数据路径
data_paths =[]
for i in range(6):
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
                feature_vector = []
                for lm in landmarks:
                    feature_vector.extend([lm["x"], lm["y"], lm["z"]])
                X.append(feature_vector)
                y.append(data["hand_type"])  # 使用手势类别标签
X = np.array(X)
y = np.array(y)

#将标签从字符串转换为整数编码（如果需要）
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 打乱数据并划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 训练KNN模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 评估模型
accuracy = knn.score(X_val, y_val)
print(f'Validation Accuracy: {accuracy:.2f}')

# 保存模型和标签编码器
joblib.dump(knn, 'knn_gesture_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
