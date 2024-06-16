import json
import os

jsonpath = 'my_gestures/data_6/frame_1718520941.8771842.json'

with open(jsonpath, 'r') as f:
    data = json.load(f)
    # 提取关键点信息
    landmarks = data["landmarks"]
    feature_vector = []
    for lm in landmarks:
        feature_vector.extend([lm["x"], lm["y"], lm["z"]])
    print(feature_vector)
    print(data['handtype'])
                        