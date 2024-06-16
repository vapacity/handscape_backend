import json
import os

def is_json_empty(file_path):
    if not os.path.exists(file_path):
        return True  # 如果文件不存在，则视为空文件

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return True  # 如果文件无法解析为JSON，视为空文件

        return not bool(data)  # 检查data是否为空，如果为空则返回True，否则返回False

# 示例用法
file_path = 'my_gestures/data_8/frame_1718507567.6693258.json'
if is_json_empty(file_path):
    print(f"The JSON file {file_path} is empty.")
else:
    print(f"The JSON file {file_path} is not empty.")
