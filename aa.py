import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 바이너리 모드로 파일 읽기
file_path = './00.json'
with open(file_path, 'rb') as file:
    binary_data = file.read()

# null byte 제거
binary_data = binary_data.replace(b'\x00', b'')

# 문자열로 변환한 후 JSON 파싱
data_str = binary_data.decode('utf-8')
data = json.loads(data_str)

# 데이터를 DataFrame으로 변환
frames = []
for key, value in data.items():
    frame_id = key
    gaze_x, gaze_y = value['gaze']
    pose_x, pose_y = value['pose']
    frames.append([frame_id, gaze_x, gaze_y, pose_x, pose_y])

df = pd.DataFrame(frames, columns=['frame_id', 'gaze_x', 'gaze_y', 'pose_x', 'pose_y'])

# 시각화
plt.figure(figsize=(14, 6))

# 시선 데이터 시각화
plt.subplot(1, 2, 1)
sns.scatterplot(x='frame_id', y='gaze_x', data=df, label='Gaze X')
sns.scatterplot(x='frame_id', y='gaze_y', data=df, label='Gaze Y')
plt.title('Gaze Data')
plt.xlabel('Frame ID')
plt.ylabel('Gaze Value')
plt.xticks(rotation=90)
plt.legend()

# 자세 데이터 시각화
plt.subplot(1, 2, 2)
sns.scatterplot(x='frame_id', y='pose_x', data=df, label='Pose X')
sns.scatterplot(x='frame_id', y='pose_y', data=df, label='Pose Y')
plt.title('Pose Data')
plt.xlabel('Frame ID')
plt.ylabel('Pose Value')
plt.xticks(rotation=90)
plt.legend()

plt.tight_layout()
plt.show()
