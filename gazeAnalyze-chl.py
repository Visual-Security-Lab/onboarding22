import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(files):
    gaze_x = []
    gaze_y = []

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                gaze_x.append(value['gaze'][0])
                gaze_y.append(value['gaze'][1])
    
    return gaze_x, gaze_y


files = [f'{i:02d}.json' for i in range(15)]  

plt.figure(figsize=(10, 5))

for file in files: #한 figure에 파일 00~14까지 색상 다르개 해서 축 올림
    if not os.path.exists(file):
        print(f"{file} 불러오기 오류")
        continue
    
    gaze_x, gaze_y = load_data([file])
    gaze_x = np.array(gaze_x)
    gaze_y = np.array(gaze_y)
    color = np.random.rand(3,)
    
    plt.scatter(gaze_x, gaze_y, color=color, label=file)

plt.title('Gaze Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

plt.show()
