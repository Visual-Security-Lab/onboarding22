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

fig, axs = plt.subplots(3, 5, figsize=(15, 15))  # 3x5 그리드의 서브플롯
fig.suptitle('Visualization for Each Frame')

files = [f'{i:02d}.json' for i in range(15)]  


for file in files: #한 figure에 파일 00~14까지 색상 다르개 해서 축 올림
    if not os.path.exists(file):
        print(f"{file} 불러오기 오류")
        continue
    i=0
    row = i // 5
    col = i % 3
    ax = axs[row, col]
    
    gaze_x, gaze_y = load_data([file])
    gaze_x = np.array(gaze_x)
    gaze_y = np.array(gaze_y)
    color = np.random.rand(3,)
    
    ax.scatter(gaze_x, gaze_y, color=color, label=file)
    for j in range(len(gaze_x)):
        ax.text(gaze_x[j], gaze_y[j], f'{j}', fontsize=8, ha='right')
    
    ax.set_title(\d i".json")
    ax.set_xlabel("gaze_x")
    ax.set_ylabel("gaze_y")
    ax.grid(True)
    i++

plt.title('Gaze Coordinates')
#plt.xlabel('X Coordinate')
#plt.ylabel('Y Coordinate')
#plt.legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
