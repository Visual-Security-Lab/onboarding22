import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_eye_data(file):
    left_eye_x = []
    left_eye_y = []
    right_eye_x = []
    right_eye_y = []

    if not os.path.exists(file):
        print(f"{file} 불러오기 오류")
        return left_eye_x, left_eye_y, right_eye_x, right_eye_y

    with open(file, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            lmk = value['lmk']
            left_eye_x.append(lmk[0])
            left_eye_y.append(lmk[1])
            right_eye_x.append(lmk[2])
            right_eye_y.append(lmk[3])
    
    return left_eye_x, left_eye_y, right_eye_x, right_eye_y

def find_global_range(files):
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')
    
    for file in files:
        left_eye_x, left_eye_y, right_eye_x, right_eye_y = load_eye_data(file)
        if left_eye_x and right_eye_x and left_eye_y and right_eye_y:
            global_min_x = min(global_min_x, min(left_eye_x), min(right_eye_x))
            global_max_x = max(global_max_x, max(left_eye_x), max(right_eye_x))
            global_min_y = min(global_min_y, min(left_eye_y), min(right_eye_y))
            global_max_y = max(global_max_y, max(left_eye_y), max(right_eye_y))
        
    return global_min_x, global_max_x, global_min_y, global_max_y

files = [f'{i:02d}.json' for i in range(15)]

global_min_x, global_max_x, global_min_y, global_max_y = find_global_range(files)

fig, axs = plt.subplots(3, 5, figsize=(15, 15)) 
fig.suptitle('Visualization of Eye Landmarks for Each Frame')

i = 0  
for file in files:  
    if not os.path.exists(file):
        print(f"{file} 불러오기 오류")
        continue

    row = i // 5
    col = i % 5
    ax = axs[row, col]
    
    left_eye_x, left_eye_y, right_eye_x, right_eye_y = load_eye_data(file)
    if left_eye_x and right_eye_x and left_eye_y and right_eye_y:
        left_eye_x = np.array(left_eye_x)
        left_eye_y = np.array(left_eye_y)
        right_eye_x = np.array(right_eye_x)
        right_eye_y = np.array(right_eye_y)
        color_left = np.random.rand(3,)
        color_right = np.random.rand(3,)
        
        ax.scatter(left_eye_x, left_eye_y, color=color_left, label='Left Eye')
        ax.scatter(right_eye_x, right_eye_y, color=color_right, label='Right Eye')
        
        ax.set_title(file)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)
        
        ax.set_xlim(global_min_x, global_max_x)
        ax.set_ylim(global_min_y, global_max_y)
        
        ax.legend()
    
    i += 1  

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)  

plt.show()
