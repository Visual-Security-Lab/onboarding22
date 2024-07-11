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

def find_global_range(files):
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')
    
    for file in files:
        if not os.path.exists(file):
            continue
        
        gaze_x, gaze_y = load_data([file])
        global_min_x = min(global_min_x, min(gaze_x))
        global_max_x = max(global_max_x, max(gaze_x))
        global_min_y = min(global_min_y, min(gaze_y))
        global_max_y = max(global_max_y, max(gaze_y))
        
    return global_min_x, global_max_x, global_min_y, global_max_y

files = [f'{i:02d}.json' for i in range(15)]

global_min_x, global_max_x, global_min_y, global_max_y = find_global_range(files)

fig, axs = plt.subplots(3, 5, figsize=(15, 15)) 
fig.suptitle('Visualization for Each Frame')

i = 0  
for file in files:  
    if not os.path.exists(file):
        print(f"{file} 불러오기 오류")
        continue

    row = i // 5
    col = i % 5
    ax = axs[row, col]
    
    gaze_x, gaze_y = load_data([file])
    gaze_x = np.array(gaze_x)
    gaze_y = np.array(gaze_y)
    color = np.random.rand(3,)
    
    ax.scatter(gaze_x, gaze_y, color=color, label=file)
    
    ax.set_title(file)
    ax.set_xlabel("gaze_x")
    ax.set_ylabel("gaze_y")
    ax.grid(True)
    
    ax.set_xlim(global_min_x, global_max_x)
    ax.set_ylim(global_min_y, global_max_y)
    
    i += 1  

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)  

plt.show()
