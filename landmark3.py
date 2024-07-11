import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(files):
    lmk_x = []
    lmk_y = []

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                lmk_x.append(value['lmk'][0::2])
                lmk_y.append(value['lmk'][1::2])
    
    return lmk_x, lmk_y

def find_global_range(files):
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')
    
    for file in files:
        if not os.path.exists(file):
            continue
        
        lmk_x, lmk_y = load_data([file])
        global_min_x = min( min(lmk_x))
        global_max_x = max(max(lmk_x))
        global_min_y = min(min(lmk_y))
        global_max_y = max( max(lmk_y))
        
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
    
    lmk_x, lmk_y = load_data([file])
    lmk_x = np.array(lmk_x)
    lmk_y = np.array(lmk_y)
    color = np.random.rand(3,)
    
    ax.scatter(lmk_x, lmk_y, color=color, label=file)
    
    ax.set_title(file)
    ax.set_xlabel("lmk_x")
    ax.set_ylabel("lmk_y")
    ax.grid(True)
    
    ax.set_xlim(global_min_x, global_max_x)
    ax.set_ylim(global_min_y, global_max_y)
    
    i += 1  

plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)  

plt.show()
