import json
import matplotlib as plt

for i in range(15):
# JSON 파일 로드
    file_path = f'/DLData/vc-onboarding/MPIIFaceGaze_normalized_folder/labels/{i:02d}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

# 5x5 그리드의 서브플롯으로 변경
fig, axs = plt.subplots(3, 5, figsize=(15, 15))  # 5x5 그리드의 서브플롯
fig.suptitle('Landmark Visualization for Each Frame')

for i, (frame, details) in enumerate(data.items()):
    if i >= 15:
        break
    row = i // 5
    col = i % 3
    ax = axs[row, col]
    
    lmk = details["lmk"]
    x_coords = lmk[0::2]
    y_coords = lmk[1::2]
    
    ax.scatter(x_coords, y_coords, color='red')
        #for j in range(len(x_coords)):
        #    ax.text(x_coords[j], y_coords[j], f'{j}', fontsize=8, ha='right')
    
    ax.set_title(f"Frame {frame}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
