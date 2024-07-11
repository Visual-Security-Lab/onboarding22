import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data(files):
    gaze_x = []
    gaze_y = []
    pose_x = []
    pose_y = []

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                gaze_x.append(value['gaze'][0])
                gaze_y.append(value['gaze'][1])
                pose_x.append(value['pose'][0])
                pose_y.append(value['pose'][1])
    
    return gaze_x, gaze_y, pose_x, pose_y


files = [f'{i:02d}.json' for i in range(1)]#파일수따라 변경할것


gaze_x, gaze_y, pose_x, pose_y = load_data(files)


gaze_x = np.array(gaze_x)
gaze_y = np.array(gaze_y)
pose_x = np.array(pose_x)
pose_y = np.array(pose_y)

X = np.vstack((pose_x, pose_y)).T
y = np.vstack((gaze_x, gaze_y)).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

gaze_mean = [np.mean(gaze_x), np.mean(gaze_y)]
pose_mean = [np.mean(pose_x), np.mean(pose_y)]


plt.figure(figsize=(10, 5))

plt.scatter(gaze_x, gaze_y, color='blue', label='Actual Gaze Data')
plt.scatter(gaze_mean[0], gaze_mean[1], color='red', marker='x', s=100, label='Mean Gaze')

plt.scatter(pose_x, pose_y, color='green', label='Pose Data')
plt.scatter(pose_mean[0], pose_mean[1], color='orange', marker='x', s=100, label='Mean Pose')

plt.scatter(y_pred[:, 0], y_pred[:, 1], color='purple', label='Predicted Gaze Data', alpha=0.5)

plt.title('Scatter Plot of Gaze and Pose Data with Predictions')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

plt.show()
