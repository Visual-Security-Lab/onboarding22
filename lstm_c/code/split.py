import os
import json
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

true_folder = './filteredData/true'
false_folder = './filteredData/false'

from dataset import preprocess_gaze_data

def load_data_from_folder(folder_path, target_area, max_length, label):
    X = []
    y = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            gaze_data = json.load(file)
            gaze_data = gaze_data["gaze"]
            if len(gaze_data) > 1:  
                preprocessed_data = preprocess_gaze_data(gaze_data, target_area, max_length)
                X.append(preprocessed_data.tolist()) 
                y.append(label) 
                
    return X, y

target_area = {
    "x1": 270, "y1": 1810,
    "x2": 270+550, "y2": 1810+150
}
max_length = 20 

X_true, y_true = load_data_from_folder(true_folder, target_area, max_length, label=1)
X_false, y_false = load_data_from_folder(false_folder, target_area, max_length, label=0)

X = X_true + X_false
y = y_true + y_false

X, y = shuffle(X, y, random_state=42)

test_size = 0.2
num_splits = 5 

for split_idx in range(num_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=split_idx)

    train_file_path = f'./dataset/gaze_train_{split_idx + 1}.json'
    test_file_path = f'./dataset/gaze_test_{split_idx + 1}.json'
    train_data = [{'gaze': X_train[i], 'success': y_train[i]} for i in range(len(X_train))]
    with open(train_file_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    test_data = [{'gaze': X_test[i], 'success': y_test[i]} for i in range(len(X_test))]
    with open(test_file_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"Saved train/test split {split_idx + 1} to {train_file_path} and {test_file_path}")
