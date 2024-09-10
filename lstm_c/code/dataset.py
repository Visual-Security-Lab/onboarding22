import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. 시간 차이 계산
def calculate_time_differences(gaze_data):
    time_diffs = []
    if len(gaze_data) < 2:
        return time_diffs  # 시퀀스 길이가 2 미만인 경우 빈 리스트 반환
    for i in range(1, len(gaze_data)):
        time_diff = gaze_data[i]['t'] - gaze_data[i-1]['t']  # 현재와 이전 좌표의 시간 차이 계산
        time_diffs.append(time_diff)
    return time_diffs

# 2. 특정 영역에 머문 시간 계산
def calculate_focus_time(gaze_data, target_area):
    focus_time = 0
    for i in range(1, len(gaze_data)):
        if target_area['x1'] <= gaze_data[i]['x'] <= target_area['x2'] and target_area['y1'] <= gaze_data[i]['y'] <= target_area['y2']:
            focus_time += gaze_data[i]['t'] - gaze_data[i-1]['t']  # 영역 내에 머문 시간 합산
    return focus_time

# 3. 시선 이동 속도 계산
def calculate_gaze_speed(gaze_data):
    speeds = []
    if len(gaze_data) < 2:
        return speeds  # 시퀀스 길이가 2 미만인 경우 빈 리스트 반환
    for i in range(1, len(gaze_data)):
        distance = np.sqrt((gaze_data[i]['x'] - gaze_data[i-1]['x'])**2 + (gaze_data[i]['y'] - gaze_data[i-1]['y'])**2)
        time_diff = gaze_data[i]['t'] - gaze_data[i-1]['t']
        if time_diff > 0:  # time_diff가 0인 경우 속도를 계산하지 않음
            speed = distance / time_diff
            speeds.append(speed)
    return speeds

# 4. 패딩을 통해 시퀀스 길이를 동일하게 맞춤 (짧은 시퀀스는 0으로 채움)
def pad_gaze_sequences(gaze_features, max_length):
    padded_sequences = pad_sequences(gaze_features, maxlen=max_length, padding='post', dtype='float32')
    return padded_sequences

# 5. 정규화
def normalize_features(gaze_features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(gaze_features)
    return normalized_features

# 6. 전체 전처리 과정
def preprocess_gaze_data(gaze_data, target_area, max_length):
    if len(gaze_data) < 2:
        return []  # gaze 데이터가 2개 미만이면 빈 배열 반환
    
    # 1) 시간 차이 계산
    time_diffs = calculate_time_differences(gaze_data)
    
    # 2) 특정 영역에 머문 시간 계산 (예: 구매 버튼 영역)
    focus_time = calculate_focus_time(gaze_data, target_area)
    
    # 3) 시선 이동 속도 계산
    gaze_speeds = calculate_gaze_speed(gaze_data)
    
    # 4) 각 세션의 피처를 시퀀스 형태로 정리
    if len(time_diffs) == 0 or len(gaze_speeds) == 0:
        return []  # 유효한 데이터가 없을 경우 빈 배열 반환

    gaze_features = np.column_stack((time_diffs, gaze_speeds, [focus_time] * len(time_diffs)))
    
    # 5) 정규화
    normalized_features = normalize_features(gaze_features)
    
    # 6) 시퀀스 길이 맞추기 (패딩)
    padded_sequences = pad_gaze_sequences([normalized_features], max_length=max_length)
    
    return padded_sequences


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    X = []
    y = []

    for entry in data:
        gaze_data = entry['gaze']
        success = entry['success']

        if gaze_data: 
            X.append(gaze_data)
            y.append(success)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int')
    if X.ndim == 4:
        X = np.squeeze(X, axis=1)
    
    return X, y