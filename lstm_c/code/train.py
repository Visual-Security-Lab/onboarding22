import os
import neptune
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import load_data


neptune_api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMWNjYjEzNi1lYzI5LTRjZWQtOWNiOC00NTEzOGNjZDFlZjcifQ==' 
neptune_project = 'lstm/food'  
run = neptune.init_run(
    api_token=neptune_api_token,
    project=neptune_project,
)

X_train, y_train = load_data('./dataset/gaze_train_1.json')
X_test, y_test = load_data('./dataset/gaze_test_1.json')

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 체크포인트 콜백 설정
checkpoint_path = "model_checkpoints/best_model.keras"

# 디렉토리 생성
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# 체크포인트 콜백
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

class NeptuneCallback(neptune.keras.NeptuneCallback):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        run["train/epoch"].log(epoch)
        run["train/accuracy"].log(logs["accuracy"])
        run["train/loss"].log(logs["loss"])
        run["val/accuracy"].log(logs["val_accuracy"])
        run["val/loss"].log(logs["val_loss"])

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[checkpoint_callback, NeptuneCallback(run)]  # Neptune 콜백 추가
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

run.stop()
