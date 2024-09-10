import numpy as np
from tensorflow.keras.models import load_model
from dataset import load_data
import neptune
neptune_api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMWNjYjEzNi1lYzI5LTRjZWQtOWNiOC00NTEzOGNjZDFlZjcifQ==' 
neptune_project = 'lstm/food'  
run = neptune.init_run(
    api_token=neptune_api_token,
    project=neptune_project,
)
model = load_model('model_checkpoints/best_model.keras')

test_file = 'gaze_test_1.json' 
target_area = {
    "x1": 270, "y1": 1810,
    "x2": 270 + 550, "y2": 1810 + 150
}
max_length = 20
X_test, y_test = load_data('./dataset/gaze_test_1.json')

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

for i, prediction in enumerate(predictions):
    print(f"Sample {i+1} - Predicted: {prediction[0]}, Actual: {y_test[i]}")


run.stop()