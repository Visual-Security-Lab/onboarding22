import os
import json

processed_folder = './gazeData00' 
output_folder = './filteredData' 

true_folder = os.path.join(output_folder, 'true')
false_folder = os.path.join(output_folder, 'false')

os.makedirs(true_folder, exist_ok=True)
os.makedirs(false_folder, exist_ok=True)

processed_files = sorted([f for f in os.listdir(processed_folder) if f.endswith('.json')])

for processed_file in processed_files:
    processed_file_path = os.path.join(processed_folder, processed_file)

    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {processed_file_path}")
        continue

    for i, item in enumerate(data):
        gaze = item['gaze']
        success = item['success']

        if not gaze:
            print(f"Skipping empty gaze data for {processed_file} at index {i}")
            continue

        if success:
            save_folder = true_folder
        else:
            save_folder = false_folder

        output_file = os.path.join(save_folder, f"{processed_file[:-5]}_part_{i}.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=4)

        print(f"Saved {output_file}")
