import os
import json
import csv

sales_006 = [
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0]
]

exp_folder = './../data'
sales_folder = './../data/Data_prepare/sales'
output_folder = './gazeData00'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

gaze_csv_path = './../data/Data_prepare/gazedata.csv'
gaze_data_rows = []

with open(gaze_csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        gaze_data_rows.append(row)

sales_files = sorted([f for f in os.listdir(sales_folder) if f.endswith('.csv')])

for idx, row in enumerate(gaze_data_rows):
    gaze_file_path = os.path.join(exp_folder, row['raw_gaze_path'])
    sales_file_name = f"COUNT_{idx:03d}.csv"
    if sales_file_name in sales_files:
        sales_file_path = os.path.join(sales_folder, sales_file_name)
        try:
            with open(sales_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                sales_data = [list(map(int, row)) for row in reader]
        except FileNotFoundError:
            print(f"File not found: {sales_file_path}")
            continue
        except ValueError:
            print(f"Invalid data format in sales file: {sales_file_path}")
            continue
    else:
        if idx == 6:
            print(f"Using predefined sales_006 data for gaze file at index {idx}.")
            sales_data = sales_006
        else:
            print(f"No matching sales file for gaze file at index {idx}. Skipping this gaze file.")
            continue

    try:
        with open(gaze_file_path, 'r', encoding='utf-8') as f:
            gaze_data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {gaze_file_path}")
        continue

    filtered_data = []
    for page_data in gaze_data:
        if page_data['page'] == 'menu_detail':
            s_num = page_data['s_num'] - 1
            f_num = page_data['f_num'] - 1

            try:
                success = sales_data[s_num][f_num] == 1
            except IndexError:
                success = False  
            filtered_data.append({
                'index': idx,
                's_num': s_num,
                'f_num': f_num,
                'gaze': page_data['gaze'],
                'success': success
            })

    output_file = os.path.join(output_folder, f'processed_{idx:03d}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Processed {gaze_file_path} and saved to {output_file}")
