import pandas as pd
import os

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# Đọc file WHO và tạo bảng tra cứu
lookup_data = []
for sex in [0, 1]:
    # Chiều cao
    height_file = os.path.join(data_dir, f'WHO-lhfa-{"boys" if sex == 0 else "girls"}-zscore-expanded-tables-0-5y.xlsx')
    if os.path.exists(height_file):
        height_df = pd.read_excel(height_file)
        height_df['sex'] = sex
        height_df['mean_height'] = height_df['M']
        height_df['std_height'] = height_df['S'] * height_df['M']
    
    # Cân nặng (giả định bạn có file WHO cho cân nặng)
    weight_file = os.path.join(data_dir, f'WHO-wfa-{"boys" if sex == 0 else "girls"}-zscore-expanded-tables-0-5y.xlsx')
    if os.path.exists(weight_file):
        weight_df = pd.read_excel(weight_file)
        weight_df['sex'] = sex
        weight_df['mean_weight'] = weight_df['M']
        weight_df['std_weight'] = weight_df['S'] * weight_df['M']
    
    # Kết hợp dữ liệu chiều cao và cân nặng
    merged_df = height_df[['Day', 'sex', 'mean_height', 'std_height']].merge(
        weight_df[['Day', 'sex', 'mean_weight', 'std_weight']],
        on=['Day', 'sex']
    )
    lookup_data.append(merged_df)

# Gộp dữ liệu và lưu vào file CSV
lookup_df = pd.concat(lookup_data, ignore_index=True)
lookup_df.to_csv(os.path.join(data_dir, 'who_lookup.csv'), index=False)
print("Đã tạo bảng tra cứu: who_lookup.csv")