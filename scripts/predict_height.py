import pandas as pd
import numpy as np
import os
import joblib

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# Đọc bảng tra cứu
lookup_df = pd.read_csv(os.path.join(data_dir, 'who_lookup.csv'))

# Đọc mô hình đã huấn luyện
haz_model = joblib.load(os.path.join(project_dir, 'haz_prediction_model.pkl'))

# Dữ liệu từ người dùng (ví dụ)
user_data = {
    'age_current': 90,  # ngày
    'htcm': 65.0,       # cm
    'wtkg': 4.5,        # kg
    'sex': 1,           # nữ
    'gagebrth': 280     # ngày
}

# Tính haz_current và waz_current từ bảng tra cứu
lookup_row = lookup_df[(lookup_df['Day'] == user_data['age_current']) & (lookup_df['sex'] == user_data['sex'])].iloc[0]
haz_current = (user_data['htcm'] - lookup_row['mean_height']) / lookup_row['std_height']
waz_current = (user_data['wtkg'] - lookup_row['mean_weight']) / lookup_row['std_weight']

# Dự đoán chiều cao tại các mốc thời gian
prediction_ages = [180, 365, 1095, 1825]  # 6 tháng, 1 năm, 3 năm, 5 năm
results = []
for age_future in prediction_ages:
    # Dự đoán haz
    haz_predicted = haz_model.predict([[user_data['age_current'], age_future, user_data['sex'], user_data['gagebrth'], haz_current, waz_current]])[0]
    
    # Tính chiều cao dự đoán (vẫn cần tra cứu WHO cho age_future)
    lookup_row_future = lookup_df[(lookup_df['Day'] == age_future) & (lookup_df['sex'] == user_data['sex'])].iloc[0]
    mean_height = lookup_row_future['mean_height']
    std_height = lookup_row_future['std_height']
    predicted_height = mean_height + (haz_predicted * std_height)
    results.append({
        'age_days': age_future,
        'predicted_height': round(predicted_height, 2)
    })

# In kết quả
for result in results:
    print(f"Tuổi: {result['age_days']} ngày, Chiều cao dự đoán: {result['predicted_height']} cm")