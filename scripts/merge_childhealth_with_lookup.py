import pandas as pd
import os

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# Đọc bảng tra cứu
lookup_df = pd.read_csv(os.path.join(data_dir, 'who_lookup.csv'))

# Đọc dữ liệu chính
df = pd.read_csv(os.path.join(data_dir, 'childhealth-dutch.csv'), na_values=['\\N'])

# Chuyển đổi cột sex
df['sex'] = df['sex'].map({'Female': 1, 'Male': 0})

# Loại bỏ các hàng có NaN trong các cột cần thiết
df = df.dropna(subset=['agedays', 'sex', 'htcm', 'wtkg', 'haz', 'waz', 'gagebrth'])

# Thêm các cột từ bảng tra cứu
df = df.merge(
    lookup_df,
    left_on=['agedays', 'sex'],
    right_on=['Day', 'sex'],
    how='left'
)

# Lưu dữ liệu đã cập nhật
df.drop(columns=['Day'], inplace=True)  # Bỏ cột Day từ lookup
df.to_csv(os.path.join(data_dir, 'childhealth_processed_with_zscores.csv'), index=False)
print("Đã cập nhật dữ liệu với haz và waz: childhealth_processed_with_zscores.csv")