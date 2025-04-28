import pandas as pd
import os

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')

# Đọc file CSV
data = pd.read_csv(os.path.join(data_dir, 'childhealth-dutch-male.csv'))

# Thêm cột is_premature (1 nếu gagebrth < 259 ngày, 0 nếu không)
data['is_premature'] = (data['gagebrth'] < 259).astype(int)

# Lưu dữ liệu vào file mới
data.to_csv(os.path.join(data_dir, 'childhealth-dutch-male-updated.csv'), index=False)
print("Đã thêm cột is_premature và lưu vào childhealth-dutch-male-updated.csv")