import os
import pandas as pd

# Thư mục chứa các file .dta
input_folder = "D:/Child_height_prediction/data/hh14_all_dta"
# Thư mục muốn lưu file .csv
output_folder = "D:/Child_height_prediction/data/csv"

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Lặp qua tất cả các file trong thư mục input
for filename in os.listdir(input_folder):
    if filename.endswith(".dta"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".csv"
        output_path = os.path.join(output_folder, output_filename)
        try:
            df = pd.read_stata(input_path)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"✅ Chuyển {filename} thành {output_filename}")
        except Exception as e:
            print(f"❌ Lỗi với file {filename}: {e}")