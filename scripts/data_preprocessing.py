import pandas as pd
import numpy as np
import os

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

def load_and_preprocess_data(file_path=os.path.join(data_dir, 'childhealth-dutch.csv')):
    """
    Tải dữ liệu, xử lý giá trị thiếu, mã hóa biến và tạo đặc trưng mới.

    Args:
        file_path (str): Đường dẫn đến file CSV.

    Returns:
        pandas.DataFrame: DataFrame đã được tiền xử lý.
    """
    print(f"Đang tải dữ liệu từ: {file_path}")
    # Đọc file, nhận diện '\N' là NaN
    df = pd.read_csv(file_path, na_values='\\N')
    print(f"Số dòng ban đầu: {len(df)}")

    # --- Xử lý giá trị thiếu ---
    initial_rows = len(df)
    df.dropna(inplace=True) # Bỏ tất cả các hàng có giá trị NA
    rows_dropped = initial_rows - len(df)
    print(f"Đã loại bỏ {rows_dropped} hàng chứa giá trị NA.")
    print(f"Số dòng sau khi loại bỏ NA: {len(df)}")

    if df.empty:
        raise ValueError("Không còn dữ liệu sau khi loại bỏ NA. Kiểm tra lại file gốc hoặc chiến lược xử lý NA.")

    # --- Mã hóa biến 'sex' ---
    df['sex_encoded'] = df['sex'].map({'Female': 1, 'Male': 0})
    print("Đã mã hóa cột 'sex' (Female=1, Male=0) thành 'sex_encoded'.")

    # --- Tạo đặc trưng 'is_premature' ---
    # Sinh non thường được định nghĩa là < 37 tuần thai (37 * 7 = 259 ngày)
    premature_threshold = 259
    df['is_premature'] = (df['gagebrth'] < premature_threshold).astype(int)
    print(f"Đã tạo cột 'is_premature' (1 nếu gagebrth < {premature_threshold}, 0 nếu ngược lại).")

    # --- Đảm bảo kiểu dữ liệu số ---
    numeric_cols = ['agedays', 'gagebrth', 'htcm', 'wtkg', 'haz', 'waz']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # Chuyển đổi, ép lỗi thành NaN (sẽ bị drop ở bước sau nếu chưa)

    # Kiểm tra lại NA sau khi ép kiểu (phòng trường hợp 'coerce' tạo ra NaN)
    df.dropna(subset=numeric_cols + ['sex_encoded', 'is_premature'], inplace=True)
    print(f"Số dòng sau khi đảm bảo kiểu số và kiểm tra NA lần cuối: {len(df)}")

    return df

def classify_waz(waz_score):
    """Phân loại tình trạng dinh dưỡng dựa trên WAZ score."""
    if waz_score < -3:
        return 'Severely Underweight' # Thiếu cân nặng
    elif waz_score < -2:
        return 'Underweight'        
    else:
        return 'Normal'             

def classify_haz(haz_score):
    """Phân loại tình trạng dinh dưỡng dựa trên HAZ score."""
    if haz_score < -3:
        return 'Severely stunted' 
    elif haz_score < -2:
        return 'Stunted'        
    else:
        return 'Trẻ bình thường'             

def add_nutrition_status_columns(df):
    """Thêm các cột phân loại tình trạng dinh dưỡng vào DataFrame."""
    if 'waz' in df.columns:
        df['nutrition_status_waz'] = df['waz'].apply(classify_waz)
        print("Đã tạo cột 'nutrition_status_waz'.")
    else:
         print("Cảnh báo: Không tìm thấy cột 'waz' để tạo 'nutrition_status_waz'.")

    if 'haz' in df.columns:
        df['nutrition_status_haz'] = df['haz'].apply(classify_haz)
        print("Đã tạo cột 'nutrition_status_haz'.")
    else:
        print("Cảnh báo: Không tìm thấy cột 'haz' để tạo 'nutrition_status_haz'.")
    return df

# Có thể thêm phần chạy thử nghiệm nếu muốn
if __name__ == "__main__":
    try:
        df_processed = load_and_preprocess_data()
        df_with_status = add_nutrition_status_columns(df_processed.copy()) # Dùng copy để tránh thay đổi df_processed gốc
        print("\n5 dòng đầu sau khi tiền xử lý và thêm cột trạng thái:")
        print(df_with_status.head())
        print("\nThông tin dữ liệu sau xử lý:")
        df_with_status.info()
        print("\nPhân bố trạng thái WAZ:")
        print(df_with_status['nutrition_status_waz'].value_counts(normalize=True))
        print("\nPhân bố trạng thái HAZ:")
        print(df_with_status['nutrition_status_haz'].value_counts(normalize=True))
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'childhealth-dutch.csv'. Đặt file vào cùng thư mục.")
    except ValueError as e:
        print(f"Lỗi xử lý dữ liệu: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")