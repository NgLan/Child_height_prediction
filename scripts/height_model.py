import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # Scaling có thể cải thiện RF một chút
from sklearn.pipeline import Pipeline

# Import các hàm từ file khác
from data_preprocessing import load_and_preprocess_data
from plotting import plot_regression_results, plot_feature_vs_prediction

def train_evaluate_height_model(df, gender_code, gender_name):
    """
    Huấn luyện và đánh giá mô hình dự đoán chiều cao cho một giới tính cụ thể.

    Args:
        df (pandas.DataFrame): DataFrame đã được tiền xử lý.
        gender_code (int): 0 cho Male, 1 cho Female.
        gender_name (str): Tên giới tính ('Male' hoặc 'Female').

    Returns:
        tuple: (model, X_test, y_test, y_pred) nếu thành công, None nếu không đủ dữ liệu.
    """
    print(f"\n--- Bắt đầu xử lý cho giới tính: {gender_name} ---")
    df_gender = df[df['sex_encoded'] == gender_code].copy()

    if len(df_gender) < 10: # Kiểm tra số lượng mẫu tối thiểu
        print(f"Không đủ dữ liệu cho {gender_name} (chỉ có {len(df_gender)} mẫu). Bỏ qua.")
        return None

    # Chọn Features và Target
    # KHÔNG dùng 'haz', 'waz', 'nutrition_status_...'
    # KHÔNG cần dùng 'sex' hoặc 'sex_encoded' vì đã lọc theo giới tính
    features = ['agedays', 'wtkg', 'gagebrth', 'is_premature']
    target = 'htcm'

    X = df_gender[features]
    y = df_gender[target]

    # Chia dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Số mẫu huấn luyện ({gender_name}): {len(X_train)}")
    print(f"Số mẫu kiểm tra ({gender_name}): {len(X_test)}")

    # Xây dựng Pipeline (Scaling + Model)
    # Scaling không bắt buộc cho RF nhưng có thể hữu ích và là thực hành tốt
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])


    # Huấn luyện mô hình
    print(f"Đang huấn luyện mô hình hồi quy cho {gender_name}...")
    pipeline.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred = pipeline.predict(X_test)

    # Đánh giá mô hình
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nKết quả đánh giá mô hình ({gender_name}):")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} cm")
    print(f"  R-squared (R2): {r2:.4f}")

    # Vẽ biểu đồ
    print(f"Đang vẽ biểu đồ cho {gender_name}...")
    plot_regression_results(y_test, y_pred, gender_name)
    # Vẽ biểu đồ với đặc trưng 'agedays'
    plot_feature_vs_prediction(X_test, y_test, y_pred, 'agedays', 'Chiều cao (cm)', gender_name)
     # Vẽ biểu đồ với đặc trưng 'wtkg'
    plot_feature_vs_prediction(X_test, y_test, y_pred, 'wtkg', 'Chiều cao (cm)', gender_name)


    return pipeline, X_test, y_test, y_pred

# --- Hàm chính để chạy ---
if __name__ == "__main__":
    try:
        # Tải và tiền xử lý dữ liệu
        df_processed = load_and_preprocess_data()

        # Huấn luyện và đánh giá cho Nữ (Female = 1)
        result_female = train_evaluate_height_model(df_processed, 1, 'Nữ')

        # Huấn luyện và đánh giá cho Nam (Male = 0)
        result_male = train_evaluate_height_model(df_processed, 0, 'Nam')

        # (Tùy chọn) Lưu mô hình nếu cần
        # from joblib import dump
        # if result_female:
        #     dump(result_female[0], 'height_model_female.joblib')
        # if result_male:
        #     dump(result_male[0], 'height_model_male.joblib')
        # print("\nĐã lưu các mô hình (nếu có đủ dữ liệu).")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'childhealth-dutch.csv'.")
    except ValueError as e:
        print(f"Lỗi xử lý dữ liệu: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")