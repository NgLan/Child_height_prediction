import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder # OneHotEncoder có thể không cần nếu chỉ dùng _encoded
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Cần thiết nếu bỏ NA ở bước đầu nhưng scaler/encoder cần input đầy đủ

# Import các hàm từ file khác
from data_preprocessing import load_and_preprocess_data, add_nutrition_status_columns
from plotting import plot_classification_confusion_matrix

def train_evaluate_nutrition_model(df, target_column, model_name):
    """
    Huấn luyện và đánh giá mô hình phân loại tình trạng dinh dưỡng.

    Args:
        df (pandas.DataFrame): DataFrame đã tiền xử lý và có cột target phân loại.
        target_column (str): Tên cột mục tiêu ('nutrition_status_waz' hoặc 'nutrition_status_haz').
        model_name (str): Tên mô hình để hiển thị (ví dụ: 'Phân loại WAZ').

    Returns:
        tuple: (model, X_test, y_test, y_pred) nếu thành công, None nếu không đủ dữ liệu.
    """
    print(f"\n--- Bắt đầu xử lý cho: {model_name} ---")

    if target_column not in df.columns:
        print(f"Lỗi: Không tìm thấy cột mục tiêu '{target_column}' trong DataFrame.")
        return None

    # Chọn Features và Target
    # KHÔNG dùng 'haz', 'waz', hoặc các cột nutrition_status khác làm feature
    # Bao gồm cả 'sex_encoded' và 'is_premature'
    features = ['sex_encoded', 'agedays', 'gagebrth', 'is_premature', 'htcm', 'wtkg']
    target = target_column

    # Loại bỏ các hàng mà target bị thiếu (nếu còn sót)
    df_model = df.dropna(subset=[target] + features).copy()

    if len(df_model) < 10:
        print(f"Không đủ dữ liệu cho {model_name} sau khi lọc (chỉ có {len(df_model)} mẫu). Bỏ qua.")
        return None

    X = df_model[features]
    y = df_model[target]

    # Lấy danh sách các lớp và kiểm tra số lớp
    classes = sorted(y.unique())
    if len(classes) < 2:
        print(f"Chỉ tìm thấy {len(classes)} lớp trong cột target '{target}'. Cần ít nhất 2 lớp để phân loại. Bỏ qua.")
        return None
    print(f"Các lớp tìm thấy cho {model_name}: {classes}")


    # Chia dữ liệu Train/Test (Stratify rất quan trọng do lớp có thể không cân bằng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Số mẫu huấn luyện ({model_name}): {len(X_train)}")
    print(f"Số mẫu kiểm tra ({model_name}): {len(X_test)}")

    # --- Định nghĩa Pipeline tiền xử lý ---
    # Các cột cần scaling
    numeric_features_scale = ['agedays', 'gagebrth', 'htcm', 'wtkg']
    # Các cột số nhị phân (đã là 0/1) không cần scale, chỉ cần đảm bảo không có NA
    numeric_features_passthrough = ['sex_encoded', 'is_premature']

    # Tạo transformer cho các cột cần scale
    numeric_transformer_scale = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Đề phòng NA nếu có
        ('scaler', StandardScaler())
    ])

    # Tạo transformer cho các cột không scale (chỉ imputer nếu cần)
    # Nếu chắc chắn không có NA ở đây (do dropna ban đầu) thì có thể dùng 'passthrough'
    numeric_transformer_passthrough = Pipeline(steps=[
         ('imputer', SimpleImputer(strategy='median')) # An toàn hơn là có imputer
         # Hoặc: ('passthrough', 'passthrough') # Nếu chắc chắn không có NA
    ])


    # Kết hợp bằng ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scale', numeric_transformer_scale, numeric_features_scale),
            ('num_pass', numeric_transformer_passthrough, numeric_features_passthrough)
            # Không có categorical features ở đây vì 'sex' đã thành 'sex_encoded'
        ],
        remainder='passthrough' # Giữ lại các cột không được chỉ định nếu có (an toàn)
        )

    # --- Xây dựng Pipeline hoàn chỉnh (Preprocessor + Model) ---
    # Sử dụng class_weight='balanced' để xử lý mất cân bằng lớp
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100,
                                              random_state=42,
                                              class_weight='balanced',
                                              n_jobs=-1))
    ])

    # Huấn luyện mô hình
    print(f"Đang huấn luyện mô hình {model_name}...")
    model_pipeline.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred = model_pipeline.predict(X_test)
    # y_pred_proba = model_pipeline.predict_proba(X_test) # Lấy xác suất nếu cần cho ROC AUC

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=classes, zero_division=0)
    # conf_matrix = confusion_matrix(y_test, y_pred, labels=classes) # Lấy matrix để vẽ

    print(f"\nKết quả đánh giá ({model_name}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(report)
    # print("\n  Confusion Matrix:") # Sẽ được vẽ bởi hàm plotting
    # print(conf_matrix)

    # Vẽ biểu đồ Confusion Matrix
    print(f"Đang vẽ Confusion Matrix cho {model_name}...")
    plot_classification_confusion_matrix(y_test, y_pred, classes, model_name)

    return model_pipeline, X_test, y_test, y_pred

# --- Hàm chính để chạy ---
if __name__ == "__main__":
    try:
        # Tải và tiền xử lý dữ liệu
        df_processed = load_and_preprocess_data()

        # Thêm các cột trạng thái dinh dưỡng
        df_with_status = add_nutrition_status_columns(df_processed)

        # --- Huấn luyện và đánh giá mô hình Phân loại WAZ ---
        result_waz = train_evaluate_nutrition_model(df_with_status,
                                                  'nutrition_status_waz',
                                                  'Phân loại WAZ')

        # --- Huấn luyện và đánh giá mô hình Phân loại HAZ ---
        result_haz = train_evaluate_nutrition_model(df_with_status,
                                                  'nutrition_status_haz',
                                                  'Phân loại HAZ')

        # (Tùy chọn) Lưu mô hình
        # from joblib import dump
        # if result_waz:
        #     dump(result_waz[0], 'nutrition_model_waz.joblib')
        # if result_haz:
        #     dump(result_haz[0], 'nutrition_model_haz.joblib')
        # print("\nĐã lưu các mô hình phân loại (nếu có đủ dữ liệu).")

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'childhealth-dutch.csv'.")
    except ValueError as e:
        print(f"Lỗi xử lý dữ liệu hoặc không đủ lớp: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")