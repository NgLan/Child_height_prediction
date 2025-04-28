import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

# Thiết lập phong cách bằng Seaborn
sns.set(style='whitegrid')

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')

# Đọc dữ liệu train và test riêng cho nam và nữ
train_df_male = pd.read_csv(os.path.join(data_dir, 'train_data_male.csv'))
train_df_female = pd.read_csv(os.path.join(data_dir, 'train_data_female.csv'))
test_df_male = pd.read_csv(os.path.join(data_dir, 'test_data_male.csv'))
test_df_female = pd.read_csv(os.path.join(data_dir, 'test_data_female.csv'))

# Chuẩn bị dữ liệu cho nam
X_train_male = train_df_male[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_train_male = train_df_male['haz_future']
X_test_male = test_df_male[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_test_male = test_df_male['haz_future']

# Chuẩn bị dữ liệu cho nữ
X_train_female = train_df_female[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_train_female = train_df_female['haz_future']
X_test_female = test_df_female[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_test_female = test_df_female['haz_future']

# Chuẩn hóa đặc trưng (đồng bộ với train_model.py)
scaler_male = StandardScaler()
X_train_male = scaler_male.fit_transform(X_train_male)
X_test_male = scaler_male.transform(X_test_male)

scaler_female = StandardScaler()
X_train_female = scaler_female.fit_transform(X_train_female)
X_test_female = scaler_female.transform(X_test_female)

# Load mô hình cho nam và nữ
model_male = joblib.load(os.path.join(models_dir, 'haz_model_male.pkl'))
model_female = joblib.load(os.path.join(models_dir, 'haz_model_female.pkl'))

# Kiểm tra kiểu của mô hình
print("Kiểu của mô hình nam:", type(model_male))
print("Kiểu của mô hình nữ:", type(model_female))

# Dự đoán trên tập train và test
y_train_pred_male = model_male.predict(X_train_male)
y_test_pred_male = model_male.predict(X_test_male)
y_train_pred_female = model_female.predict(X_train_female)
y_test_pred_female = model_female.predict(X_test_female)

# Hàm vẽ biểu đồ (sử dụng lại cho cả nam và nữ)
def plot_charts(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, model, title_prefix):
    # 1. Biểu đồ: Trục X là agedays (0-60 ngày), trục Y là haz, điểm train/test, đường dự đoán
    train_mask = (X_train[:, 1] >= 0) & (X_train[:, 1] <= 60)  # age_future là cột thứ 2 sau khi chuẩn hóa
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    y_train_pred_filtered = y_train_pred[train_mask]

    test_mask = (X_test[:, 1] >= 0) & (X_test[:, 1] <= 60)
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]
    y_test_pred_filtered = y_test_pred[test_mask]

    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_filtered[:, 1], y_train_filtered, color='blue', alpha=0.5, label='Train (thực tế)', s=30)
    plt.scatter(X_test_filtered[:, 1], y_test_filtered, color='green', alpha=0.5, label='Test (thực tế)', s=30)

    X_all_filtered = np.concatenate([X_train_filtered, X_test_filtered], axis=0)
    y_all_pred_filtered = np.concatenate([y_train_pred_filtered, y_test_pred_filtered])
    sorted_indices = X_all_filtered[:, 1].argsort()
    X_all_sorted = X_all_filtered[sorted_indices]
    y_all_pred_sorted = y_all_pred_filtered[sorted_indices]

    plt.plot(X_all_sorted[:, 1], y_all_pred_sorted, color='orange', label='Dự đoán', linewidth=2)
    plt.xlabel('Tuổi (agedays, ngày)')
    plt.ylabel('haz')
    plt.title(f'{title_prefix} - Biểu đồ: Điểm Train/Test và Đường Dự đoán (0-60 ngày)')
    plt.legend()
    plt.show()

    # 2. Biểu đồ đánh giá: 2 biểu đồ con cho tập train và test
    plt.figure(figsize=(12, 5))

    # Tập train
    plt.subplot(1, 2, 1)
    train_sorted_indices = X_train[:, 1].argsort()
    X_train_sorted = X_train[train_sorted_indices]
    y_train_sorted = y_train[train_sorted_indices]
    y_train_pred_sorted = y_train_pred[train_sorted_indices]

    plt.plot(X_train_sorted[:, 1], y_train_sorted, color='blue', label='Thực tế (y_train)', linewidth=2)
    plt.plot(X_train_sorted[:, 1], y_train_pred_sorted, color='orange', linestyle='--', label='Dự đoán (y_train_pred)', linewidth=2)
    plt.xlabel('Tuổi (age_future, ngày)')
    plt.ylabel('haz_future')
    plt.title(f'{title_prefix} - Tập Train: Thực tế vs Dự đoán')
    plt.legend()

    # Tập test
    plt.subplot(1, 2, 2)
    test_sorted_indices = X_test[:, 1].argsort()
    X_test_sorted = X_test[test_sorted_indices]
    y_test_sorted = y_test[test_sorted_indices]
    y_test_pred_sorted = y_test_pred[test_sorted_indices]

    plt.plot(X_test_sorted[:, 1], y_test_sorted, color='blue', label='Thực tế (y_test)', linewidth=2)
    plt.plot(X_test_sorted[:, 1], y_test_pred_sorted, color='orange', linestyle='--', label='Dự đoán (y_test_pred)', linewidth=2)
    plt.xlabel('Tuổi (age_future, ngày)')
    plt.ylabel('haz_future')
    plt.title(f'{title_prefix} - Tập Test: Thực tế vs Dự đoán')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 3. Biểu đồ mức độ quan trọng (Ridge không có feature_importances_, bỏ phần này)
    print(f"Biểu đồ mức độ quan trọng không áp dụng cho mô hình {title_prefix} vì sử dụng Ridge Regression.")

# Vẽ biểu đồ cho nam
print("\nBiểu đồ cho mô hình nam:")
plot_charts(X_train_male, y_train_male, y_train_pred_male, X_test_male, y_test_male, y_test_pred_male, model_male, "Nam")

# Vẽ biểu đồ cho nữ
print("\nBiểu đồ cho mô hình nữ:")
plot_charts(X_train_female, y_train_female, y_train_pred_female, X_test_female, y_test_female, y_test_pred_female, model_female, "Nữ")