import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Thiết lập phong cách bằng Seaborn
sns.set(style='whitegrid')

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')

# Đọc dữ liệu train và test từ file CSV
train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

# Chia dữ liệu thành X và y
X_train = train_df[['age_current', 'age_future', 'sex', 'gagebrth', 'haz_current', 'waz_current']]
y_train = train_df['haz_future']
X_test = test_df[['age_current', 'age_future', 'sex', 'gagebrth', 'haz_current', 'waz_current']]
y_test = test_df['haz_future']

# Load mô hình
file_path = os.path.join(models_dir, 'haz_prediction_model.pkl')
model = joblib.load(file_path)

# Kiểm tra kiểu của model
print("Kiểu của model:", type(model))

# Dự đoán trên tập train và test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 1. Biểu đồ: Trục X là agedays (0-60 ngày), trục Y là haz, điểm train/test, đường dự đoán
# Lọc dữ liệu trong khoảng 0-60 ngày
train_mask = (X_train['age_future'] >= 0) & (X_train['age_future'] <= 60)
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]
y_train_pred_filtered = y_train_pred[train_mask]

test_mask = (X_test['age_future'] >= 0) & (X_test['age_future'] <= 60)
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]
y_test_pred_filtered = y_test_pred[test_mask]

plt.figure(figsize=(10, 5))

# Điểm dữ liệu của tập train
plt.scatter(X_train_filtered['age_future'], y_train_filtered, color='blue', alpha=0.5, label='Train (thực tế)', s=30)
# Điểm dữ liệu của tập test
plt.scatter(X_test_filtered['age_future'], y_test_filtered, color='green', alpha=0.5, label='Test (thực tế)', s=30)

# Gộp dữ liệu train và test để vẽ đường dự đoán
X_all_filtered = pd.concat([X_train_filtered, X_test_filtered], ignore_index=True)
y_all_pred_filtered = np.concatenate([y_train_pred_filtered, y_test_pred_filtered])

# Sắp xếp theo age_future để vẽ đường mượt mà
sorted_indices = X_all_filtered['age_future'].argsort()
X_all_sorted = X_all_filtered.iloc[sorted_indices]
y_all_pred_sorted = y_all_pred_filtered[sorted_indices]

# Đường dự đoán
plt.plot(X_all_sorted['age_future'], y_all_pred_sorted, color='orange', label='Dự đoán', linewidth=2)

plt.xlabel('Tuổi (agedays, ngày)')
plt.ylabel('haz')
plt.title('Biểu đồ: Điểm Train/Test và Đường Dự đoán (0-60 ngày)')
plt.legend()
plt.show()

# 2. Biểu đồ đánh giá: 2 biểu đồ con cho tập train và test
plt.figure(figsize=(12, 5))

# Tập train
plt.subplot(1, 2, 1)
train_sorted_indices = X_train['age_future'].argsort()
X_train_sorted = X_train.iloc[train_sorted_indices]
y_train_sorted = y_train.iloc[train_sorted_indices]
y_train_pred_sorted = y_train_pred[train_sorted_indices]

plt.plot(X_train_sorted['age_future'], y_train_sorted, color='blue', label='Thực tế (y_train)', linewidth=2)
plt.plot(X_train_sorted['age_future'], y_train_pred_sorted, color='orange', linestyle='--', label='Dự đoán (y_train_pred)', linewidth=2)
plt.xlabel('Tuổi (age_future, ngày)')
plt.ylabel('haz_future')
plt.title('Tập Train: Thực tế vs Dự đoán')
plt.legend()

# Tập test
plt.subplot(1, 2, 2)
test_sorted_indices = X_test['age_future'].argsort()
X_test_sorted = X_test.iloc[test_sorted_indices]
y_test_sorted = y_test.iloc[test_sorted_indices]
y_test_pred_sorted = y_test_pred[test_sorted_indices]

plt.plot(X_test_sorted['age_future'], y_test_sorted, color='blue', label='Thực tế (y_test)', linewidth=2)
plt.plot(X_test_sorted['age_future'], y_test_pred_sorted, color='orange', linestyle='--', label='Dự đoán (y_test_pred)', linewidth=2)
plt.xlabel('Tuổi (age_future, ngày)')
plt.ylabel('haz_future')
plt.title('Tập Test: Thực tế vs Dự đoán')
plt.legend()

plt.tight_layout()
plt.show()

# 3. Biểu đồ mức độ quan trọng
# Kiểm tra xem model có thuộc tính feature_importances_ không
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', hue='feature', data=feature_importance, palette='viridis', legend=False)
    plt.title('Độ Quan trọng của Đặc trưng (Feature Importance)')
    plt.xlabel('Tầm quan trọng')
    plt.ylabel('Đặc trưng')
    plt.show()
else:
    print("Lỗi: Mô hình không có thuộc tính 'feature_importances_'. Vui lòng kiểm tra lại file mô hình.")

# 4. Biểu đồ phân bố dữ liệu
plt.hist(X_train['age_future'], bins=30, alpha=0.5, label='Train')
plt.hist(X_test['age_future'], bins=30, alpha=0.5, label='Test')
plt.xlabel('age_future')
plt.ylabel('Số lượng')
plt.legend()
plt.show()