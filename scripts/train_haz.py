import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import os

# Định nghĩa thư mục dữ liệu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Đọc dữ liệu đã tiền xử lý
df = pd.read_csv(os.path.join(data_dir, 'childhealth_processed_with_zscores.csv'))

# Loại bỏ các hàng có NaN
df = df.dropna(subset=['haz', 'waz', 'sex', 'agedays', 'gagebrth'])

# Chia danh sách subjid thành tập train và test
subjids = df['subjid'].unique()
train_subjids, test_subjids = train_test_split(subjids, test_size=0.2, random_state=42)

# Hàm tạo dữ liệu từ danh sách subjid
def create_training_data(df, subjids):
    training_data = []
    for subjid in subjids:
        child_data = df[df['subjid'] == subjid].sort_values('agedays')
        for i in range(len(child_data)):
            for j in range(i + 1, len(child_data)):
                current_row = child_data.iloc[i]
                future_row = child_data.iloc[j]
                training_data.append({
                    'age_current': current_row['agedays'],
                    'age_future': future_row['agedays'],
                    'sex': current_row['sex'],
                    'gagebrth': current_row['gagebrth'],
                    'haz_current': current_row['haz'],
                    'waz_current': current_row['waz'],
                    'haz_future': future_row['haz']
                })
    return pd.DataFrame(training_data)

# Tạo dữ liệu train và test
train_df = create_training_data(df, train_subjids)
test_df = create_training_data(df, test_subjids)

# Chia dữ liệu thành X và y
X_train = train_df[['age_current', 'age_future', 'sex', 'gagebrth', 'haz_current', 'waz_current']]
y_train = train_df['haz_future']
X_test = test_df[['age_current', 'age_future', 'sex', 'gagebrth', 'haz_current', 'waz_current']]
y_test = test_df['haz_future']

# Huấn luyện mô hình với tham số điều chỉnh để giảm overfitting
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,  # Giới hạn độ sâu của cây
    min_samples_split=5,  # Số mẫu tối thiểu để chia một nút
    min_samples_leaf=2,  # Số mẫu tối thiểu tại mỗi lá
    max_features='sqrt',  # Số đặc trưng tối đa khi chia nhánh
    random_state=42
)
model.fit(X_train, y_train)

# Dự đoán trên tập train và test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Đánh giá mô hình
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train Mean Squared Error: {train_mse:.4f}")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test Mean Squared Error: {test_mse:.4f}")
print(f"Test R² Score: {test_r2:.4f}")

# Đánh giá mô hình bằng cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R² Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation CV R² Score: {cv_scores.std():.4f}")

# Lưu mô hình
model_path = os.path.join(models_dir, 'haz_prediction_model.pkl')
joblib.dump(model, model_path)
print(f"Đã huấn luyện và lưu mô hình tại: {model_path}")

# Lưu dữ liệu train và test để sử dụng trong file visualize_model.py
train_df.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
print("Đã lưu dữ liệu train và test vào thư mục data.")