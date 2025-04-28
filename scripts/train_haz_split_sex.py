import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# Chia dữ liệu thành tập nam và nữ
train_df_male = train_df[train_df['sex'] == 1]
train_df_female = train_df[train_df['sex'] == 0]
test_df_male = test_df[test_df['sex'] == 1]
test_df_female = test_df[test_df['sex'] == 0]

# Chuẩn bị dữ liệu cho mô hình nam
X_train_male = train_df_male[['age_current', 'age_future', 'gagebrth', 'haz_current']]  # Loại bỏ 'waz_current'
y_train_male = train_df_male['haz_future']
X_test_male = test_df_male[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_test_male = test_df_male['haz_future']

# Chuẩn bị dữ liệu cho mô hình nữ
X_train_female = train_df_female[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_train_female = train_df_female['haz_future']
X_test_female = test_df_female[['age_current', 'age_future', 'gagebrth', 'haz_current']]
y_test_female = test_df_female['haz_future']

# Loại bỏ outliers trong tập train nam
Q1_male = y_train_male.quantile(0.25)
Q3_male = y_train_male.quantile(0.75)
IQR_male = Q3_male - Q1_male
train_mask_male = (y_train_male >= Q1_male - 1.5*IQR_male) & (y_train_male <= Q3_male + 1.5*IQR_male)
X_train_male = X_train_male[train_mask_male]
y_train_male = y_train_male[train_mask_male]

# Loại bỏ outliers trong tập test nam
Q1_male = y_test_male.quantile(0.25)
Q3_male = y_test_male.quantile(0.75)
IQR_male = Q3_male - Q1_male
test_mask_male = (y_test_male >= Q1_male - 1.5*IQR_male) & (y_test_male <= Q3_male + 1.5*IQR_male)
X_test_male = X_test_male[test_mask_male]
y_test_male = y_test_male[test_mask_male]

# Loại bỏ outliers trong tập train nữ
Q1_female = y_train_female.quantile(0.25)
Q3_female = y_train_female.quantile(0.75)
IQR_female = Q3_female - Q1_female
train_mask_female = (y_train_female >= Q1_female - 1.5*IQR_female) & (y_train_female <= Q3_female + 1.5*IQR_female)
X_train_female = X_train_female[train_mask_female]
y_train_female = y_train_female[train_mask_female]

# Loại bỏ outliers trong tập test nữ
Q1_female = y_test_female.quantile(0.25)
Q3_female = y_test_female.quantile(0.75)
IQR_female = Q3_female - Q1_female
test_mask_female = (y_test_female >= Q1_female - 1.5*IQR_female) & (y_test_female <= Q3_female + 1.5*IQR_female)
X_test_female = X_test_female[test_mask_female]
y_test_female = y_test_female[test_mask_female]

# Chuẩn hóa đặc trưng
scaler_male = StandardScaler()
X_train_male = scaler_male.fit_transform(X_train_male)
X_test_male = scaler_male.transform(X_test_male)

scaler_female = StandardScaler()
X_train_female = scaler_female.fit_transform(X_train_female)
X_test_female = scaler_female.transform(X_test_female)

# Huấn luyện mô hình cho nam (thử Ridge Regression)
model_male = Ridge(alpha=1.0, random_state=42)
# model_male = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=10,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features='sqrt',
#     random_state=42
# )
model_male.fit(X_train_male, y_train_male)

# Dự đoán và đánh giá trên tập nam
y_train_pred_male = model_male.predict(X_train_male)
y_test_pred_male = model_male.predict(X_test_male)

train_mse_male = mean_squared_error(y_train_male, y_train_pred_male)
train_r2_male = r2_score(y_train_male, y_train_pred_male)
test_mse_male = mean_squared_error(y_test_male, y_test_pred_male)
test_r2_male = r2_score(y_test_male, y_test_pred_male)

print("Kết quả cho mô hình nam:")
print(f"Train Mean Squared Error: {train_mse_male:.4f}")
print(f"Train R² Score: {train_r2_male:.4f}")
print(f"Test Mean Squared Error: {test_mse_male:.4f}")
print(f"Test R² Score: {test_r2_male:.4f}")

# Huấn luyện mô hình cho nữ (thử Ridge Regression)
model_female = Ridge(alpha=1.0, random_state=42)
# model_female = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=10,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features='sqrt',
#     random_state=42
# )
model_female.fit(X_train_female, y_train_female)

# Dự đoán và đánh giá trên tập nữ
y_train_pred_female = model_female.predict(X_train_female)
y_test_pred_female = model_female.predict(X_test_female)

train_mse_female = mean_squared_error(y_train_female, y_train_pred_female)
train_r2_female = r2_score(y_train_female, y_train_pred_female)
test_mse_female = mean_squared_error(y_test_female, y_test_pred_female)
test_r2_female = r2_score(y_test_female, y_test_pred_female)

print("\nKết quả cho mô hình nữ:")
print(f"Train Mean Squared Error: {train_mse_female:.4f}")
print(f"Train R² Score: {train_r2_female:.4f}")
print(f"Test Mean Squared Error: {test_mse_female:.4f}")
print(f"Test R² Score: {test_r2_female:.4f}")

# Lưu hai mô hình
model_male_path = os.path.join(models_dir, 'haz_model_male.pkl')
model_female_path = os.path.join(models_dir, 'haz_model_female.pkl')
joblib.dump(model_male, model_male_path)
joblib.dump(model_female, model_female_path)
print(f"Đã lưu mô hình nam tại: {model_male_path}")
print(f"Đã lưu mô hình nữ tại: {model_female_path}")

# Lưu dữ liệu train và test riêng cho nam và nữ
train_df_male.to_csv(os.path.join(data_dir, 'train_data_male.csv'), index=False)
train_df_female.to_csv(os.path.join(data_dir, 'train_data_female.csv'), index=False)
test_df_male.to_csv(os.path.join(data_dir, 'test_data_male.csv'), index=False)
test_df_female.to_csv(os.path.join(data_dir, 'test_data_female.csv'), index=False)
print("Đã lưu dữ liệu train và test riêng cho nam và nữ vào thư mục data.")