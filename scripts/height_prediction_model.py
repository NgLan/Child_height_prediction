import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Xác định đường dẫn
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')

# Đọc dữ liệu
data_file = os.path.join(data_dir, 'childhealth-dutch-female.csv')
data = pd.read_csv(data_file)

# Chọn các đặc trưng và nhãn
features = ['agedays', 'wtkg', 'gagebrth']
label = 'htcm'

# Loại bỏ hàng có giá trị thiếu
data = data[features + [label]].dropna()

# Chia dữ liệu thành train và test
X = data[features]
y = data[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Đánh giá mô hình
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# In kết quả
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Lưu mô hình và tập train/test
model_file = os.path.join(models_dir, 'height_prediction_model.pkl')
joblib.dump(model, model_file)
joblib.dump(X_train, os.path.join(models_dir, 'X_train_height.pkl'))
joblib.dump(y_train, os.path.join(models_dir, 'y_train_height.pkl'))
joblib.dump(X_test, os.path.join(models_dir, 'X_test_height.pkl'))
joblib.dump(y_test, os.path.join(models_dir, 'y_test_height.pkl'))
print(f"Mô hình đã được lưu vào {model_file}")
print(f"Tập train/test đã được lưu vào {models_dir}")