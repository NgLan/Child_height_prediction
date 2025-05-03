import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Xác định đường dẫn
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')
figures_dir = os.path.join(project_dir, 'figures')

# Tạo thư mục figures nếu chưa tồn tại
os.makedirs(figures_dir, exist_ok=True)

# Đọc dữ liệu (cần wtkg, htcm cho scatter plot)
data_file = os.path.join(data_dir, 'childhealth-dutch-female.csv')
data = pd.read_csv(data_file)

# Tải mô hình dự đoán chiều cao và tập train/test
height_model = joblib.load(os.path.join(models_dir, 'height_prediction_model.pkl'))
X_train_h = joblib.load(os.path.join(models_dir, 'X_train_height.pkl'))
y_train_h = joblib.load(os.path.join(models_dir, 'y_train_height.pkl'))
X_test_h = joblib.load(os.path.join(models_dir, 'X_test_height.pkl'))
y_test_h = joblib.load(os.path.join(models_dir, 'y_test_height.pkl'))

# Tải mô hình phân loại, LabelEncoder, scaler và tập train/test
classification_model = joblib.load(os.path.join(models_dir, 'child_classification_model.pkl'))
label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler_classification.pkl'))
X_test_c = joblib.load(os.path.join(models_dir, 'X_test_classification.pkl'))
y_test_c = joblib.load(os.path.join(models_dir, 'y_test_classification.pkl'))

# Dự đoán chiều cao
y_train_pred_h = height_model.predict(X_train_h)
y_test_pred_h = height_model.predict(X_test_h)

# Dự đoán phân loại
y_class_pred = classification_model.predict(X_test_c)
y_class_pred_labels = label_encoder.inverse_transform(y_class_pred)

# Lấy wtkg, htcm từ dữ liệu gốc cho scatter plot (khớp với X_test_c)
data_class = data.loc[X_test_c.index, ['wtkg', 'htcm']].dropna()

# Biểu đồ 1: Tuổi (ngày) vs Chiều cao (cm)
plt.figure(figsize=(10, 6))
plt.scatter(X_test_h['agedays'], y_test_h, color='blue', label='Thực tế', alpha=0.5)
plt.scatter(X_test_h['agedays'], y_test_pred_h, color='red', label='Dự đoán', alpha=0.5)
plt.xlabel('Tuổi (ngày)')
plt.ylabel('Chiều cao (cm)')
plt.title('Mối quan hệ giữa Tuổi và Chiều cao (Nữ)')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'height_age_plot.png'))
plt.show()

# Biểu đồ 2: Cân nặng (kg) vs Chiều cao (cm)
plt.figure(figsize=(10, 6))
plt.scatter(X_test_h['wtkg'], y_test_h, color='blue', label='Thực tế', alpha=0.5)
plt.scatter(X_test_h['wtkg'], y_test_pred_h, color='red', label='Dự đoán', alpha=0.5)
plt.xlabel('Cân nặng (kg)')
plt.ylabel('Chiều cao (cm)')
plt.title('Mối quan hệ giữa Cân nặng và Chiều cao (Nữ)')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'height_weight_plot.png'))
plt.show()

# Biểu đồ 3: Độ quan trọng của đặc trưng (Dự đoán chiều cao)
plt.figure(figsize=(10, 6))
plt.bar(['agedays', 'wtkg', 'gagebrth'], height_model.coef_)
plt.xlabel('Đặc trưng')
plt.ylabel('Hệ số hồi quy')
plt.title('Độ quan trọng của các đặc trưng trong dự đoán chiều cao')
plt.savefig(os.path.join(figures_dir, 'feature_importance_height.png'))
plt.show()

# Biểu đồ 4: Độ quan trọng của đặc trưng (Phân loại trẻ)
plt.figure(figsize=(10, 6))
plt.bar(['agedays', 'wtkg', 'htcm', 'wtkg_per_agedays', 'bmi'], classification_model.feature_importances_)
plt.xlabel('Đặc trưng')
plt.ylabel('Độ quan trọng')
plt.title('Độ quan trọng của các đặc trưng trong phân loại trẻ')
plt.savefig(os.path.join(figures_dir, 'feature_importance_classification.png'))
plt.show()

# Biểu đồ 5: Phân loại trẻ theo Cân nặng (wtkg) và Chiều cao (htcm) với decision boundaries
plt.figure(figsize=(10, 6))

# Tạo lưới cho decision boundaries
wtkg_min, wtkg_max = data_class['wtkg'].min() - 1, data_class['wtkg'].max() + 1
htcm_min, htcm_max = data_class['htcm'].min() - 1, data_class['htcm'].max() + 1
wtkg_grid, htcm_grid = np.meshgrid(
    np.linspace(wtkg_min, wtkg_max, 500),  # Tăng số điểm lên 500
    np.linspace(htcm_min, htcm_max, 500)
)

# Tạo dữ liệu đầu vào cho mô hình (cố định agedays, wtkg_per_agedays, bmi)
agedays_mean = data['agedays'].mean()
wtkg_per_agedays_grid = wtkg_grid.ravel() / agedays_mean  # Tính wtkg_per_agedays
bmi_grid = wtkg_grid.ravel() / (htcm_grid.ravel() / 100) ** 2  # Tính bmi
X_grid = np.c_[np.ones(wtkg_grid.ravel().shape) * agedays_mean,
               wtkg_grid.ravel(),
               htcm_grid.ravel(),
               wtkg_per_agedays_grid,
               bmi_grid]

# Chuyển X_grid thành DataFrame với tên cột
X_grid_df = pd.DataFrame(X_grid, columns=['agedays', 'wtkg', 'htcm', 'wtkg_per_agedays', 'bmi'])

# Chuẩn hóa dữ liệu lưới trước khi dự đoán
X_grid_scaled = scaler.transform(X_grid_df)
X_grid_scaled_df = pd.DataFrame(X_grid_scaled, columns=['agedays', 'wtkg', 'htcm', 'wtkg_per_agedays', 'bmi'])

# Dự đoán danh mục trên lưới
Z = classification_model.predict(X_grid_scaled_df)
Z = Z.reshape(wtkg_grid.shape)

# Vẽ contour plot cho decision boundaries với màu giống hình mẫu
from matplotlib.colors import ListedColormap
# Định nghĩa màu cho 3 lớp: xanh nhạt, hồng nhạt, xanh lá nhạt
cmap = ListedColormap(['#ADD8E6', '#FFB6C1', '#90EE90'])  # Light blue, light pink, light green
plt.contourf(wtkg_grid, htcm_grid, Z, alpha=0.5, cmap=cmap, levels=np.arange(len(label_encoder.classes_) + 1) - 0.5)

# Vẽ scatter plot cho các điểm dữ liệu với màu giống hình mẫu
categories = np.unique(y_class_pred_labels)  # 3 lớp: Normal, Overweight, Undernourished
# Định nghĩa màu cho điểm dữ liệu: xanh dương, đỏ, xanh lá
colors = ['blue', 'red', 'green']
for i, category in enumerate(categories):
    mask = y_class_pred_labels == category
    plt.scatter(data_class[mask]['wtkg'], data_class[mask]['htcm'],
                c=colors[i], label=category, marker='o', alpha=0.6)

# Cài đặt tiêu đề, nhãn trục và legend
plt.xlabel('Cân nặng (kg)')
plt.ylabel('Chiều cao (cm)')
plt.title('Data with Categorical Response')
# Tạo legend với tiêu đề "Predicted Regions"
plt.legend(title="Predicted Regions")
plt.savefig(os.path.join(figures_dir, 'classification_scatter.png'))
plt.show()