import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from scipy.stats import mstats

# Xác định đường dẫn
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
models_dir = os.path.join(project_dir, 'models')
figures_dir = os.path.join(project_dir, 'figures')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(models_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Đọc dữ liệu
data_file = os.path.join(data_dir, 'childhealth-dutch-female.csv')
data = pd.read_csv(data_file)

# Hàm phân loại trẻ dựa trên Z-score
def classify_child(row):
    if row['haz'] < -2 or row['waz'] < -2 or row['wlhz'] < -2:
        return 'Undernourished'
    elif row['wlhz'] > 2 or row['baz'] > 2:
        return 'Overweight'
    else:
        return 'Normal'

# Thêm cột category dựa trên Z-score
data['category'] = data.apply(classify_child, axis=1)

# Kiểm tra phân bố lớp
print("Phân bố lớp trong dữ liệu gốc:")
print(data['category'].value_counts())

# Feature engineering
data['agedays'] = data['agedays'].replace(0, 1)
data['wtkg_per_agedays'] = data['wtkg'] / data['agedays']

# Chọn đặc trưng và nhãn
features = ['agedays', 'wtkg', 'htcm', 'wtkg_per_agedays', 'bmi']
label = 'category'

# Loại bỏ hàng có giá trị thiếu
data = data[features + [label]].dropna()

# Kiểm tra và xử lý giá trị inf, -inf, và giá trị quá lớn
for feature in features:
    data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
    if feature in ['agedays', 'wtkg', 'htcm', 'bmi']:
        data = data[data[feature] > 0]
    data = data.dropna(subset=[feature])

# Kiểm tra dữ liệu sau khi xử lý
print("\nKiểm tra dữ liệu sau khi xử lý giá trị bất thường:")
for feature in features:
    print(f"{feature} - min: {data[feature].min()}, max: {data[feature].max()}")

# Kiểm tra phân bố lớp
print("\nPhân bố lớp sau khi xử lý:")
print(data['category'].value_counts())

# Vẽ histogram của wtkg_per_agedays trước khi Winsorization
plt.figure(figsize=(10, 6))
plt.hist(data['wtkg_per_agedays'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Phân phối của wtkg_per_agedays (Trước Winsorization)')
plt.xlabel('wtkg_per_agedays')
plt.ylabel('Số lượng')
plt.savefig(os.path.join(figures_dir, 'wtkg_per_agedays_distribution_before.png'))
plt.close()

# Áp dụng Winsorization cho wtkg_per_agedays (giới hạn ở percentile 99)
data['wtkg_per_agedays'] = mstats.winsorize(data['wtkg_per_agedays'], limits=[0, 0.01])
print("\nSau Winsorization, wtkg_per_agedays - min:", data['wtkg_per_agedays'].min(), "max:", data['wtkg_per_agedays'].max())

# Vẽ histogram của wtkg_per_agedays sau khi Winsorization
plt.figure(figsize=(10, 6))
plt.hist(data['wtkg_per_agedays'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Phân phối của wtkg_per_agedays (Sau Winsorization)')
plt.xlabel('wtkg_per_agedays')
plt.ylabel('Số lượng')
plt.savefig(os.path.join(figures_dir, 'wtkg_per_agedays_distribution_after.png'))
plt.close()

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
X = pd.DataFrame(X, columns=features)
joblib.dump(scaler, os.path.join(models_dir, 'scaler_classification.pkl'))

# Mã hóa nhãn
le = LabelEncoder()
data['category'] = le.fit_transform(data['category'])
# In ra mapping của LabelEncoder để làm rõ lớp nào tương ứng với 0, 1, 2
print("\nMapping của LabelEncoder (lớp -> số):")
for cls, idx in zip(le.classes_, range(len(le.classes_))):
    print(f"{cls}: {idx}")

# Chia dữ liệu
X = X
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kiểm tra phân bố lớp trong tập train
print("\nPhân bố lớp trong tập train:")
train_class_counts = pd.Series(y_train).value_counts()
print(train_class_counts)

# Tinh chỉnh siêu tham số
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [4, 6],
    # Trọng số tùy chỉnh: ưu tiên lớp hiếm để tăng recall
    # - Normal (0): trọng số 1 (lớp đa số, ít ưu tiên)
    # - Overweight (1): trọng số 10 (lớp hiếm, ưu tiên trung bình)
    # - Undernourished (2): trọng số 20 (lớp hiếm, ưu tiên cao vì tầm quan trọng trong y tế)
    'class_weight': ['balanced', {0: 1, 1: 10, 2: 20}]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1_macro'  # Sử dụng f1_macro để ưu tiên lớp thiểu số
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Dự đoán
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Đánh giá mô hình
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Tính classification error rate
train_error = 1 - train_accuracy
test_error = 1 - test_accuracy

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Train Error Rate: {train_error:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Error Rate: {test_error:.4f}")

# Classification Report
print("\nClassification Report (Test):")
present_classes = le.classes_[np.unique(y_test)]
report = classification_report(y_test, y_test_pred, labels=np.unique(y_test), target_names=present_classes, output_dict=True)
print(classification_report(y_test, y_test_pred, labels=np.unique(y_test), target_names=present_classes))

# Confusion Matrix với nhãn
print("\nConfusion Matrix (Test) with Labels:")
conf_matrix = confusion_matrix(y_test, y_test_pred, labels=np.unique(y_test))
conf_matrix_df = pd.DataFrame(conf_matrix, index=present_classes, columns=present_classes)
print(conf_matrix_df)

# Phân tích hiệu suất chi tiết
print("\nPhân tích hiệu suất chi tiết:")
# Tìm lớp có recall thấp nhất
min_recall_class = min(report.items(), key=lambda x: x[1]['recall'] if x[0] in present_classes else float('inf'))
print(f"Lớp có recall thấp nhất: {min_recall_class[0]} (Recall: {min_recall_class[1]['recall']:.2f})")

# Tìm lớp có precision thấp nhất
min_precision_class = min(report.items(), key=lambda x: x[1]['precision'] if x[0] in present_classes else float('inf'))
print(f"Lớp có precision thấp nhất: {min_precision_class[0]} (Precision: {min_precision_class[1]['precision']:.2f})")

# Tìm lớp có F1-score thấp nhất
min_f1_class = min(report.items(), key=lambda x: x[1]['f1-score'] if x[0] in present_classes else float('inf'))
print(f"Lớp có F1-score thấp nhất: {min_f1_class[0]} (F1-score: {min_f1_class[1]['f1-score']:.2f})")

# Lưu ý về wtkg_per_agedays
print("\nLưu ý: Kiểm tra độ quan trọng của đặc trưng 'wtkg_per_agedays' trong 'feature_importance_classification.png'.")
print("Nếu độ quan trọng thấp (e.g., dưới 0.1), cân nhắc loại bỏ đặc trưng này để giảm nhiễu.")

# Lưu mô hình, LabelEncoder, scaler và tập train/test
model_file = os.path.join(models_dir, 'child_classification_model.pkl')
le_file = os.path.join(models_dir, 'label_encoder.pkl')
joblib.dump(model, model_file)
joblib.dump(le, le_file)
joblib.dump(X_train, os.path.join(models_dir, 'X_train_classification.pkl'))
joblib.dump(y_train, os.path.join(models_dir, 'y_train_classification.pkl'))
joblib.dump(X_test, os.path.join(models_dir, 'X_test_classification.pkl'))
joblib.dump(y_test, os.path.join(models_dir, 'y_test_classification.pkl'))
print(f"Mô hình đã được lưu vào {model_file}")
print(f"LabelEncoder đã được lưu vào {le_file}")
print(f"Scaler đã được lưu vào {models_dir}")
print(f"Tập train/test đã được lưu vào {models_dir}")