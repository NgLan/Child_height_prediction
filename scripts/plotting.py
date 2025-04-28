import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_regression_results(y_test, y_pred, gender):
    """
    Vẽ biểu đồ scatter plot so sánh giá trị thực tế và dự đoán cho mô hình hồi quy.

    Args:
        y_test (pandas.Series): Giá trị thực tế.
        y_pred (numpy.array): Giá trị dự đoán.
        gender (str): Giới tính ('Nam' hoặc 'Nữ') để đặt tiêu đề.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Dự đoán')
    # Vẽ đường y=x (dự đoán hoàn hảo)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Hoàn hảo (y=x)')
    plt.xlabel("Chiều cao thực tế (cm)")
    plt.ylabel("Chiều cao dự đoán (cm)")
    plt.title(f"Hồi quy Chiều cao: Thực tế vs. Dự đoán ({gender})")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_vs_prediction(X_test, y_test, y_pred, feature_name, target_name, gender):
    """
    Vẽ biểu đồ thể hiện mối quan hệ của một đặc trưng đầu vào
    với giá trị thực tế và giá trị dự đoán.

    Args:
        X_test (pandas.DataFrame): DataFrame chứa các đặc trưng của tập test.
        y_test (pandas.Series): Giá trị thực tế.
        y_pred (numpy.array): Giá trị dự đoán.
        feature_name (str): Tên của cột đặc trưng muốn vẽ trên trục X.
        target_name (str): Tên của biến mục tiêu (ví dụ: 'Chiều cao (cm)').
        gender (str): Giới tính ('Nam' hoặc 'Nữ').
    """
    if feature_name not in X_test.columns:
        print(f"Lỗi: Không tìm thấy đặc trưng '{feature_name}' trong X_test.")
        return

    # Tạo DataFrame tạm để dễ dàng sắp xếp và vẽ
    plot_df = pd.DataFrame({
        feature_name: X_test[feature_name],
        'Actual': y_test,
        'Predicted': y_pred
    }).sort_values(by=feature_name) # Sắp xếp theo feature để vẽ đường dễ nhìn hơn (tùy chọn)

    plt.figure(figsize=(10, 7))
    plt.scatter(plot_df[feature_name], plot_df['Actual'], alpha=0.5, label='Giá trị thực tế', s=15)
    plt.scatter(plot_df[feature_name], plot_df['Predicted'], alpha=0.5, label='Giá trị dự đoán', marker='x', s=15)
    # Có thể vẽ đường nếu muốn, ví dụ đường trung bình trượt hoặc hồi quy lowess
    # plt.plot(plot_df[feature_name], plot_df['Actual'].rolling(window=20, center=True).mean(), color='blue', linestyle='-', label='Xu hướng thực tế')
    # plt.plot(plot_df[feature_name], plot_df['Predicted'].rolling(window=20, center=True).mean(), color='orange', linestyle='--', label='Xu hướng dự đoán')

    plt.xlabel(f"Đặc trưng: {feature_name}")
    plt.ylabel(target_name)
    plt.title(f"{target_name} vs. {feature_name} ({gender}) - Thực tế và Dự đoán")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_classification_confusion_matrix(y_test, y_pred, classes, model_name):
    """
    Vẽ confusion matrix cho mô hình phân loại.

    Args:
        y_test (pandas.Series): Nhãn thực tế.
        y_pred (numpy.array): Nhãn dự đoán.
        classes (list): Danh sách tên các lớp (theo thứ tự).
        model_name (str): Tên mô hình (ví dụ: 'Phân loại WAZ').
    """
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Có thể thêm phần chạy thử nghiệm nếu muốn
if __name__ == "__main__":
    # Tạo dữ liệu giả để test hàm vẽ
    print("Chạy thử nghiệm các hàm vẽ biểu đồ (dữ liệu giả):")

    # Test Regression Plots
    y_test_reg = pd.Series(np.random.rand(50) * 30 + 50)
    y_pred_reg = y_test_reg + np.random.randn(50) * 3
    X_test_reg = pd.DataFrame({'agedays': np.random.rand(50) * 700, 'wtkg': np.random.rand(50)*10 + 3})
    plot_regression_results(y_test_reg, y_pred_reg, 'Thử nghiệm')
    plot_feature_vs_prediction(X_test_reg, y_test_reg, y_pred_reg, 'agedays', 'Chiều cao (cm)', 'Thử nghiệm')

    # Test Classification Plot
    class_names = ['Normal', 'Stunted', 'Severe Stunted']
    y_test_cls = pd.Series(np.random.choice(class_names, 100, p=[0.7, 0.2, 0.1]))
    y_pred_cls = np.random.choice(class_names, 100, p=[0.6, 0.25, 0.15]) # Giả lập dự đoán hơi lệch
    plot_classification_confusion_matrix(y_test_cls, y_pred_cls, class_names, 'Phân loại thử nghiệm')