from evaluate_height_model import evaluate_height_model # Import hàm chung

def evaluate_weight_model(model, train_data, test_data):
    """Đánh giá mô hình dự đoán cân nặng bằng cách gọi hàm đánh giá chung."""
    return evaluate_height_model(model, train_data, test_data, target_name="cân nặng")