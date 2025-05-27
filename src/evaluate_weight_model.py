# --- START OF FILE src/evaluate_weight_model.py ---
from .evaluate_model import evaluate_model # Import hàm chung

def evaluate_model_for_weight(model, train_data, test_data):
    """Đánh giá mô hình dự đoán cân nặng bằng cách gọi hàm đánh giá chung."""
    return evaluate_model(model, train_data, test_data, target_name="cân nặng")
# --- END OF FILE src/evaluate_weight_model.py ---