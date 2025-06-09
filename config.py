import os

# --- 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THƯ MỤC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
WHO_REF_DIR = os.path.join(DATA_DIR, "who_references")

INPUT_DATA_FILE = os.path.join(DATA_DIR, "childhealth-dutch.csv")

INTERPOLATED_HEIGHT_FILE = "interpolated_height.csv"
INTERPOLATED_WEIGHT_FILE = "interpolated_weight.csv"

# --- 2. CẤU HÌNH XỬ LÝ DỮ LIỆU VÀ CHUỖI THỜI GIAN ---
REFERENCE_START_DATE = '2000-01-01'
TIME_SERIES_FREQ = '30D'
INTERPOLATION_STEP = 30 # Ngày

# --- 3. CẤU HÌNH HUẤN LUYỆN MÔ HÌNH ---
TRAIN_TEST_SPLIT_RATIO = 0.8
MIN_SERIES_LENGTH = 5 # Độ dài tối thiểu của chuỗi thời gian để xử lý

MODEL_PARAMS = {
    "lags": 2,
    "lags_future_covariates": [0],
    "output_chunk_length": 1,
}

# --- 4. CẤU HÌNH CHI TIẾT CHO TỪNG MÔ HÌNH (CHIỀU CAO & CÂN NẶNG) ---
MODEL_CONFIGS = {
    "height": {
        "display_name": "Chiều cao",
        "unit": "cm",
        "target_col": "htcm",
        # Loại bỏ 'haz' và 'waz' khỏi covariates để tránh rò rỉ dữ liệu
        "covariate_cols": ['wtkg', 'sex', 'gagebrth'],
        "model_filename": "linear_regression_height_model.pkl",
    },
    "weight": {
        "display_name": "Cân nặng",
        "unit": "kg",
        "target_col": "wtkg",
        "covariate_cols": ['htcm', 'sex', 'gagebrth'],
        "model_filename": "linear_regression_weight_model.pkl",
    }
}

# --- 5. CẤU HÌNH PHÂN TÍCH VÀ TRỰC QUAN HÓA ---
ZSCORE_ANALYSIS_LIMIT = 5
INDIVIDUAL_PLOTS_LIMIT = 3
COMBINED_PLOTS_LIMIT = 4 # Số cặp biểu đồ, tức 8 biểu đồ con