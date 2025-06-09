# src/train_model.py

import pandas as pd
import pickle
from darts import TimeSeries
from darts.models import LinearRegressionModel

def train_model(df, model_save_path, target_col, covariate_cols, model_params, 
                start_date_str, freq, split_ratio, min_series_len):
    """
    Hàm huấn luyện mô hình.
    Nhận các tham số từ file config.
    """
    if df.empty:
        raise ValueError(f"Train {target_col}: DataFrame đầu vào rỗng.")

    series_dict = {}
    covariate_dict = {}
    start_date = pd.Timestamp(start_date_str)

    # Lọc ra các subjid không đủ dài trước khi xử lý
    subjid_counts = df['subjid'].value_counts()
    valid_subjids = subjid_counts[subjid_counts >= min_series_len].index
    df_filtered = df[df['subjid'].isin(valid_subjids)]

    grouped = df_filtered.groupby('subjid')

    for subjid, group_original in grouped:
        group = group_original.sort_values('agedays').copy()
        
        # Xử lý NaN cho covariates (ffill/bfill)
        for cov_col in covariate_cols:
            if group[cov_col].isnull().any():
                group[cov_col] = group[cov_col].ffill().bfill()
        group = group.dropna(subset=covariate_cols) # Xóa nếu vẫn còn NaN

        if len(group) < min_series_len:
            continue

        try:
            group['time_idx'] = pd.to_datetime(start_date) + pd.to_timedelta(group['agedays'].astype(int), unit='D')
            group = group.drop_duplicates(subset=['time_idx']).set_index('time_idx')
            
            series = TimeSeries.from_dataframe(group, value_cols=[target_col], freq=freq)
            covariates = TimeSeries.from_dataframe(group, value_cols=covariate_cols, freq=freq)

            series_dict[subjid] = series
            covariate_dict[subjid] = covariates
        except Exception as e:
            print(f"Lỗi khi tạo TimeSeries cho subjid {subjid} (Train {target_col}): {e}")
            continue

    if not series_dict:
        raise ValueError(f"Train {target_col}: Không có dữ liệu TimeSeries hợp lệ để huấn luyện.")

    # Chia Train/Test
    train_series_list, test_series_list = [], []
    train_cov_list, test_cov_list = [], []

    for subjid, series in series_dict.items():
        covariates = covariate_dict.get(subjid)
        try:
            train_series, test_series = series.split_after(split_ratio)
            # Covariates cần được giữ nguyên toàn bộ cho việc dự đoán trong tương lai
            train_cov, _ = covariates.split_after(split_ratio)
            
            train_series_list.append(train_series)
            test_series_list.append(test_series)
            train_cov_list.append(train_cov)
            test_cov_list.append(covariates) # Lưu toàn bộ covariates cho tập test
        except (ValueError, IndexError):
            continue

    if not train_series_list:
        raise ValueError(f"Train {target_col}: Không có dữ liệu train hợp lệ sau khi chia.")

    # Khởi tạo và huấn luyện mô hình
    model = LinearRegressionModel(**model_params)
    
    print(f"Train {target_col}: Huấn luyện với {len(train_series_list)} series.")
    model.fit(series=train_series_list, future_covariates=train_cov_list)

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Trả về dữ liệu dưới dạng danh sách, không phải dict
    train_data = (train_series_list, train_cov_list)
    test_data = (test_series_list, test_cov_list)

    return model, train_data, test_data