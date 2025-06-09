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

    series_list = [] 
    covariate_list = []
    start_date = pd.Timestamp(start_date_str)

    subjid_counts = df['subjid'].value_counts()
    valid_subjids = subjid_counts[subjid_counts >= min_series_len].index
    df_filtered = df[df['subjid'].isin(valid_subjids)]

    grouped = df_filtered.groupby('subjid')

    for subjid, group_original in grouped:
        group = group_original.sort_values('agedays').copy()
        
        for cov_col in covariate_cols:
            if group[cov_col].isnull().any():
                group[cov_col] = group[cov_col].ffill().bfill()
        group = group.dropna(subset=covariate_cols)

        if len(group) < min_series_len:
            continue

        try:
            group['time_idx'] = pd.to_datetime(start_date) + pd.to_timedelta(group['agedays'].astype(int), unit='D')
            group = group.drop_duplicates(subset=['time_idx']).set_index('time_idx')
            
            # Tạo một DataFrame nhỏ chứa các biến tĩnh (static covariates)
            # Chúng phải có cùng giá trị cho tất cả các hàng của một subjid
            static_cov_df = group[['subjid', 'sex']].iloc[[0]] # Lấy dòng đầu tiên là đủ
            static_cov_df = static_cov_df.reset_index(drop=True)

            # Truyền `static_covariates` vào khi tạo TimeSeries
            series = TimeSeries.from_dataframe(
                group, 
                value_cols=[target_col], 
                freq=freq,
                static_covariates=static_cov_df
            )
            
            covariates = TimeSeries.from_dataframe(
                group, 
                value_cols=covariate_cols, 
                freq=freq,
                static_covariates=static_cov_df 
            )

            series_list.append(series)
            covariate_list.append(covariates)
            
        except Exception as e:
            print(f"Lỗi khi tạo TimeSeries cho subjid {subjid} (Train {target_col}): {e}")
            continue

    if not series_list:
        raise ValueError(f"Train {target_col}: Không có dữ liệu TimeSeries hợp lệ để huấn luyện.")

    # Chia Train/Test
    train_series_list, test_series_list = [], []
    train_cov_list, test_cov_list = [], []

    # Lặp qua danh sách đã tạo
    for i in range(len(series_list)):
        series = series_list[i]
        covariates = covariate_list[i]
        try:
            train_series, test_series = series.split_after(split_ratio)
            train_cov, _ = covariates.split_after(split_ratio)
            
            train_series_list.append(train_series)
            test_series_list.append(test_series)
            train_cov_list.append(train_cov)
            test_cov_list.append(covariates)
        except (ValueError, IndexError):
            continue

    if not train_series_list:
        raise ValueError(f"Train {target_col}: Không có dữ liệu train hợp lệ sau khi chia.")

    model = LinearRegressionModel(**model_params)
    
    print(f"Train {target_col}: Huấn luyện với {len(train_series_list)} series.")
    model.fit(series=train_series_list, future_covariates=train_cov_list)

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    
    train_data = (train_series_list, train_cov_list)
    test_data = (test_series_list, test_cov_list)

    return model, train_data, test_data