# --- START OF FILE src/train_model.py --- (PHIÊN BẢN ĐÚNG)
import pandas as pd
import pickle
from darts import TimeSeries
from darts.models import LinearRegressionModel

def train_linear_regression_model(df, model_save_path, target_col='htcm',
                                  covariate_cols=['wtkg', 'haz', 'waz', 'sex', 'gagebrth']):
    """Huấn luyện mô hình Linear Regression (chung)."""
    if df.empty:
        raise ValueError(f"Train {target_col}: DataFrame đầu vào rỗng.")

    series_dict = {}
    covariate_dict = {}
    start_date = pd.Timestamp('2000-01-01')

    grouped = df.groupby('subjid')

    for subjid, group_original in grouped:
        group = group_original.sort_values('agedays').copy()

        for cov_col in covariate_cols:
            if cov_col in group.columns:
                if group[cov_col].isnull().any():
                    group[cov_col] = group[cov_col].ffill().bfill().fillna(0)
            else:
                group[cov_col] = 0

        group.dropna(subset=[target_col], inplace=True)

        min_len_for_series = 2 + 1 + 1
        if len(group) < min_len_for_series:
            continue

        try:
            group['time_idx'] = pd.to_datetime(start_date) + pd.to_timedelta(group['agedays'].astype(int), unit='D')
            group = group.set_index('time_idx')
            # print(f"Subjid {subjid} - Inferred freq: {pd.infer_freq(group.index)}") # debug

            if not group.index.is_unique:
                group = group[~group.index.duplicated(keep='last')]
                if len(group) < min_len_for_series:
                    continue

            series = TimeSeries.from_dataframe(
                group,
                value_cols=[target_col],
                freq='30D'
            )

            actual_covariate_cols = [col for col in covariate_cols if col in group.columns]
            covariates = None
            if actual_covariate_cols:
                covariates = TimeSeries.from_dataframe(
                    group,
                    value_cols=actual_covariate_cols,
                    freq='30D'
                )

            if len(series) < min_len_for_series:
                continue
            if covariates is not None and len(covariates) != len(series):
                covariates = None

            series_dict[subjid] = series
            covariate_dict[subjid] = covariates
        except Exception as e:
            # print(f"Lỗi khi tạo TimeSeries cho subjid {subjid} (Train {target_col}): {e}")
            continue

    if not series_dict:
        raise ValueError(f"Train {target_col}: Không có dữ liệu TimeSeries hợp lệ để huấn luyện.")

    final_train_series_dict = {}
    final_test_series_dict = {}
    final_train_covariates_dict = {}
    final_test_covariates_full_dict = {}

    for subjid_key, current_series in series_dict.items():
        current_covariates = covariate_dict.get(subjid_key, None)
        if len(current_series) < 3:
            continue
        split_point = int(len(current_series) * 0.8)
        if split_point < 2 or (len(current_series) - split_point < 1):
            continue

        final_train_series_dict[subjid_key] = current_series[:split_point]
        final_test_series_dict[subjid_key] = current_series[split_point:]

        if current_covariates:
            final_train_covariates_dict[subjid_key] = current_covariates[:split_point]
            final_test_covariates_full_dict[subjid_key] = current_covariates
        else:
            final_train_covariates_dict[subjid_key] = None
            final_test_covariates_full_dict[subjid_key] = None

    if not final_train_series_dict:
        raise ValueError(f"Train {target_col}: Không có dữ liệu train hợp lệ sau khi chia train/test.")

    list_to_fit_series = []
    list_to_fit_covariates = []
    has_any_covariate = any(cov is not None for cov in final_train_covariates_dict.values())

    for subjid_key, ts_train in final_train_series_dict.items():
        cov_train = final_train_covariates_dict.get(subjid_key, None)
        if has_any_covariate:
            if cov_train is not None:
                list_to_fit_series.append(ts_train)
                list_to_fit_covariates.append(cov_train)
        else:
            list_to_fit_series.append(ts_train)

    if not list_to_fit_series:
        raise ValueError(f"Train {target_col}: Không có series nào phù hợp để đưa vào model.fit.")

    lags_future_covariates_param = None
    if has_any_covariate and list_to_fit_covariates:
        lags_future_covariates_param = [0] # Sử dụng covariate tại thời điểm hiện tại

    model = LinearRegressionModel(
        lags=2,
        lags_future_covariates=lags_future_covariates_param,
        # lags_future_covariates=[0] if has_any_covariate and list_to_fit_covariates else None,
        output_chunk_length=1,
        # likelihood="quantile",
        # quantiles=[0.05, 0.5, 0.95]
    )

    print(f"Train {target_col}: Huấn luyện với {len(list_to_fit_series)} series.")
    # if has_any_covariate and list_to_fit_covariates:
    #     print(f"Train {target_col}: Sử dụng {len(list_to_fit_covariates)} future covariates.")
    #     model.fit(series=list_to_fit_series, future_covariates=list_to_fit_covariates)
    # else:
    #     print(f"Train {target_col}: Huấn luyện không sử dụng future covariates.")
    #     model.fit(series=list_to_fit_series)
    if lags_future_covariates_param is not None: # Kiểm tra dựa trên param đã xác định
        print(f"Train {target_col}: Sử dụng future covariates với lags: {lags_future_covariates_param}.")
        model.fit(series=list_to_fit_series, future_covariates=list_to_fit_covariates)
    else:
        print(f"Train {target_col}: Huấn luyện không sử dụng future covariates.")
        model.fit(series=list_to_fit_series) # future_covariates=None là mặc định

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    train_data_tuple = (final_train_series_dict, final_train_covariates_dict)
    test_data_tuple = (final_test_series_dict, final_test_covariates_full_dict)

    return model, train_data_tuple, test_data_tuple
# --- END OF FILE src/train_model.py ---