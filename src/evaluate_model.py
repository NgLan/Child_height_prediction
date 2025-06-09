import pandas as pd
from darts import TimeSeries
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error 

def evaluate_model(model, train_data, test_data, target_name):
    """
    Hàm này có nhiệm vụ:
    1. Sử dụng mô hình đã huấn luyện (`model`) để tạo dự đoán trên dữ liệu kiểm tra (`test_data`).
    2. Áp dụng phương pháp backtesting (cửa sổ trượt).
    3. Tính toán các chỉ số lỗi (RMSE, MAE) giữa dự đoán và giá trị thực tế.
    4. Trả về các chỉ số lỗi, các chuỗi dự đoán, và dữ liệu test để vẽ biểu đồ.
    """
    train_series_list, _ = train_data
    test_series_list, test_cov_list = test_data

    predictions_dict = {}
    
    min_len_for_forecast_start = model.min_train_series_length
    
    print(f"\nĐánh giá cho mô hình '{target_name}'. Yêu cầu start_index = {min_len_for_forecast_start}.")
    print(f"Tổng số series trong tập test ban đầu: {len(test_series_list)}")

    valid_series_to_eval = []
    valid_covariates_to_eval = []
    original_indices_map = [] 

    for i, series in enumerate(test_series_list):
        if len(series) > min_len_for_forecast_start:
            valid_series_to_eval.append(series)
            if test_cov_list:
                valid_covariates_to_eval.append(test_cov_list[i])
            original_indices_map.append(i)
        else:
            subjid_debug = f"SubjID {series.static_covariates['subjid'].iloc[0]}" if series.has_static_covariates else f"series_index_{i}"
            print(f"CẢNH BÁO: Loại bỏ series '{subjid_debug}' (index {i}) vì có độ dài {len(series)}, không đủ cho start_index={min_len_for_forecast_start}.")

    if not valid_series_to_eval:
        print(f"LỖI: Không có series nào trong tập test đủ dài để đánh giá cho mô hình '{target_name}'.")
        empty_metrics = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
        return empty_metrics, {}, ({}, {})

    print(f"Bắt đầu đánh giá trên {len(valid_series_to_eval)} series hợp lệ (đủ dài).")

    try:
        list_of_forecast_lists = model.historical_forecasts(
            series=valid_series_to_eval,
            future_covariates=valid_covariates_to_eval if valid_covariates_to_eval else None,
            start=min_len_for_forecast_start, 
            forecast_horizon=1,
            stride=1,
            retrain=False, # Không huấn luyện lại mô hình trong quá trình backtesting.
            verbose=False,
            last_points_only=False # Trả về toàn bộ chuỗi dự đoán, không chỉ điểm cuối.
        )
    except Exception as e:
        print(f"\nLỖI KHI DÙNG historical_forecasts: {e}.")
        import traceback
        traceback.print_exc()
        list_of_forecast_lists = []

    if not list_of_forecast_lists:
        print(f"Không có dự đoán nào thành công cho mô hình {target_name}.")
        empty_metrics = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
        return empty_metrics, {}, ({}, {})

    # --- TÍNH METRICS ---
    
    all_rmse_scores = []
    all_mae_scores = []

    for i, single_child_forecasts in enumerate(list_of_forecast_lists):
        # Lấy series thực tế tương ứng
        actual_series = valid_series_to_eval[i]
        
        if not single_child_forecasts:
            continue
            
        # 1. Nối tất cả các đoạn dự đoán lại thành 1 chuỗi dài
        combined_pred_df = pd.concat([ts.to_dataframe() for ts in single_child_forecasts])
        combined_pred_df = combined_pred_df[~combined_pred_df.index.duplicated(keep='first')].sort_index()
        
        # 2. Cắt chuỗi thực tế để khớp với khoảng thời gian của chuỗi dự đoán
        actual_df = actual_series.to_dataframe()
        common_index = actual_df.index.intersection(combined_pred_df.index)
        
        y_true = actual_df.loc[common_index].values.flatten()
        y_pred = combined_pred_df.loc[common_index].values.flatten()

        if len(y_true) > 0:
            # 3. Tính RMSE và MAE bằng sklearn
            current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            current_mae = mean_absolute_error(y_true, y_pred)
            
            all_rmse_scores.append(current_rmse)
            all_mae_scores.append(current_mae)
        
        # Xây dựng `predictions_dict` để trả về cho việc vẽ biểu đồ
        original_test_index = original_indices_map[i]
        corresponding_test_series = test_series_list[original_test_index]
        subjid = corresponding_test_series.static_covariates['subjid'].iloc[0]

        # Tạo TimeSeries từ DataFrame đã nối để vẽ biểu đồ
        predictions_dict[subjid] = TimeSeries.from_dataframe(combined_pred_df)

    # Tính toán giá trị trung bình của các scores
    mean_rmse = np.mean(all_rmse_scores) if all_rmse_scores else float('nan')
    mean_mae = np.mean(all_mae_scores) if all_mae_scores else float('nan')

    metrics_summary = {
        'Mean_RMSE': mean_rmse,
        'Mean_MAE': mean_mae
    }
    
    # Tạo lại test_series_dict để trả về cho việc vẽ biểu đồ
    test_series_dict = {}
    for series in test_series_list:
        subjid = series.static_covariates['subjid'].iloc[0]
        test_series_dict[subjid] = series
    
    test_data_for_plotting = (test_series_dict, {})

    return metrics_summary, predictions_dict, test_data_for_plotting