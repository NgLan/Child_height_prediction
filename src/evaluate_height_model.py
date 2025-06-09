from darts.metrics import rmse, mae
import pandas as pd

def evaluate_height_model(model, train_data, test_data, target_name="chiều cao"):
    """Đánh giá mô hình dự đoán (chung cho chiều cao/cân nặng)."""
    train_series_dict, train_covariates_dict = train_data
    test_series_dict, test_covariates_full_dict = test_data # Đây là toàn bộ covariates cho mỗi subjid

    predictions = {}
    successful_predictions = 0

    for subjid in list(test_series_dict.keys()):
        if subjid not in train_series_dict:
            continue
        
        current_train_series = train_series_dict[subjid]
        current_test_series = test_series_dict[subjid]
        
        # future_covariates cho hàm predict nên là toàn bộ chuỗi covariates
        # có sẵn cho khoảng thời gian mà chúng ta muốn dự đoán.
        # Darts sẽ tự động chọn phần cần thiết.
        future_covariates_for_prediction_period = None
        if model.uses_future_covariates:
            if subjid not in test_covariates_full_dict or test_covariates_full_dict[subjid] is None:
                # print(f"Cảnh báo (Đánh giá {target_name}, Subjid {subjid}): Mô hình yêu cầu future_covariates nhưng không có hoặc là None. Bỏ qua.")
                continue
            future_covariates_for_prediction_period = test_covariates_full_dict[subjid]
            # Không cần slice ở đây. Model.predict sẽ xử lý.
            # Đảm bảo rằng test_covariates_full_dict[subjid] bao phủ ít nhất khoảng thời gian của current_test_series
            if future_covariates_for_prediction_period.end_time() < current_test_series.end_time() or \
               future_covariates_for_prediction_period.start_time() > current_test_series.start_time():
                # print(f"Cảnh báo (Đánh giá {target_name}, Subjid {subjid}): Covariates không bao phủ toàn bộ test period. Bỏ qua.")
                # print(f"  Test Series: {current_test_series.time_index[[0,-1]]}")
                # print(f"  Covariates:  {future_covariates_for_prediction_period.time_index[[0,-1]]}")
                continue
        
        try:
            # SỬA ĐỔI Ở ĐÂY: sử dụng model.min_train_series_length thay vì model.lags
            if len(current_train_series) < model.min_train_series_length:
                 # print(f"Cảnh báo (Đánh giá {target_name}, Subjid {subjid}): Train series ({len(current_train_series)}) quá ngắn so với model.min_train_series_length ({model.min_train_series_length}). Bỏ qua.")
                 continue

            pred = model.predict(
                n=len(current_test_series),
                series=current_train_series,
                future_covariates=future_covariates_for_prediction_period # Truyền toàn bộ covariates có sẵn
            )
            predictions[subjid] = pred
            successful_predictions += 1
        except Exception as e:
            print(f"Lỗi khi dự đoán cho subjid {subjid} (Đánh giá {target_name}): {e}")
            # import traceback
            # traceback.print_exc() # Để xem chi tiết lỗi
            # print(f"  Train series len: {len(current_train_series)}, Test series len: {len(current_test_series)}")
            # if future_covariates_for_prediction_period is not None:
            #     print(f"  Future covariates for pred period start: {future_covariates_for_prediction_period.start_time()}, end: {future_covariates_for_prediction_period.end_time()}, len: {len(future_covariates_for_prediction_period)}")
            continue
            
    if successful_predictions == 0:
        print(f"Không có dự đoán nào thành công cho mô hình {target_name}.")
        return {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}, {}

    metrics = {'RMSE': [], 'MAE': []}
    for subjid_key, pred_series in predictions.items():
        if subjid_key in test_series_dict:
            actual_series = test_series_dict[subjid_key]
            try:
                rmse_val = rmse(actual_series, pred_series)
                mae_val = mae(actual_series, pred_series)
                metrics['RMSE'].append(rmse_val)
                metrics['MAE'].append(mae_val)
            except Exception as e_metric:
                # print(f"Lỗi khi tính metrics cho subjid {subjid_key} (Đánh giá {target_name}): {e_metric}")
                continue
    
    if not metrics['RMSE']: 
         print(f"Không thể tính toán metrics nào cho {target_name}.")
         return {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}, predictions

    metrics_summary = {
        'Mean_RMSE': sum(metrics['RMSE']) / len(metrics['RMSE']) if metrics['RMSE'] else float('nan'),
        'Mean_MAE': sum(metrics['MAE']) / len(metrics['MAE']) if metrics['MAE'] else float('nan')
    }
    
    return metrics_summary, predictions