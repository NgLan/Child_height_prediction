import os
import pandas as pd
import config  

from src.data_processing import preprocess_data, interpolate_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.visualization import plot_individual_predictions, plot_evaluation_metrics, plot_combined_child_predictions, plot_z_score_trends
from src.z_score_calculator import load_who_references, analyze_child_growth_from_predictions

def main():
    # --- 1. THIẾT LẬP ---
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.WHO_REF_DIR, exist_ok=True)

    print("\n===== TẢI DỮ LIỆU THAM CHIẾU WHO =====")
    who_references = load_who_references()
    z_score_start_date_ref = pd.Timestamp(config.REFERENCE_START_DATE)

    # --- 2. VÒNG LẶP XỬ LÝ CHÍNH CHO TỪNG MÔ HÌNH ---
    all_predictions = {}
    all_metrics = {}
    all_test_data_for_plot = {}
    all_subj_sex_map = {}

    for model_key, model_conf in config.MODEL_CONFIGS.items():
        display_name = model_conf['display_name']
        print(f"\n===== BẮT ĐẦU QUY TRÌNH DỰ ĐOÁN {display_name.upper()} =====")
        
        try:
            # Bước 1: Tiền xử lý dữ liệu
            df_processed = preprocess_data(
                config.INPUT_DATA_FILE,
                model_conf['target_col'],
                model_conf['covariate_cols']
            )
            
            # Cập nhật bản đồ giới tính
            if not df_processed.empty and 'subjid' in df_processed.columns and 'sex' in df_processed.columns:
                temp_sex_map = df_processed.drop_duplicates(subset=['subjid'])[['subjid', 'sex']].set_index('subjid')['sex'].to_dict()
                all_subj_sex_map.update(temp_sex_map)

            # Bước 2: Nội suy dữ liệu
            df_interpolated = interpolate_data(
                df_processed,
                model_conf['target_col'],
                model_conf['covariate_cols'],
                step=config.INTERPOLATION_STEP
            )

            if not df_interpolated.empty:
                if model_key == 'height':
                    save_path = os.path.join(config.PROCESSED_DATA_DIR, config.INTERPOLATED_HEIGHT_FILE)
                else:
                    save_path = os.path.join(config.PROCESSED_DATA_DIR, config.INTERPOLATED_WEIGHT_FILE)
                
                # Lưu DataFrame vào file CSV
                df_interpolated.to_csv(save_path, index=False)
                print(f"Đã lưu dữ liệu nội suy cho '{display_name}' tại: {save_path}")
            else:
                print(f"LỖI ({display_name}): Không có dữ liệu sau khi nội suy.")
                all_metrics[model_key] = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
                continue

            # Bước 3: Huấn luyện mô hình
            model_save_path = os.path.join(config.MODELS_DIR, model_conf['model_filename'])
            trained_model, train_data, test_data = train_model(
                df=df_interpolated,
                model_save_path=model_save_path,
                target_col=model_conf['target_col'],
                covariate_cols=model_conf['covariate_cols'],
                model_params=config.MODEL_PARAMS,
                start_date_str=config.REFERENCE_START_DATE,
                freq=config.TIME_SERIES_FREQ,
                split_ratio=config.TRAIN_TEST_SPLIT_RATIO,
                min_series_len=config.MIN_SERIES_LENGTH
            )
            print(f"Đã lưu mô hình {display_name} tại: {model_save_path}")

            # Bước 4: Đánh giá mô hình
            # Lưu ý: Hàm evaluate_model đã được sửa để trả về dữ liệu test đã định dạng cho plotting
            metrics, predictions, test_data_for_plot = evaluate_model(
                trained_model, train_data, test_data, display_name
            )
            
            all_metrics[model_key] = metrics
            all_predictions[model_key] = predictions
            all_test_data_for_plot[model_key] = test_data_for_plot
            print(f"Kết quả đánh giá {display_name}:", metrics)

        except Exception as e:
            print(f"LỖI NGHIÊM TRỌNG trong quy trình {display_name}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[model_key] = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}

    # --- 3. PHÂN TÍCH Z-SCORE ---
    print("\n===== TÍNH TOÁN Z-SCORE VÀ PHÂN LOẠI =====")
    predictions_h = all_predictions.get('height', {})
    predictions_w = all_predictions.get('weight', {})
    
    all_zscore_analyses = {}

    if who_references and predictions_h and predictions_w:
        common_subjids = set(predictions_h.keys()) & set(predictions_w.keys())

        print(f"Tìm thấy {len(predictions_h)} dự đoán chiều cao.")
        print(f"Tìm thấy {len(predictions_w)} dự đoán cân nặng.")
        print(f"Tìm thấy {len(common_subjids)} trẻ em có chung cả hai dự đoán.")
        
        subjids_for_analysis = list(common_subjids)[:config.ZSCORE_ANALYSIS_LIMIT]
        
        for subjid in subjids_for_analysis:
            if subjid in all_subj_sex_map:
                analysis_df = analyze_child_growth_from_predictions(
                    subjid, 
                    all_subj_sex_map[subjid], 
                    predictions_h.get(subjid), 
                    predictions_w.get(subjid), 
                    who_references, 
                    z_score_start_date_ref
                )
                if analysis_df is not None:
                    all_zscore_analyses[subjid] = analysis_df
            else:
                print(f"Cảnh báo: Không tìm thấy giới tính cho subjid {subjid}, bỏ qua phân tích Z-score.")
    else:
        print("Không thể tính Z-score do thiếu dữ liệu dự đoán hoặc tham chiếu WHO.")

    # --- 4. TRỰC QUAN HÓA ---
    print("\n===== ĐANG TẠO BIỂU ĐỒ =====")
    for model_key, model_conf in config.MODEL_CONFIGS.items():
        if model_key in all_predictions and model_key in all_test_data_for_plot:
            predictions = all_predictions[model_key]
            test_data_for_plot = all_test_data_for_plot[model_key]
            if predictions and test_data_for_plot:
                test_series_dict, _ = test_data_for_plot
                plot_individual_predictions(
                    predictions, 
                    test_series_dict, 
                    model_conf['display_name'], 
                    model_conf['unit'], 
                    config.FIGURES_DIR, 
                    model_key, 
                    z_score_start_date_ref, # <-- THÊM THAM SỐ MỚI
                    limit=config.INDIVIDUAL_PLOTS_LIMIT
                )
        if model_key in all_metrics:
            metrics = all_metrics[model_key]
            if metrics and not pd.isna(metrics.get('Mean_RMSE')):
                plot_evaluation_metrics(metrics, config.FIGURES_DIR, model_key)

    # Vẽ biểu đồ Z-score 
    if all_zscore_analyses:
        print("\nĐang tạo biểu đồ xu hướng Z-score...")
        subjids_for_zscore_plot = list(all_zscore_analyses.keys())[:config.INDIVIDUAL_PLOTS_LIMIT]
        for subjid in subjids_for_zscore_plot:
            plot_z_score_trends(subjid, all_zscore_analyses[subjid], config.FIGURES_DIR)
        
    # Vẽ biểu đồ kết hợp
    if 'height' in all_predictions and 'weight' in all_predictions:
        test_data_h = all_test_data_for_plot.get('height')
        test_data_w = all_test_data_for_plot.get('weight')
        if test_data_h and test_data_w and all_zscore_analyses:
            test_series_h_dict, _ = test_data_h
            test_series_w_dict, _ = test_data_w
            
            # Tìm các subjid có đủ tất cả các loại dữ liệu
            pred_h_keys = set(predictions_h.keys())
            pred_w_keys = set(predictions_w.keys())
            test_h_keys = set(test_series_h_dict.keys())
            test_w_keys = set(test_series_w_dict.keys())
            zscore_keys = set(all_zscore_analyses.keys())
            
            common_plot_subjids = list(pred_h_keys & pred_w_keys & test_h_keys & test_w_keys & zscore_keys)
            
            if common_plot_subjids:
                plot_combined_child_predictions(
                    predictions_h, test_series_h_dict,
                    predictions_w, test_series_w_dict,
                    all_zscore_analyses,  
                    common_plot_subjids, 
                    config.FIGURES_DIR, 
                    z_score_start_date_ref,
                    limit=config.COMBINED_PLOTS_LIMIT
                )

    print("\n===== HOÀN THÀNH TẤT CẢ QUY TRÌNH =====")

if __name__ == "__main__":
    main()