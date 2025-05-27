# # --- START OF FILE main.py ---
# import os
# import pandas as pd

# # Modules cho Chiều cao
# from src.data_processing import load_and_preprocess_data as load_and_preprocess_height_data, interpolate_data as interpolate_height_data
# from src.train_model import train_linear_regression_model # Đã đổi tên trong train_model.py
# from src.evaluate_model import evaluate_model # Hàm evaluate chung

# # Modules cho Cân nặng
# from src.data_processing_weight import load_and_preprocess_data_for_weight, interpolate_weight_data
# from src.train_weight_model import train_linear_regression_model_for_weight
# from src.evaluate_weight_model import evaluate_model_for_weight # Gọi hàm evaluate chung với tên khác

# # Module trực quan hóa chung
# from src.visualization import plot_individual_predictions, plot_evaluation_metrics, plot_combined_child_predictions

# def main():
#     # Đường dẫn file
#     # Giả sử main.py nằm ở thư mục gốc của dự án (ví dụ: D:\Child_height_prediction)
#     # và các thư mục con data, models, figures, src nằm cùng cấp với main.py
#     base_path = os.path.dirname(os.path.abspath(__file__)) # Lấy đường dẫn của thư mục chứa main.py
    
#     data_path = os.path.join(base_path, "data", "childhealth-dutch.csv")
#     model_height_save_path = os.path.join(base_path, "models", "linear_regression_model_height.pkl")
#     model_weight_save_path = os.path.join(base_path, "models", "linear_regression_model_weight.pkl")
#     figures_path = os.path.join(base_path, "figures") # figures_path chỉ là thư mục

#     # Tạo thư mục nếu chưa tồn tại
#     os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
#     os.makedirs(figures_path, exist_ok=True)

#     # --- QUY TRÌNH CHO CHIỀU CAO ---
#     print("===== BẮT ĐẦU QUY TRÌNH DỰ ĐOÁN CHIỀU CAO =====")
#     df_height_interpolated = None
#     predictions_height = {}
#     metrics_height = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
#     test_data_height_for_plot = None

#     try:
#         print("Đang xử lý dữ liệu chiều cao...")
#         df_height_raw = load_and_preprocess_height_data(data_path)
#         if not df_height_raw.empty:
#             df_height_interpolated = interpolate_height_data(df_height_raw)
        
#         if df_height_interpolated is None or df_height_interpolated.empty:
#             print("LỖI (Chiều cao): Không có dữ liệu sau tiền xử lý/nội suy.")
#         else:
#             print("Đang huấn luyện mô hình chiều cao...")
#             model_height, train_data_height, test_data_height = train_linear_regression_model(
#                 df_height_interpolated, 
#                 model_height_save_path,
#                 target_col='htcm',
#                 covariate_cols=['wtkg', 'haz', 'waz', 'sex', 'gagebrth'] # Covariates cho chiều cao
#             )
#             print(f"Đã lưu mô hình chiều cao tại: {model_height_save_path}")
#             test_data_height_for_plot = test_data_height # Lưu lại để vẽ

#             print("Đang đánh giá mô hình chiều cao...")
#             metrics_height, predictions_height = evaluate_model(model_height, train_data_height, test_data_height, target_name="chiều cao")
#             print("Kết quả đánh giá chiều cao:", metrics_height)

#     except Exception as e:
#         print(f"Lỗi nghiêm trọng trong quy trình chiều cao: {e}")
#         import traceback
#         traceback.print_exc()


#     # --- QUY TRÌNH CHO CÂN NẶNG ---
#     print("\n===== BẮT ĐẦU QUY TRÌNH DỰ ĐOÁN CÂN NẶNG =====")
#     df_weight_interpolated = None
#     predictions_weight = {}
#     metrics_weight = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
#     test_data_weight_for_plot = None
#     try:
#         print("Đang xử lý dữ liệu cân nặng...")
#         df_weight_raw = load_and_preprocess_data_for_weight(data_path)
#         if not df_weight_raw.empty:
#             df_weight_interpolated = interpolate_weight_data(df_weight_raw)

#         if df_weight_interpolated is None or df_weight_interpolated.empty:
#             print("LỖI (Cân nặng): Không có dữ liệu sau tiền xử lý/nội suy.")
#         else:
#             print("Đang huấn luyện mô hình cân nặng...")
#             model_weight, train_data_weight, test_data_weight = train_linear_regression_model_for_weight(
#                 df_weight_interpolated, 
#                 model_weight_save_path
#             )
#             print(f"Đã lưu mô hình cân nặng tại: {model_weight_save_path}")
#             test_data_weight_for_plot = test_data_weight # Lưu lại để vẽ

#             print("Đang đánh giá mô hình cân nặng...")
#             metrics_weight, predictions_weight = evaluate_model_for_weight(model_weight, train_data_weight, test_data_weight)
#             print("Kết quả đánh giá cân nặng:", metrics_weight)
            
#     except Exception as e:
#         print(f"Lỗi nghiêm trọng trong quy trình cân nặng: {e}")
#         import traceback
#         traceback.print_exc()


#     # --- TRỰC QUAN HÓA ---
#     print("\n===== ĐANG TẠO BIỂU ĐỒ =====")
#     if predictions_height and test_data_height_for_plot:
#         test_series_height, _ = test_data_height_for_plot
#         plot_individual_predictions(predictions_height, test_series_height, "Chiều cao", "cm", figures_path, "height")
#     if not pd.isna(metrics_height['Mean_RMSE']): # Chỉ vẽ nếu có metric hợp lệ
#         plot_evaluation_metrics(metrics_height, figures_path, "height")

#     if predictions_weight and test_data_weight_for_plot:
#         test_series_weight, _ = test_data_weight_for_plot
#         plot_individual_predictions(predictions_weight, test_series_weight, "Cân nặng", "kg", figures_path, "weight")
#     if not pd.isna(metrics_weight['Mean_RMSE']):
#         plot_evaluation_metrics(metrics_weight, figures_path, "weight")

#     if predictions_height and predictions_weight and test_data_height_for_plot and test_data_weight_for_plot:
#         test_series_h, _ = test_data_height_for_plot
#         test_series_w, _ = test_data_weight_for_plot
        
#         common_subjids = set(predictions_height.keys()) & set(test_series_h.keys()) & \
#                          set(predictions_weight.keys()) & set(test_series_w.keys())
        
#         if common_subjids:
#             plot_combined_child_predictions(
#                 predictions_height, test_series_h,
#                 predictions_weight, test_series_w,
#                 list(common_subjids), figures_path
#             )
#         else:
#             print("Không có subjid chung giữa dự đoán chiều cao và cân nặng để vẽ biểu đồ gộp.")
            
#     print("\n===== HOÀN THÀNH TẤT CẢ QUY TRÌNH =====")

# if __name__ == "__main__":
#     main()
# # --- END OF FILE main.py ---

# --- START OF FILE main.py ---
import os
import pandas as pd

# Modules cho Chiều cao
from src.data_processing import load_and_preprocess_data as load_and_preprocess_height_data, interpolate_data as interpolate_height_data
from src.train_model import train_linear_regression_model
from src.evaluate_model import evaluate_model

# Modules cho Cân nặng
from src.data_processing_weight import load_and_preprocess_data_for_weight, interpolate_weight_data
from src.train_weight_model import train_linear_regression_model_for_weight
from src.evaluate_weight_model import evaluate_model_for_weight

# Module trực quan hóa chung
from src.visualization import plot_individual_predictions, plot_evaluation_metrics, plot_combined_child_predictions

# Module tính Z-score
from src.z_score_calculator import load_who_references, analyze_child_growth_from_predictions # ĐÃ THÊM

def main():
    # Đường dẫn file
    base_path = os.path.dirname(os.path.abspath(__file__)) 
    
    data_path = os.path.join(base_path, "data", "childhealth-dutch.csv")
    model_height_save_path = os.path.join(base_path, "models", "linear_regression_model_height.pkl")
    model_weight_save_path = os.path.join(base_path, "models", "linear_regression_model_weight.pkl")
    figures_path = os.path.join(base_path, "figures") 

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    # Đảm bảo thư mục tham chiếu WHO tồn tại (nếu bạn đặt file CSV ở đó)
    os.makedirs(os.path.join(base_path, "data", "who_references"), exist_ok=True) 

    # --- TẢI DỮ LIỆU THAM CHIẾU WHO CHO Z-SCORE ---
    print("\n===== TẢI DỮ LIỆU THAM CHIẾU WHO CHO Z-SCORE =====")
    who_references = load_who_references()
    # Định nghĩa ngày bắt đầu tham chiếu được sử dụng trong train_model.py để tính agedays
    # Đây là pd.Timestamp('2000-01-01') theo train_model.py
    z_score_start_date_ref = pd.Timestamp('2000-01-01')


    # --- QUY TRÌNH CHO CHIỀU CAO ---
    print("\n===== BẮT ĐẦU QUY TRÌNH DỰ ĐOÁN CHIỀU CAO =====")
    df_height_interpolated = None
    predictions_height = {}
    metrics_height = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
    test_data_height_for_plot = None
    all_subj_sex_map = {} # Để lưu trữ giới tính cho mỗi subjid

    try:
        print("Đang xử lý dữ liệu chiều cao...")
        df_height_raw = load_and_preprocess_height_data(data_path)
        if not df_height_raw.empty:
            # Lưu trữ giới tính cho tính toán Z-score sau này
            # Giả sử 'subjid' và 'sex' có mặt sau tiền xử lý
            # Và sex là 0 cho nữ, 1 cho nam
            if 'subjid' in df_height_raw.columns and 'sex' in df_height_raw.columns:
                temp_sex_map_h = df_height_raw.drop_duplicates(subset=['subjid'])[['subjid', 'sex']].set_index('subjid')['sex'].to_dict()
                all_subj_sex_map.update(temp_sex_map_h)
            
            df_height_interpolated = interpolate_height_data(df_height_raw)
        
        if df_height_interpolated is None or df_height_interpolated.empty:
            print("LỖI (Chiều cao): Không có dữ liệu sau tiền xử lý/nội suy.")
        else:
            print("Đang huấn luyện mô hình chiều cao...")
            model_height, train_data_height, test_data_height = train_linear_regression_model(
                df_height_interpolated, 
                model_height_save_path,
                target_col='htcm',
                covariate_cols=['wtkg', 'haz', 'waz', 'sex', 'gagebrth']
            )
            print(f"Đã lưu mô hình chiều cao tại: {model_height_save_path}")
            test_data_height_for_plot = test_data_height

            print("Đang đánh giá mô hình chiều cao...")
            metrics_height, predictions_height = evaluate_model(model_height, train_data_height, test_data_height, target_name="chiều cao")
            print("Kết quả đánh giá chiều cao:", metrics_height)

    except Exception as e:
        print(f"Lỗi nghiêm trọng trong quy trình chiều cao: {e}")
        import traceback
        traceback.print_exc()


    # --- QUY TRÌNH CHO CÂN NẶNG ---
    print("\n===== BẮT ĐẦU QUY TRÌNH DỰ ĐOÁN CÂN NẶNG =====")
    df_weight_interpolated = None
    predictions_weight = {}
    metrics_weight = {'Mean_RMSE': float('nan'), 'Mean_MAE': float('nan')}
    test_data_weight_for_plot = None
    try:
        print("Đang xử lý dữ liệu cân nặng...")
        df_weight_raw = load_and_preprocess_data_for_weight(data_path)
        if not df_weight_raw.empty:
            # Lưu trữ/cập nhật giới tính cho tính toán Z-score
            if 'subjid' in df_weight_raw.columns and 'sex' in df_weight_raw.columns:
                temp_sex_map_w = df_weight_raw.drop_duplicates(subset=['subjid'])[['subjid', 'sex']].set_index('subjid')['sex'].to_dict()
                all_subj_sex_map.update(temp_sex_map_w) # Cập nhật, phòng trường hợp một số subjid chỉ có trong dữ liệu cân nặng

            df_weight_interpolated = interpolate_weight_data(df_weight_raw)

        if df_weight_interpolated is None or df_weight_interpolated.empty:
            print("LỖI (Cân nặng): Không có dữ liệu sau tiền xử lý/nội suy.")
        else:
            print("Đang huấn luyện mô hình cân nặng...")
            model_weight, train_data_weight, test_data_weight = train_linear_regression_model_for_weight(
                df_weight_interpolated, 
                model_weight_save_path
            )
            print(f"Đã lưu mô hình cân nặng tại: {model_weight_save_path}")
            test_data_weight_for_plot = test_data_weight

            print("Đang đánh giá mô hình cân nặng...")
            metrics_weight, predictions_weight = evaluate_model_for_weight(model_weight, train_data_weight, test_data_weight)
            print("Kết quả đánh giá cân nặng:", metrics_weight)
            
    except Exception as e:
        print(f"Lỗi nghiêm trọng trong quy trình cân nặng: {e}")
        import traceback
        traceback.print_exc()

    # --- TÍNH TOÁN VÀ PHÂN LOẠI Z-SCORE ---
    print("\n===== TÍNH TOÁN Z-SCORE VÀ PHÂN LOẠI =====")
    if who_references and predictions_height and predictions_weight:
        # Lặp qua các subjid có cả dự đoán chiều cao và cân nặng
        common_pred_subjids = set(predictions_height.keys()) & set(predictions_weight.keys())
        
        # Giới hạn số trẻ em được phân tích Z-score để output không quá dài (ví dụ: 5 trẻ đầu tiên)
        subjids_for_zscore_analysis = list(common_pred_subjids)[:5] 
        print(f"Phân tích Z-score cho {len(subjids_for_zscore_analysis)} trẻ em đầu tiên có đủ dữ liệu dự đoán.")

        for subjid in subjids_for_zscore_analysis: # Thay common_pred_subjids bằng subjids_for_zscore_analysis
            if subjid not in all_subj_sex_map:
                print(f"Subjid {subjid}: Không tìm thấy thông tin giới tính. Bỏ qua Z-score.")
                continue
            
            child_sex = all_subj_sex_map[subjid]
            pred_h_ts = predictions_height[subjid]
            pred_w_ts = predictions_weight[subjid]
            
            analyze_child_growth_from_predictions(
                subjid, 
                child_sex, 
                pred_h_ts, 
                pred_w_ts, 
                who_references,
                z_score_start_date_ref # Truyền ngày bắt đầu tham chiếu
            )
    else:
        if not who_references:
            print("Không thể tính Z-score do thiếu dữ liệu tham chiếu WHO.")
        elif not predictions_height or not predictions_weight:
             print("Không thể tính Z-score do thiếu dữ liệu dự đoán chiều cao hoặc cân nặng.")
        else:
            print("Không thể tính Z-score vì một lý do không xác định (thiếu dữ liệu chung).")


    # --- TRỰC QUAN HÓA ---
    print("\n===== ĐANG TẠO BIỂU ĐỒ =====")
    if predictions_height and test_data_height_for_plot and test_data_height_for_plot[0]:
        test_series_height, _ = test_data_height_for_plot
        plot_individual_predictions(predictions_height, test_series_height, "Chiều cao", "cm", figures_path, "height")
    # Kiểm tra metrics_height['Mean_RMSE'] có phải là float NaN không
    if isinstance(metrics_height.get('Mean_RMSE'), float) and not pd.isna(metrics_height['Mean_RMSE']):
        plot_evaluation_metrics(metrics_height, figures_path, "height")

    if predictions_weight and test_data_weight_for_plot and test_data_weight_for_plot[0]:
        test_series_weight, _ = test_data_weight_for_plot
        plot_individual_predictions(predictions_weight, test_series_weight, "Cân nặng", "kg", figures_path, "weight")
    if isinstance(metrics_weight.get('Mean_RMSE'), float) and not pd.isna(metrics_weight['Mean_RMSE']):
        plot_evaluation_metrics(metrics_weight, figures_path, "weight")

    if predictions_height and predictions_weight and \
       test_data_height_for_plot and test_data_height_for_plot[0] and \
       test_data_weight_for_plot and test_data_weight_for_plot[0]:
        
        test_series_h, _ = test_data_height_for_plot
        test_series_w, _ = test_data_weight_for_plot
        
        valid_common_subjids_for_plot = []
        # Sử dụng tập hợp các subjid đã có dự đoán thành công cho cả chiều cao và cân nặng
        # và cũng có trong test_series tương ứng.
        subjids_with_preds = set(predictions_height.keys()) & set(predictions_weight.keys())
        subjids_in_test_h = set(test_series_h.keys())
        subjids_in_test_w = set(test_series_w.keys())

        potential_common_subjids_plot = subjids_with_preds & subjids_in_test_h & subjids_in_test_w
        
        for s_id in potential_common_subjids_plot:
            pred_h_valid = predictions_height[s_id] is not None and len(predictions_height[s_id]) > 0
            pred_w_valid = predictions_weight[s_id] is not None and len(predictions_weight[s_id]) > 0
            actual_h_valid = test_series_h[s_id] is not None and len(test_series_h[s_id]) > 0
            actual_w_valid = test_series_w[s_id] is not None and len(test_series_w[s_id]) > 0

            if pred_h_valid and pred_w_valid and actual_h_valid and actual_w_valid:
                valid_common_subjids_for_plot.append(s_id)
        
        if valid_common_subjids_for_plot:
            plot_combined_child_predictions(
                predictions_height, test_series_h,
                predictions_weight, test_series_w,
                list(valid_common_subjids_for_plot), figures_path
            )
        else:
            print("Không có subjid chung với dữ liệu hợp lệ giữa dự đoán chiều cao và cân nặng để vẽ biểu đồ gộp.")
            
    print("\n===== HOÀN THÀNH TẤT CẢ QUY TRÌNH =====")

if __name__ == "__main__":
    main()
# --- END OF FILE main.py ---