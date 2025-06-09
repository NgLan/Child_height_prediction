import matplotlib.pyplot as plt
import os
import math # For math.ceil

def plot_individual_predictions(predictions, test_series, target_name_display, unit_display, figures_path, model_type_prefix, limit):
    """Vẽ biểu đồ so sánh dự đoán và thực tế cho một loại (chiều cao hoặc cân nặng)."""
    
    subjids_to_plot = list(predictions.keys())
    if len(subjids_to_plot) > limit:
        print(f"Giới hạn vẽ {limit} biểu đồ cá nhân (có tổng cộng {len(subjids_to_plot)}).")
        subjids_to_plot = subjids_to_plot[:limit]
    
    for subjid in subjids_to_plot:
        if subjid not in test_series:
            print(f"Bỏ qua vẽ cho subjid {subjid} ({model_type_prefix}) do không có trong test_series.")
            continue
            
        actual = test_series[subjid]
        predicted = predictions[subjid]
        
        # Thêm kiểm tra None và độ dài trước khi vẽ
        if actual is None or predicted is None or len(actual) == 0 or len(predicted) == 0:
            print(f"Dữ liệu không đủ hoặc None cho subjid {subjid} ({model_type_prefix}). Bỏ qua vẽ biểu đồ cá nhân.")
            continue

        plt.figure(figsize=(10, 6))
        try:
            actual.plot(label='Thực tế')
            predicted.plot(label='Dự đoán')
            plt.title(f'Dự đoán {target_name_display} cho trẻ {subjid}')
            plt.xlabel('Thời gian (ngày)')
            plt.ylabel(f'{target_name_display} ({unit_display})')
            plt.legend()
            
            os.makedirs(figures_path, exist_ok=True)
            plt.savefig(os.path.join(figures_path, f'{model_type_prefix}_prediction_{subjid}.png'))
            plt.show() # Gỡ comment nếu muốn hiển thị
            # plt.close() # Đóng figure sau khi lưu để tránh hiển thị nhiều cửa sổ
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ cá nhân cho {subjid} ({model_type_prefix}): {e}")
            plt.close()


def plot_evaluation_metrics(metrics, figures_path, model_type_prefix):
    """Vẽ biểu đồ các chỉ số đánh giá cho một loại mô hình."""
    if not metrics or all(math.isnan(v) for v in metrics.values()): 
        print(f"Không có metrics hợp lệ để vẽ cho {model_type_prefix}.")
        return

    plt.figure(figsize=(8, 5))
    
    valid_metrics = {k: v for k, v in metrics.items() if not (isinstance(v, float) and math.isnan(v))} 
    if not valid_metrics:
        print(f"Tất cả giá trị metrics là NaN cho {model_type_prefix}. Không vẽ biểu đồ.")
        plt.close()
        return

    plt.bar(valid_metrics.keys(), valid_metrics.values(), color=['blue', 'orange'])
    plt.title(f'Chỉ số đánh giá mô hình ({model_type_prefix.replace("_", " ").title()})')
    plt.ylabel('Giá trị')
    for i, (k, v) in enumerate(valid_metrics.items()): 
        plt.text(i, v + 0.01 * abs(v) if v != 0 else 0.01, f'{v:.2f}', ha='center', va='bottom') 
    
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, f'{model_type_prefix}_evaluation_metrics.png'))
    plt.show() 
    # plt.close()

def plot_combined_child_predictions(
    predictions_h, test_series_h,
    predictions_w, test_series_w,
    common_subjids_list, figures_path,
    limit
):
    """Vẽ biểu đồ dự đoán chiều cao và cân nặng của cùng 1 đứa trẻ trên cùng 1 dòng."""
    num_children_to_plot = min(len(common_subjids_list), limit)
    if num_children_to_plot == 0:
        print("Không có trẻ em chung nào để vẽ biểu đồ gộp.")
        return

    num_rows = math.ceil(num_children_to_plot / 2)
    
    fig, axs = plt.subplots(num_rows, 4, figsize=(22, 5 * num_rows), squeeze=False) 
    fig.suptitle('So sánh Dự đoán Chiều cao và Cân nặng', fontsize=16, y=1.02) 

    plot_count = 0
    for i in range(num_children_to_plot):
        subjid = common_subjids_list[i]
        
        current_row = i // 2
        col_offset_height = (i % 2) * 2       
        col_offset_weight = col_offset_height + 1 

        # Plot chiều cao
        ax_h = axs[current_row, col_offset_height]
        title_h_set = False
        if subjid in test_series_h and subjid in predictions_h:
            ts_actual_h = test_series_h[subjid]
            ts_pred_h = predictions_h[subjid]

            if ts_actual_h is not None and ts_pred_h is not None:
                try:
                    # SỬA ĐỔI: Kiểm tra độ dài thay vì .pd_dataframe().empty
                    if len(ts_actual_h) > 0 and len(ts_pred_h) > 0:
                        ts_actual_h.plot(label='Thực tế (Cao)', ax=ax_h, color='blue')
                        ts_pred_h.plot(label='Dự đoán (Cao)', ax=ax_h, color='skyblue', linestyle='--')
                        ax_h.set_title(f'ID {subjid} - Chiều cao')
                        ax_h.set_ylabel('Chiều cao (cm)')
                        ax_h.legend()
                        title_h_set = True
                        plot_count +=1
                    else:
                        # print(f"Dữ liệu rỗng (kiểm tra bằng len) cho chiều cao của subjid {subjid}")
                        pass 
                except Exception as e:
                    print(f"Lỗi vẽ chiều cao cho {subjid}: {e}")
            # else:
                # print(f"Một trong các series chiều cao là None cho subjid {subjid}")
        
        if not title_h_set: 
            ax_h.set_title(f'ID {subjid} - Thiếu/Lỗi dữ liệu chiều cao')
            ax_h.axis('off')
        ax_h.set_xlabel('Thời gian (ngày)')


        # Plot cân nặng
        ax_w = axs[current_row, col_offset_weight]
        title_w_set = False
        if subjid in test_series_w and subjid in predictions_w:
            ts_actual_w = test_series_w[subjid]
            ts_pred_w = predictions_w[subjid]

            if ts_actual_w is not None and ts_pred_w is not None:
                try:
                    # SỬA ĐỔI: Kiểm tra độ dài thay vì .pd_dataframe().empty
                    if len(ts_actual_w) > 0 and len(ts_pred_w) > 0:
                        ts_actual_w.plot(label='Thực tế (Nặng)', ax=ax_w, color='red')
                        ts_pred_w.plot(label='Dự đoán (Nặng)', ax=ax_w, color='salmon', linestyle='--')
                        ax_w.set_title(f'ID {subjid} - Cân nặng')
                        ax_w.set_ylabel('Cân nặng (kg)')
                        ax_w.legend()
                        title_w_set = True
                        plot_count +=1
                    else:
                        # print(f"Dữ liệu rỗng (kiểm tra bằng len) cho cân nặng của subjid {subjid}")
                        pass
                except Exception as e:
                    print(f"Lỗi vẽ cân nặng cho {subjid}: {e}")
            # else:
                # print(f"Một trong các series cân nặng là None cho subjid {subjid}")


        if not title_w_set:
            ax_w.set_title(f'ID {subjid} - Thiếu/Lỗi dữ liệu cân nặng')
            ax_w.axis('off')
        ax_w.set_xlabel('Thời gian (ngày)')

    total_axes_per_child_pair = 4
    total_child_pairs = num_rows
    
    for r_idx_flat in range(total_axes_per_child_pair * total_child_pairs):
        row_idx = r_idx_flat // 4 
        col_idx = r_idx_flat % 4
        
        child_idx_for_axis = row_idx * 2 + (col_idx // 2) 

        if child_idx_for_axis >= num_children_to_plot:
            if row_idx < axs.shape[0] and col_idx < axs.shape[1]: 
                if not axs[row_idx, col_idx].get_title(): # Chỉ ẩn nếu chưa được dùng
                    axs[row_idx, col_idx].axis('off')
    
    if plot_count == 0: 
        plt.close(fig) 
        print("Không có dữ liệu hợp lệ nào để vẽ trong biểu đồ gộp.")
        return

    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    
    os.makedirs(figures_path, exist_ok=True)
    save_path = os.path.join(figures_path, 'combined_height_weight_predictions.png')
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ gộp tại: {save_path}")
    plt.show() 