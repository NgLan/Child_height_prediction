import matplotlib.pyplot as plt
import os
import math 
import pandas as pd

def plot_individual_predictions(
    predictions, 
    test_series, 
    target_name_display, 
    unit_display, 
    figures_path, 
    model_type_prefix, 
    start_date_ref, 
    limit
):
    """
    Vẽ biểu đồ so sánh dự đoán và thực tế cho một loại (chiều cao hoặc cân nặng).
    Trục x được chuyển đổi thành agedays.
    """
    subjids_to_plot = list(predictions.keys())
    if len(subjids_to_plot) > limit:
        print(f"Giới hạn vẽ {limit} biểu đồ cá nhân (có tổng cộng {len(subjids_to_plot)}).")
        subjids_to_plot = subjids_to_plot[:limit]
    
    for subjid in subjids_to_plot:
        if subjid not in test_series:
            print(f"Bỏ qua vẽ cho subjid {subjid} ({model_type_prefix}) do không có trong test_series.")
            continue
            
        actual_ts = test_series.get(subjid)
        predicted_ts = predictions.get(subjid)
        
        if actual_ts is None or predicted_ts is None or len(actual_ts) == 0 or len(predicted_ts) == 0:
            print(f"Dữ liệu không đủ hoặc None cho subjid {subjid} ({model_type_prefix}). Bỏ qua vẽ biểu đồ cá nhân.")
            continue

        plt.figure(figsize=(10, 6))
        try:
            # Chuyển đổi TimeSeries sang DataFrame và tính agedays
            df_actual = actual_ts.to_dataframe()
            df_actual['agedays'] = (df_actual.index - start_date_ref).days
            
            df_predicted = predicted_ts.to_dataframe()
            df_predicted['agedays'] = (df_predicted.index - start_date_ref).days
            
            # Vẽ bằng cách sử dụng cột agedays làm trục x
            plt.plot(df_actual['agedays'], df_actual.iloc[:, 0], label='Thực tế', color='blue')
            plt.plot(df_predicted['agedays'], df_predicted.iloc[:, 0], label='Dự đoán', color='orange', linestyle='--')
            
            plt.title(f'Dự đoán {target_name_display} cho trẻ {subjid}')
            plt.xlabel('Tuổi (ngày)') # Đổi nhãn trục x
            plt.ylabel(f'{target_name_display} ({unit_display})')
            plt.legend()
            plt.grid(True, linestyle=':')
            
            os.makedirs(figures_path, exist_ok=True)
            plt.savefig(os.path.join(figures_path, f'{model_type_prefix}_prediction_{subjid}.png'))
            plt.show()
            plt.close()
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
    z_score_analyses,
    common_subjids_list, 
    figures_path,
    start_date_ref,  
    limit
):
    """
    Vẽ biểu đồ dự đoán và Z-score cho cùng một đứa trẻ.
    Trục x của tất cả các biểu đồ đều là 'agedays'.
    """
    num_children_to_plot = min(len(common_subjids_list), limit)
    if num_children_to_plot == 0:
        print("Không có trẻ em chung nào để vẽ biểu đồ gộp.")
        return

    num_rows = num_children_to_plot
    fig, axs = plt.subplots(num_rows, 3, figsize=(24, 7 * num_rows), squeeze=False)
    fig.suptitle('Phân tích và Dự đoán Tăng trưởng Toàn diện', fontsize=20)

    for i in range(num_children_to_plot):
        subjid = common_subjids_list[i]
        
        ax_h = axs[i, 0]
        ax_w = axs[i, 1]
        ax_z = axs[i, 2]

        # --- Plot 1: Dự đoán Chiều cao (với trục x là agedays) ---
        if subjid in test_series_h and subjid in predictions_h:
            ts_actual_h = test_series_h.get(subjid)
            ts_pred_h = predictions_h.get(subjid)
            if ts_actual_h and ts_pred_h:
                # Chuyển đổi sang DataFrame và tính agedays
                df_actual_h = ts_actual_h.to_dataframe()
                df_actual_h['agedays'] = (df_actual_h.index - start_date_ref).days
                
                df_pred_h = ts_pred_h.to_dataframe()
                df_pred_h['agedays'] = (df_pred_h.index - start_date_ref).days
                
                # Vẽ bằng cách sử dụng cột agedays
                ax_h.plot(df_actual_h['agedays'], df_actual_h.iloc[:, 0], label='Thực tế (Chiều Cao)', color='blue')
                ax_h.plot(df_pred_h['agedays'], df_pred_h.iloc[:, 0], label='Dự đoán (Chiều Cao)', color='skyblue', linestyle='--')
        
        ax_h.set_title(f'ID {subjid} - Dự đoán Chiều cao')
        ax_h.set_xlabel('Tuổi (ngày)') # Đổi nhãn trục x
        ax_h.set_ylabel('Chiều cao (cm)')
        ax_h.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax_h.grid(True, linestyle=':')

        # --- Plot 2: Dự đoán Cân nặng (với trục x là agedays) ---
        if subjid in test_series_w and subjid in predictions_w:
            ts_actual_w = test_series_w.get(subjid)
            ts_pred_w = predictions_w.get(subjid)
            if ts_actual_w and ts_pred_w:
                df_actual_w = ts_actual_w.to_dataframe()
                df_actual_w['agedays'] = (df_actual_w.index - start_date_ref).days
                
                df_pred_w = ts_pred_w.to_dataframe()
                df_pred_w['agedays'] = (df_pred_w.index - start_date_ref).days
                
                ax_w.plot(df_actual_w['agedays'], df_actual_w.iloc[:, 0], label='Thực tế (Cân Nặng)', color='red')
                ax_w.plot(df_pred_w['agedays'], df_pred_w.iloc[:, 0], label='Dự đoán (Cân Nặng)', color='salmon', linestyle='--')

        ax_w.set_title(f'ID {subjid} - Dự đoán Cân nặng')
        ax_w.set_xlabel('Tuổi (ngày)') # Đổi nhãn trục x
        ax_w.set_ylabel('Cân nặng (kg)')
        ax_w.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        ax_w.grid(True, linestyle=':')

        # --- Plot 3: Xu hướng Z-score (vốn đã dùng agedays) ---
        if subjid in z_score_analyses:
            analysis_df = z_score_analyses[subjid]
            if not analysis_df.empty:
                analysis_df.plot(x='agedays', y='hfa_z', label='HFA (Chiều Cao/Tuổi)', ax=ax_z, marker='.')
                analysis_df.plot(x='agedays', y='wfa_z', label='WFA (Cân Nặng/Tuổi)', ax=ax_z, marker='.')
                analysis_df.plot(x='agedays', y='wfh_z', label='WFH (Cân Nặng/Chiều Cao)', ax=ax_z, marker='.')

                ax_z.axhline(y=0, color='green', linestyle='--', lw=1)
                ax_z.axhline(y=-2, color='orange', linestyle='--', lw=1)
                ax_z.axhline(y=2, color='orange', linestyle='--', lw=1)
                ax_z.axhline(y=-3, color='red', linestyle='--', lw=1)
                ax_z.axhline(y=3, color='red', linestyle='--', lw=1)

                ax_z.fill_between(analysis_df['agedays'], -3, -2, color='red', alpha=0.1)
                ax_z.fill_between(analysis_df['agedays'], 2, 3, color='red', alpha=0.1)

        ax_z.set_title(f'ID {subjid} - Xu hướng Z-Score')
        ax_z.set_xlabel('Tuổi (ngày)') # Thêm nhãn trục x cho nhất quán
        ax_z.set_ylabel('Z-score')
        ax_z.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        ax_z.grid(True, linestyle=':')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    os.makedirs(figures_path, exist_ok=True)
    save_path = os.path.join(figures_path, 'combined_growth_analysis.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Đã lưu biểu đồ phân tích gộp tại: {save_path}")
    plt.show()
    plt.close(fig)

def plot_z_score_trends(subjid, analysis_df, figures_path):
    """
    Vẽ biểu đồ xu hướng Z-score (HFA, WFA, WFH) theo thời gian cho một đứa trẻ.
    3 biểu đồ con được vẽ trên cùng một hàng.
    Legend được di chuyển ra ngoài và chỉ hiển thị một lần.
    """
    if analysis_df.empty:
        print(f"Subjid {subjid}: Không có dữ liệu Z-score để vẽ biểu đồ.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
    fig.suptitle(f'Xu hướng Z-Score cho trẻ ID: {subjid}', fontsize=16)

    z_scores_to_plot = {
        'hfa_z': ('Chiều cao theo tuổi (HFA)', axs[0]),
        'wfa_z': ('Cân nặng theo tuổi (WFA)', axs[1]),
        'wfh_z': ('Cân nặng theo chiều cao (WFH/L)', axs[2])
    }

    # Vẽ và tắt legend của từng subplot
    for z_col, (title, ax) in z_scores_to_plot.items():
        ax.plot(analysis_df['agedays'], analysis_df[z_col], label='Dự đoán Z-score', color='blue', marker='.', linestyle='-')
        ax.axhline(y=0, color='green', linestyle='--', label='0 SD (Median)')
        ax.axhline(y=2, color='orange', linestyle='--', label='+2 SD / -2 SD')
        ax.axhline(y=-2, color='orange', linestyle='--')
        ax.axhline(y=3, color='red', linestyle='--', label='+3 SD / -3 SD')
        ax.axhline(y=-3, color='red', linestyle='--')
        
        # Tô màu vùng nguy cơ
        ax.fill_between(analysis_df['agedays'], -3, -2, color='orange', alpha=0.2)
        ax.fill_between(analysis_df['agedays'], 2, 3, color='orange', alpha=0.2)
        
        ax.set_title(title)
        ax.set_xlabel('Tuổi (ngày)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.legend().set_visible(False) # Tắt legend của từng subplot

    axs[0].set_ylabel('Z-score')

    # Lấy handles và labels từ subplot ở giữa để tạo legend chung
    handles, labels = axs[1].get_legend_handles_labels()
    
    # Sửa lỗi lặp: Chỉ lấy các label duy nhất
    by_label = dict(zip(labels, handles))
    
    # Tạo legend chung nằm bên dưới các subplot
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(by_label))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Điều chỉnh rect để chừa không gian

    os.makedirs(figures_path, exist_ok=True)
    save_path = os.path.join(figures_path, f'zscore_trends_{subjid}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Đã lưu biểu đồ Z-score cho trẻ {subjid} tại: {save_path}")
    plt.show()
    # plt.close(fig)
