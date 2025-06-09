import pandas as pd
import numpy as np

def preprocess_data(file_path, target_col, all_covariate_cols):
    """
    Tải và tiền xử lý dữ liệu chung.
    Loại bỏ các hàng thiếu dữ liệu ở cột mục tiêu và các cột covariate.
    """
    try:
        df = pd.read_csv(file_path, na_values=['\\N'])
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu tại: {file_path}")
        return pd.DataFrame()

    print(f"Tiền xử lý cho mục tiêu: '{target_col}'")
    
    # --- BƯỚC 1: XỬ LÝ CÁC CỘT PHI SỐ TRƯỚC ---
    # Ánh xạ cột 'sex' từ chuỗi sang số (0/1) 
    if 'sex' in df.columns:
        print("Đang ánh xạ cột 'sex' từ chuỗi sang số...")
        df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
        # Sau khi map, một số giá trị có thể thành NaN nếu không phải 'Female'/'Male',
        # chúng sẽ được xử lý ở bước dropna chung bên dưới.

    # --- BƯỚC 2: CHUYỂN ĐỔI SANG KIỂU SỐ ---
    all_needed_numeric_cols = list(dict.fromkeys([target_col] + all_covariate_cols + ['agedays']))

    for col in all_needed_numeric_cols:
        if col in df.columns:
            # Chuyển đổi sang số, các giá trị không hợp lệ sẽ thành NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- BƯỚC 3: LỌC DỮ LIỆU ---
    # Gộp tất cả các cột cần thiết (bao gồm cả các cột định danh)
    # để thực hiện một lần dropna duy nhất.
    required_cols_for_row = list(dict.fromkeys(
        ['subjid', 'agedays', 'sex', target_col] + all_covariate_cols
    ))
    
    # Loại bỏ bất kỳ hàng nào có giá trị thiếu trong các cột bắt buộc.
    # Bước này sẽ xử lý các NaN từ file gốc, từ việc to_numeric, và từ việc map 'sex'.
    df = df.dropna(subset=required_cols_for_row)

    if df.empty:
        print(f"CẢNH BÁO: DataFrame rỗng sau khi lọc dữ liệu cho mục tiêu '{target_col}'.")
        return df

    # --- BƯỚC 4: CHUẨN HÓA KIỂU DỮ LIỆU CUỐI CÙNG ---
    # Bây giờ không còn NaN trong các cột này, ta có thể ép kiểu an toàn.
    df['subjid'] = df['subjid'].astype(int)
    df['agedays'] = df['agedays'].astype('Int64')
    df['sex'] = df['sex'].astype(int)
    
    print(f"Số dòng sau tiền xử lý: {len(df)}")
    return df

def interpolate_data(df, target_col, all_covariate_cols, step):
    """Nội suy dữ liệu để đảm bảo các mốc thời gian cách đều."""
    if df.empty:
        print(f"DataFrame đầu vào cho nội suy ('{target_col}') rỗng. Bỏ qua.")
        return pd.DataFrame()

    max_days = df['agedays'].max()
    if pd.isna(max_days):
        print(f"Không có giá trị 'agedays' hợp lệ để nội suy ('{target_col}').")
        return pd.DataFrame()
        
    time_points = np.arange(0, int(max_days) + step, step)
    
    interpolated_dfs = []
    grouped = df.groupby(['subjid', 'sex'])
    
    cols_to_interpolate = [target_col] + all_covariate_cols

    for (subjid, sex), group in grouped:
        group = group.sort_values('agedays')
        
        if len(group['agedays'].unique()) < 2:
            continue

        new_df = pd.DataFrame({'agedays': time_points})
        new_df['subjid'] = subjid
        new_df['sex'] = sex
        
        for col in cols_to_interpolate:
            if col == 'sex': continue

            # Đếm số giá trị khác nhau trong cột
            if group[col].nunique() == 1:
                new_df[col] = group[col].iloc[0]
            else:
                interpolated_values = np.interp(
                    time_points, # Mảng các điểm x mới
                    group['agedays'].values, # Mảng các điểm x cũ
                    group[col].values, # Mảng các điểm y cũ
                    left=np.nan, # Không ngoại suy
                    right=np.nan
                )
                new_df[col] = interpolated_values
        
        interpolated_dfs.append(new_df)
    
    if not interpolated_dfs:
        print(f"Không có dữ liệu nào được nội suy cho '{target_col}'.")
        return pd.DataFrame()

    result_df = pd.concat(interpolated_dfs, ignore_index=True)
    
    result_df = result_df.dropna(subset=[target_col] + all_covariate_cols)
    
    print(f"Số dòng sau nội suy: {len(result_df)}")
    return result_df