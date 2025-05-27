# --- START OF FILE src/data_processing_weight.py ---
import pandas as pd
import numpy as np

def load_and_preprocess_data_for_weight(file_path):
    """Tải và tiền xử lý dữ liệu để dự đoán cân nặng."""
    df = pd.read_csv(file_path, na_values=['\\N'])
    
    print("Cân nặng - Dữ liệu thô:")
    print(df.dtypes)
    print(df.head())

    essential_cols = ['subjid', 'sex', 'agedays', 'wtkg'] # Target là wtkg
    numeric_cols_to_check = ['agedays', 'htcm', 'wtkg', 'haz', 'waz', 'gagebrth']

    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Cảnh báo (Cân nặng): Cột {col} không có trong DataFrame.")

    df = df.dropna(subset=essential_cols)
    # Covariates cho cân nặng: htcm, haz, waz, gagebrth, sex
    df = df.dropna(subset=['htcm', 'haz', 'waz', 'gagebrth'])

    if 'agedays' in df.columns and not df['agedays'].empty:
        df['agedays'] = df['agedays'].astype('Int64')
    
    if 'sex' in df.columns and not df['sex'].empty:
        df['sex_original'] = df['sex']
        df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
        if df['sex'].isnull().any():
            df = df.dropna(subset=['sex'])
        if not df['sex'].empty:
            df['sex'] = df['sex'].astype(int)
        df = df.drop(columns=['sex_original'], errors='ignore')
    
    print("\nCân nặng - Dữ liệu sau tiền xử lý cơ bản:")
    print(df.dtypes)
    print(df.head())
    print(f"Số dòng còn lại: {len(df)}")
    if df.empty:
        print("CẢNH BÁO (Cân nặng): DataFrame rỗng sau tiền xử lý cơ bản.")

    return df

def interpolate_weight_data(df):
    """Nội suy dữ liệu cân nặng để đảm bảo các mốc thời gian cách đều 30 ngày."""
    if df.empty:
        print("Cân nặng - DataFrame đầu vào cho nội suy rỗng. Bỏ qua nội suy.")
        return pd.DataFrame()
        
    required_cols_for_interpolation = ['subjid', 'sex', 'agedays', 'wtkg', 'htcm', 'haz', 'waz', 'gagebrth']
    if not all(col in df.columns for col in required_cols_for_interpolation):
        print(f"Lỗi (Cân nặng): Thiếu cột cần thiết cho nội suy. Cần: {required_cols_for_interpolation}")
        return pd.DataFrame()

    max_days = df['agedays'].max()
    if pd.isna(max_days):
        print("Cân nặng - Không có giá trị 'agedays' hợp lệ để nội suy.")
        return pd.DataFrame()
        
    time_points = np.arange(0, int(max_days) + 30, 30)
    
    interpolated_dfs = []
    grouped = df.groupby(['subjid', 'sex'])
    
    cols_to_interpolate = ['wtkg', 'htcm', 'haz', 'waz', 'gagebrth']

    for (subjid, sex_val), group in grouped:
        group = group.sort_values('agedays')
        
        if len(group['agedays'].unique()) < 2:
            continue

        new_df = pd.DataFrame({'agedays': time_points})
        new_df['subjid'] = subjid
        new_df['sex'] = sex_val
        
        for col in cols_to_interpolate:
            if col in group.columns and not group[col].isnull().all():
                interpolated_values = np.interp(
                    time_points,
                    group['agedays'].astype(float).values,
                    group[col].astype(float).values,
                    left=np.nan,
                    right=np.nan
                )
                new_df[col] = interpolated_values
            else:
                new_df[col] = np.nan
        
        interpolated_dfs.append(new_df)
    
    if not interpolated_dfs:
        print("Cân nặng - Không có dữ liệu nào được nội suy.")
        return pd.DataFrame()

    result_df = pd.concat(interpolated_dfs, ignore_index=True)
    result_df = result_df.dropna(subset=['wtkg']) # Đảm bảo target không NaN

    print("\nCân nặng - Dữ liệu sau nội suy:")
    # print(result_df.head())
    print(f"Số dòng còn lại: {len(result_df)}")
    if result_df.empty:
        print("CẢNH BÁO (Cân nặng): DataFrame rỗng sau nội suy.")
        
    return result_df
# --- END OF FILE src/data_processing_weight.py ---