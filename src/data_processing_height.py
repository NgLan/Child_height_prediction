import pandas as pd
import numpy as np

def preprocess_height_data(file_path):
    """Tải và tiền xử lý dữ liệu cho chiều cao."""
    df = pd.read_csv(file_path, na_values=['\\N'])
    
    print("Chiều cao - Dữ liệu thô:")
    print(df.dtypes)
    print(df.head())

    # Các cột cần thiết cho mô hình chiều cao và các cột số khác để làm sạch
    essential_cols = ['subjid', 'sex', 'agedays', 'htcm'] # Target là htcm
    numeric_cols_to_check = ['agedays', 'htcm', 'wtkg', 'haz', 'waz', 'gagebrth']

    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Cảnh báo (Chiều cao): Cột {col} không có trong DataFrame.")

    # Loại bỏ các hàng nếu các cột thiết yếu bị thiếu SAU KHI chuyển đổi
    df = df.dropna(subset=essential_cols) # Đảm bảo các cột chính không NaN

    # Loại bỏ thêm các hàng nếu các cột covariate tiềm năng bị thiếu
    # (wtkg, haz, waz, gagebrth là covariates cho chiều cao)
    df = df.dropna(subset=['wtkg', 'haz', 'waz', 'gagebrth'])


    if 'agedays' in df.columns and not df['agedays'].empty:
        df['agedays'] = df['agedays'].astype('Int64')
    
    if 'sex' in df.columns and not df['sex'].empty:
        df['sex_original'] = df['sex'] # Giữ lại để debug nếu map lỗi
        df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
        if df['sex'].isnull().any():
            # print("Cảnh báo (Chiều cao): Giá trị null trong cột 'sex' sau khi map. Các hàng này sẽ bị xóa.")
            # print(df[df['sex'].isnull()][['sex_original', 'sex']])
            df = df.dropna(subset=['sex'])
        if not df['sex'].empty:
            df['sex'] = df['sex'].astype(int)
        df = df.drop(columns=['sex_original'], errors='ignore')
    
    print("\nChiều cao - Dữ liệu sau tiền xử lý cơ bản:")
    print(df.dtypes)
    print(df.head())
    print(f"Số dòng còn lại: {len(df)}")
    if df.empty:
        print("CẢNH BÁO (Chiều cao): DataFrame rỗng sau tiền xử lý cơ bản.")
    
    return df

def interpolate_height_data(df):
    """Nội suy dữ liệu chiều cao để đảm bảo các mốc thời gian cách đều 30 ngày."""
    if df.empty:
        print("Chiều cao - DataFrame đầu vào cho nội suy rỗng. Bỏ qua nội suy.")
        return pd.DataFrame()

    # Các cột cần thiết cho nội suy (bao gồm target và các covariates sẽ được dùng)
    required_cols_for_interpolation = ['subjid', 'sex', 'agedays', 'htcm', 'wtkg', 'haz', 'waz', 'gagebrth']
    if not all(col in df.columns for col in required_cols_for_interpolation):
        print(f"Lỗi (Chiều cao): Thiếu cột cần thiết cho nội suy. Cần: {required_cols_for_interpolation}")
        return pd.DataFrame()

    max_days = df['agedays'].max()
    if pd.isna(max_days):
        print("Chiều cao - Không có giá trị 'agedays' hợp lệ để nội suy.")
        return pd.DataFrame()
        
    time_points = np.arange(0, int(max_days) + 30, 30) # Đảm bảo max_days là int
    
    interpolated_dfs = []
    grouped = df.groupby(['subjid', 'sex'])
    
    cols_to_interpolate = ['htcm', 'wtkg', 'haz', 'waz', 'gagebrth']

    for (subjid, sex_val), group in grouped:
        group = group.sort_values('agedays')
        
        if len(group['agedays'].unique()) < 2: # Cần ít nhất 2 điểm để nội suy
            continue

        new_df = pd.DataFrame({'agedays': time_points})
        new_df['subjid'] = subjid
        new_df['sex'] = sex_val # sex_val đã là 0 hoặc 1
        
        for col in cols_to_interpolate:
            if col in group.columns and not group[col].isnull().all(): # Chỉ nội suy nếu cột tồn tại và không toàn NaN
                interpolated_values = np.interp(
                    time_points,
                    group['agedays'].astype(float).values, # Đảm bảo là float
                    group[col].astype(float).values,       # Đảm bảo là float
                    left=np.nan,
                    right=np.nan
                )
                new_df[col] = interpolated_values
            else:
                new_df[col] = np.nan # Nếu cột không có hoặc toàn NaN, điền NaN
        
        interpolated_dfs.append(new_df)
    
    if not interpolated_dfs:
        print("Chiều cao - Không có dữ liệu nào được nội suy.")
        return pd.DataFrame()

    result_df = pd.concat(interpolated_dfs, ignore_index=True)
    
    # Loại bỏ các hàng có giá trị NaN sau nội suy, đặc biệt là ở target 'htcm'
    result_df = result_df.dropna(subset=['htcm'])
    # Các covariates khác có thể fill sau nếu cần, hoặc để Darts xử lý (một số mô hình có thể)
    # Nhưng với LinearRegressionModel và lags_future_covariates=[0], NaN sẽ gây lỗi.
    # Chúng ta sẽ xử lý NaN cho covariates trong hàm train.

    print("\nChiều cao - Dữ liệu sau nội suy:")
    # print(result_df.head())
    print(f"Số dòng còn lại: {len(result_df)}")
    if result_df.empty:
        print("CẢNH BÁO (Chiều cao): DataFrame rỗng sau nội suy.")

    return result_df