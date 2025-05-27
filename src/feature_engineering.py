import pandas as pd
from src.data_processing import load_data, interpolate_missing_values
from pathlib import Path

class FeatureEngineer:
    """Custom transformer để thực hiện feature engineering."""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.mean_encoders = {}
    
    def fit(self, df):
        """Tính toán target mean encoding dựa trên dữ liệu quá khứ."""
        # Tính mean của htcm cho mỗi subjid dựa trên quá khứ
        self.mean_encoders['htcm_mean_by_subjid'] = df.groupby('subjid')['htcm'].mean()
        return self
    
    def transform(self, df):
        """Thêm các đặc trưng mới: rolling mean, rolling std, target mean encoding."""
        df = df.copy()
        df = df.sort_values(['subjid', 'sex', 'agedays'])
        
        # Thêm rolling mean và rolling std cho các cột wtkg, haz, waz
        for col in ['wtkg', 'haz', 'waz']:
            df[f'{col}_rolling_mean'] = df.groupby(['subjid', 'sex'])[col].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).mean()
            )
            df[f'{col}_rolling_std'] = df.groupby(['subjid', 'sex'])[col].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).std()
            )
        
        # Thêm target mean encoding cho subjid
        df['htcm_mean_by_subjid'] = df['subjid'].map(self.mean_encoders.get('htcm_mean_by_subjid', pd.Series()))
        
        # Điền giá trị thiếu cho rolling std (nếu không đủ window)
        df = df.ffill() 
        return df

def save_engineered_data(df, output_path):
    """Lưu dữ liệu đã qua feature engineering."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Đường dẫn file
    data_path = Path("../data/childhealth-dutch.csv")
    engineered_output_path = Path("../data/childhealth_engineered.csv")
    
    # Tải và nội suy dữ liệu
    df = load_data(data_path)
    df = interpolate_missing_values(df)
    
    # Feature engineering
    engineer = FeatureEngineer(window_size=3)
    engineer.fit(df)
    engineered_df = engineer.transform(df)
    
    # Lưu dữ liệu
    save_engineered_data(engineered_df, engineered_output_path)