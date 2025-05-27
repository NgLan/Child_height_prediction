import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path):
    """Load and clean the childhealth-dutch.csv data."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data with shape: {df.shape}")
        # Replace '\N' with NaN
        df = df.replace(r'\N', np.nan)
        # Convert numeric columns
        numeric_cols = ['agedays', 'gagebrth', 'htcm', 'wtkg', 'haz', 'waz']
        df[numeric_cols] = df[numeric_cols].astype(float)
        # Check unique subjects and measurements
        logging.info(f"Number of unique subjid: {df['subjid'].nunique()}")
        logging.info(f"Average measurements per subjid: {df.groupby('subjid').size().mean()}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def fill_missing_values(df):
    """Fill missing values with group mean based on sex and agedays."""
    for col in ['htcm', 'wtkg', 'haz', 'waz']:
        df[col] = df.groupby(['sex', 'agedays'])[col].transform(lambda x: x.fillna(x.mean()))
        # If still NaN, fill with overall mean for the column
        df[col] = df[col].fillna(df[col].mean())
    return df

def interpolate_data(df, target_days, min_measurements=2):
    """Resample data to fixed time points using linear interpolation."""
    interpolated_dfs = []
    for subjid in df['subjid'].unique():
        sub_df = df[df['subjid'] == subjid].sort_values('agedays')
        if len(sub_df) < min_measurements:
            logging.warning(f"Skipping subjid {subjid}: only {len(sub_df)} measurements")
            continue

        # Initialize output DataFrame for this subject
        new_df = pd.DataFrame({
            'subjid': subjid,
            'agedays': target_days,
            'sex': sub_df['sex'].iloc[0],
            'gagebrth': sub_df['gagebrth'].iloc[0],
            'is_interpolated': False
        })

        # Interpolate numeric columns
        for col in ['htcm', 'wtkg', 'haz', 'waz']:
            valid_data = sub_df[['agedays', col]].dropna()
            if len(valid_data) >= 2:
                interp_func = interp1d(
                    valid_data['agedays'],
                    valid_data[col],
                    kind='linear',
                    fill_value='extrapolate'
                )
                new_df[col] = interp_func(target_days)
                new_df['is_interpolated'] = new_df['is_interpolated'] | (~np.isin(target_days, sub_df['agedays']))
            else:
                new_df[col] = np.nan

        # Add missing value indicators
        new_df['is_missing_htcm'] = new_df['htcm'].isna()
        new_df['is_missing_wtkg'] = new_df['wtkg'].isna()

        interpolated_dfs.append(new_df)

    result = pd.concat(interpolated_dfs, ignore_index=True) if interpolated_dfs else pd.DataFrame()
    logging.info(f"Interpolated data shape: {result.shape}")
    return result

def create_features(df):
    """Create derived features: growth rates and delta days."""
    df = df.sort_values(['subjid', 'agedays'])

    # Calculate delta days
    df['delta_days'] = df.groupby('subjid')['agedays'].diff().fillna(0)

    # Calculate growth rates
    df['growth_rate_ht'] = df.groupby('subjid')['htcm'].diff() / df['delta_days']
    df['growth_rate_wt'] = df.groupby('subjid')['wtkg'].diff() / df['delta_days']

    # Fill NaN for first measurement of each subject
    df.loc[df['delta_days'] == 0, ['growth_rate_ht', 'growth_rate_wt']] = 0

    logging.info(f"Features added, data shape: {df.shape}")
    return df

def create_supervised_data(df, prediction_horizon=180, window_size=2):
    """Restructure data into supervised format with sliding window."""
    features = ['htcm', 'wtkg', 'haz', 'waz', 'growth_rate_ht', 'growth_rate_wt', 'sex', 'gagebrth']
    supervised_data = []

    for subjid in df['subjid'].unique():
        sub_df = df[df['subjid'] == subjid].sort_values('agedays')
        if len(sub_df) < window_size + 1:
            logging.warning(f"Skipping subjid {subjid}: only {len(sub_df)} points for supervised data")
            continue

        # Create sliding windows
        for i in range(len(sub_df) - window_size):
            window = sub_df.iloc[i:i + window_size]
            target_idx = i + window_size

            # Check if target exists and within prediction horizon
            if target_idx < len(sub_df):
                target = sub_df.iloc[target_idx]
                if (target['agedays'] - window['agedays'].iloc[-1]) <= prediction_horizon * 1.5:
                    # Collect features
                    instance = {
                        'subjid': subjid,
                        'window_start_age': window['agedays'].iloc[0],
                        'window_end_age': window['agedays'].iloc[-1],
                        'target_age': target['agedays'],
                        'target_htcm': target['htcm'],
                        'target_wtkg': target['wtkg']
                    }

                    # Add window features
                    for j, row in enumerate(window.itertuples()):
                        for feat in features:
                            if feat == 'sex':
                                instance[f'window_{j+1}_sex'] = 1 if row.sex == 'Male' else 0
                            else:
                                instance[f'window_{j+1}_{feat}'] = getattr(row, feat)

                    supervised_data.append(instance)

    result = pd.DataFrame(supervised_data)
    logging.info(f"Supervised data shape: {result.shape}")
    return result

def main(file_path):
    # Define fixed time points for resampling
    target_days = [0, 180, 365, 730, 1095]

    # Load and clean data
    df = load_and_clean_data(file_path)

    # Fill missing values
    df = fill_missing_values(df)

    # Resample and interpolate
    df_interpolated = interpolate_data(df, target_days, min_measurements=2)

    if df_interpolated.empty:
        logging.error("No data after interpolation. Check input data or interpolation logic.")
        return pd.DataFrame()

    # Create derived features
    df_features = create_features(df_interpolated)

    # Create supervised dataset
    df_supervised = create_supervised_data(df_features, prediction_horizon=180, window_size=2)

    # Save processed data
    output_path = 'childhealth_processed.csv'
    df_supervised.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to '{output_path}'. Shape: {df_supervised.shape}")
    return df_supervised

if __name__ == '__main__':
    file_path = 'childhealth-dutch.csv'
    processed_data = main(file_path)
    print(f"Processed data saved to 'childhealth_processed.csv'. Shape: {processed_data.shape}")