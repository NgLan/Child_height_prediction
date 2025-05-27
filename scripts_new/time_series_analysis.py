import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import detrend
import seaborn as sns

# Loading the CSV file
df = pd.read_csv('../data/childhealth-dutch.csv')

# Converting columns to appropriate types, handling missing values
df['htcm'] = pd.to_numeric(df['htcm'], errors='coerce')
df['wtkg'] = pd.to_numeric(df['wtkg'], errors='coerce')
df['agedays'] = pd.to_numeric(df['agedays'], errors='coerce')

# Dropping rows with missing critical columns
df = df.dropna(subset=['subjid', 'agedays', 'htcm', 'wtkg'])

# Analyzing data for a single subject (e.g., subjid=10001) for clarity
subject_id = 10001
df_subject = df[df['subjid'] == subject_id].sort_values('agedays')

# Visualizing raw time-series data (height and weight)
plt.figure(figsize=(10, 6))
plt.plot(df_subject['agedays'], df_subject['htcm'], marker='o', label='Height (cm)')
plt.plot(df_subject['agedays'], df_subject['wtkg'], marker='s', label='Weight (kg)')
plt.title(f'Height and Weight over Age (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Checking for outliers using IQR method
def detect_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

# Applying outlier detection to height and weight
htcm_outliers = detect_outliers(df_subject['htcm'])
wtkg_outliers = detect_outliers(df_subject['wtkg'])
print(f"Height outliers: {htcm_outliers}")
print(f"Weight outliers: {wtkg_outliers}")

# Interpolating missing or uneven data points (mimicking interpolate_data)
def interpolate_time_series(agedays, values, target_days):
    # Removing NaN values for interpolation
    valid_idx = ~np.isnan(values)
    agedays_valid = agedays[valid_idx]
    values_valid = values[valid_idx]
    
    # Creating interpolation function
    f = interp1d(agedays_valid, values_valid, kind='linear', fill_value='extrapolate')
    
    # Interpolating at target days
    interpolated_values = f(target_days)
    return interpolated_values

# Defining target days for interpolation (similar to code)
target_days = [0, 180, 365, 730, 1095]
interpolated_htcm = interpolate_time_series(df_subject['agedays'].values, df_subject['htcm'].values, target_days)
interpolated_wtkg = interpolate_time_series(df_subject['agedays'].values, df_subject['wtkg'].values, target_days)

# Visualizing interpolated data
plt.figure(figsize=(10, 6))
plt.plot(df_subject['agedays'], df_subject['htcm'], 'o-', label='Original Height')
plt.plot(target_days, interpolated_htcm, 's--', label='Interpolated Height')
plt.title(f'Interpolated Height (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)

# Detrending the height data to check for seasonality or noise
detrended_htcm = detrend(df_subject['htcm'].values)
plt.figure(figsize=(10, 6))
plt.plot(df_subject['agedays'], df_subject['htcm'], label='Original Height')
plt.plot(df_subject['agedays'], detrended_htcm, label='Detrended Height')
plt.title(f'Detrended Height (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)

# Creating supervised data (mimicking create_supervised_data)
def create_supervised_data(data, window_size=2):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Applying supervised data creation to interpolated height
X, y = create_supervised_data(interpolated_htcm, window_size=2)
print("Supervised data example:")
print("X (input windows):", X)
print("y (target values):", y)

# Showing all plots
plt.show()