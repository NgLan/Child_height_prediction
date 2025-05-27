import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Loading the CSV file
df = pd.read_csv('../data/childhealth-dutch.csv')

# Converting columns to appropriate types, handling missing values
df['htcm'] = pd.to_numeric(df['htcm'], errors='coerce')
df['wtkg'] = pd.to_numeric(df['wtkg'], errors='coerce')
df['agedays'] = pd.to_numeric(df['agedays'], errors='coerce')

# Dropping rows with missing critical columns
df = df.dropna(subset=['subjid', 'agedays', 'htcm', 'wtkg'])

# Selecting data for a single subject (e.g., subjid=10001)
subject_id = 10001
df_subject = df[df['subjid'] == subject_id].sort_values('agedays')

# Defining interpolation function
def interpolate_time_series(agedays, values, target_days):
    # Removing NaN values for interpolation
    valid_idx = ~np.isnan(values)
    agedays_valid = agedays[valid_idx]
    values_valid = values[valid_idx]
    
    # Ensuring there are enough points for interpolation
    if len(agedays_valid) < 2:
        return np.full_like(target_days, np.nan, dtype=float)
    
    # Creating interpolation function
    f = interp1d(agedays_valid, values_valid, kind='linear', fill_value='extrapolate')
    
    # Interpolating at target days
    interpolated_values = f(target_days)
    return interpolated_values

# Original target days
original_target_days = [0, 180, 365, 730, 1095]

# Monthly target days (every 30 days up to 3 years = 1080 days)
monthly_target_days = np.arange(0, 1081, 30)

# Interpolating height and weight for original target days
interpolated_htcm_original = interpolate_time_series(df_subject['agedays'].values, df_subject['htcm'].values, original_target_days)
interpolated_wtkg_original = interpolate_time_series(df_subject['agedays'].values, df_subject['wtkg'].values, original_target_days)

# Interpolating height and weight for monthly target days
interpolated_htcm_monthly = interpolate_time_series(df_subject['agedays'].values, df_subject['htcm'].values, monthly_target_days)
interpolated_wtkg_monthly = interpolate_time_series(df_subject['agedays'].values, df_subject['wtkg'].values, monthly_target_days)

# Plotting original vs interpolated data (height)
plt.figure(figsize=(12, 6))
plt.plot(df_subject['agedays'], df_subject['htcm'], 'o-', label='Original Height', color='blue')
plt.plot(original_target_days, interpolated_htcm_original, 's--', label='Interpolated Height (Original Target Days)', color='red')
plt.plot(monthly_target_days, interpolated_htcm_monthly, 'x-', label='Interpolated Height (Monthly)', color='green')
plt.title(f'Height Interpolation Comparison (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)

# Plotting original vs interpolated data (weight)
plt.figure(figsize=(12, 6))
plt.plot(df_subject['agedays'], df_subject['wtkg'], 'o-', label='Original Weight', color='blue')
plt.plot(original_target_days, interpolated_wtkg_original, 's--', label='Interpolated Weight (Original Target Days)', color='red')
plt.plot(monthly_target_days, interpolated_wtkg_monthly, 'x-', label='Interpolated Weight (Monthly)', color='green')
plt.title(f'Weight Interpolation Comparison (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.grid(True)

# Calculating number of supervised samples for each approach
def count_supervised_samples(data, window_size=2):
    return len(data) - window_size

window_size = 2
original_samples = count_supervised_samples(interpolated_htcm_original, window_size)
monthly_samples = count_supervised_samples(interpolated_htcm_monthly, window_size)
print(f"Number of supervised samples (original target days): {original_samples}")
print(f"Number of supervised samples (monthly): {monthly_samples}")

# Showing all plots
plt.show()