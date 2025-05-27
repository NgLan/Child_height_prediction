import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Normalizing height and weight using StandardScaler
scaler = StandardScaler()
features = df_subject[['htcm', 'wtkg']].values
features_scaled = scaler.fit_transform(features)
df_subject['htcm_scaled'] = features_scaled[:, 0]
df_subject['wtkg_scaled'] = features_scaled[:, 1]

# Feature Engineering
# 1. Lagging feature (previous height)
df_subject['htcm_lag1'] = df_subject['htcm'].shift(1)

# 2. Growth rate (change in height per day)
df_subject['growth_rate'] = (df_subject['htcm'] - df_subject['htcm_lag1']) / (df_subject['agedays'] - df_subject['agedays'].shift(1))

# 3. Height-to-weight ratio
df_subject['ht_wt_ratio'] = df_subject['htcm'] / df_subject['wtkg']

# Plotting original vs normalized data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_subject['agedays'], df_subject['htcm'], 'o-', label='Original Height (cm)')
plt.plot(df_subject['agedays'], df_subject['wtkg'], 's-', label='Original Weight (kg)')
plt.title(f'Original Data (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df_subject['agedays'], df_subject['htcm_scaled'], 'o-', label='Normalized Height')
plt.plot(df_subject['agedays'], df_subject['wtkg_scaled'], 's-', label='Normalized Weight')
plt.title(f'Normalized Data (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)

# Plotting new features
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(df_subject['agedays'], df_subject['htcm_lag1'], 'o-', label='Lagged Height (cm)')
plt.title(f'Lagged Height (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Height (cm)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(df_subject['agedays'], df_subject['growth_rate'], 'o-', label='Growth Rate (cm/day)')
plt.title(f'Growth Rate (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Growth Rate (cm/day)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(df_subject['agedays'], df_subject['ht_wt_ratio'], 'o-', label='Height/Weight Ratio')
plt.title(f'Height/Weight Ratio (Subject {subject_id})')
plt.xlabel('Age (days)')
plt.ylabel('Ratio (cm/kg)')
plt.legend()
plt.grid(True)

# Preparing data for create_supervised_data (using normalized data)
def create_supervised_data(data, window_size=2):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Using normalized height for supervised data
X, y = create_supervised_data(df_subject['htcm_scaled'].values, window_size=2)
print("Supervised data example (normalized height):")
print("X (input windows):", X)
print("y (target values):", y)

# Showing all plots
plt.show()