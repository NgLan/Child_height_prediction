import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check TensorFlow version
logging.info(f"TensorFlow version: {tf.__version__}")

# Mount Google Drive (uncomment and run this cell first in Colab)
from google.colab import drive
drive.mount('/content/drive')

# File paths (update as needed)
INPUT_FILE_PATH = '/content/drive/MyDrive/MachineLearning/data/childhealth_processed.csv'
MODEL_PATH = '/content/drive/MyDrive/MachineLearning/model/lstm_model.keras'
METRICS_PATH = '/content/drive/MyDrive/MachineLearning/model/evaluation_metrics.txt'
PREDICTIONS_PATH = '/content/drive/MyDrive/MachineLearning/data/predictions.csv'
PLOT_PATH = '/content/drive/MyDrive/MachineLearning/plot/loss_plot.png'

def load_processed_data(file_path):
    """Load processed data from childhealth_processed.csv."""
    try:
        logging.info(f"Attempting to load file from: {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Loaded processed data with shape: {df.shape}")
        if not df.empty:
            logging.info(f"Sample data:\n{df.head()}")
        else:
            logging.warning("Data frame is empty.")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_data(df, window_size=2):
    """Prepare data for LSTM model."""
    if df.empty or len(df) < 10:
        logging.error("Data is empty or too small for training. Check input file.")
        return None, None, None, None, None

    feature_cols = [col for col in df.columns if col.startswith('window_')]
    label_cols = ['target_htcm', 'target_wtkg']

    if not all(col in df.columns for col in label_cols + feature_cols):
        logging.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
        return None, None, None, None, None

    X = df[feature_cols].values
    y = df[label_cols].values

    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    features_per_timestep = len(feature_cols) // window_size
    X = X.reshape(X.shape[0], window_size, features_per_timestep)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler_y

def build_lstm_model(input_shape, output_dim=2):
    """Build LSTM model for height and weight prediction."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the LSTM model with early stopping."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, scaler_y):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test, verbose=0)

    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)

    rmse_htcm = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_wtkg = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    mae_htcm = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_wtkg = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

    metrics = {
        'rmse_htcm': rmse_htcm,
        'rmse_wtkg': rmse_wtkg,
        'mae_htcm': mae_htcm,
        'mae_wtkg': mae_wtkg
    }

    logging.info(f"RMSE htcm: {rmse_htcm:.4f}, wtkg: {rmse_wtkg:.4f}")
    logging.info(f"MAE htcm: {mae_htcm:.4f}, wtkg: {mae_wtkg:.4f}")
    return metrics, y_pred, y_test

def save_results(metrics, y_pred, y_test):
    """Save evaluation results and predictions."""
    with open(METRICS_PATH, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    pred_df = pd.DataFrame({
        'pred_htcm': y_pred[:, 0],
        'pred_wtkg': y_pred[:, 1],
        'true_htcm': y_test[:, 0],
        'true_wtkg': y_test[:, 1]
    })
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    logging.info(f"Results saved to {METRICS_PATH} and {PREDICTIONS_PATH}")

def plot_loss(history):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    plt.close()
    logging.info(f"Loss plot saved to {PLOT_PATH}")

def main():
    df = load_processed_data(INPUT_FILE_PATH)

    if df.empty:
        logging.error("No data loaded. Check input file path or upload file to Colab.")
        return

    X_train, X_test, y_train, y_test, scaler_y = prepare_data(df, window_size=2)

    if X_train is None:
        logging.error("Data preparation failed. Exiting.")
        return

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_test, y_test)

    metrics, y_pred, y_test = evaluate_model(model, X_test, y_test, scaler_y)

    model.save(MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    save_results(metrics, y_pred, y_test)

    plot_loss(history)

if __name__ == '__main__':
    main()