"""
04_train_lstm.py — Entrenamiento LSTM (Long Short-Term Memory)
Dataset: microgrid_15min.csv (datos cada 15min)
Targets: Fre, VA, VB, Generacion_Solar
seq_len=12 (3h historia @ 15min) → output_len=96 (24h predicción @ 15min)
"""
import os, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "kaggle_data", "microgrid_15min.csv")

SEQ_LEN = 12
OUTPUT_LEN = 96
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5

FEATURE_COLS = ['VA', 'VB', 'Fre', 'Temp', 'temp_2m', 'shortwave_radiation',
                'wind_speed_10m', 'pressure_msl', 'cloud_cover', 'Generacion_Solar']
TARGET_COLS = ['Fre', 'VA', 'VB', 'Generacion_Solar']


def build_lstm(seq_len, n_features, n_targets, output_len, lstm_units=128):
    inputs = keras.Input(shape=(seq_len, n_features), name='input')

    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    n_outputs = n_targets * output_len
    outputs = layers.Dense(n_outputs)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='LSTM_Microgrid')


def create_sequences(data, feature_cols, target_cols, seq_len, output_len):
    X, y = [], []
    for i in range(len(data) - seq_len - output_len + 1):
        X.append(data[feature_cols].iloc[i:i+seq_len].values)
        y.append(data[target_cols].iloc[i+seq_len:i+seq_len+output_len].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, patience):
    initial_lr = 0.001
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr, first_decay_steps=10, t_mul=2.0, m_mul=1.0
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.perf_counter() - t0

    return history, train_time


def evaluate_model(model, X_test, y_test, scaler_y, target_cols, output_len):
    y_pred = model.predict(X_test, verbose=0)

    y_pred_2d = y_pred.reshape(-1, output_len, len(target_cols))
    y_test_2d = y_test.reshape(-1, output_len, len(target_cols))

    results = {}
    for i, col in enumerate(target_cols):
        pred_col = y_pred_2d[:, :, i]
        true_col = y_test_2d[:, :, i]

        mae = np.mean(np.abs(pred_col - true_col))
        mse = np.mean((pred_col - true_col) ** 2)

        pred_flat = pred_col.reshape(-1, 1)
        true_flat = true_col.reshape(-1, 1)

        dummy_pred = np.zeros((pred_flat.shape[0], len(target_cols)))
        dummy_pred[:, i] = pred_flat[:, 0]
        pred_orig = scaler_y.inverse_transform(dummy_pred)[:, i].reshape(pred_col.shape)

        dummy_true = np.zeros((true_flat.shape[0], len(target_cols)))
        dummy_true[:, i] = true_flat[:, 0]
        true_orig = scaler_y.inverse_transform(dummy_true)[:, i].reshape(true_col.shape)

        mae_orig = np.mean(np.abs(pred_orig - true_orig))
        mse_orig = np.mean((pred_orig - true_orig) ** 2)

        results[col] = {
            'mae_normalized': float(mae),
            'mse_normalized': float(mse),
            'mae_original': float(mae_orig),
            'mse_original': float(mse_orig),
        }

    return results


def main():
    print("=" * 70)
    print("PASO 4: Entrenamiento LSTM (Long Short-Term Memory)")
    print("=" * 70)

    print("\n[1] Cargando dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"    Shape: {df.shape}")
    print(f"    Período: {df.index.min()} → {df.index.max()}")
    print(f"    Features: {FEATURE_COLS}")
    print(f"    Targets: {TARGET_COLS}")

    print(f"\n[2] Creando secuencias (seq_len={SEQ_LEN}, output_len={OUTPUT_LEN})...")
    X, y = create_sequences(df, FEATURE_COLS, TARGET_COLS, SEQ_LEN, OUTPUT_LEN)
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    val_idx = int(len(X_train) * 0.9)
    X_train_split, X_val = X_train[:val_idx], X_train[val_idx:]
    y_train_split, y_val = y_train[:val_idx], y_train[val_idx:]

    print(f"    Train: {X_train_split.shape[0]} samples")
    print(f"    Val: {X_val.shape[0]} samples")
    print(f"    Test: {X_test.shape[0]} samples")

    print("\n[3] Normalizando...")
    scaler_X = StandardScaler()
    X_train_flat = X_train_split.reshape(-1, X_train_split.shape[-1])
    scaler_X.fit(X_train_flat)
    X_train_norm = scaler_X.transform(X_train_flat).reshape(X_train_split.shape)
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_norm = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_norm = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_flat = y_train_split.reshape(-1, y_train_split.shape[-1])
    scaler_y.fit(y_train_flat)
    y_train_norm = scaler_y.transform(y_train_flat).reshape(y_train_split.shape)
    y_val_flat = y_val.reshape(-1, y_val.shape[-1])
    y_val_norm = scaler_y.transform(y_val_flat).reshape(y_val.shape)
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    y_test_norm = scaler_y.transform(y_test_flat).reshape(y_test.shape)

    y_train_flat_norm = y_train_norm.reshape(y_train_norm.shape[0], -1)
    y_val_flat_norm = y_val_norm.reshape(y_val_norm.shape[0], -1)
    y_test_flat_norm = y_test_norm.reshape(y_test_norm.shape[0], -1)

    print(f"    X_train_norm: {X_train_norm.shape}")
    print(f"    y_train_norm: {y_train_norm.shape}")

    print("\n[4] Construyendo modelo LSTM...")
    model = build_lstm(SEQ_LEN, len(FEATURE_COLS), len(TARGET_COLS), OUTPUT_LEN, LSTM_UNITS)
    model.summary()

    print("\n[5] Entrenando...")
    history, train_time = train_model(
        model, X_train_norm, y_train_flat_norm,
        X_val_norm, y_val_flat_norm,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE
    )

    print(f"\n    Tiempo total: {train_time:.2f}s")
    print(f"    Epochs completados: {len(history.history['loss'])}")

    print("\n[6] Evaluando en test set...")
    eval_results = evaluate_model(model, X_test_norm, y_test_flat_norm, scaler_y, TARGET_COLS, OUTPUT_LEN)
    print("\n    Métricas por target (normalizado):")
    for col, metrics in eval_results.items():
        print(f"    {col}: MAE={metrics['mae_normalized']:.6f}, MSE={metrics['mse_normalized']:.6f}")

    print("\n    Métricas por target (escala original):")
    for col, metrics in eval_results.items():
        print(f"    {col}: MAE={metrics['mae_original']:.6f}, MSE={metrics['mse_original']:.6f}")

    print("\n[7] Guardando modelo y parámetros...")
    model_dir = os.path.join(SCRIPT_DIR, "exported_model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.keras"))

    params = {
        'seq_len': SEQ_LEN,
        'output_len': OUTPUT_LEN,
        'lstm_units': LSTM_UNITS,
        'n_features': len(FEATURE_COLS),
        'n_targets': len(TARGET_COLS),
        'feature_cols': FEATURE_COLS,
        'target_cols': TARGET_COLS,
        'X_mean': scaler_X.mean_.tolist(),
        'X_scale': scaler_X.scale_.tolist(),
        'y_mean': scaler_y.mean_.tolist(),
        'y_scale': scaler_y.scale_.tolist(),
        'train_samples': int(X_train_split.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
    }
    with open(os.path.join(model_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(model_dir, "history.json"), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"    Guardado en: {model_dir}")

    print("\n" + "=" * 70)
    print("RESUMEN ENTRENAMIENTO LSTM")
    print("=" * 70)
    print(f"  Dataset: {df.shape[0]} registros @ 15min ({df.shape[0]*15/60:.1f} horas)")
    print(f"  Secuencia: {SEQ_LEN} × 15min = {SEQ_LEN*15}min → predicción {OUTPUT_LEN} × 15min = {OUTPUT_LEN*15}min")
    print(f"  Arquitectura: LSTM units={LSTM_UNITS}")
    print(f"  Parámetros: {model.count_params():,}")
    print(f"  Tiempo: {train_time:.2f}s")
    print(f"  Targets: {TARGET_COLS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
