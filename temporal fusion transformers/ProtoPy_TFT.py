"""
ProtoPy_TFT.py — Prototipo TFT (Temporal Fusion Transformer) para microrredes
Carga dataset real de Kaggle, entrena TFT con Variable Selection,
LSTM encoder, Multi-Head Self-Attention y gating.
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

def create_sequences(data, feature_cols, target_cols, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len][feature_cols].values)
        y.append(data.iloc[i+seq_len][target_cols].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_tft(input_shape, n_outputs, d_model=64, num_heads=4):
    """
    Temporal Fusion Transformer (simplificado pero con componentes clave):
    1. Variable Selection Network (GRN + softmax weights)
    2. LSTM Encoder
    3. Multi-Head Self-Attention
    4. Gating Residual Network
    5. Skip connections + LayerNorm
    """
    inputs = keras.Input(shape=input_shape)
    seq_len, n_features = input_shape

    # ── 1. Variable Selection Network ──
    flat = layers.Flatten()(inputs)
    context = layers.Dense(d_model, activation='elu')(flat)
    context = layers.Dropout(0.1)(context)
    context = layers.Dense(d_model)(context)
    feat_weights = layers.Dense(n_features, activation='softmax', name='feat_weights')(context)
    feat_weights = layers.Lambda(lambda w: tf.expand_dims(w, 1), name='expand_weights')(feat_weights)
    x = layers.Multiply(name='apply_weights')([inputs, feat_weights])

    # ── 2. Input projection ──
    x = layers.Dense(d_model)(x)
    skip = x

    # ── 3. LSTM Encoder ──
    lstm_out = layers.LSTM(d_model, return_sequences=True)(x)
    x = layers.Add()([x, lstm_out])
    x = layers.LayerNormalization()(x)

    # ── 4. Multi-Head Self-Attention ──
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    # ── 5. Gating with residual ──
    gate = layers.Dense(d_model, activation='sigmoid')(x)
    x = layers.Multiply()([x, gate])
    skip_proj = layers.Dense(d_model)(skip) if skip.shape[-1] != d_model else skip
    x = layers.Add()([x, skip_proj])

    # ── 6. Output head ──
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_outputs)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='TFT_Microgrid')

def train_and_benchmark(model, X_train, y_train, X_test, y_test, epochs=20):
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    t0 = time.perf_counter()
    history = model.fit(X_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=64, verbose=0)
    train_time = time.perf_counter() - t0
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return {
        'model_name': model.name,
        'params': model.count_params(),
        'train_time_s': round(train_time, 2),
        'time_per_epoch_s': round(train_time / epochs, 3),
        'test_loss_mse': round(loss, 4),
        'test_mae': round(mae, 4),
        'history': history.history,
    }

def benchmark_inference(model, X_test, n_runs=500):
    for i in range(10):
        model.predict(X_test[i:i+1], verbose=0)
    latencies = []
    for i in range(n_runs):
        idx = i % len(X_test)
        t0 = time.perf_counter()
        model.predict(X_test[idx:idx+1], verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies = np.array(latencies)
    return {
        'mean_ms': round(float(np.mean(latencies)), 3),
        'std_ms': round(float(np.std(latencies)), 3),
        'p50_ms': round(float(np.median(latencies)), 3),
        'p95_ms': round(float(np.percentile(latencies, 95)), 3),
        'p99_ms': round(float(np.percentile(latencies, 99)), 3),
        'min_ms': round(float(np.min(latencies)), 3),
        'max_ms': round(float(np.max(latencies)), 3),
    }

def export_to_tfjs(model, path='exported_model'):
    os.makedirs(path, exist_ok=True)
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, path)
        print(f"[Export] Modelo guardado en '{path}/'")
    except (ImportError, Exception) as e:
        print(f"    [WARN] Export TFJS falló: {e}")
        model.save(os.path.join(path, 'model.keras'))

def load_rye_dataset():
    """Carga el dataset real Rye Microgrid (Zenodo)."""
    csv_path = 'kaggle_data/dataset.csv'
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)
    print(f"    Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

if __name__ == '__main__':
    print("="*65)
    print("PROTOTIPO TFT (TEMPORAL FUSION TRANSFORMER) — MICRORREDES")
    print("="*65)

    SEQ_LEN = 24

    print(f"\n[1] Cargando dataset real de Kaggle (seq_len={SEQ_LEN})...")
    df = load_rye_dataset()

    target_cols = ['Consumption', 'Solar', 'Wind']
    feature_cols = ['Consumption', 'Solar', 'Wind']

    print(f"    Features ({len(feature_cols)}): {feature_cols}")
    print(f"    Targets ({len(target_cols)}): {target_cols}")

    X, y = create_sequences(df, feature_cols, target_cols, seq_len=SEQ_LEN)
    print(f"    Tensores: X {X.shape}, y {y.shape}")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler_X = StandardScaler().fit(X_flat)
    X_train_norm = scaler_X.transform(X_flat).reshape(X_train.shape)
    X_flat_test = X_test.reshape(-1, X_test.shape[-1])
    X_test_norm = scaler_X.transform(X_flat_test).reshape(X_test.shape)

    scaler_y = StandardScaler().fit(y_train)
    y_train_norm = scaler_y.transform(y_train)
    y_test_norm = scaler_y.transform(y_test)

    print(f"    Train: {X_train_norm.shape}  Test: {X_test_norm.shape}")

    os.makedirs('exported_model', exist_ok=True)
    scaler_params = {
        'seq_len': SEQ_LEN, 'n_features': X.shape[-1], 'n_targets': y.shape[-1],
        'feature_cols': feature_cols, 'target_cols': target_cols,
        'X_mean': scaler_X.mean_.tolist(), 'X_scale': scaler_X.scale_.tolist(),
        'y_mean': scaler_y.mean_.tolist(), 'y_scale': scaler_y.scale_.tolist(),
    }
    with open('exported_model/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print("    Parámetros de normalización exportados")

    print("\n[2] Construyendo TFT...")
    model = build_tft(X.shape[1:], len(target_cols))
    model.summary()
    stats = train_and_benchmark(model, X_train_norm, y_train_norm,
                                X_test_norm, y_test_norm)
    print(f"\n    Params: {stats['params']:,}")
    print(f"    Tiempo: {stats['train_time_s']}s total  |  {stats['time_per_epoch_s']}s/epoch")
    print(f"    Test MAE: {stats['test_mae']}  |  MSE: {stats['test_loss_mse']}")

    print("\n[3] Benchmark de inferencia (500 predicciones)...")
    inf_stats = benchmark_inference(model, X_test_norm, n_runs=500)
    print(f"    Mean:  {inf_stats['mean_ms']} ms")
    print(f"    P50:   {inf_stats['p50_ms']} ms")
    print(f"    P95:   {inf_stats['p95_ms']} ms")
    print(f"    P99:   {inf_stats['p99_ms']} ms")
    print(f"    Min:   {inf_stats['min_ms']} ms")
    print(f"    Max:   {inf_stats['max_ms']} ms")

    print("\n[4] Exportando modelo TFT a TensorFlow.js...")
    try:
        export_to_tfjs(model, path='exported_model')
    except ImportError:
        print("    [WARN] tensorflowjs no instalado.")
        os.makedirs('exported_model', exist_ok=True)
        model.save('exported_model/model.keras')

    metrics_py = {
        model.name: {
            'params': stats['params'],
            'train_time_s': stats['train_time_s'],
            'time_per_epoch_s': stats['time_per_epoch_s'],
            'test_loss_mse': stats['test_loss_mse'],
            'test_mae': stats['test_mae'],
        },
        'inference': {
            'model': model.name,
            'mean_ms': inf_stats['mean_ms'],
            'std_ms': inf_stats['std_ms'],
            'p50_ms': inf_stats['p50_ms'],
            'p95_ms': inf_stats['p95_ms'],
            'p99_ms': inf_stats['p99_ms'],
            'throughput_pred_s': round(1000 / inf_stats['mean_ms'], 1),
        },
    }
    with open('exported_model/metrics_python.json', 'w') as f:
        json.dump(metrics_py, f, indent=2)
    print("    Métricas exportadas")

    print("\n" + "="*65)
    print("RESULTADOS TFT — ARQUITECTURA CON TRANSFORMER")
    print("="*65)
    print(f"  Params:         {stats['params']:,}")
    print(f"  Train time:     {stats['train_time_s']}s total  |  {stats['time_per_epoch_s']}s/epoch")
    print(f"  Test MAE:       {stats['test_mae']}")
    print(f"  Test MSE:       {stats['test_loss_mse']}")
    print(f"  Inferencia:     {inf_stats['mean_ms']} ms mean | P95: {inf_stats['p95_ms']} ms")
    print(f"  Throughput:     {metrics_py['inference']['throughput_pred_s']} pred/s")
    print(f"  Rye Microgrid:  {df.shape[0]} filas, {len(feature_cols)} features, seq_len={SEQ_LEN}")
    print("="*65)
