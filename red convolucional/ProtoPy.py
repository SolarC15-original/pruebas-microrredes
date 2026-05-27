"""
ProtoPy.py — Prototipo 1D-CNN / TCN para microrredes
Carga dataset real de Kaggle (Renewable Energy Microgrid),
entrena arquitecturas convolucionales y exporta a TFJS + scaler params.
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

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
def load_rye_dataset():
    """Carga el dataset real Rye Microgrid (Zenodo)."""
    csv_path = 'kaggle_data/dataset.csv'
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)
    print(f"    Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def create_sequences(data, feature_cols, target_cols, seq_len=24):
    """Convierte el DataFrame en tensores 3D [muestras, pasos, features]."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len][feature_cols].values)
        y.append(data.iloc[i+seq_len][target_cols].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ─────────────────────────────────────────────
# 2. ARQUITECTURAS CONVOLUCIONALES
# ─────────────────────────────────────────────
def build_cnn_1d(input_shape, n_outputs):
    """1D-CNN estándar con 2 capas Conv1D + Pooling."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_outputs)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='CNN1D_Microgrid')

def build_tcn_block(x, filters, kernel_size, dilation_rate, dropout=0.2):
    """Bloque convolucional temporal con dilatación + skip connection."""
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='causal',
                      dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, padding='causal',
                      dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1)(shortcut)
    return layers.Add()([shortcut, x])

def build_tcn(input_shape, n_outputs):
    """Red Temporal Convolutional (TCN) con dilatación creciente."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    filters_list = [32, 64, 64]
    dilations = [1, 2, 4]
    for f, d in zip(filters_list, dilations):
        x = build_tcn_block(x, filters=f, kernel_size=3, dilation_rate=d)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_outputs)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='TCN_Microgrid')

# ─────────────────────────────────────────────
# 3. ENTRENAMIENTO Y EVALUACIÓN
# ─────────────────────────────────────────────
def train_and_benchmark(model, X_train, y_train, X_test, y_test, epochs=20):
    """Entrena y mide tiempo por época, pérdida final y MAE."""
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

# ─────────────────────────────────────────────
# 4. INFERENCIA (simula edge/VPS)
# ─────────────────────────────────────────────
def benchmark_inference(model, X_test, n_runs=500):
    """Mide latencia de inferencia: warm-up vs. steady-state."""
    # Warm-up (10 pasos)
    for i in range(10):
        model.predict(X_test[i:i+1], verbose=0)

    # Steady-state
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

# ─────────────────────────────────────────────
# 5. EXPORTACIÓN A FORMATO TFJS
# ─────────────────────────────────────────────
def export_to_tfjs(model, path='exported_model'):
    """Guarda modelo en formato TensorFlow.js JSON."""
    os.makedirs(path, exist_ok=True)
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, path)
        print(f"[Export] Modelo guardado en '{path}/'")
        files = os.listdir(path)
        for f in sorted(files):
            fpath = os.path.join(path, f)
            size = os.path.getsize(fpath)
            print(f"  {f}: {size/1024:.1f} KB")
    except (ImportError, Exception) as e:
        print(f"    [WARN] Export TFJS falló: {e}")
        # Fallback: guardar como SavedModel + .keras
        tf.saved_model.save(model, os.path.join(path, 'saved_model'))
        model.save(os.path.join(path, 'model.keras'))
        print(f"    Modelo guardado como SavedModel + .keras en '{path}/'")
        print("    Para convertir: tensorflowjs_converter --input_format=keras \\")
        print(f"      {path}/model.keras {path}/tfjs")



# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("="*65)
    print("PROTOTIPO 1D-CNN / TCN — MICRORREDES ELÉCTRICAS")
    print("="*65)

    SEQ_LEN = 24

    # ── Datos ──
    print(f"\n[1] Cargando dataset Rye Microgrid (seq_len={SEQ_LEN})...")
    df = load_rye_dataset()

    target_cols = ['Consumption', 'Solar', 'Wind']
    feature_cols = ['Consumption', 'Solar', 'Wind']

    print(f"    Features ({len(feature_cols)}): {feature_cols}")
    print(f"    Targets ({len(target_cols)}): {target_cols}")

    X, y = create_sequences(df, feature_cols, target_cols, seq_len=SEQ_LEN)
    print(f"    Tensores: X {X.shape}, y {y.shape}")

    # Partición cronológica (time-series: primeros 80% train, últimos 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalización Z-score (fit SOLO en train)
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler_X = StandardScaler().fit(X_flat)
    X_train_norm = scaler_X.transform(X_flat).reshape(X_train.shape)
    X_flat_test = X_test.reshape(-1, X_test.shape[-1])
    X_test_norm = scaler_X.transform(X_flat_test).reshape(X_test.shape)

    scaler_y = StandardScaler().fit(y_train)
    y_train_norm = scaler_y.transform(y_train)
    y_test_norm = scaler_y.transform(y_test)

    print(f"    Train: {X_train_norm.shape}  Test: {X_test_norm.shape}")

    # Exportar parámetros de normalización para JS
    os.makedirs('exported_model', exist_ok=True)
    scaler_params = {
        'seq_len': SEQ_LEN,
        'n_features': X.shape[-1],
        'n_targets': y.shape[-1],
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'X_mean': scaler_X.mean_.tolist(),
        'X_scale': scaler_X.scale_.tolist(),
        'y_mean': scaler_y.mean_.tolist(),
        'y_scale': scaler_y.scale_.tolist(),
    }
    with open('exported_model/scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print("    Parámetros de normalización exportados a exported_model/scaler_params.json")

    # ── Modelos ──
    results = []
    models_to_train = [
        ('CNN 1D', build_cnn_1d(X.shape[1:], len(target_cols))),
        ('TCN', build_tcn(X.shape[1:], len(target_cols))),
    ]

    for name, model in models_to_train:
        print(f"\n[2] Entrenando: {name}")
        model.summary()
        stats = train_and_benchmark(model, X_train_norm, y_train_norm,
                                     X_test_norm, y_test_norm)
        results.append(stats)
        print(f"    Params: {stats['params']:,}")
        print(f"    Tiempo: {stats['train_time_s']}s total  |  {stats['time_per_epoch_s']}s/epoch")
        print(f"    Test MAE: {stats['test_mae']}  |  MSE: {stats['test_loss_mse']}")

    # ── Benchmark de inferencia (TCN) ──
    print("\n[3] Benchmark de inferencia (500 predicciones)...")
    best_model = [m for _, m in models_to_train if m.name == 'TCN_Microgrid'][0]
    inf_stats = benchmark_inference(best_model, X_test_norm, n_runs=500)
    print(f"    Mean:  {inf_stats['mean_ms']} ms")
    print(f"    P50:   {inf_stats['p50_ms']} ms")
    print(f"    P95:   {inf_stats['p95_ms']} ms")
    print(f"    P99:   {inf_stats['p99_ms']} ms")
    print(f"    Min:   {inf_stats['min_ms']} ms")
    print(f"    Max:   {inf_stats['max_ms']} ms")

    # ── Exportación ──
    print("\n[4] Exportando modelo TCN a TensorFlow.js...")
    try:
        export_to_tfjs(best_model, path='exported_model')
    except ImportError:
        print("    [WARN] tensorflowjs no instalado. Instalar: pip install tensorflowjs")
        os.makedirs('exported_model', exist_ok=True)
        best_model.save('exported_model/model.keras')
        print("    Modelo guardado en formato .keras (exportación TFJS manual requerida)")

    # ── Exportar métricas a JSON ──
    metrics_py = {}
    for r in results:
        metrics_py[r['model_name']] = {
            'params': r['params'],
            'train_time_s': r['train_time_s'],
            'time_per_epoch_s': r['time_per_epoch_s'],
            'test_loss_mse': r['test_loss_mse'],
            'test_mae': r['test_mae'],
        }
    metrics_py['inference'] = {
        'model': best_model.name,
        'mean_ms': inf_stats['mean_ms'],
        'std_ms': inf_stats['std_ms'],
        'p50_ms': inf_stats['p50_ms'],
        'p95_ms': inf_stats['p95_ms'],
        'p99_ms': inf_stats['p99_ms'],
        'throughput_pred_s': round(1000 / inf_stats['mean_ms'], 1),
    }
    with open('exported_model/metrics_python.json', 'w') as f:
        json.dump(metrics_py, f, indent=2)
    print("    Métricas exportadas a exported_model/metrics_python.json")

    # ── Tabla comparativa ──
    print("\n" + "="*65)
    print("TABLA COMPARATIVA — ARQUITECTURAS CONVOLUCIONALES")
    print("="*65)
    header = f"{'Modelo':<12} {'Params':<12} {'Tiempo(s)':<12} {'s/epoch':<12} {'MAE':<12} {'MSE':<12}"
    print(header)
    print("-"*65)
    for r in results:
        row = f"{r['model_name']:<12} {r['params']:<12,} {r['train_time_s']:<12} {r['time_per_epoch_s']:<12} {r['test_mae']:<12} {r['test_loss_mse']:<12}"
        print(row)
    print("-"*65)
    print(f"\nRye Microgrid: {df.shape[0]} filas, {len(feature_cols)} features, seq_len={SEQ_LEN}")
    print(f"Inferencia TCN (500 runs): {inf_stats['mean_ms']} ms mean | P95: {inf_stats['p95_ms']} ms")
    print("="*65)
