"""
graficas_pred_vs_real.py — Predicciones vs valores reales en test
Carga modelo entrenado + scaler params, predice sobre test
y grafica comparativa para cada target.
"""
import json, os, sys, warnings, zipfile
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

base = os.path.dirname(__file__)
folder_name = os.path.basename(base)

# ── Cargar scaler params ──
with open(os.path.join(base, 'exported_model', 'scaler_params.json')) as f:
    sp = json.load(f)
SEQ_LEN = sp['seq_len']
feature_cols = sp['feature_cols']
target_cols = sp['target_cols']
y_mean = np.array(sp['y_mean'])
y_scale = np.array(sp['y_scale'])
X_mean = np.array(sp['X_mean'])
X_scale = np.array(sp['X_scale'])

# ── Cargar datos ──
csv_path = os.path.join(base, 'kaggle_data', 'dataset.csv')
df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

X, y = [], []
for i in range(len(df) - SEQ_LEN):
    X.append(df.iloc[i:i+SEQ_LEN][feature_cols].values)
    y.append(df.iloc[i+SEQ_LEN][target_cols].values)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split cronológico 80/20
split = int(len(X) * 0.8)
X_test, y_test = X[split:], y[split:]

# Normalizar
X_test_flat = X_test.reshape(-1, X_test.shape[-1])
X_test_norm = ((X_test_flat - X_mean) / X_scale).reshape(X_test.shape)
y_test_norm = (y_test - y_mean) / y_scale

# ── Cargar modelo ──
model_path = os.path.join(base, 'exported_model', 'model.keras')
if not os.path.exists(model_path):
    print(f"ERROR: No se encuentra {model_path}. Ejecutá el entrenamiento primero.")
    sys.exit(1)

# Extraer pesos del .keras y reconstruir modelo desde la arquitectura de ProtoPy_TFT
with zipfile.ZipFile(model_path) as z:
    z.extract('model.weights.h5', base + '/tmp_weights')
weights_path = os.path.join(base, 'tmp_weights/model.weights.h5')

# Reconstruir modelo TFT (misma arquitectura que ProtoPy_TFT.py)
n_features = len(feature_cols)
n_outputs = len(target_cols)
d_model = 64
num_heads = 4

inputs = keras.Input(shape=(SEQ_LEN, n_features))
flat = layers.Flatten()(inputs)
context = layers.Dense(d_model, activation='elu')(flat)
context = layers.Dropout(0.1)(context)
context = layers.Dense(d_model)(context)
feat_weights = layers.Dense(n_features, activation='softmax', name='feat_weights')(context)
feat_weights = layers.Lambda(lambda w: tf.expand_dims(w, 1),
                              output_shape=lambda s: (s[0], 1, s[1]),
                              name='expand_weights')(feat_weights)
x = layers.Multiply(name='apply_weights')([inputs, feat_weights])
x = layers.Dense(d_model)(x)
skip = x
lstm_out = layers.LSTM(d_model, return_sequences=True)(x)
x = layers.Add()([x, lstm_out])
x = layers.LayerNormalization()(x)
attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
x = layers.Add()([x, attn_out])
x = layers.LayerNormalization()(x)
gate = layers.Dense(d_model, activation='sigmoid')(x)
x = layers.Multiply()([x, gate])
skip_proj = layers.Dense(d_model)(skip) if skip.shape[-1] != d_model else skip
x = layers.Add()([x, skip_proj])
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(n_outputs)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='TFT_Microgrid')

model.load_weights(weights_path)
# Limpiar
import shutil
shutil.rmtree(os.path.join(base, 'tmp_weights'), ignore_errors=True)
print(f"Modelo cargado: {model.name}")

# ── Predecir ──
y_pred_norm = model.predict(X_test_norm, verbose=0)
y_pred = y_pred_norm * y_scale + y_mean
y_actual = y_test

# ── Graficar ──
n_show = min(150, len(y_pred))
target_labels = [c.replace('_', ' ').title() for c in target_cols]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (ax, tlabel) in enumerate(zip(axes, target_labels)):
    ax.plot(range(n_show), y_actual[:n_show, i], 'b-', linewidth=1.5, alpha=0.7, label='Real')
    ax.plot(range(n_show), y_pred[:n_show, i], 'r--', linewidth=1.5, alpha=0.7, label='Predicho')
    ax.set_title(tlabel, fontsize=13, fontweight='bold')
    ax.set_xlabel('Muestra en test')
    ax.set_ylabel('Valor')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

fig.suptitle(f'{folder_name.title()} — Predicciones vs Reales\n({len(y_pred)} muestras test, mostrando {n_show})',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(base, 'pred_vs_real.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[OK] Gráfica guardada: {out}")
