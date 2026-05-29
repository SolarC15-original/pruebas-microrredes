"""
analyze_tft_importance.py — Extrae pesos de importancia de features del modelo TFT
Usa los pesos del VariableSelectionNetwork para ver cuáles features contribuyen más
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "exported_model")
DATA_PATH = os.path.join(SCRIPT_DIR, "kaggle_data", "microgrid_15min.csv")

FEATURE_COLS = ['VA', 'VB', 'Fre', 'Temp', 'temp_2m', 'shortwave_radiation',
                'wind_speed_10m', 'pressure_msl', 'cloud_cover', 'Generacion_Solar']
TARGET_COLS = ['Fre', 'VA', 'VB', 'Generacion_Solar']

FEATURE_GROUPS = {
    'Eléctricas': ['VA', 'VB', 'Fre'],
    'Temperatura': ['Temp', 'temp_2m'],
    'Radiación Solar': ['shortwave_radiation', 'Generacion_Solar'],
    'Viento': ['wind_speed_10m'],
    'Presión': ['pressure_msl'],
    'Nubes': ['cloud_cover'],
}


class GatedResidualNetwork(keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.ffn = keras.Sequential([
            keras.layers.Dense(units, activation='elu'),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(units),
            keras.layers.Dropout(dropout_rate),
        ])
        self.gate = keras.Sequential([keras.layers.Dense(units, activation='sigmoid')])
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x, training=False):
        skip = x
        out = self.ffn(x, training=training)
        gate = self.gate(x)
        out = out * gate
        return self.layer_norm(out + skip)


class VariableSelectionNetwork(keras.layers.Layer):
    def __init__(self, num_features, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.d_model = d_model
        self.flatten = keras.layers.Flatten()
        self.context = keras.Sequential([
            keras.layers.Dense(d_model, activation='elu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(d_model),
        ])
        self.feat_weights = keras.layers.Dense(num_features, activation='softmax')

    def call(self, x):
        flat = self.flatten(x)
        ctx = self.context(flat)
        w = self.feat_weights(ctx)
        w = tf.expand_dims(w, 1)
        return x * w, w

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_features': self.num_features, 'd_model': self.d_model})
        return cfg


def extract_feature_weights(model, df, feature_cols, n_samples=100):
    print("\n[1] Extrayendo pesos de importancia de features...")

    scaler_X = StandardScaler()
    X_data = df[feature_cols].values
    X_scaled = scaler_X.fit_transform(X_data)
    indices = np.random.choice(len(X_scaled), min(n_samples, len(X_scaled)), replace=False)

    vsn_layer = None
    for layer in model.layers:
        if layer.name == 'variable_selection_network':
            vsn_layer = layer
            break

    if vsn_layer is None:
        for layer in model.layers:
            if 'VariableSelection' in str(type(layer).__name__):
                vsn_layer = layer
                break

    if vsn_layer is None:
        print("  No se encontró VariableSelectionNetwork")
        return None

    weights = []
    for idx in indices:
        seq = X_scaled[idx:idx+12]
        if len(seq) < 12:
            continue
        seq = seq.reshape(1, 12, len(feature_cols))
        _, w = vsn_layer(seq, training=False)
        weights.append(w.numpy().flatten())

    weights = np.array(weights)
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)

    df_weights = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': mean_weights,
        'Std': std_weights
    })
    df_weights = df_weights.sort_values('Importance', ascending=False)

    print(f"  Muestras procesadas: {len(weights)}")
    return df_weights


def analyze_by_group(df_weights, feature_groups):
    print("\n[2] Importancia por grupo de features...")

    group_scores = {}
    for group, features in feature_groups.items():
        mask = df_weights['Feature'].isin(features)
        group_importance = df_weights.loc[mask, 'Importance'].sum()
        group_scores[group] = group_importance

    df_groups = pd.DataFrame({
        'Group': list(group_scores.keys()),
        'Importance': list(group_scores.values())
    })
    df_groups = df_groups.sort_values('Importance', ascending=False)
    df_groups['Pct'] = (df_groups['Importance'] / df_groups['Importance'].sum() * 100).round(1)

    return df_groups


def analyze_target_contribution(df, feature_cols, target_cols):
    print("\n[3] Contribución de features por target (análisis de correlación)...")

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(df[feature_cols].values)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(df[target_cols].values)

    correlations = np.zeros((len(target_cols), len(feature_cols)))

    for i in range(len(target_cols)):
        for j in range(len(feature_cols)):
            corr = np.corrcoef(X[:, j], y[:, i])[0, 1]
            correlations[i, j] = abs(corr) if not np.isnan(corr) else 0

    df_corr = pd.DataFrame(
        correlations,
        index=target_cols,
        columns=feature_cols
    )

    print("\n  Correlación absoluta promaipo por target:")
    for target in target_cols:
        top3 = df_corr.loc[target].nlargest(3)
        print(f"    {target}: {', '.join([f'{f}({v:.3f})' for f, v in top3.items()])}")

    return df_corr


def print_analysis(df_weights, df_groups, df_corr):
    print("\n" + "="*70)
    print("ANÁLISIS DE IMPORTANCIA DE FEATURES — TFT")
    print("="*70)

    print("\n[重要性 Individual de Features]")
    print("-"*50)
    for _, row in df_weights.iterrows():
        bar = "█" * int(row['Importance'] * 50)
        print(f"  {row['Feature']:22s}: {row['Importance']:.4f} ± {row['Std']:.4f} {bar}")

    print("\n[Importancia por Grupo]")
    print("-"*50)
    for _, row in df_groups.iterrows():
        bar = "█" * int(row['Pct'] / 2)
        print(f"  {row['Group']:22s}: {row['Pct']:5.1f}% {bar}")

    print("\n[Matriz de Correlación Target vs Feature]")
    print("-"*50)
    print(df_corr.round(3).to_string())


def main():
    print("="*70)
    print("ANÁLISIS DE IMPORTANCIA — TFT (Variable Selection Network)")
    print("="*70)

    print("\n[Cargando modelo y dataset...]")
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "model.keras"),
        custom_objects={
            'GatedResidualNetwork': GatedResidualNetwork,
            'VariableSelectionNetwork': VariableSelectionNetwork
        }
    )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"  Dataset: {df.shape}")

    df_weights = extract_feature_weights(model, df, FEATURE_COLS)
    df_groups = analyze_by_group(df_weights, FEATURE_GROUPS)
    df_corr = analyze_target_contribution(df, FEATURE_COLS, TARGET_COLS)

    print_analysis(df_weights, df_groups, df_corr)

    csv_path = os.path.join(SCRIPT_DIR, "feature_importance.csv")
    df_weights.to_csv(csv_path, index=False)
    print(f"\n  Guardado: {csv_path}")

    print("\n" + "="*70)
    print("INTERPRETACIÓN DEL MODELO")
    print("="*70)
    print("""
1. VARIABLE SELECTION NETWORK (VSN)
   - Cada timestep recibe pesos softmax que determinan cuánto contribuye
   - Features con peso alto = el modelo las usa más para predecir
   - Features con peso bajo = el modelo las ignora o son redundantes

2. GRN (GATED RESIDUAL NETWORK)
   - El gate (sigmoid) controla cuánta información pasa
   - Gate cercano a 0 = suprime la skip connection por completo
   - Gate cercano a 1 = pasa toda la información

3. ARQUITECTURA TEMPORAL
   - LSTM encoder: procesa la secuencia de 12 timesteps (3 horas)
   - Multi-head attention: captura dependencias a largo plazo
   - LSTM decoder: genera los 96 outputs (24 horas)

4. IMPORTANCIA DE FEATURES
   - Features con alta importancia son los que más influyen en la predicción
   - Features correlacionados con targets ayudan a predecirlos
    """)


if __name__ == "__main__":
    main()
