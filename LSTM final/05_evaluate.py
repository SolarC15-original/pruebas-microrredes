"""
05_evaluate.py — Gráficas de evaluación LSTM con Plotly
Genera: pérdida entrenamiento, scatter plots, series temporales, métricas R²
"""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "exported_model")
DATA_PATH = os.path.join(SCRIPT_DIR, "kaggle_data", "microgrid_15min.csv")

SEQ_LEN = 12
OUTPUT_LEN = 96
FEATURE_COLS = ['VA', 'VB', 'Fre', 'Temp', 'temp_2m', 'shortwave_radiation',
                'wind_speed_10m', 'pressure_msl', 'cloud_cover', 'Generacion_Solar']
TARGET_COLS = ['Fre', 'VA', 'VB', 'Generacion_Solar']

TARGET_LABELS = {
    'Fre': 'Frecuencia (Hz)',
    'VA': 'Voltaje A (V)',
    'VB': 'Voltaje B (V)',
    'Generacion_Solar': 'Generación Solar (kWh)',
}


def load_history():
    with open(os.path.join(MODEL_DIR, "history.json")) as f:
        return json.load(f)


def create_sequences(data, feature_cols, target_cols, seq_len, output_len):
    X, y = [], []
    for i in range(len(data) - seq_len - output_len + 1):
        X.append(data[feature_cols].iloc[i:i+seq_len].values)
        y.append(data[target_cols].iloc[i+seq_len:i+seq_len+output_len].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def plot_loss_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history['loss'])+1)),
        y=history['loss'],
        name='Train Loss',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history['val_loss'])+1)),
        y=history['val_loss'],
        name='Val Loss',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig.update_layout(
        title='Pérdida durante entrenamiento LSTM',
        xaxis_title='Epoch',
        yaxis_title='MSE Loss',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        height=450, width=800
    )
    fig.write_html(os.path.join(SCRIPT_DIR, "loss_history.html"))
    print(f"  Guardado: loss_history.html")


def plot_scatter(y_true, y_pred, target_cols, output_len):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[TARGET_LABELS[c] for c in target_cols])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, col in enumerate(target_cols):
        row, col_idx = (idx // 2) + 1, (idx % 2) + 1
        pred_vals = y_pred[:, :, idx].flatten()
        true_vals = y_true[:, :, idx].flatten()

        mae = np.mean(np.abs(pred_vals - true_vals))
        rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))

        mean_true = np.mean(true_vals)
        ss_tot = np.sum((true_vals - mean_true) ** 2)
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        fig.add_trace(go.Scatter(
            x=true_vals, y=pred_vals, mode='markers',
            marker=dict(color=colors[idx], opacity=0.5, size=5),
            name=f'{col} (R²={r2:.3f})'
        ), row=row, col=col_idx)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='gray', dash='dash'),
            showlegend=False
        ), row=row, col=col_idx)

        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'MAE={mae:.4f} | R²={r2:.3f}',
            showarrow=False, font=dict(size=9),
            bgcolor='white', borderpad=3,
            row=row, col=col_idx
        )

    fig.update_layout(
        height=650, width=850,
        title='Predicción vs Real — LSTM (datos normalizados)',
        showlegend=True
    )
    fig.write_html(os.path.join(SCRIPT_DIR, "scatter_predictions.html"))
    print(f"  Guardado: scatter_predictions.html")


def plot_time_series(y_true, y_pred, target_cols, output_len, sample_idx=0):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[TARGET_LABELS[c] for c in target_cols])
    colors_true = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_pred = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']

    for idx, col in enumerate(target_cols):
        row, col_idx = (idx // 2) + 1, (idx % 2) + 1
        true_seq = y_true[sample_idx, :, idx]
        pred_seq = y_pred[sample_idx, :, idx]
        hours = list(range(output_len))

        fig.add_trace(go.Scatter(
            x=hours, y=true_seq, mode='lines+markers',
            marker=dict(color=colors_true[idx], size=6),
            name='Real', line=dict(width=2)
        ), row=row, col=col_idx)

        fig.add_trace(go.Scatter(
            x=hours, y=pred_seq, mode='lines+markers',
            marker=dict(color=colors_pred[idx], size=6, symbol='diamond'),
            name='Predicción', line=dict(width=2, dash='dash')
        ), row=row, col=col_idx)

        mae = np.mean(np.abs(pred_seq - true_seq))
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'MAE={mae:.4f}',
            showarrow=False, font=dict(size=9),
            bgcolor='white', borderpad=3,
            row=row, col=col_idx
        )

    fig.update_layout(
        height=650, width=850,
        title=f'Serie temporal: predicción 24h vs real — {output_len} intervalos de 15min (muestra {sample_idx})',
        showlegend=True, xaxis_title='Intervalos de 15min ahead'
    )
    fig.write_html(os.path.join(SCRIPT_DIR, "time_series_prediction.html"))
    print(f"  Guardado: time_series_prediction.html")


def plot_all_targets_24h(y_true, y_pred, target_cols, output_len):
    colors_true = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_pred = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    n_samples = 1

    for idx, col in enumerate(target_cols):
        fig = go.Figure()

        for sample in range(n_samples):
            fig.add_trace(go.Scatter(
                x=list(range(output_len)),
                y=y_true[sample, :, idx],
                mode='lines',
                line=dict(color=colors_true[idx], width=2),
                opacity=0.8,
                name='Real' if sample == 0 else None,
                showlegend=sample == 0
            ))
            fig.add_trace(go.Scatter(
                x=list(range(output_len)),
                y=y_pred[sample, :, idx],
                mode='lines',
                line=dict(color=colors_pred[idx], width=2, dash='dash'),
                opacity=0.8,
                name='Pred' if sample == 0 else None,
                showlegend=sample == 0
            ))

        mae_all = np.mean(np.abs(y_pred[:n_samples, :, idx] - y_true[:n_samples, :, idx]))
        fig.update_layout(
            title=f'{TARGET_LABELS[col]} — Predicción 24h vs Real (muestra {sample})',
            xaxis_title='Intervalos de 15min ahead',
            yaxis_title=col,
            template='plotly_white',
            height=400, width=800
        )
        fig.write_html(os.path.join(SCRIPT_DIR, f"ts_{col}_24h.html"))
        print(f"  Guardado: ts_{col}_24h.html")


def compute_metrics_table(y_true, y_pred, target_cols, output_len):
    rows = []
    for idx, col in enumerate(target_cols):
        pred = y_pred[:, :, idx].flatten()
        true = y_true[:, :, idx].flatten()

        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)

        mean_true = np.mean(true)
        ss_tot = np.sum((true - mean_true) ** 2)
        ss_res = np.sum((true - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))

        mape = np.mean(np.abs((pred - true) / (true + 1e-10))) * 100

        rows.append({
            'Target': col,
            'MAE': f"{mae:.6f}",
            'RMSE': f"{rmse:.6f}",
            'MSE': f"{mse:.6f}",
            'R²': f"{r2:.6f}",
            'MAPE (%)': f"{mape:.2f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("MÉTRICAS DE EVALUACIÓN — LSTM (datos normalizados)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    csv_path = os.path.join(SCRIPT_DIR, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Guardado: metrics_summary.csv")
    return df


def main():
    print("="*70)
    print("PASO 5: Evaluación LSTM — Gráficas y Métricas (Plotly)")
    print("="*70)

    print("\n[1] Cargando historial de entrenamiento...")
    history = load_history()
    print(f"    Epochs: {len(history['loss'])}")

    print("\n[2] Cargando modelo...")
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))
    print(f"    Parámetros: {model.count_params():,}")

    print("\n[3] Preparando datos de test...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    X, y = create_sequences(df, FEATURE_COLS, TARGET_COLS, SEQ_LEN, OUTPUT_LEN)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    val_idx = int(len(X_train) * 0.9)
    X_train_split, X_val = X_train[:val_idx], X_train[val_idx:]
    y_train_split, y_val = y_train[:val_idx], y_train[val_idx:]

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

    print(f"    Train: {X_train_split.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    print("\n[4] Generando predicciones...")
    y_pred_norm = model.predict(X_test_norm, verbose=0)
    y_pred_2d = y_pred_norm.reshape(-1, OUTPUT_LEN, len(TARGET_COLS))
    y_test_2d = y_test_norm.reshape(-1, OUTPUT_LEN, len(TARGET_COLS))

    print("\n[5] Generando gráficas...")
    plot_loss_history(history)
    plot_scatter(y_test_2d, y_pred_2d, TARGET_COLS, OUTPUT_LEN)
    plot_time_series(y_test_2d, y_pred_2d, TARGET_COLS, OUTPUT_LEN, sample_idx=0)
    plot_all_targets_24h(y_test_2d, y_pred_2d, TARGET_COLS, OUTPUT_LEN)

    print("\n[6] Calculando métricas...")
    compute_metrics_table(y_test_2d, y_pred_2d, TARGET_COLS, OUTPUT_LEN)

    print("\n" + "="*70)
    print("EVALUACIÓN COMPLETADA")
    print("Archivos HTML generados en:", SCRIPT_DIR)
    print("="*70)


if __name__ == "__main__":
    main()
