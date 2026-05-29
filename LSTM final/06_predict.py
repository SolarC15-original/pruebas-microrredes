"""
06_predict.py — Predicción LSTM con datos frescos de MongoDB + Open-Meteo
Carga modelo entrenado, obtiene últimos datos de sensores, y predice 24h ahead.
"""
import os, json, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import pymongo
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "exported_model")
DATA_DIR = os.path.join(SCRIPT_DIR, "kaggle_data")

MONGO_URI = "mongodb+srv://root:SISTEMA2025qwer@clusterinteligente.qnejnxi.mongodb.net/"
DATABASE = "sistema_inteligente_db"
COLLECTIONS = {
    'Energy Tablero 2': '69e50396cc47b3a12324e64f',
    'Temperatura PV': '69ee990dcc47b3a12324e851',
}

TARGET_COLS = ['Fre', 'VA', 'VB', 'Generacion_Solar']
FEATURE_COLS = ['VA', 'VB', 'Fre', 'Temp', 'temp_2m', 'shortwave_radiation',
                'wind_speed_10m', 'pressure_msl', 'cloud_cover', 'Generacion_Solar']
TARGET_LABELS = {
    'Fre': 'Frecuencia (Hz)',
    'VA': 'Voltaje A (V)',
    'VB': 'Voltaje B (V)',
    'Generacion_Solar': 'Generación Solar (kWh)',
}


def fetch_mongo_collection(collection_id, hours=72):
    client = pymongo.MongoClient(MONGO_URI)
    col = client[DATABASE][collection_id]
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
    docs = list(col.find({'createAt': {'$gte': cutoff}}).sort('createAt', 1))
    client.close()
    if not docs:
        return None
    df = pd.DataFrame(docs)
    df['createAt'] = pd.to_datetime(df['createAt'])
    df = df.rename(columns={'createAt': 'timestamp'})
    df = df.set_index('timestamp')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def resample_to_15min(df, cols):
    if df is None or df.empty:
        return None
    df = df[cols].resample('15min').mean()
    df = df.dropna()
    return df


def fetch_openmeteo(hours=72):
    lat, lon = 0.7, -77.6
    now = datetime.datetime.utcnow()
    start = (now - datetime.timedelta(hours=hours)).strftime('%Y-%m-%d')
    end = now.strftime('%Y-%m-%d')
    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&hourly=temperature_2m,shortwave_radiation,wind_speed_10m,pressure_msl,cloud_cover"
           f"&timezone=auto")
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df = df.rename(columns={
            'time': 'timestamp',
            'temperature_2m': 'temp_2m',
            'shortwave_radiation': 'shortwave_radiation',
            'wind_speed_10m': 'wind_speed_10m',
            'pressure_msl': 'pressure_msl',
            'cloud_cover': 'cloud_cover',
        })
        df = df.set_index('timestamp')
        return df
    except Exception as e:
        print(f"  Open-Meteo error: {e}")
        return None


def estimate_solar_generation(df):
    eficiencia = 0.18
    area_panel = 1.6
    factor_temp = 1 - 0.004 * (df['Temp'] - 25)
    factor_temp = factor_temp.clip(lower=0.8)
    df = df.copy()
    df['Generacion_Solar'] = (df['shortwave_radiation'] / 1000.0 * eficiencia *
                              area_panel * factor_temp)
    df['Generacion_Solar'] = df['Generacion_Solar'].clip(lower=0)
    return df


def merge_data(mongo_df, weather_df):
    if mongo_df is None or weather_df is None:
        return None
    merged = mongo_df.join(weather_df, how='inner')
    merged = merged.dropna()
    if 'Generacion_Solar' not in merged.columns:
        merged = estimate_solar_generation(merged)
    return merged


def prepare_input(df, feature_cols, seq_len):
    if len(df) < seq_len:
        print(f"  ERROR: Se necesitan {seq_len} horas, solo hay {len(df)}")
        return None
    last_seq = df[feature_cols].iloc[-seq_len:].values.astype(np.float32)
    return last_seq


def load_lstm_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model.keras"))
    with open(os.path.join(MODEL_DIR, "params.json")) as f:
        params = json.load(f)
    return model, params


def predict_24h(model, X_input, scaler_X, scaler_y, target_cols, output_len):
    X_norm = scaler_X.transform(X_input.reshape(-1, X_input.shape[-1])).reshape(X_input.shape)
    X_norm = np.expand_dims(X_norm, 0)
    y_pred_norm = model.predict(X_norm, verbose=0)
    y_pred_2d = y_pred_norm.reshape(-1, output_len, len(target_cols))
    dummy = np.zeros((output_len, len(target_cols)))
    for i in range(len(target_cols)):
        dummy[:, i] = y_pred_2d[0, :, i]
    y_pred_orig = scaler_y.inverse_transform(dummy)
    result = {}
    for i, col in enumerate(target_cols):
        vals = y_pred_orig[:, i]
        if col == 'Generacion_Solar':
            vals = np.clip(vals, 0, None)
        result[col] = vals
    return result


def plot_predictions(predictions, output_len):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    hours = list(range(1, output_len + 1))
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[TARGET_LABELS[c] for c in TARGET_COLS])
    for idx, col in enumerate(TARGET_COLS):
        row, col_idx = (idx // 2) + 1, (idx % 2) + 1
        fig.add_trace(go.Scatter(
            x=hours, y=predictions[col],
            mode='lines+markers',
            marker=dict(color=colors[idx], size=6),
            name=col,
            line=dict(width=2)
        ), row=row, col=col_idx)
    fig.update_layout(
        height=650, width=850,
        title='Predicción LSTM — Próximas 24 horas (96 intervalos de 15min)',
        showlegend=True,
        xaxis_title='Intervalo de 15min ahead'
    )
    return fig


def main():
    print("="*70)
    print("PASO 6: Predicción LSTM — Datos Frescos MongoDB + Open-Meteo")
    print("="*70)

    print("\n[1] Cargando modelo y parámetros...")
    model, params = load_lstm_model()
    seq_len = params['seq_len']
    output_len = params['output_len']
    print(f"    seq_len={seq_len}, output_len={output_len}, lstm_units={params['lstm_units']}")
    print(f"    Targets: {params['target_cols']}")

    print("\n[2] Obteniendo datos de MongoDB...")
    print("    - Energy Tablero 2...")
    df_energy_raw = fetch_mongo_collection(COLLECTIONS['Energy Tablero 2'], hours=seq_len+24)
    df_energy = resample_to_15min(df_energy_raw, ['VA', 'VB', 'Fre'])
    print(f"      {len(df_energy) if df_energy is not None else 0} horas disponibles")

    print("    - Temperatura PV...")
    df_temp_raw = fetch_mongo_collection(COLLECTIONS['Temperatura PV'], hours=seq_len+24)
    df_temp = resample_to_15min(df_temp_raw, ['Temp'])
    print(f"      {len(df_temp) if df_temp is not None else 0} horas disponibles")

    print("    - Open-Meteo...")
    df_weather = fetch_openmeteo(hours=seq_len+24)
    print(f"      {len(df_weather) if df_weather is not None else 0} horas disponibles")

    if df_energy is None:
        print("    ERROR: No hay datos de Energy Tablero 2")
        return
    if df_weather is None:
        print("    ERROR: No hay datos de Open-Meteo")
        return

    print("\n[3] Merging datos...")
    if df_temp is not None and len(df_temp) > 0:
        df_merged = df_energy.join(df_temp, how='outer')
    else:
        print("    Temperatura PV no disponible — usando solo datos de Energy + Weather")
        df_merged = df_energy.copy()
        df_merged['Temp'] = df_merged[['VA', 'VB']].mean(axis=1) * 0.0 + 25.0
    df_merged = df_merged.join(df_weather, how='inner')
    df_merged = df_merged.dropna()
    if 'Generacion_Solar' not in df_merged.columns:
        df_merged = estimate_solar_generation(df_merged)
    print(f"    Shape final: {df_merged.shape}")
    print(f"    Período: {df_merged.index.min()} → {df_merged.index.max()}")
    print(f"    Columnas: {df_merged.columns.tolist()}")

    if len(df_merged) < seq_len:
        print(f"\n    ERROR: Se necesitan {seq_len} horas, solo hay {len(df_merged)}")
        print("    Necesita esperar más datos de sensores...")
        return

    print("\n[4] Preparando input...")
    X_input = prepare_input(df_merged, FEATURE_COLS, seq_len)
    if X_input is None:
        return
    print(f"    Input shape: {X_input.shape}")

    print("\n[5] Cargando scalers y preparando predicción...")
    scaler_X = StandardScaler()
    scaler_X.mean_ = np.array(params['X_mean'])
    scaler_X.scale_ = np.array(params['X_scale'])
    scaler_X.n_features_in_ = len(params['feature_cols'])

    scaler_y = StandardScaler()
    scaler_y.mean_ = np.array(params['y_mean'])
    scaler_y.scale_ = np.array(params['y_scale'])
    scaler_y.n_features_in_ = params['n_targets']

    print("\n[6] Prediciendo 24h ahead...")
    predictions = predict_24h(model, X_input, scaler_X, scaler_y,
                                TARGET_COLS, output_len)

    print("\n" + "-"*50)
    print("PREDICCIONES LSTM — Próximas 24 horas")
    print("-"*50)
    for col in TARGET_COLS:
        vals = predictions[col]
        print(f"  {col:20s}: min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}")
    print("-"*50)

    print("\n[7] Generando gráficas...")
    fig = plot_predictions(predictions, output_len)

    output_html = os.path.join(SCRIPT_DIR, "prediction_output.html")
    fig.write_html(output_html)
    print(f"  Guardado: prediction_output.html")

    output_csv = os.path.join(SCRIPT_DIR, "prediction_24h.csv")
    df_pred = pd.DataFrame(predictions, index=[f"h{i+1}" for i in range(output_len)])
    df_pred.index.name = 'hora'
    df_pred.to_csv(output_csv)
    print(f"  Guardado: prediction_24h.csv")

    print("\n" + "="*70)
    print("PREDICCIÓN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    main()
