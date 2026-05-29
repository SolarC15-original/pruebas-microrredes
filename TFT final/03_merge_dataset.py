"""
03_merge_dataset.py — Merge de datos MongoDB + Open-Meteo (15min)
- Combina Energy Tablero 2 + Temperatura PV + Clima
- Calcula Generacion_Solar
- Período: 26-abr → 17-may (20 días)
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"    {filename}: {df.shape} | {df.index.min()} → {df.index.max()}")
    return df

def calculate_solar_generation(df, temp_col='Temp'):
    """
    Calcula generación solar estimada (kWh) por hora.
    Fórmula simplificada:
    P_solar = GHI * A * n * n_temp
    Donde:
    - GHI = shortwave_radiation (W/m²)
    - A = área panel (asumimos 1 m² normalizado)
    - n = eficiencia panel (20%)
    - n_temp = factor temperatura (derate por calor)
    """
    GHI = df['shortwave_radiation'].values
    temp = df[temp_col].values if temp_col in df.columns else df['temp_2m'].values

    eficiencia_panel = 0.20
    factor_temp = 1 - 0.004 * (temp - 25)
    factor_temp = np.clip(factor_temp, 0.8, 1.0)

    generacion = GHI * eficiencia_panel * factor_temp / 1000
    df['Generacion_Solar'] = generacion
    return df

def main():
    print("=" * 60)
    print("PASO 3: Merge de datasets + Generación Solar")
    print("=" * 60)

    print("\n[Cargando CSVs...]")
    df_energy = load_csv("energy_tablero_2_15min.csv")
    df_temp = load_csv("temperatura_pv_15min.csv")
    df_weather = load_csv("weather_15min.csv")

    print("\n[Merge Energy + Temperatura PV...]")
    df_merged = df_energy.join(df_temp, how='inner')
    print(f"    After join with Temp: {df_merged.shape}")

    print("\n[Merge con Weather (Open-Meteo)...]")
    df_merged = df_merged.join(df_weather, how='inner')
    print(f"    After join with Weather: {df_merged.shape}")

    print("\n[Eliminando NaN...]")
    before = len(df_merged)
    df_merged = df_merged.dropna()
    print(f"    Eliminados: {before - len(df_merged)} | Restantes: {len(df_merged)}")
    print(f"    Rango final: {df_merged.index.min()} → {df_merged.index.max()}")

    print("\n[Calculando Generacion_Solar...]")
    df_merged = calculate_solar_generation(df_merged)
    print(f"    Generacion_Solar: min={df_merged['Generacion_Solar'].min():.4f}, "
          f"max={df_merged['Generacion_Solar'].max():.4f}, "
          f"mean={df_merged['Generacion_Solar'].mean():.4f} kWh")

    print("\n[Features y Targets finales]")
    feature_cols = ['VA', 'VB', 'Fre', 'Temp', 'temp_2m', 'shortwave_radiation',
                    'wind_speed_10m', 'pressure_msl', 'cloud_cover', 'Generacion_Solar']
    target_cols = ['Fre', 'VA', 'VB', 'Generacion_Solar']

    print(f"    Features ({len(feature_cols)}): {feature_cols}")
    print(f"    Targets ({len(target_cols)}): {target_cols}")

    print(f"\n[Dataset final]")
    print(df_merged[feature_cols + target_cols].head(15).to_string())

    print(f"\n[Estadísticas]")
    print(df_merged[target_cols].describe().to_string())

    output_path = os.path.join(SCRIPT_DIR, "kaggle_data", "microgrid_15min.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path)
    print(f"\nGuardado: {output_path}")
    print(f"Shape final: {df_merged.shape}")

    print("\n" + "=" * 60)
    print("ANÁLISIS:")
    print(f"- Período: {df_merged.index.min().date()} → {df_merged.index.max().date()}")
    print(f"- Registros totales (15min): {len(df_merged)}")
    print(f"- Features: {len(feature_cols)}")
    print(f"- Targets: {target_cols}")
    print("=" * 60)

if __name__ == "__main__":
    main()
