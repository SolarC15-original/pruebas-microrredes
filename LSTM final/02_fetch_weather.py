"""
02_fetch_weather.py — Extraer datos climáticos de Open-Meteo e interpolar a 15min
Período: 26-abr-2026 → 17-may-2026
Coordenadas: 0.7°N, 77.6°W (zona andina, Colombia)
Variables: temp_2m, shortwave_radiation, wind_speed_10m, pressure_msl, cloud_cover
"""
import os
import requests
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

LAT = 0.7
LON = -77.6

DATE_FROM = "2026-04-26"
DATE_TO = "2026-05-17"

VARIABLES = [
    "temperature_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "surface_pressure",
    "cloud_cover",
]

def fetch_weather_openmeteo(lat, lon, date_from, date_to, variables):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_from,
        "end_date": date_to,
        "hourly": ",".join(variables),
        "timezone": "America/Bogota",
    }
    print(f"    URL: {url}")
    print(f"    Params: {params}")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data

def parse_weather_data(data, variables):
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise ValueError("No hourly data in response")

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    df = df.set_index("time")

    for var in variables:
        values = hourly.get(var, [])
        if len(values) == len(times):
            df[var] = values
        else:
            print(f"    [WARN] {var}: {len(values)} values vs {len(times)} times")
            df[var] = pd.Series(values).reindex(range(len(times))).values

    df = df[variables]
    df.columns = [
        "temp_2m", "shortwave_radiation", "wind_speed_10m", "pressure_msl", "cloud_cover"
    ]
    return df


def interpolate_to_15min(df):
    df_15min = df.resample("15min").asfreq()
    numeric_cols = df_15min.select_dtypes(include=["number"]).columns
    df_15min[numeric_cols] = df_15min[numeric_cols].interpolate(method="linear")
    non_numeric = df_15min.select_dtypes(exclude=["number"]).columns
    for col in non_numeric:
        df_15min[col] = df_15min[col].ffill().bfill()
    return df_15min

def main():
    print("=" * 60)
    print("PASO 2: Extraer datos de Open-Meteo")
    print("=" * 60)
    print(f"\nCoordenadas: {LAT}°N, {LON}°W")
    print(f"Período: {DATE_FROM} → {DATE_TO}")

    print(f"\n[Descargando datos de Open-Meteo Archive API...]")
    data = fetch_weather_openmeteo(LAT, LON, DATE_FROM, DATE_TO, VARIABLES)

    print(f"\n[Procesando respuesta...]")
    df = parse_weather_data(data, VARIABLES)

    print(f"\n[Datos horarios Open-Meteo]")
    print(f"    Shape: {df.shape}")
    print(f"    Rango: {df.index.min()} → {df.index.max()}")

    print(f"\n[Interpolando a 15min...]")
    df = interpolate_to_15min(df)
    print(f"    Shape 15min: {df.shape}")

    print(f"\n    Preview:")
    print(df.head(10).to_string())

    print(f"\n    Estadísticas:")
    print(df.describe().to_string())

    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, "weather_15min.csv")
    df.to_csv(csv_path)
    print(f"\nGuardado: {csv_path}")

    print("\n" + "=" * 60)
    print("ANÁLISIS:")
    print(f"- Variables climáticas: {list(df.columns)}")
    print(f"- Período: {df.index.min().date()} → {df.index.max().date()}")
    print(f"- Total registros (15min): {len(df)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
