"""
01_fetch_mongodb.py — Extraer datos de MongoDB y resamplear a 15min
Sensores: Energy Tablero 2 (VA, VB, Fre) y Temperatura PV (Temp)
Período: 26-abr-2026 → 17-may-2026 (20 días continuos)
"""
import os
from pymongo import MongoClient
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

MONGO_URI = "mongodb+srv://root:SISTEMA2025qwer@clusterinteligente.qnejnxi.mongodb.net/"
DATABASE = "sistema_inteligente_db"

SENSORS = {
    "energy_tablero_2": {
        "collection_id": "69e50396cc47b3a12324e64f",
        "fields": ["VA", "VB", "Fre"],
    },
    "temperatura_pv": {
        "collection_id": "69ee990dcc47b3a12324e851",
        "fields": ["Temp"],
    },
}

DATE_FROM = pd.Timestamp("2026-04-26T00:00:00Z")
DATE_TO = pd.Timestamp("2026-05-17T23:59:59Z")

def fetch_sensor(collection_id, fields):
    client = MongoClient(MONGO_URI)
    db = client[DATABASE]
    col = db[collection_id]

    query = {"createAt": {"$gte": DATE_FROM, "$lte": DATE_TO}}
    projection = {f: 1 for f in fields}
    projection["createAt"] = 1

    docs = list(col.find(query, projection).sort("createAt", 1))
    client.close()

    if not docs:
        print(f"    [WARN] Colección {collection_id}: 0 documentos")
        return None

    df = pd.DataFrame(docs)
    df["createAt"] = pd.to_datetime(df["createAt"])
    df = df.set_index("createAt")
    df = df[fields]
    print(f"    {collection_id}: {len(df)} registros crudos")
    return df

def resample_to_15min(df):
    if df is None:
        return None
    df_resampled = df.resample("15min").mean()
    return df_resampled

def main():
    print("=" * 60)
    print("PASO 1: Extraer datos de MongoDB")
    print("=" * 60)
    print(f"\nPeríodo: {DATE_FROM.date()} → {DATE_TO.date()}")

    sensor_data = {}
    for sensor_name, config in SENSORS.items():
        print(f"\n[{sensor_name}]")
        df = fetch_sensor(config["collection_id"], config["fields"])
        if df is not None:
            df_15min = resample_to_15min(df)
            sensor_data[sensor_name] = df_15min
            print(f"    → {len(df_15min)} registros de 15min después de resamplear")

    print("\n[Datos crudos 15min]")
    for name, df in sensor_data.items():
        print(f"    {name}: {df.shape}")
        print(f"    Rango: {df.index.min()} → {df.index.max()}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    for name, df in sensor_data.items():
        csv_path = os.path.join(output_dir, f"{name}_15min.csv")
        df.to_csv(csv_path)
        print(f"\nGuardado: {csv_path}")

    print("\n" + "=" * 60)
    print("ANÁLISIS:")
    print(f"- Energy Tablero 2: {len(sensor_data.get('energy_tablero_2', []))} registros 15min")
    print(f"- Temperatura PV: {len(sensor_data.get('temperatura_pv', []))} registros 15min")
    print(f"- Período overlap: {DATE_FROM.date()} → {DATE_TO.date()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
