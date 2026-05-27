import pymongo
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

uri = 'mongodb+srv://root:SISTEMA2025qwer@clusterinteligente.qnejnxi.mongodb.net/'
client = pymongo.MongoClient(uri)
db = client['sistema_inteligente_db']

SENSORS = {
    '69e50396cc47b3a12324e64f': 'Energy Tablero 2',
    '69ee990dcc47b3a12324e851': 'Temperatura PV',
    '69de4a392a1d2949a3c94226': 'panel01',
    '69da6927ff371e558d1a33aa': 'Sensor_de_potencia',
}

EXCLUDE = {'_id', 'createAt', 'timestamp', 'Time'}

for coll_id, sensor_name in SENSORS.items():
    print(f"\n=== {sensor_name} ===")
    col = db[coll_id]
    total = col.count_documents({})
    print(f"  Total docs: {total}")

    sample = col.find_one()
    if not sample:
        continue

    fields = [k for k in sample.keys() if k not in EXCLUDE]
    fields = [f for f in fields if f not in ('createAt', 'timestamp', 'Time')]

    # Determine sort field and limit
    sort_field = 'createAt' if 'createAt' in sample else ('timestamp' if 'timestamp' in sample else 'Time')
    limit = 10000 if sensor_name in ('Energy Tablero 2', 'Temperatura PV') else 5000

    cursor = col.find({}, {f: 1 for f in fields} | {sort_field: 1}).sort(sort_field, 1).limit(limit)
    docs = list(cursor)
    print(f"  Cargados: {len(docs)}")
    if not docs:
        continue

    df = pd.DataFrame(docs)

    if 'createAt' in df.columns:
        df['datetime'] = pd.to_datetime(df['createAt'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = pd.to_datetime(df['Time'])
    df = df.set_index('datetime').sort_index()
    df = df[fields]

    # Drop constant columns
    varying = [c for c in df.columns if df[c].nunique() > 1]
    if not varying:
        print("  (sin variables con variación)")
        continue

    print(f"  Variables: {varying}")
    print(f"  Rango: {df.index.min()} → {df.index.max()}")

    df = df[varying]

    # Create grouped plot: subplots compartiendo eje X
    n = len(varying)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'{v}' for v in varying],
        vertical_spacing=0.04
    )

    for i, var in enumerate(varying):
        unit = ''
        if var in ('VA', 'VB', 'VC', 'v'):
            unit = 'V'
        elif var in ('IA', 'IB', 'IC', 'c', 'c2'):
            unit = 'A'
        elif var in ('PA', 'PB', 'PC', 'PT', 'p'):
            unit = 'W'
        elif var in ('QA', 'QB', 'QC', 'QT'):
            unit = 'VAR'
        elif var in ('SA', 'SB', 'SC', 'ST'):
            unit = 'VA'
        elif var in ('Fre', 'f'):
            unit = 'Hz'
        elif var in ('Temp',):
            unit = '°C'
        elif var in ('e',):
            unit = 'kWh'
        elif var in ('pf', 'FPA', 'FPB', 'FPC', 'FPT', 'FP'):
            unit = ''

        y_label = f'{var} ({unit})' if unit else var
        fig.add_trace(
            go.Scatter(x=df.index, y=df[var], mode='lines', name=var,
                       line=dict(width=1), hovertemplate=f'{var}: %{{y:.4f}}<br>%{{x|%b %d %H:%M}}<extra></extra>'),
            row=i+1, col=1
        )
        fig.update_yaxes(title_text=y_label, row=i+1, col=1)

    fig.update_layout(
        title=f'{sensor_name} — {len(df)} registros ({df.index.min().strftime("%b %d")} → {df.index.max().strftime("%b %d")})',
        template='plotly_white',
        hovermode='x unified',
        height=220 * n,
        margin=dict(l=70, r=30, t=60, b=60),
        showlegend=False,
    )

    safe_name = sensor_name.lower().replace(' ', '_')
    filename = f'plotly_{safe_name}.html'
    fig.write_html(f'/home/solarc/Desktop/pruebas microrredes/{filename}')
    print(f"  ✓ {filename}")

print(f"\n✅ {len(SENSORS)} gráficas generadas (una por sensor)")
