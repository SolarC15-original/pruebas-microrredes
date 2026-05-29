# Microredes — Resumen del proyecto

## Datasets anteriores (Rye Microgrid - Zenodo)
- **Rye Microgrid** (Zenodo, Noruega): `rye_generation_and_load.csv`
- ~8,266 filas horarias (2020-01-01 → 2020-12-11), 3 columnas: `Consumption`, `Solar`, `Wind`
- Autocorrelación real: Solar ρ=0.91 lag-1, ρ=0.70 lag-24 (ciclo día/noche)
- `met_data.h5` disponible con clima (temperatura, presión, radiación, viento)
- 14 NaN en Wind (ya manejados con `.dropna()`)
- URL: https://zenodo.org/records/4448894

## Estructura anterior
```
pruebas microrredes/
├── red convolucional/          — CNN 1D + TCN (Python, 20 epochs)
├── Long Short-Term Memory/     — LSTM + BiLSTM (Python, 20 epochs)
└── temporal fusion transformers/ — TFT simplificado (Python, 20 epochs)
```

## Resultados modelos anteriores (Rye dataset, 20 epochs)

| Modelo | Mejor target | MAE Z (mejor) | vs naive | R² |
|---|---|---|---|---|
| TCN | Solar | 0.172 | −70 % | — |
| **LSTM** | **Solar** | **0.123** | **−78 %** | — |
| TFT | Solar | 0.150 | −74 % | +0.69 (Wind) |

---

# TFT FINAL — Nuevo proyecto (2026-05-26)

## Contexto
Se migró del dataset Rye Microgrid (Zenodo) a **datos propios de MongoDB** de una microrred multisensor en Colombia (coordenadas aproximadas: 0.7°N, 77.6°W — zona andina). El objetivo es **predicción horaria multi-variable orientada a estimación de energía para la carga y predicción de consumo**.

## Base de datos MongoDB
```
mongodb+srv://root:SISTEMA2025qwer@clusterinteligente.qnejnxi.mongodb.net/
Database: sistema_inteligente_db
```

## Sensores disponibles (datos reales)

| Sensor | Colección MongoDB | Datos | Período | Frecuencia |
|---|---|---|---|---|
| Energy Tablero 2 | `69e50396cc47b3a12324e64f` | VA, VB, Fre | 26-abr → 26-may (30 días) | ~57s |
| Temperatura PV | `69ee990dcc47b3a12324e851` | Temp | 26-abr → 17-may (21 días) | ~65s |
| Sensor_de_potencia | `69da6927ff371e558d1a33aa` | **OMITIR** — solo 3h de prueba | — | — |
| panel01 | `69de4a392a1d2949a3c94226` | **OMITIR** — solo 3h de prueba | — | — |
| Energy Tablero Principal | `69e502d6cc47b3a12324e644` | **Sin uso** (colección vacía) | — | — |

## Relación sensores → energía

| Variable | Relación con energía |
|---|---|
| **Fre** (~60Hz) | Desviaciones de 60Hz ↔ carga de red. Proxy de consumo/demanda |
| **VA, VB** (~125V) | Estabilidad de voltaje — indica demanda vs suministro |
| **Temp panel PV** | Proxy de irradiación solar → generación PV |
| **shortwave_radiation** (Open-Meteo) | **Directo** — predicción generación solar real |

## Targets del modelo

| Target | Interpretación energética |
|---|---|
| **Fre** | Proxy de carga/consumo de red |
| **Temp** | Proxy de generación solar (irradiación) |
| **VA** | Estabilidad del sistema |
| **VB** | Estabilidad del sistema |

## Relación clima → generación energética

```
shortwave_radiation (Open-Meteo) → generación solar estimada (kWh)
wind_speed_10m → pérdidas por convección en paneles
temp_2m → eficiencia paneles PV (derate factor)
cloud_cover → atenuación solar
```

## Arquitectura TFT completa
- GRN real con gating (mantiene eje temporal, no Flatten+Dense)
- Variable Selection Network (GRN + softmax weights por feature)
- LSTM Encoder + Decoder LSTM con Multi-Head Self-Attention
- d_model=128, num_heads=8
- seq_len=48 (48 horas de historia)
- Salida multi-step: 24 horas ahead × 4 targets = 96 salidas
- Cosine decay + gradient clipping + early stopping

## Estructura TFT final
```
TFT final/
├── 01_fetch_mongodb.py     # Extraer VA, VB, Fre, Temp, resamplear a 1h
├── 02_fetch_weather.py     # Open-Meteo API para el período
├── 03_merge_dataset.py     # Merge hourly + interpolar gap + guardar microgrid_hourly.csv
├── 04_train_tft.py         # TFT completo + entrenamiento
├── 05_evaluate.py          # Gráficas MAE/MSE/R²
├── 06_predict.py           # Script de inferencia
└── kaggle_data/
    └── microgrid_hourly.csv
```

## Pipeline de datos
1. MongoDB → `createAt` (datetime BSON) → resample hourly (promedio ~60 lecturas/hora)
2. Open-Meteo → datos horarios del mismo período
3. Join por timestamp de hora exacta
4. **Interpolación lineal del gap de 9 días en Temp PV (17-may → 26-may)**

## Pending para TFT final
1. Temperatura PV se detuvo el 17-may → gap de 9 días a interpolar linealmente
2. Más datos necesarios para entrenamiento sólido (idealmente 30+ días después de que Temp PV retome)
3. Considerar instalar sensores de corriente/potencia para targets más completos (I, P, E)
4. Energy Tablero Principal (`69e502d6..`) coleccion vacía — sin datos disponibles

## Comandos recurrentes (proyectos anteriores)
```bash
# Entrenar
cd "red convolucional" && python3 ProtoPy.py
cd "Long Short-Term Memory" && python3 ProtoPy_LSTM.py
cd "temporal fusion transformers" && python3 ProtoPy_TFT.py

# Gráficas
cd "red convolucional" && python3 graficas_error.py && python3 graficas_pred_vs_real.py
cd "Long Short-Term Memory" && python3 graficas_error.py && python3 graficas_pred_vs_real.py
cd "temporal fusion transformers" && python3 graficas_error.py && python3 graficas_pred_vs_real.py
```
