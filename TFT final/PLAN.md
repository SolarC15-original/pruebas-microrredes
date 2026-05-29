# TFT Final — Plan de implementación

## Objetivo
Predicción horaria de **carga/consumo** y **generación de energía** usando TFT completo con datos de MongoDB + Open-Meteo.

## Sensores útiles (2)

| Sensor | Colección MongoDB | Datos | Período | Frecuencia |
|---|---|---|---|---|
| Energy Tablero 2 | `69e50396cc47b3a12324e64f` | VA, VB, Fre | 26-abr → 26-may (30 días) | ~57s |
| Temperatura PV | `69ee990dcc47b3a12324e851` | Temp | 26-abr → 17-may (21 días) | ~65s |

**Sensores omitidos:**
- Sensor_de_potencia (`69da6927..`) — solo 3h de prueba
- panel01 (`69de4a392..`) — solo 3h de prueba
- Energy Tablero Principal (`69e502d6..`) — colección vacía

## Período de entrenamiento: 20 días

- Período overlap sensores: **26-abr → 17-may (21 días)**
- Se usan **20 días** para entrenamiento (seq_len=48h + buffer)
- Sin gap de interpolación — se usa el período continuo donde ambos sensores tienen datos

## Targets (variables a predecir)

### 1. Carga/Consumo (Energy Tablero 2)
| Target | Fuente | Descripción |
|---|---|---|
| **Fre** | Energy Tablero 2 | Frecuencia ~60Hz — proxy directo de carga/demanda de red |
| **VA** | Energy Tablero 2 | Voltaje fase A — indica estabilidad y demanda |
| **VB** | Energy Tablero 2 | Voltaje fase B — indica estabilidad y demanda |

### 2. Generación de Energía (Open-Meteo + Temp PV)
| Target | Fuente | Descripción |
|---|---|---|
| **Generacion_Solar** | Open-Meteo + Temp PV | Estimación de energía solar generada (kWh) |

**Fórmula de generación solar:**
```
P_solar = shortwave_radiation * eficiencia_PV * area_panel * factor_temp
```

Donde:
- `shortwave_radiation` (W/m²) — radiación directa de Open-Meteo
- `Temp` (°C) — temperatura panel PV (ajusta eficiencia)
- Factor de temperatura: `1 - 0.004 * (Temp - 25)` (por grado sobre 25°C)
- Eficiencia y area_panel son constantes del sistema

## Relación sensores → energía

| Variable | Relación con energía |
|---|---|
| **Fre** (~60Hz) | Desviaciones ↔ carga de red. Más carga → frecuencia baja |
| **VA, VB** (~125V) | Estabilidad de voltaje — indica demanda vs suministro |
| **shortwave_radiation** | **Principal** — radiación solar incidente → generación PV |
| **Temp panel PV** | **Secundario** — eficiencia del panel (derate factor) |
| **temp_2m** (Open-Meteo) | Temperatura ambiente — afecta eficiencia PV |
| **cloud_cover** | Atenuación de radiación solar |
| **wind_speed_10m** | Enfriamiento de paneles → mejora eficiencia |

## Features de entrada (12 dimensiones)

- **Eléctricas (3):** VA, VB, Fre (lags 48h → seq_len=48)
- **Clima Open-Meteo (6):** temp_2m, shortwave_radiation, wind_speed_10m, pressure_msl, cloud_cover, estimated_solar
- **Temporales (3):** hora (sin/cos), día_semana (sin/cos), mes (sin/cos)

## Arquitectura TFT completa

- GRN real con gating (mantiene eje temporal, no Flatten+Dense)
- Variable Selection Network (GRN + softmax weights por feature)
- LSTM Encoder + Decoder LSTM con Multi-Head Self-Attention
- d_model=128, num_heads=8
- seq_len=48 (48 horas de historia)
- Salida multi-step: 24 horas ahead × 4 targets = 96 salidas
- Cosine decay + gradient clipping + early stopping

## Pipeline de datos

1. **MongoDB** → `createAt` (datetime BSON) → resample hourly (promedio ~60 lecturas/hora)
   - Extraer: VA, VB, Fre, Temp
2. **Open-Meteo** → datos horarios del mismo período
   - Extraer: temp_2m, shortwave_radiation, wind_speed_10m, pressure_msl, cloud_cover
3. **Calcular Generación Solar** → estimar P_solar por hora
4. **Join** por timestamp de hora exacta
5. **20 días continuos** (26-abr → 17-may) — sin gaps

## Estructura de archivos

```
TFT final/
├── 01_fetch_mongodb.py     # Extraer VA, VB, Fre, Temp, resamplear a 1h
├── 02_fetch_weather.py     # Open-Meteo API para el período
├── 03_merge_dataset.py    # Calcular Generacion_Solar + Merge + microgrid_hourly.csv
├── 04_train_tft.py        # TFT completo + entrenamiento (20 días, 4 targets)
├── 05_evaluate.py          # Gráficas MAE/MSE/R² por target
├── 06_predict.py          # Script de inferencia
└── kaggle_data/
    └── microgrid_hourly.csv
```

## Targets del modelo

| # | Target | Tipo | Fuente |
|---|---|---|---|
| 1 | **Fre** | Carga/Consumo | Energy Tablero 2 |
| 2 | **VA** | Carga/Consumo | Energy Tablero 2 |
| 3 | **VB** | Carga/Consumo | Energy Tablero 2 |
| 4 | **Generacion_Solar** | Generación | Open-Meteo + Temp |

## Pending

- Más datos necesarios para entrenamiento sólido (idealmente 30+ días)
- Considerar instalar sensores de corriente/potencia para targets más completos (I, P, E)
