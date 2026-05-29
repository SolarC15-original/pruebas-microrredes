# Plan 02: TFT con datos cada 15 minutos

## Objetivo
Migrar el pipeline de datos a resolución **15 minutos** manteniendo la arquitectura TFT completa.

## Cambios respecto a Plan anterior

| Aspecto | Antes (Plan.md) | Ahora (Plan 02) |
|---|---|---|
| Resampling MongoDB | 1 hora | **15 minutos** |
| Datos climáticos | 1 hora | **Interpolación a 15 min** |
|seq_len | 12 tokens (3h con 15min) | **96 tokens (24h con 15min)** |
| Tokens/48h historia | 48 | 192 |

**Nota**: seq_len=96 significa 96 timesteps × 15 min = 1440 min = 24 horas de historia. Más contexto temporal para capturar ciclos diarios completos.

## Pipeline de datos

### 1. MongoDB (01_fetch_mongodb.py)
- Resampling: `df.resample("15min").mean()`
- Energy Tablero 2 (~57s entre lecturas): ~16 lecturas/15min
- Temperatura PV (~65s entre lecturas): ~14 lecturas/15min
- Campos: VA, VB, Fre, Temp
- Índice: `createAt` (datetime BSON)

### 2. Open-Meteo (02_fetch_weather.py)
- API ofrece datos horarios → **interpolar linealmente a 15 min**
- Variables: temp_2m, shortwave_radiation, wind_speed_10m, pressure_msl, cloud_cover
- Coordenadas: 0.7°N, 77.6°W

### 3. Merge (03_merge_dataset.py)
- Join interno por timestamp (15 min)
- Calcular `Generacion_Solar`:
  ```
  Generacion_Solar = shortwave_radiation * 0.20 * factor_temp / 1000
  factor_temp = 1 - 0.004 * (Temp - 25), clipped [0.8, 1.0]
  ```
- Guardar `microgrid_15min.csv`

## Features y Targets

### Features entrada (10 + temporal)
- **Eléctricas (3)**: VA, VB, Fre
- **Clima (5)**: temp_2m, shortwave_radiation, wind_speed_10m, pressure_msl, cloud_cover
- **Generación Solar (1)**: Generacion_Solar
- **Temporales (3)**: hora (sin/cos), día_semana (sin/cos), mes (sin/cos)

### Targets (3) — Fre como feature de contexto
| # | Target | Fuente |
|---|---|---|
| 1 | VA | Energy Tablero 2 |
| 2 | VB | Energy Tablero 2 |
| 3 | Generacion_Solar | Open-Meteo + Temp PV |

**Nota:** Fre se mantiene como feature de entrada pero NO es target. La frecuencia de red es demasiado estable (~60Hz ±0.05Hz) y no correlaciona con ningún otro feature, making it unpredictable from our sensor set. Ahora es útil como contexto de carga de red sin ser predicho.

## Arquitectura TFT (sin cambios)

- GRN con gating
- Variable Selection Network
- LSTM Encoder + Decoder con Multi-Head Self-Attention
- d_model=128, num_heads=8
- **seq_len=96** (96 timesteps × 15 min = 24 horas)
- Output: **24 horas ahead × 3 targets = 288 salidas**
- Loss: MSE, Optimizer: Adam, Cosine decay + gradient clipping + early stopping

## Estructura archivos

```
TFT final/
├── 01_fetch_mongodb.py      # ← resample 15min
├── 02_fetch_weather.py       # ← interpolar clima a 15min
├── 03_merge_dataset.py       # ← merge + microgrid_15min.csv
├── 04_train_tft.py            # ← seq_len=12, 96 outputs cada 15min
├── 05_evaluate.py             # ← gráficas MAE/MSE/R²
├── 06_predict.py              # ← inferencia
└── kaggle_data/
    └── microgrid_15min.csv   # ← nuevo dataset
```

- Feature de contexto: Fre (~60Hz) — indica carga de red, no se predice pero ayuda al modelo

- [ ] Modificar 01_fetch_mongodb.py → resample 15min
- [ ] Modificar 02_fetch_weather.py → interpolar weather a 15min
- [ ] Modificar 03_merge_dataset.py → output microgrid_15min.csv
- [ ] Ajustar lógica de inferencia/gráficas para 15min
- [ ] Entrenar y evaluar
