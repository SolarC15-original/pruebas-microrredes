# Microredes — Resumen del proyecto

## Dataset
- **Rye Microgrid** (Zenodo, Noruega): `rye_generation_and_load.csv`
- ~8,266 filas horarias (2020-01-01 → 2020-12-11), 3 columnas: `Consumption`, `Solar`, `Wind`
- Autocorrelación real: Solar ρ=0.91 lag-1, ρ=0.70 lag-24 (ciclo día/noche)
- `met_data.h5` disponible con clima (temperatura, presión, radiación, viento)
- 14 NaN en Wind (ya manejados con `.dropna()`)
- URL: https://zenodo.org/records/4448894

## Estructura
```
pruebas microrredes/
├── red convolucional/
│   ├── ProtoPy.py              — CNN 1D + TCN (Python, 20 epochs)
│   ├── node_project/ProtoJS.js  — CNN 1D (Node.js/tfjs-node)
│   ├── graficas.py              — comparativa_py_vs_js.png
│   ├── graficas_error.py        — error_metrics_comparativa.png (MAE/MSE bar)
│   ├── graficas_pred_vs_real.py — pred_vs_real.png (3 targets, 150 muestras)
│   ├── kaggle_data/dataset.csv
│   └── exported_model/
│       ├── model.keras          — TCN_Microgrid
│       ├── scaler_params.json    — seq_len, feature_cols, target_cols, X_mean, X_scale, y_mean, y_scale
│       ├── metrics_python.json
│       └── metrics_js.json
├── Long Short-Term Memory/
│   ├── ProtoPy_LSTM.py          — LSTM + BiLSTM (Python, 20 epochs)
│   ├── node_project/ProtoJS_LSTM.js
│   ├── graficas_LSTM.py
│   ├── graficas_error.py
│   ├── graficas_pred_vs_real.py
│   ├── kaggle_data/dataset.csv
│   └── exported_model/ (igual que CNN, BiLSTM_Microgrid)
└── temporal fusion transformers/
    ├── ProtoPy_TFT.py           — TFT (Python, 20 epochs)
    ├── node_project/ProtoJS_TFT.js
    ├── graficas_TFT.py
    ├── graficas_error.py
    ├── graficas_pred_vs_real.py
    ├── kaggle_data/dataset.csv
    └── exported_model/ (TFT_Microgrid, con Lambda layer)
```

## Columnas (feature_cols == target_cols)
```python
feature_cols = ['Consumption', 'Solar', 'Wind']
target_cols  = ['Consumption', 'Solar', 'Wind']
```
Las 3 son autoregresivas (se usan como features en los 24 pasos anteriores para predecir el próximo paso).

## Pipeline estándar
1. Cargar CSV con `load_rye_dataset()` del proto local (`kaggle_data/dataset.csv`)
2. `create_sequences(data, feature_cols, target_cols, seq_len=24)` → X (N, 24, 3), y (N, 3)
3. Split cronológico 80/20
4. Normalización Z-score fit en train, transform en test
5. Exportar `scaler_params.json` con medias/std de X e y
6. Entrenar, guardar `model.keras`, exportar `metrics_python.json`

## Resultados (20 epochs, Rye dataset)

| Modelo | Mejor target | MAE Z (mejor) | vs naive | R² |
|---|---|---|---|---|
| TCN | Solar | 0.172 | −70 % | — |
| **LSTM** | **Solar** | **0.123** | **−78 %** | — |
| TFT | Solar | 0.150 | −74 % | +0.69 (Wind) |

## Problemas conocidos
- `tfjs.converters.save_keras_model` falla por incompatibilidad de protobuf (gencode 6.31.1 vs runtime 5.29.6). Workaround: guardar `model.keras` y convertir manualmente.
- TFT guardado con `Lambda` layer → al cargar requiere `safe_mode=False` y `custom_objects`. El script `graficas_pred_vs_real.py` de TFT reconstruye el modelo desde cero y carga solo los pesos.
- JS LSTM/TFT usan `@tensorflow/tfjs-node 4.23.0-rc.0` — no tiene `multiHeadAttention`.

## Mejoras pendientes para TFT
1. GRN real (no Flatten + Dense) manteniendo eje temporal
2. Decoder LSTM + attention (no GlobalAveragePooling1D)
3. GRN con gating real vs sigmoid simple
4. Cosine decay + gradient clipping + early stopping
5. seq_len=48, d_model=128, num_heads=8
6. Agregar features meteorológicos de `met_data.h5`
7. Lags 24h/48h como features adicionales
8. Salida por cuantiles (p10/p50/p90)

## Comandos recurrentes
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
