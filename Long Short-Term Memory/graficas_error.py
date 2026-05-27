"""
graficas_error.py — Gráfica comparativa de MAE y MSE entre modelos
(python y JS) entrenados en esta carpeta.
Lee automáticamente de exported_model/metrics_{python,js}.json
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base = os.path.dirname(__file__)
folder_name = os.path.basename(base)

py_path = os.path.join(base, 'exported_model', 'metrics_python.json')
js_path = os.path.join(base, 'exported_model', 'metrics_js.json')

if not os.path.exists(py_path) or not os.path.exists(js_path):
    print("ERROR: Ejecutá los scripts de entrenamiento primero.")
    sys.exit(1)

with open(py_path) as f:
    py = json.load(f)
with open(js_path) as f:
    js = json.load(f)

models = []
mae_vals = []
mse_vals = []
colors = []

for name, data in py.items():
    if name == 'inference':
        continue
    if 'test_mae' in data and 'test_loss_mse' in data:
        label = name.replace('_Microgrid', '').replace('_', ' ')
        models.append(f"Python\n{label}")
        mae_vals.append(data['test_mae'])
        mse_vals.append(data['test_loss_mse'])
        colors.append('#2e86ab')

if 'model' in js and 'test_mae' in js['model'] and 'test_loss_mse' in js['model']:
    label = js['model']['name'].replace('_Microgrid_JS', '').replace('_', ' ')
    models.append(f"Node.js\n{label}")
    mae_vals.append(js['model']['test_mae'])
    mse_vals.append(js['model']['test_loss_mse'])
    colors.append('#a23b72')

if len(models) == 0:
    print("ERROR: No se encontraron métricas de modelos.")
    sys.exit(1)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, mae_vals, w, label='MAE (menor = mejor)',
               color='#e74c3c', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, mse_vals, w, label='MSE (menor = mejor)',
               color='#3498db', edgecolor='white', linewidth=0.5)

ax.set_ylabel('Error (unidades Z-score)')
ax.set_title(f'Métricas de Error — {folder_name.title()}\nMAE y MSE en test (normalizado)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f'{h:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
            f'{h:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out = os.path.join(base, 'error_metrics_comparativa.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"[OK] Gráfica guardada: {out}")
