"""
graficas_TFT.py — Visualización comparativa Python vs JavaScript (tfjs-node)
Métricas de inferencia, throughput y memoria para TFT.
Carga automática desde exported_model/metrics_{python,js}.json
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')

def load_metrics(path, label):
    if not os.path.exists(path):
        print(f"[WARN] No se encuentra {path}")
        return None
    with open(path) as f:
        return json.load(f)

base = os.path.dirname(__file__)
py_path = os.path.join(base, 'exported_model', 'metrics_python.json')
js_path = os.path.join(base, 'exported_model', 'metrics_js.json')

py_metrics = load_metrics(py_path, 'Python')
js_metrics = load_metrics(js_path, 'Node.js')

if py_metrics is None or js_metrics is None:
    print("ERROR: Ejecutá ProtoPy_TFT.py y ProtoJS_TFT.js primero.")
    sys.exit(1)

py_inf = py_metrics.get('inference', {})
py_model = py_metrics.get('TFT_Microgrid', {})
js_inf = js_metrics.get('inference', {})
js_model = js_metrics.get('model', {})
js_mem = js_metrics.get('memory', {})

metrics = {
    'Inferencia\nMedia (ms)':      {'Python': py_inf.get('mean_ms', 0),  'Node.js': js_inf.get('mean_ms', 0)},
    'Inferencia\nP95 (ms)':        {'Python': py_inf.get('p95_ms', 0),   'Node.js': js_inf.get('p95_ms', 0)},
    'Training\ns/epoch':           {'Python': py_model.get('time_per_epoch_s', 0), 'Node.js': js_model.get('time_per_epoch_s', 0)},
    'Throughput\n(pred/s)':        {'Python': py_inf.get('throughput_pred_s', 0),  'Node.js': js_inf.get('throughput_pred_s', 0)},
    'Memoria RSS\n(MB)':           {'Python': 0,  'Node.js': js_mem.get('rss_mb', 0)},
}

categories = list(metrics.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
inf_cats = ['Inferencia\nMedia (ms)', 'Inferencia\nP95 (ms)']
inf_py   = [metrics[c]['Python'] for c in inf_cats]
inf_node = [metrics[c]['Node.js'] for c in inf_cats]
x = np.arange(len(inf_cats))
w = 0.35
bars1 = ax.bar(x - w/2, inf_py,   w, label='Python', color='#2e86ab', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, inf_node, w, label='Node.js', color='#a23b72', edgecolor='white', linewidth=0.5)
ax.set_title('Latencia de Inferencia', fontsize=14, fontweight='bold')
ax.set_ylabel('Milisegundos (ms)')
ax.set_xticks(x)
ax.set_xticklabels(['Media', 'P95'], fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax = axes[1]
t_cats = ['Throughput\n(pred/s)']
t_py   = [metrics[c]['Python'] for c in t_cats]
t_node = [metrics[c]['Node.js'] for c in t_cats]
x = np.arange(len(t_cats))
bars1 = ax.bar(x - w/2, t_py,   w, label='Python', color='#2e86ab', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, t_node, w, label='Node.js', color='#a23b72', edgecolor='white', linewidth=0.5)
ax.set_title('Throughput de Inferencia', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicciones / segundo')
ax.set_xticks(x)
ax.set_xticklabels([''], fontsize=12)
ax.legend(fontsize=11)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax = axes[2]
extra_cats = ['Training\ns/epoch', 'Memoria RSS\n(MB)']
extra_py   = [metrics[c]['Python'] for c in extra_cats]
extra_node = [metrics[c]['Node.js'] for c in extra_cats]
x = np.arange(len(extra_cats))
bars1 = ax.bar(x - w/2, extra_py,   w, label='Python', color='#2e86ab', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + w/2, extra_node, w, label='Node.js', color='#a23b72', edgecolor='white', linewidth=0.5)
ax.set_title('Training y Memoria', fontsize=14, fontweight='bold')
ax.set_ylabel('Valor')
ax.set_xticks(x)
ax.set_xticklabels(['s/epoch', 'RSS (MB)'], fontsize=12)
ax.legend(fontsize=11)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('Benchmark TFT (Temporal Fusion Transformer) — Python vs Node.js\nMicrorredes Eléctricas — CPU (sin GPU)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('comparativa_py_vs_js.png', dpi=150, bbox_inches='tight')
print("[OK] Gráfica guardada: comparativa_py_vs_js.png")

fig2, ax2 = plt.subplots(figsize=(10, 5))

ratio_cats = ['Inferencia\nMedia', 'Inferencia\nP95', 'Training\ns/epoch', 'Throughput', 'Memoria\nRSS']

inf_mean_py = py_inf.get('mean_ms', 1)
inf_mean_js = js_inf.get('mean_ms', 1)
inf_p95_py = py_inf.get('p95_ms', 1)
inf_p95_js = js_inf.get('p95_ms', 1)
train_py = py_model.get('time_per_epoch_s', 1)
train_js = js_model.get('time_per_epoch_s', 1)
tp_py = py_inf.get('throughput_pred_s', 1)
tp_js = js_inf.get('throughput_pred_s', 1)
mem_js = js_mem.get('rss_mb', 1)

ratios = [
    inf_mean_py / inf_mean_js if inf_mean_js > 0 else 0,
    inf_p95_py / inf_p95_js if inf_p95_js > 0 else 0,
    train_py / train_js if train_js > 0 else 0,
    tp_js / tp_py if tp_py > 0 else 0,
    mem_js / 190,
]
bar_colors = ['#2ecc71' if r > 1 else '#e74c3c' for r in ratios]
labels_ratio = [
    f'{ratios[0]:.1f}x más\nrápido (JS)' if ratios[0] > 1 else f'{1/ratios[0]:.1f}x más\nlento (JS)',
    f'{ratios[1]:.1f}x más\nrápido (JS)' if ratios[1] > 1 else f'{1/ratios[1]:.1f}x más\nlento (JS)',
    f'{1/ratios[2]:.1f}x más\nlento (JS)' if ratios[2] < 1 else f'{ratios[2]:.1f}x más\nrápido (JS)',
    f'{ratios[3]:.1f}x más\n(JS)' if ratios[3] > 1 else f'{1/ratios[3]:.1f}x menos\n(JS)',
    f'{ratios[4]:.1f}x más\n(JS)' if ratios[4] > 1 else f'{1/ratios[4]:.1f}x menos\n(JS)',
]

bars = ax2.barh(ratio_cats, ratios, color=bar_colors, edgecolor='white', linewidth=0.8, height=0.6)
for bar, label in zip(bars, labels_ratio):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             label, va='center', fontsize=10, fontweight='bold')

ax2.axvline(1, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Factor de mejora (escala log)', fontsize=12)
ax2.set_title('TFT — Ratio Node.js / Python\n(>1 favorece a Node.js, <1 favorece a Python)',
              fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.set_xlim(0.1, 200)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ratio_mejora_js_vs_py.png', dpi=150, bbox_inches='tight')
print("[OK] Gráfica guardada: ratio_mejora_js_vs_py.png")
print("\nArchivos generados:")
print("  - comparativa_py_vs_js.png  (barras agrupadas)")
print("  - ratio_mejora_js_vs_py.png (factor de mejora)")
