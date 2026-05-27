/**
 * ProtoJS.js — Prototipo 1D-CNN en Node.js/tfjs-node
 * Carga dataset real de Kaggle + scaler params exportados por Python,
 * entrena modelo, mide latencia de inferencia (warm-up vs steady-state)
 * y reporta uso de memoria.
 */
require('./patchnode');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const perf_hooks = require('perf_hooks');

// ─────────────────────────────────────────────
// 1. CARGA DE DATASET REAL (Kaggle)
// ─────────────────────────────────────────────
function loadCSV(filePath) {
  const raw = fs.readFileSync(filePath, 'utf-8').trim().split('\n');
  const headers = raw[0].split(',');
  const rows = [];
  for (let i = 1; i < raw.length; i++) {
    if (!raw[i].trim()) continue;
    const vals = raw[i].split(',');
    const row = {};
    for (let j = 0; j < headers.length; j++) {
      const num = parseFloat(vals[j]);
      row[headers[j]] = isNaN(num) ? vals[j] : num;
    }
    rows.push(row);
  }
  return { headers, rows };
}

function createSequences(rows, featureCols, targetCols, seqLen) {
  const X = [], y = [];
  for (let i = 0; i < rows.length - seqLen; i++) {
    const seq = [];
    for (let j = i; j < i + seqLen; j++) {
      const feat = [];
      for (const col of featureCols) feat.push(rows[j][col]);
      seq.push(feat);
    }
    X.push(seq);
    const tgt = [];
    for (const col of targetCols) tgt.push(rows[i + seqLen][col]);
    y.push(tgt);
  }
  return {
    data: tf.tensor3d(X),
    targets: tf.tensor2d(y)
  };
}

function loadScalerParams(filePath) {
  const raw = fs.readFileSync(filePath, 'utf-8');
  return JSON.parse(raw);
}

// ─────────────────────────────────────────────
// 2. CONSTRUCCIÓN DE MODELO 1D-CNN (equivalente)
// ─────────────────────────────────────────────
function buildCNN1D(inputShape) {
  const inputs = tf.input({ shape: inputShape });
  let x = tf.layers.conv1d({
    filters: 64, kernelSize: 3, activation: 'relu',
    padding: 'same'
  }).apply(inputs);
  x = tf.layers.maxPooling1d({ poolSize: 2 }).apply(x);
  x = tf.layers.conv1d({
    filters: 64, kernelSize: 3, activation: 'relu',
    padding: 'same'
  }).apply(x);
  x = tf.layers.maxPooling1d({ poolSize: 2 }).apply(x);
  x = tf.layers.flatten().apply(x);
  x = tf.layers.dropout({ rate: 0.3 }).apply(x);
  x = tf.layers.dense({ units: 32, activation: 'relu' }).apply(x);
  const outputs = tf.layers.dense({ units: 3 }).apply(x);
  return tf.model({ inputs, outputs, name: 'CNN1D_Microgrid_JS' });
}

// ─────────────────────────────────────────────
// 3. ENTRENAMIENTO (limitado en JS para benchmark)
// ─────────────────────────────────────────────
async function trainJS(model, xs, ys, epochs = 5) {
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['mae'] });
  const t0 = perf_hooks.performance.now();
  const history = await model.fit(xs, ys, {
    epochs,
    batchSize: 64,
    validationSplit: 0.1,
    verbose: 0,
  });
  const trainTime = (perf_hooks.performance.now() - t0) / 1000;
  const lastLoss = history.history.loss[history.history.loss.length - 1];
  const lastMae = history.history.mae[history.history.mae.length - 1];
  return { trainTime, epochs, lastLoss, lastMae, history };
}

// ─────────────────────────────────────────────
// 4. BENCHMARK DE INFERENCIA
// ─────────────────────────────────────────────
async function benchmarkInference(model, xs, nRuns = 500) {
  const nSamples = xs.shape[0];
  const totalTime = tf.util.now(); // high-res timestamp

  // Warm-up (10 pasos)
  for (let i = 0; i < 10; i++) {
    const idx = i % nSamples;
    const input = xs.slice([idx, 0, 0], [1, xs.shape[1], xs.shape[2]]);
    model.predict(input);
    input.dispose();
    tf.dispose();
  }

  // Steady-state
  const latencies = [];
  for (let i = 0; i < nRuns; i++) {
    const idx = i % nSamples;
    const input = xs.slice([idx, 0, 0], [1, xs.shape[1], xs.shape[2]]);
    const t0 = tf.util.now();
    const output = model.predict(input);
    const elapsed = tf.util.now() - t0;
    latencies.push(elapsed);
    output.dispose();
    input.dispose();
    tf.dispose();
  }

  latencies.sort((a, b) => a - b);
  const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const std = Math.sqrt(latencies.reduce((s, v) => s + (v - mean) ** 2, 0) / latencies.length);
  const p50 = latencies[Math.floor(latencies.length * 0.5)];
  const p95 = latencies[Math.floor(latencies.length * 0.95)];
  const p99 = latencies[Math.floor(latencies.length * 0.99)];
  const minVal = latencies[0];
  const maxVal = latencies[latencies.length - 1];

  return { mean, std, p50, p95, p99, min: minVal, max: maxVal, nRuns };
}

// ─────────────────────────────────────────────
// 5. CARGA DE MODELO EXPORTADO (TFJS)
// ─────────────────────────────────────────────
async function loadExportedModel(modelDir = '../exported_model') {
  const modelJson = path.join(modelDir, 'model.json');
  if (fs.existsSync(modelJson)) {
    console.log(`[Carga] Modelo encontrado en ${modelJson}`);
    const model = await tf.loadLayersModel(`file://${modelJson}`);
    return model;
  }
  console.log('[Carga] Modelo exportado no encontrado. Usando modelo construido in-situ.');
  return null;
}

// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────
async function main() {
  console.log('='.repeat(65));
  console.log('PROTOTIPO 1D-CNN (NODE.JS/TFJS) — MICRORREDES');
  console.log('='.repeat(65));

  // ── Cargar scaler params de Python ──
  console.log('\n[0] Cargando parámetros de normalización (desde Python)...');
  const scalerParams = loadScalerParams('../exported_model/scaler_params.json');
  const SEQ_LEN = scalerParams.seq_len;
  const featureCols = scalerParams.feature_cols;
  const targetCols = scalerParams.target_cols;
  const Xmean = tf.tensor1d(scalerParams.X_mean);
  const Xstd = tf.tensor1d(scalerParams.X_scale);
  const yMean = tf.tensor1d(scalerParams.y_mean);
  const yStd = tf.tensor1d(scalerParams.y_scale);
  console.log(`    seq_len=${SEQ_LEN}, features=${featureCols.length}, targets=${targetCols.length}`);

  // ── Cargar datos ──
  console.log('\n[1] Cargando dataset real de Kaggle...');
  const csvPath = path.join(__dirname, '..', 'kaggle_data', 'dataset.csv');
  const { headers, rows } = loadCSV(csvPath);
  console.log(`    Filas: ${rows.length}, Columnas: ${headers.length}`);

  const { data, targets } = createSequences(rows, featureCols, targetCols, SEQ_LEN);
  const totalSamples = data.shape[0];

  // Partición cronológica (primero 80% train, último 20% test)
  const splitIdx = Math.floor(totalSamples * 0.8);
  const trainXsRaw = data.slice([0, 0, 0], [splitIdx, data.shape[1], data.shape[2]]);
  const trainYsRaw = targets.slice([0, 0], [splitIdx, targets.shape[1]]);
  const testXsRaw = data.slice([splitIdx, 0, 0], [totalSamples - splitIdx, data.shape[1], data.shape[2]]);
  const testYsRaw = targets.slice([splitIdx, 0], [totalSamples - splitIdx, targets.shape[1]]);

  // Normalización Z-score usando los mismos parámetros que Python
  const trainXs = trainXsRaw.sub(Xmean).div(Xstd);
  const testXs = testXsRaw.sub(Xmean).div(Xstd);
  const trainYs = trainYsRaw.sub(yMean).div(yStd);
  const testYs = testYsRaw.sub(yMean).div(yStd);

  console.log(`    Tensores: X [${data.shape}], y [${targets.shape}]`);
  console.log(`    Train: ${trainXs.shape[0]} muestras  Test: ${testXs.shape[0]} muestras`);

  // ── Cargar o construir modelo ──
  console.log('\n[2] Inicializando modelo...');
  let model = await loadExportedModel();
  let params = 0, testLoss = 0, testMae = 0;
  let trainStats = { trainTime: 0, epochs: 0, lastLoss: 0, lastMae: 0 };
  if (!model) {
    model = buildCNN1D([SEQ_LEN, featureCols.length]);
    model.summary();
    params = model.countParams();
    console.log(`    Parámetros totales: ${params.toLocaleString()}`);

    console.log(`\n[3] Entrenando modelo (20 épocas)...`);
    trainStats = await trainJS(model, trainXs, trainYs, 20);
    console.log(`    Tiempo: ${trainStats.trainTime.toFixed(2)}s total  |  ${(trainStats.trainTime / 20).toFixed(3)}s/epoch`);
    console.log(`    Loss final: ${trainStats.lastLoss.toFixed(4)}`);
    console.log(`    MAE final:  ${trainStats.lastMae.toFixed(4)}`);

    const evalResult = model.evaluate(testXs, testYs, { batchSize: 64 });
    testLoss = evalResult[0].dataSync()[0];
    testMae = evalResult[1].dataSync()[0];
    console.log(`    Test Loss:  ${testLoss.toFixed(4)}`);
    console.log(`    Test MAE:   ${testMae.toFixed(4)}`);
    evalResult.forEach(t => t.dispose());
  } else {
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['mae'] });
    console.log('    Modelo cargado desde archivo.');
  }

  // ── Benchmark de inferencia ──
  console.log(`\n[4] Benchmark de inferencia (500 predicciones)...`);
  const infStats = await benchmarkInference(model, testXs, 500);
  console.log(`    Mean:  ${infStats.mean.toFixed(3)} ms`);
  console.log(`    Std:   ${infStats.std.toFixed(3)} ms`);
  console.log(`    P50:   ${infStats.p50.toFixed(3)} ms`);
  console.log(`    P95:   ${infStats.p95.toFixed(3)} ms`);
  console.log(`    P99:   ${infStats.p99.toFixed(3)} ms`);
  console.log(`    Min:   ${infStats.min.toFixed(3)} ms`);
  console.log(`    Max:   ${infStats.max.toFixed(3)} ms`);

  // ── Memoria ──
  const memUsage = process.memoryUsage();
  console.log(`\n[5] Uso de memoria (Node.js):`);
  console.log(`    RSS:        ${(memUsage.rss / 1024 / 1024).toFixed(1)} MB`);
  console.log(`    Heap Total: ${(memUsage.heapTotal / 1024 / 1024).toFixed(1)} MB`);
  console.log(`    Heap Used:  ${(memUsage.heapUsed / 1024 / 1024).toFixed(1)} MB`);
  console.log(`    External:   ${(memUsage.external / 1024 / 1024).toFixed(1)} MB`);

  // ── Exportar métricas a JSON ──
  const metricsJs = {
    model: {
      name: model.name,
      params,
      train_time_s: trainStats.trainTime,
      time_per_epoch_s: trainStats.epochs > 0 ? trainStats.trainTime / trainStats.epochs : 0,
      test_loss_mse: testLoss,
      test_mae: testMae,
    },
    inference: {
      mean_ms: infStats.mean,
      std_ms: infStats.std,
      p50_ms: infStats.p50,
      p95_ms: infStats.p95,
      p99_ms: infStats.p99,
      throughput_pred_s: 1000 / infStats.mean,
    },
    memory: {
      rss_mb: memUsage.rss / 1024 / 1024,
      heap_total_mb: memUsage.heapTotal / 1024 / 1024,
      heap_used_mb: memUsage.heapUsed / 1024 / 1024,
    },
  };
  const metricsPath = path.join(__dirname, '..', 'exported_model', 'metrics_js.json');
  fs.writeFileSync(metricsPath, JSON.stringify(metricsJs, null, 2));
  console.log(`\n    Métricas exportadas a exported_model/metrics_js.json`);

  // ── Cargar métricas de Python para comparación ──
  let pyInfMean = 'N/A', pyInfP95 = 'N/A', pyTrainEpoch = 'N/A', pyThroughput = 'N/A', pyMemRss = 'N/A';
  const pyMetricsPath = path.join(__dirname, '..', 'exported_model', 'metrics_python.json');
  if (fs.existsSync(pyMetricsPath)) {
    try {
      const pyMetrics = JSON.parse(fs.readFileSync(pyMetricsPath, 'utf-8'));
      if (pyMetrics.inference) {
        pyInfMean = pyMetrics.inference.mean_ms.toFixed(2);
        pyInfP95 = pyMetrics.inference.p95_ms.toFixed(2);
        pyThroughput = (pyMetrics.inference.throughput_pred_s || (1000 / pyMetrics.inference.mean_ms)).toFixed(0);
      }
      if (pyMetrics.TCN_Microgrid) {
        pyTrainEpoch = pyMetrics.TCN_Microgrid.time_per_epoch_s.toFixed(3);
      }
    } catch (e) {}
  }

  // ── Resumen ──
  console.log('\n' + '='.repeat(65));
  console.log('RESUMEN — INFERENCIA EN NODE.JS (tfjs-node nativo)');
  console.log('='.repeat(65));
  console.log(`  Dataset real: ${rows.length} filas, ${featureCols.length} features`);
  console.log(`  Tiempo medio:       ${infStats.mean.toFixed(3)} ms`);
  console.log(`  P95:                ${infStats.p95.toFixed(3)} ms`);
  console.log(`  Throughput estimado: ${(1000 / infStats.mean).toFixed(0)} pred/s`);
  console.log(`  Memoria (RSS):       ${(memUsage.rss / 1024 / 1024).toFixed(1)} MB`);
  console.log(`  Memoria (Heap):      ${(memUsage.heapUsed / 1024 / 1024).toFixed(1)} MB`);

  // ── Tabla comparativa Python vs JS ──
  console.log('\n' + '='.repeat(65));
  console.log('COMPARATIVA PYTHON vs JAVASCRIPT (mismo dataset real)');
  console.log('='.repeat(65));
  console.log('  Métrica               Python (TF/Keras)   Node.js (tfjs-node)');
  console.log('  ' + '-'.repeat(55));
  console.log(`  Training s/epoch      ${pyTrainEpoch.padEnd(18)}           ${(metricsJs.model.time_per_epoch_s).toFixed(3).padEnd(18)}s`);
  console.log(`  Inferencia mean       ${pyInfMean.padEnd(18)}           ${infStats.mean.toFixed(2).padEnd(18)} ms`);
  console.log(`  Inferencia P95        ${pyInfP95.padEnd(18)}           ${infStats.p95.toFixed(2).padEnd(18)} ms`);
  console.log(`  Throughput            ${pyThroughput.padEnd(18)}           ${(1000/infStats.mean).toFixed(0).padEnd(18)} pred/s`);
  console.log(`  Memoria (RSS)         ${pyMemRss.padEnd(18)}           ${(memUsage.rss/1024/1024).toFixed(0).padEnd(14)} MB`);
  console.log('  ' + '-'.repeat(55));
  console.log('  *Ejecutar ProtoPy.py primero para generar scaler_params.json + metrics_python.json');
  console.log(`  *Dataset: ${rows.length} filas, ${featureCols.length} features, seq_len=${SEQ_LEN}`);

  // ── Liberar memoria TF ──
  model.dispose();
  data.dispose();
  targets.dispose();
  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  Xmean.dispose();
  Xstd.dispose();
  yMean.dispose();
  yStd.dispose();
  tf.dispose();

  console.log('='.repeat(65));
}

main().catch(console.error);
