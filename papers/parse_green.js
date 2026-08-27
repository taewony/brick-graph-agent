const fs = require('fs');
const file = process.argv[2];
const lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);

const metrics = [
  'decode_step_p50_ms', 'decode_step_p95_ms', 'decode_step_p99_ms', 'decode_step_max_ms',
  'decode_gap_p50_ms', 'decode_gap_p95_ms', 'decode_gap_p99_ms', 'decode_gap_max_ms',
  'throughput'
];

const rows = lines.map(l => JSON.parse(l));
console.log('total runs:', rows.length);

// per-run delta for key metrics
for (const r of rows) {
  const b = r.baseline, g = r.green, d = r.delta_pct;
  console.log(
    `run ${r.run_index}: ` +
    `green_enabled=${g.green_enabled} api=${g.green_api_type} ` +
    `gap_p99: b=${b.decode_gap_p99_ms.toFixed(1)} g=${g.decode_gap_p99_ms.toFixed(1)} (${d.decode_gap_p99_ms.toFixed(2)}%) | ` +
    `step_p99: b=${b.decode_step_p99_ms.toFixed(1)} g=${g.decode_step_p99_ms.toFixed(1)} (${d.decode_step_p99_ms.toFixed(2)}%) | ` +
    `tp: b=${b.throughput.toFixed(0)} g=${g.throughput.toFixed(0)} (${d.throughput.toFixed(2)}%)`
  );
}

// summary stats
for (const m of metrics) {
  const bs = rows.map(r => r.baseline[m]);
  const gs = rows.map(r => r.green[m]);
  const ds = rows.map(r => r.delta_pct[m]);
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const lowerIsBetter = m.startsWith('decode');
  const improved = ds.filter(d => lowerIsBetter ? d < 0 : d > 0).length;
  console.log(
    `${m}: baseline_mean=${mean(bs).toFixed(2)} green_mean=${mean(gs).toFixed(2)} ` +
    `delta_mean=${mean(ds).toFixed(2)}% delta_median=${med(ds).toFixed(2)}% improved=${improved}/${rows.length}`
  );
}
