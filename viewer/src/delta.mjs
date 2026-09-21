export function computeDelta(a, b) {
  if (a.length !== b.length) {
    throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = b[i] - a[i];
  return out;
}

// Mirrors megido/volexport.py's compare_volumes verdict thresholds exactly.
export function deltaVerdict(rms, scale) {
  if (rms < 1e-3 * scale) return 'unchanged';
  if (rms < 1e-1 * scale) return 'shifted slightly';
  return 'shifted substantially';
}
