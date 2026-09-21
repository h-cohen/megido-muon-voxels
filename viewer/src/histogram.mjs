export function computeHistogram(data, lo, hi, nbins) {
  const out = new Uint32Array(nbins);
  const span = hi - lo;
  if (span <= 0) return out;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v) || v < lo || v > hi) continue;
    let bin = Math.floor(((v - lo) / span) * nbins);
    if (bin >= nbins) bin = nbins - 1;
    if (bin < 0) bin = 0;
    out[bin] += 1;
  }
  return out;
}
