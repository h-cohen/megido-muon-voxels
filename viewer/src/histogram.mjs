// Robust display window for a layer: [0, value at the given cumulative
// percentile], computed via a fixed-bin histogram over the data's own
// [min, max] so a single outlier voxel (real campaign: value_range[1] is
// one such outlier) never sets the window and hides the bulk distribution.
export function robustWindow(data, percentile = 0.99) {
  let dataMin = Infinity, dataMax = -Infinity, n = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isNaN(v)) continue;
    if (v < dataMin) dataMin = v;
    if (v > dataMax) dataMax = v;
    n++;
  }
  if (n === 0 || !Number.isFinite(dataMin) || !Number.isFinite(dataMax)) return [0, 1];
  if (dataMax <= 0 || dataMax === dataMin) return [dataMin, dataMax];

  const nbins = 1024;
  const hist = computeHistogram(data, dataMin, dataMax, nbins);
  const target = percentile * n;
  let cum = 0;
  for (let b = 0; b < nbins; b++) {
    cum += hist[b];
    if (cum >= target) {
      const span = dataMax - dataMin;
      const hi = dataMin + ((b + 1) / nbins) * span;
      return [0, hi];
    }
  }
  return [0, dataMax];
}

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
