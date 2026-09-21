export function defaultStops() {
  return [
    { t: 0.0, r: 20, g: 20, b: 120, a: 0 },
    { t: 0.5, r: 200, g: 120, b: 20, a: 140 },
    { t: 1.0, r: 255, g: 240, b: 200, a: 255 },
  ];
}

export function buildTransferLUT(stops, size = 256) {
  const sorted = [...stops].sort((a, b) => a.t - b.t);
  const out = new Uint8Array(size * 4);
  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    let lo = sorted[0], hi = sorted[sorted.length - 1];
    for (let k = 0; k < sorted.length - 1; k++) {
      if (t >= sorted[k].t && t <= sorted[k + 1].t) {
        lo = sorted[k]; hi = sorted[k + 1];
        break;
      }
    }
    const span = hi.t - lo.t;
    const f = span > 0 ? (t - lo.t) / span : 0;
    out[i * 4 + 0] = Math.round(lo.r + (hi.r - lo.r) * f);
    out[i * 4 + 1] = Math.round(lo.g + (hi.g - lo.g) * f);
    out[i * 4 + 2] = Math.round(lo.b + (hi.b - lo.b) * f);
    out[i * 4 + 3] = Math.round(lo.a + (hi.a - lo.a) * f);
  }
  return out;
}
