// Hillside surface mesh: turns the fitted height field H(gx, gy) (from the
// `hillside` CLI's hill_surface.npy/.json artifacts) into a triangle mesh for
// the viewer's marker/line GL program (extended with a triangle draw path).
// Pure geometry only - no GL here, mirroring markers.mjs.

// surfaceMesh(H, gx, gy) -> { positions: Float32Array, indices: Uint32Array }
// H: flat row-major array, length gx.length * gy.length, H[i * ny + j] is the
// surface height (convention metres) at world (gx[i], gy[j]); NaN off-coverage.
// One vertex is emitted per grid node, at world (gx[i], gy[j], H[i*ny+j]),
// regardless of whether it is finite. A cell (the quad spanning nodes
// (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1)) contributes two triangles to `indices`
// ONLY if all four of its corner heights are finite; a NaN corner is never
// referenced by any triangle, so off-coverage regions become holes.
export function surfaceMesh(H, gx, gy) {
  const nx = gx.length;
  const ny = gy.length;
  const positions = new Float32Array(nx * ny * 3);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const idx = i * ny + j;
      positions[idx * 3] = gx[i];
      positions[idx * 3 + 1] = gy[j];
      positions[idx * 3 + 2] = H[idx];
    }
  }

  const indices = [];
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const i00 = i * ny + j;
      const i10 = (i + 1) * ny + j;
      const i01 = i * ny + (j + 1);
      const i11 = (i + 1) * ny + (j + 1);
      const h00 = H[i00], h10 = H[i10], h01 = H[i01], h11 = H[i11];
      if (!Number.isFinite(h00) || !Number.isFinite(h10) ||
          !Number.isFinite(h01) || !Number.isFinite(h11)) {
        continue;
      }
      // Two triangles covering the quad (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1).
      indices.push(i00, i10, i01);
      indices.push(i10, i11, i01);
    }
  }

  return { positions, indices: Uint32Array.from(indices) };
}

// smoothHeightfield(H, nx, ny, iterations) -> Float32Array
// DISPLAY-ONLY de-jitter for a fitted height field: NaN-aware iterative
// neighbour averaging. Adds no information and changes no fit - it only
// makes the rendered mesh look less jumpy. Each pass replaces every FINITE
// cell with the mean of itself and its finite 4-neighbours (up/down/left/
// right, flat index i*ny+j); a missing (off-grid or NaN) neighbour is simply
// excluded from that mean, never treated as zero. NaN cells are never
// touched - they stay NaN and are never averaged into a finite neighbour's
// mean, so holes in the surface never get smoothed shut. Pure: returns a
// NEW array; `H` is never mutated. iterations=0 returns an unchanged copy.
export function smoothHeightfield(H, nx, ny, iterations) {
  let current = Float32Array.from(H);
  for (let pass = 0; pass < iterations; pass++) {
    const next = new Float32Array(current.length);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const idx = i * ny + j;
        const center = current[idx];
        if (!Number.isFinite(center)) {
          next[idx] = NaN;
          continue;
        }
        let sum = center;
        let count = 1;
        if (i > 0) {
          const v = current[(i - 1) * ny + j];
          if (Number.isFinite(v)) { sum += v; count++; }
        }
        if (i < nx - 1) {
          const v = current[(i + 1) * ny + j];
          if (Number.isFinite(v)) { sum += v; count++; }
        }
        if (j > 0) {
          const v = current[i * ny + (j - 1)];
          if (Number.isFinite(v)) { sum += v; count++; }
        }
        if (j < ny - 1) {
          const v = current[i * ny + (j + 1)];
          if (Number.isFinite(v)) { sum += v; count++; }
        }
        next[idx] = sum / count;
      }
    }
    current = next;
  }
  return current;
}

// Sequential single-hue ramp for the surface's posterior sigma:
// warm (confident) -> deep burnt orange (uncertain). Readable on both themes.
export const SIGMA_RAMP = [[0.96, 0.72, 0.36], [0.45, 0.16, 0.03]];

// surfaceVertexColors(sigma, lo, hi) -> Float32Array, length 3*sigma.length.
// Maps each sigma value onto SIGMA_RAMP by t=(sigma-lo)/(hi-lo), clamped to
// [0,1]. A non-finite sigma (NaN off-coverage) gets the ramp midpoint rather
// than either end, since "unknown" is neither confident nor uncertain.
// hi<=lo (a degenerate/flat sigma field) is treated as a unit span so every
// vertex still gets a defined (midpoint-ish) colour instead of dividing by
// zero.
export function surfaceVertexColors(sigma, lo, hi) {
  const out = new Float32Array(sigma.length * 3);
  const [a, b] = SIGMA_RAMP;
  const span = hi > lo ? hi - lo : 1;
  for (let k = 0; k < sigma.length; k++) {
    const s = sigma[k];
    let t = Number.isFinite(s) ? (s - lo) / span : 0.5;
    t = Math.min(1, Math.max(0, t));
    for (let c = 0; c < 3; c++) out[k * 3 + c] = a[c] + (b[c] - a[c]) * t;
  }
  return out;
}

// robustRange(values, pLo=0.05, pHi=0.95) -> [lo, hi]: linear-interpolated
// percentiles over the FINITE values only (NaN off-coverage cells are
// excluded, not treated as zero). Returns [NaN, NaN] when nothing is finite,
// so callers can detect "no usable sigma" rather than silently getting a
// degenerate [0, 0] range.
export function robustRange(values, pLo = 0.05, pHi = 0.95) {
  const v = Array.from(values).filter(Number.isFinite).sort((x, y) => x - y);
  if (!v.length) return [NaN, NaN];
  const q = (p) => {
    const pos = p * (v.length - 1), i = Math.floor(pos), f = pos - i;
    return i + 1 < v.length ? v[i] + (v[i + 1] - v[i]) * f : v[i];
  };
  return [q(pLo), q(pHi)];
}
