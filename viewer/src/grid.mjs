import { multiply, fromTranslation, fromScaling } from './mat4.mjs';

export function modelMatrixFromMeta(meta) {
  const [nx, ny, nz] = meta.shape;
  const s = meta.spacing_m;
  const extents = [nx * s, ny * s, nz * s];
  return multiply(fromTranslation(meta.origin_m), fromScaling(extents));
}

export function worldToVoxel(p, meta) {
  const s = meta.spacing_m;
  return [
    (p[0] - meta.origin_m[0]) / s,
    (p[1] - meta.origin_m[1]) / s,
    (p[2] - meta.origin_m[2]) / s,
  ];
}

export function voxelToWorld(v, meta) {
  const s = meta.spacing_m;
  return [
    meta.origin_m[0] + v[0] * s,
    meta.origin_m[1] + v[1] * s,
    meta.origin_m[2] + v[2] * s,
  ];
}

export function sampleNearest(data, shape, i, j, k) {
  const [nx, ny, nz] = shape;
  const ii = Math.round(i), jj = Math.round(j), kk = Math.round(k);
  if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) return NaN;
  return data[ii * ny * nz + jj * nz + kk];
}
