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

// Nearest-voxel lookup for a fractional voxel-space coordinate (as returned
// by worldToVoxel). Voxel k occupies the half-open box [k, k+1) in this
// coordinate (center at k+0.5) -- the SAME convention GL's NEAREST texture
// filtering uses for a texture coordinate scaled by the axis size (texel
// index = floor(coord)). Uses Math.floor, not Math.round: rounding to the
// nearest INTEGER (rather than the nearest voxel BOX) is off by up to half a
// voxel and disagrees with which voxel the GPU actually sampled -- exactly
// the kind of drift hover picking must not have from the renderer.
export function sampleNearest(data, shape, i, j, k) {
  const [nx, ny, nz] = shape;
  const ii = Math.floor(i), jj = Math.floor(j), kk = Math.floor(k);
  if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) return NaN;
  return data[ii * ny * nz + jj * nz + kk];
}

// Slab intersection mirroring the shader's box march (FRAGMENT_SRC in
// app.mjs): same sign-preserving 1e-8 guard on a zero/near-zero direction
// component (so 1/dir never divides by exact zero), tEnter clamped at 0, and
// null when the ray misses the box (tExit <= tEnter). castHoverRay uses this
// so CPU-side hover picking marches the same box the GPU shader does.
export function rayBox(origin, dir, min, max) {
  const invDir = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const d = dir[a];
    const safe = Math.abs(d) < 1e-8 ? (d >= 0 ? 1e-8 : -1e-8) : d;
    invDir[a] = 1 / safe;
  }
  let tEnter = -Infinity, tExit = Infinity;
  for (let a = 0; a < 3; a++) {
    const t0 = (min[a] - origin[a]) * invDir[a];
    const t1 = (max[a] - origin[a]) * invDir[a];
    tEnter = Math.max(tEnter, Math.min(t0, t1));
    tExit = Math.min(tExit, Math.max(t0, t1));
  }
  tEnter = Math.max(tEnter, 0);
  if (tExit <= tEnter) return null;
  return [tEnter, tExit];
}

// numpy C-order for shape [nx,ny,nz] is z-fastest: data[x*ny*nz + y*nz + z].
// WebGL's texImage3D reads its buffer x-fastest: texel (x,y,z) at
// buf[z*ny*nx + y*nx + x]. Reorder into a new buffer for the GPU upload only;
// `data` itself (and anything else reading it, e.g. sampleNearest) stays
// numpy-order.
export function reorderForTexture(data, shape) {
  const [nx, ny, nz] = shape;
  const out = new Float32Array(nx * ny * nz);
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        out[z * ny * nx + y * nx + x] = data[x * ny * nz + y * nz + z];
      }
    }
  }
  return out;
}

// Exact voxel-to-voxel traversal (Amanatides & Woo 1987 DDA) for the "Voxel
// cubes" render mode. Enters the box via rayBox (same slab test and 1e-8
// guard as the shader), then steps one voxel face at a time, calling
// visit(i, j, k, axis) for every voxel in order -- `axis` is the face the
// ray entered through (0 x, 1 y, 2 z; -1 when the ray starts inside the
// box). Returns [i, j, k, axis] for the first voxel visit() accepts, or null.
// FRAGMENT_SRC's cube branch runs this same algorithm; keep them in step.
export function voxelMarch(origin, dir, meta, visit) {
  const [nx, ny, nz] = meta.shape;
  const n = [nx, ny, nz];
  const s = meta.spacing_m;
  const o = meta.origin_m;
  const min = o;
  const max = [o[0] + nx * s, o[1] + ny * s, o[2] + nz * s];
  const hit = rayBox(origin, dir, min, max);
  if (!hit) return null;
  const [tEnter, tExit] = hit;

  const safe = dir.map((d) => (Math.abs(d) < 1e-8 ? (d >= 0 ? 1e-8 : -1e-8) : d));
  // Entry face: the axis whose slab entry time is the latest (the face hit).
  let axis = -1;
  if (tEnter > 0) {
    let best = -Infinity;
    for (let a = 0; a < 3; a++) {
      const t0 = (min[a] - origin[a]) / safe[a], t1 = (max[a] - origin[a]) / safe[a];
      const te = Math.min(t0, t1);
      if (te > best) { best = te; axis = a; }
    }
  }
  const p = [0, 1, 2].map((a) => origin[a] + dir[a] * tEnter);
  const idx = [0, 1, 2].map((a) => Math.min(n[a] - 1, Math.max(0, Math.floor((p[a] - o[a]) / s))));
  const step = safe.map((d) => (d > 0 ? 1 : -1));
  const tDelta = safe.map((d) => Math.abs(s / d));
  const tMax = [0, 1, 2].map((a) => {
    const boundary = o[a] + (idx[a] + (step[a] > 0 ? 1 : 0)) * s;
    return (boundary - origin[a]) / safe[a];
  });

  for (let guard = 0; guard < nx + ny + nz + 3; guard++) {
    if (visit(idx[0], idx[1], idx[2], axis)) return [idx[0], idx[1], idx[2], axis];
    let a = 0;
    if (tMax[1] < tMax[a]) a = 1;
    if (tMax[2] < tMax[a]) a = 2;
    if (tMax[a] > tExit) return null;
    idx[a] += step[a];
    if (idx[a] < 0 || idx[a] >= n[a]) return null;
    tMax[a] += tDelta[a];
    axis = a;
  }
  return null;
}
