import { rayBox, voxelMarch, voxelIndex, worldToVoxel } from './grid.mjs';

// Hover picking, CPU twin of the renderer (FRAGMENT_SRC in app.mjs): the same
// camera unprojection, box march and gates, so the voxel under the cursor is
// the voxel that was drawn. Pure: all inputs come in `scene`.
//   scene = { meta, data, window, renderMode, raySteps, minStepVoxels,
//             voxel, spatial,            // gates.mjs gate sets
//             cubeGrid, cubeThreshold }  // cube mode
// Fog: steps tEnter + (i + 0.5) * stepLen, stepLen = max(minStepVoxels *
// voxel, span / raySteps) (always the FULL step count, never the preview),
// first sample above window[0]. Cubes: voxel-to-voxel DDA (voxelMarch), gates
// at voxel centres, first voxel >= cubeThreshold.
export function pickVoxel(ndc, invViewProj, scene) {
  const { meta, data } = scene;
  if (!meta || !data || !invViewProj) return null;
  const m = invViewProj;
  const unprojectAt = (z) => {
    const c = [ndc[0], ndc[1], z, 1];
    const w = m[3] * c[0] + m[7] * c[1] + m[11] * c[2] + m[15] * c[3];
    return [
      (m[0] * c[0] + m[4] * c[1] + m[8] * c[2] + m[12] * c[3]) / w,
      (m[1] * c[0] + m[5] * c[1] + m[9] * c[2] + m[13] * c[3]) / w,
      (m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14] * c[3]) / w,
    ];
  };
  const nearP = unprojectAt(-1), farP = unprojectAt(1);
  const d = [farP[0] - nearP[0], farP[1] - nearP[1], farP[2] - nearP[2]];
  const len = Math.hypot(d[0], d[1], d[2]);
  const dir = [d[0] / len, d[1] / len, d[2] / len];
  return scene.renderMode === 'cubes' ? pickCubeAlong(nearP, dir, scene) : pickFogAlong(nearP, dir, scene);
}

function pickFogAlong(nearP, dir, scene) {
  const { meta, data } = scene;
  const min = meta.origin_m;
  const extent = meta.shape.map((n) => n * meta.spacing_m);
  const hit = rayBox(nearP, dir, min, [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]]);
  if (!hit) return null;
  const [tEnter, tExit] = hit;
  const voxel = extent[0] / meta.shape[0];   // cubic voxels (single spacing)
  const minStep = scene.minStepVoxels != null ? scene.minStepVoxels : 0.5;
  const stepLen = Math.max(minStep * voxel, (tExit - tEnter) / scene.raySteps);
  const lo = scene.window ? scene.window[0] : 0;
  for (let i = 0; i < scene.raySteps; i++) {
    const tt = tEnter + (i + 0.5) * stepLen;
    if (tt > tExit) break;
    const world = [nearP[0] + dir[0] * tt, nearP[1] + dir[1] * tt, nearP[2] + dir[2] * tt];
    const tex = [0, 1, 2].map((a) => (world[a] - min[a]) / extent[a]);
    const [vi, vj, vk] = worldToVoxel(world, meta);
    const n = voxelIndex(meta.shape, vi, vj, vk);
    if (n < 0 || !scene.spatial.keep(tex, world) || !scene.voxel.keep(n)) continue;
    const value = data[n];
    if (!Number.isNaN(value) && value > lo) {
      return { i: Math.floor(vi), j: Math.floor(vj), k: Math.floor(vk), value };
    }
  }
  return null;
}

function pickCubeAlong(nearP, dir, scene) {
  const { meta, cubeGrid: grid } = scene;
  if (!grid || !grid.values) return null;
  const [, ny, nz] = meta.shape;
  const [, cy, cz] = grid.shape;
  const s = grid.spacing, o = meta.origin_m;
  const ext = meta.shape.map((n) => n * meta.spacing_m);
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  let value = NaN;
  const hit = voxelMarch(nearP, dir, { shape: grid.shape, origin_m: o, spacing_m: s }, (i, j, k) => {
    const world = [o[0] + (i + 0.5) * s, o[1] + (j + 0.5) * s, o[2] + (k + 0.5) * s];
    const tex = [0, 1, 2].map((a) => clamp01((world[a] - o[a]) / ext[a]));
    if (!scene.spatial.keep(tex, world)) return false;
    if (!grid.baked && !scene.voxel.keep(i * ny * nz + j * nz + k)) return false;
    const v = grid.values[i * cy * cz + j * cz + k];
    if (v >= scene.cubeThreshold) { value = v; return true; }
    return false;
  });
  return hit ? { i: hit[0], j: hit[1], k: hit[2], value } : null;
}
