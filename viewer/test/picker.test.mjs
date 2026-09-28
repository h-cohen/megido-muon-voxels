import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickVoxel } from '../src/picker.mjs';
import { voxelGates, spatialGates } from '../src/gates.mjs';
import { lookAt, perspective, multiply, invert } from '../src/mat4.mjs';

// 4x4x4 grid of 1 m voxels at the origin; camera straight above the column
// (1,1,*) looking down -z.
const META = { shape: [4, 4, 4], spacing_m: 1, origin_m: [0, 0, 0] };
const idx = (i, j, k) => i * 16 + j * 4 + k;
const OFF = { sigmaGateEnabled: false, sigmaGateValue: 1, coverageGateEnabled: false, minRays: 2, snrGateEnabled: false, minSnr: 3 };
const CLIP = { clipMin: [0, 0, 0], clipMax: [1, 1, 1], clipPlaneEnabled: false, clipPlaneNormal: [0, 0, 1], clipPlaneD: 0, surfClip: false };

function camera() {
  const eye = [1.5, 1.5, 20], target = [1.5, 1.5, 0];
  const view = lookAt(eye, target, [0, 1, 0]);
  const proj = perspective(Math.PI / 8, 1, 0.05, 100);
  return invert(multiply(proj, view));
}

function scene(data, extra = {}) {
  return {
    meta: META, data, window: [0.1, 1], renderMode: 'fog', raySteps: 200, minStepVoxels: 0.5,
    voxel: voxelGates(OFF, {}), spatial: spatialGates(CLIP, null),
    cubeGrid: { shape: META.shape, spacing: 1, values: data, baked: false }, cubeThreshold: 0.5,
    ...extra,
  };
}

test('fog: returns the first voxel from the camera above the window', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.9;   // top
  data[idx(1, 1, 0)] = 0.9;   // bottom, occluded
  const hit = pickVoxel([0, 0], camera(), scene(data));
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 3]);
  assert.ok(Math.abs(hit.value - 0.9) < 1e-6);
});

test('fog: values at or below window[0] are not hits', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.05;
  data[idx(1, 1, 1)] = 0.5;
  const hit = pickVoxel([0, 0], camera(), scene(data));
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 1]);
});

test('fog: a voxel gate hides the top voxel, the pick falls through', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.9;
  data[idx(1, 1, 1)] = 0.9;
  const snr = new Float32Array(64).fill(10);
  snr[idx(1, 1, 3)] = 0.5;
  const s = scene(data, { voxel: voxelGates({ ...OFF, snrGateEnabled: true }, { snr }) });
  const hit = pickVoxel([0, 0], camera(), s);
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 1]);
});

test('fog: the clip box hides the top half', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.9;
  data[idx(1, 1, 0)] = 0.9;
  const s = scene(data, { spatial: spatialGates({ ...CLIP, clipMax: [1, 1, 0.5] }, null) });
  const hit = pickVoxel([0, 0], camera(), s);
  assert.deepEqual([hit.i, hit.j, hit.k], [1, 1, 0]);
});

test('a ray that misses the box returns null', () => {
  const data = new Float32Array(64).fill(1);
  assert.equal(pickVoxel([0.99, 0.99], camera(), scene(data)), null);
});

test('cubes: first voxel at or above threshold; data gates skipped when baked', () => {
  const data = new Float32Array(64);
  data[idx(1, 1, 3)] = 0.6;
  data[idx(1, 1, 1)] = 0.9;
  const sig = new Float32Array(64);
  sig[idx(1, 1, 3)] = 9;
  const gated = voxelGates({ ...OFF, sigmaGateEnabled: true }, { sigma: sig });
  const unbaked = pickVoxel([0, 0], camera(), scene(data, { renderMode: 'cubes', voxel: gated }));
  assert.deepEqual([unbaked.i, unbaked.j, unbaked.k], [1, 1, 1]);   // sigma hides top
  const baked = pickVoxel([0, 0], camera(), scene(data, {
    renderMode: 'cubes', voxel: gated,
    cubeGrid: { shape: META.shape, spacing: 1, values: data, baked: true },
  }));
  assert.deepEqual([baked.i, baked.j, baked.k], [1, 1, 3]);          // baked: gates already applied
});
