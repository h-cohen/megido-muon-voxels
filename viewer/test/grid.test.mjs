import { test } from 'node:test';
import assert from 'node:assert/strict';
import { modelMatrixFromMeta, worldToVoxel, voxelToWorld, sampleNearest } from '../src/grid.mjs';

const META = { shape: [4, 3, 2], spacing_m: 0.5, origin_m: [1, 2, 3] };

test('worldToVoxel and voxelToWorld are inverses', () => {
  const world = [2.5, 3.0, 3.5];
  const voxel = worldToVoxel(world, META);
  const back = voxelToWorld(voxel, META);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[i] - world[i]) < 1e-6);
});

test('worldToVoxel at the origin is voxel (0,0,0)', () => {
  const v = worldToVoxel(META.origin_m, META);
  assert.deepEqual(v.map((x) => Math.round(x * 1e6) / 1e6), [0, 0, 0]);
});

test('modelMatrixFromMeta maps the unit cube onto the world-space box', () => {
  const m = modelMatrixFromMeta(META);
  // point (1,1,1) in unit-cube/texture space -> far corner of the voxel box
  const x = m[0] * 1 + m[4] * 1 + m[8] * 1 + m[12];
  const y = m[1] * 1 + m[5] * 1 + m[9] * 1 + m[13];
  const z = m[2] * 1 + m[6] * 1 + m[10] * 1 + m[14];
  assert.ok(Math.abs(x - (1 + 4 * 0.5)) < 1e-6);
  assert.ok(Math.abs(y - (2 + 3 * 0.5)) < 1e-6);
  assert.ok(Math.abs(z - (3 + 2 * 0.5)) < 1e-6);
});

test('sampleNearest reads C-order [nx,ny,nz] data', () => {
  const shape = [2, 2, 2];
  const data = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]); // index = i*4 + j*2 + k
  assert.equal(sampleNearest(data, shape, 0, 0, 0), 0);
  assert.equal(sampleNearest(data, shape, 1, 0, 1), 5);
  assert.equal(sampleNearest(data, shape, 1, 1, 1), 7);
});

test('sampleNearest returns NaN out of bounds', () => {
  const shape = [2, 2, 2];
  const data = new Float32Array(8);
  assert.ok(Number.isNaN(sampleNearest(data, shape, -1, 0, 0)));
  assert.ok(Number.isNaN(sampleNearest(data, shape, 2, 0, 0)));
});
