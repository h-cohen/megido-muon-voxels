import { test } from 'node:test';
import assert from 'node:assert/strict';
import { modelMatrixFromMeta, worldToVoxel, voxelToWorld, sampleNearest, reorderForTexture } from '../src/grid.mjs';

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

test('reorderForTexture transposes numpy C-order (z-fastest) to GL x-fastest', () => {
  const shape = [3, 2, 2]; // nx=3, ny=2, nz=2, all distinct
  const data = Float32Array.from({ length: 12 }, (_, n) => n);
  const out = reorderForTexture(data, shape);
  const [nx, ny, nz] = shape;
  const expected = new Float32Array(nx * ny * nz);
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        expected[z * ny * nx + y * nx + x] = data[x * ny * nz + y * nz + z];
      }
    }
  }
  assert.deepEqual(Array.from(out), Array.from(expected));
  // guard against a stub that returns the input unchanged (the buggy behavior)
  assert.notDeepEqual(Array.from(out), Array.from(data));
  assert.equal(out.length, data.length);
  assert.notEqual(out, data);
  assert.deepEqual(Array.from(data), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]); // not mutated
});

test('reorderForTexture passthrough for [1,1,1]', () => {
  const data = new Float32Array([42]);
  const out = reorderForTexture(data, [1, 1, 1]);
  assert.deepEqual(Array.from(out), [42]);
  assert.notEqual(out, data);
});
