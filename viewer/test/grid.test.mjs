import { test } from 'node:test';
import assert from 'node:assert/strict';
import { modelMatrixFromMeta, worldToVoxel, voxelToWorld, sampleNearest, reorderForTexture, rayBox } from '../src/grid.mjs';

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

test('sampleNearest floors a fractional voxel coordinate to the containing box, not the nearest integer', () => {
  // Voxel k occupies [k, k+1) in worldToVoxel units (center k+0.5), matching
  // GL NEAREST texture filtering (texel index = floor(coord)). i=1.8 lies in
  // box [1,2) -- a round-to-nearest-integer implementation would wrongly
  // read voxel 2 (round(1.8) = 2) instead of voxel 1.
  const shape = [4, 1, 1];
  const data = new Float32Array([10, 20, 30, 40]);
  assert.equal(sampleNearest(data, shape, 1.8, 0, 0), 20);
  assert.equal(sampleNearest(data, shape, 1.01, 0, 0), 20);
  assert.equal(sampleNearest(data, shape, 1.99, 0, 0), 20);
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

// rayBox mirrors the shader's slab intersection (FRAGMENT_SRC in app.mjs):
// same sign-preserving 1e-8 guard on a zero direction component, tEnter
// clamped at 0, null when the ray misses the box (tExit <= tEnter).

test('rayBox: axis-aligned ray through a unit box gives the expected entry/exit', () => {
  const min = [0, 0, 0], max = [1, 1, 1];
  const hit = rayBox([0.5, 0.5, -2], [0, 0, 1], min, max);
  assert.ok(hit);
  const [tEnter, tExit] = hit;
  assert.ok(Math.abs(tEnter - 2) < 1e-6);
  assert.ok(Math.abs(tExit - 3) < 1e-6);
});

test('rayBox: a miss returns null', () => {
  const min = [0, 0, 0], max = [1, 1, 1];
  const hit = rayBox([5, 5, -2], [0, 0, 1], min, max);
  assert.equal(hit, null);
});

test('rayBox: origin inside the box clamps tEnter to 0', () => {
  const min = [0, 0, 0], max = [1, 1, 1];
  const hit = rayBox([0.5, 0.5, 0.5], [0, 0, 1], min, max);
  assert.ok(hit);
  const [tEnter, tExit] = hit;
  assert.equal(tEnter, 0);
  assert.ok(Math.abs(tExit - 0.5) < 1e-6);
});

test('rayBox: a ray with a zero direction component still works (guard)', () => {
  const min = [0, 0, 0], max = [1, 1, 1];
  const hit = rayBox([0.5, -2, 0.5], [0, 1, 0], min, max);
  assert.ok(hit);
  const [tEnter, tExit] = hit;
  assert.ok(Math.abs(tEnter - 2) < 1e-6);
  assert.ok(Math.abs(tExit - 3) < 1e-6);
});

// voxelMarch: exact voxel-to-voxel traversal (Amanatides-Woo DDA) used by the
// "Voxel cubes" render mode and its hover mirror. Visits every voxel the ray
// passes through, in order, with the axis of the face it entered through.
import { voxelMarch } from '../src/grid.mjs';

const CUBE_META = { shape: [4, 3, 5], origin_m: [0, 0, 1], spacing_m: 0.5 };

test('voxelMarch: a vertical ray from above visits one column top-down, entering through z faces', () => {
  const seen = [];
  voxelMarch([0.75, 0.25, 10], [0, 0, -1], CUBE_META, (i, j, k, axis) => { seen.push([i, j, k, axis]); return false; });
  assert.deepEqual(seen.map((v) => v.slice(0, 3)), [[1, 0, 4], [1, 0, 3], [1, 0, 2], [1, 0, 1], [1, 0, 0]]);
  assert.ok(seen.every((v) => v[3] === 2));
});

test('voxelMarch: stops at the first voxel the visitor accepts', () => {
  const seen = [];
  const hit = voxelMarch([0.75, 0.25, 10], [0, 0, -1], CUBE_META, (i, j, k) => { seen.push(k); return k === 2; });
  assert.deepEqual(hit.slice(0, 3), [1, 0, 2]);
  assert.deepEqual(seen, [4, 3, 2]);
});

test('voxelMarch: a diagonal ray in x-z steps through adjacent voxels only', () => {
  const seen = [];
  const d = [1 / Math.SQRT2, 0, 1 / Math.SQRT2];
  voxelMarch([-0.1, 0.25, 0.9], d, CUBE_META, (i, j, k, axis) => { seen.push([i, j, k, axis]); return false; });
  assert.ok(seen.length >= 4);
  for (let n = 1; n < seen.length; n++) {
    const dx = seen[n][0] - seen[n - 1][0], dz = seen[n][2] - seen[n - 1][2];
    assert.equal(Math.abs(dx) + Math.abs(dz), 1, `step ${n} jumps a voxel`);
    assert.equal(seen[n][3], dx !== 0 ? 0 : 2, `step ${n} reports the wrong entry face`);
  }
});

test('voxelMarch: a ray that misses the box visits nothing', () => {
  let n = 0;
  const hit = voxelMarch([10, 10, 10], [0, 0, 1], CUBE_META, () => { n++; return true; });
  assert.equal(hit, null);
  assert.equal(n, 0);
});
