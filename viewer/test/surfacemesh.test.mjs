import { test } from 'node:test';
import assert from 'node:assert/strict';
import { surfaceMesh, smoothHeightfield, surfaceVertexColors, robustRange, SIGMA_RAMP, surfaceHeightAt, surfaceTextureData } from '../src/surfacemesh.mjs';

test('surfaceMesh: 2x2 all-finite grid -> 4 verts, one quad -> 6 indices', () => {
  const gx = [0, 1];
  const gy = [0, 2];
  const H = new Float32Array([10, 11, 12, 13]); // row-major i*ny+j
  const { positions, indices } = surfaceMesh(H, gx, gy);
  assert.equal(positions.length, 12);
  assert.equal(indices.length, 6);
  // vertex (i=0,j=0) at (gx[0], gy[0], H[0])
  assert.deepEqual(Array.from(positions.slice(0, 3)), [0, 0, 10]);
  // vertex (i=1,j=1) at (gx[1], gy[1], H[3])
  assert.deepEqual(Array.from(positions.slice(9, 12)), [1, 2, 13]);
});

test('surfaceMesh: one NaN corner omits that cell\'s triangles', () => {
  const gx = [0, 1];
  const gy = [0, 2];
  const H = new Float32Array([10, 11, NaN, 13]);
  const { positions, indices } = surfaceMesh(H, gx, gy);
  assert.equal(positions.length, 12); // still one vertex per node
  assert.equal(indices.length, 0); // the only cell has a NaN corner
});

test('surfaceMesh: 3x3 grid with all-finite interior quad produces 6 quads worth of indices', () => {
  const gx = [0, 1, 2];
  const gy = [0, 1, 2];
  const H = new Float32Array(9).fill(5);
  const { indices } = surfaceMesh(H, gx, gy);
  assert.equal(indices.length, 4 * 6); // 2x2 cells, 6 indices each
});

test('smoothHeightfield: iterations=0 returns an unchanged copy, not the same object', () => {
  const H = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const out = smoothHeightfield(H, 3, 3, 0);
  assert.notEqual(out, H);
  assert.deepEqual(Array.from(out), Array.from(H));
});

test('smoothHeightfield: reduces a spike in an otherwise uniform field', () => {
  const nx = 5, ny = 5;
  const H = new Float32Array(nx * ny).fill(10);
  H[2 * ny + 2] = 100; // spike at center
  const out = smoothHeightfield(H, nx, ny, 1);
  assert.ok(out[2 * ny + 2] < 100);
  assert.ok(out[2 * ny + 2] > 10);
});

test('smoothHeightfield: NaN cells stay NaN and are not averaged into finite neighbours', () => {
  const nx = 3, ny = 3;
  // Row-major i*ny+j; NaN at (i=1,j=1) = idx 4 (center of a 3x3 grid).
  const H = new Float32Array([1, 1, 1, 1, NaN, 1, 1, 1, 1]);
  const out = smoothHeightfield(H, nx, ny, 1);
  assert.ok(Number.isNaN(out[4]));
  // Every finite neighbour of the NaN center is a uniform field of 1s
  // elsewhere, so no finite cell should be pulled off 1 by the NaN.
  for (let i = 0; i < out.length; i++) {
    if (i === 4) continue;
    assert.ok(Number.isFinite(out[i]));
    assert.equal(out[i], 1);
  }
});

test('smoothHeightfield: a constant field is unchanged by smoothing', () => {
  const nx = 4, ny = 4;
  const H = new Float32Array(nx * ny).fill(7);
  const out = smoothHeightfield(H, nx, ny, 5);
  for (const v of out) assert.equal(v, 7);
});

test('surfaceVertexColors maps lo/hi to ramp ends and clamps', () => {
  const c = surfaceVertexColors(Float32Array.from([0, 1, 2, -5, 9]), 0, 2);
  const [a, b] = SIGMA_RAMP;
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(c[k] - a[k]) < 1e-6);          // sigma=lo -> start
    assert.ok(Math.abs(c[6 + k] - b[k]) < 1e-6);      // sigma=hi -> end
    assert.ok(Math.abs(c[9 + k] - a[k]) < 1e-6);      // below lo clamps
    assert.ok(Math.abs(c[12 + k] - b[k]) < 1e-6);     // above hi clamps
    assert.ok(Math.abs(c[3 + k] - (a[k] + b[k]) / 2) < 1e-6); // midpoint
  }
});

test('surfaceVertexColors gives NaN sigma the mid colour', () => {
  const c = surfaceVertexColors(Float32Array.from([NaN]), 0, 1);
  const [a, b] = SIGMA_RAMP;
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(c[k] - (a[k] + b[k]) / 2) < 1e-6);
});

test('robustRange ignores NaN and returns 5th/95th percentiles', () => {
  const v = Float32Array.from([...Array(101).keys()].map(Number).concat([NaN]));
  const [lo, hi] = robustRange(v);
  assert.ok(Math.abs(lo - 5) < 1e-6 && Math.abs(hi - 95) < 1e-6);
  const [n1, n2] = robustRange(Float32Array.from([NaN]));
  assert.ok(Number.isNaN(n1) && Number.isNaN(n2));
});

test('surfaceHeightAt is exact on a plane, NaN outside or at a NaN corner', () => {
  const gx = [0, 1, 2], gy = [0, 1, 2, 3];
  const H = new Float32Array(12);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) H[i * 4 + j] = 2 * gx[i] + 3 * gy[j] + 1;
  assert.ok(Math.abs(surfaceHeightAt(H, gx, gy, 0.5, 2.25) - (1 + 6.75 + 1)) < 1e-5);
  assert.ok(Number.isNaN(surfaceHeightAt(H, gx, gy, -0.1, 1)));
  assert.ok(Number.isNaN(surfaceHeightAt(H, gx, gy, 1, 3.5)));
  const H2 = Float32Array.from(H); H2[1 * 4 + 1] = NaN;
  assert.ok(Number.isNaN(surfaceHeightAt(H2, gx, gy, 0.5, 0.5)));
});

test('surfaceTextureData transposes to x-fastest and sentinels NaN', () => {
  const nx = 3, ny = 2;
  const H = Float32Array.from([0, 1, 10, 11, NaN, 21]); // H[i*ny+j] = 10*i + j
  const t = surfaceTextureData(H, nx, ny, 1e6);
  // out[j*nx+i] = H[i*ny+j]
  assert.deepEqual(Array.from(t), [0, 10, 1e6, 1, 11, 21]);
});
