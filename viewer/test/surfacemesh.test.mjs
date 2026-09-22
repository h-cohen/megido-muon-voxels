import { test } from 'node:test';
import assert from 'node:assert/strict';
import { surfaceMesh } from '../src/surfacemesh.mjs';

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
