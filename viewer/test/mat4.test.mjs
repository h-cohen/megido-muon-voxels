import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  identity, multiply, fromTranslation, fromScaling,
  perspective, lookAt, invert, transpose,
} from '../src/mat4.mjs';

function closeTo(a, b, eps = 1e-5) {
  return Math.abs(a - b) < eps;
}
function matClose(a, b, eps = 1e-5) {
  for (let i = 0; i < 16; i++) if (!closeTo(a[i], b[i], eps)) return false;
  return true;
}
function apply(m, p) {
  const [x, y, z] = p;
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

test('identity leaves points unchanged', () => {
  assert.deepEqual([...apply(identity(), [1, 2, 3])], [1, 2, 3]);
});

test('fromTranslation moves a point', () => {
  const m = fromTranslation([1, 2, 3]);
  assert.deepEqual([...apply(m, [0, 0, 0])], [1, 2, 3]);
});

test('fromScaling scales a point', () => {
  const m = fromScaling([2, 3, 4]);
  assert.deepEqual([...apply(m, [1, 1, 1])], [2, 3, 4]);
});

test('multiply applies b then a', () => {
  const t = fromTranslation([10, 0, 0]);
  const s = fromScaling([2, 2, 2]);
  const m = multiply(t, s); // scale first, then translate
  const p = apply(m, [1, 0, 0]);
  assert.ok(closeTo(p[0], 12) && closeTo(p[1], 0) && closeTo(p[2], 0));
});

test('invert undoes a translation', () => {
  const m = fromTranslation([5, -3, 2]);
  const inv = invert(m);
  const round = multiply(m, inv);
  assert.ok(matClose(round, identity()));
});

test('invert returns null for a singular matrix', () => {
  const m = fromScaling([0, 1, 1]);
  assert.equal(invert(m), null);
});

test('transpose swaps rows and columns', () => {
  const m = fromTranslation([1, 2, 3]);
  const t = transpose(m);
  assert.equal(t[12], 0);
  assert.equal(t[3], 1);
});

test('lookAt places the eye and looks toward center', () => {
  const m = lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
  const view = apply(m, [0, 0, 0]); // center, in eye space, should be at (0,0,-5)
  assert.ok(closeTo(view[0], 0) && closeTo(view[1], 0) && closeTo(view[2], -5));
});

test('perspective is invertible and non-degenerate', () => {
  const m = perspective(Math.PI / 4, 1.5, 0.1, 100);
  const inv = invert(m);
  assert.notEqual(inv, null);
  const round = multiply(m, inv);
  assert.ok(matClose(round, identity(), 1e-4));
});

test('lookAt transforms a general (non-axis-aligned) point correctly', () => {
  // eye = [3,4,0], center = origin, up = +y. Hand-derived basis:
  //   zAxis = normalize(eye - center)        = [0.6, 0.8, 0]
  //   xAxis = normalize(cross(up, zAxis))    = [0, 0, -1]
  //   yAxis = cross(zAxis, xAxis)            = [-0.8, 0.6, 0]
  // World point [1,1,1] relative to eye is [-2,-3,1]; dotting with the
  // basis above gives the expected view-space coordinates below. A
  // swapped cross product (handedness flip) changes the basis and would
  // break this assertion, unlike a check that only reads the z row.
  const m = lookAt([3, 4, 0], [0, 0, 0], [0, 1, 0]);
  const p = apply(m, [1, 1, 1]);
  assert.ok(closeTo(p[0], -1) && closeTo(p[1], -0.2) && closeTo(p[2], -3.6));
});

test('perspective maps the near plane to NDC z=-1 and the far plane to z=+1', () => {
  // Standard GL convention: a point on the frustum axis at eye-space
  // z = -near projects to NDC z = -1, and z = -far projects to z = +1.
  // A sign-flipped out[11] or out[14] would break one or both of these.
  const near = 0.1;
  const far = 100;
  const m = perspective(Math.PI / 4, 1.5, near, far);
  const atNear = apply(m, [0, 0, -near]);
  const atFar = apply(m, [0, 0, -far]);
  assert.ok(closeTo(atNear[2], -1));
  assert.ok(closeTo(atFar[2], 1));
});
