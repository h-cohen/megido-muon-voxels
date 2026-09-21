import { test } from 'node:test';
import assert from 'node:assert/strict';
import { insideClipBox, clipPlaneDistance, insideClipPlane } from '../src/clip.mjs';

test('insideClipBox is true inside, false outside', () => {
  assert.equal(insideClipBox([0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1]), true);
  assert.equal(insideClipBox([1.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1]), false);
  assert.equal(insideClipBox([0, 0, 0], [0, 0, 0], [1, 1, 1]), true); // inclusive edge
});

test('clipPlaneDistance is signed distance from the plane through the box center', () => {
  const d = clipPlaneDistance([0.5, 0.5, 0.7], [0, 0, 1], 0.2);
  assert.ok(Math.abs(d - (0.2 - 0.2)) < 1e-9); // (0.7-0.5) - 0.2 = 0
});

test('insideClipPlane keeps the positive side', () => {
  assert.equal(insideClipPlane([0.5, 0.5, 0.9], [0, 0, 1], 0.0), true);
  assert.equal(insideClipPlane([0.5, 0.5, 0.1], [0, 0, 1], 0.0), false);
});
