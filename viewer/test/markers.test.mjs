import { test } from 'node:test';
import assert from 'node:assert/strict';
import { markerVertices, MARKER_HALF_M } from '../src/markers.mjs';

test('markerVertices builds a 3-axis cross per detector', () => {
  const v = markerVertices([{ x: 2.2, y: 0, z: 0 }]);
  assert.equal(v.length, 6 * 3);                 // 3 axes × 2 endpoints × 3 coords
  // x-arm endpoints straddle the centre on x
  assert.ok(Math.abs(v[0] - (2.2 - MARKER_HALF_M)) < 1e-6);
  assert.ok(Math.abs(v[3] - (2.2 + MARKER_HALF_M)) < 1e-6);
});

test('two detectors -> two crosses', () => {
  const v = markerVertices([{ x: 0, y: 0, z: 0 }, { x: 2.2, y: 0, z: 0 }]);
  assert.equal(v.length, 2 * 6 * 3);
});
