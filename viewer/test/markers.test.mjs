import { test } from 'node:test';
import assert from 'node:assert/strict';
import { markerVertices, MARKER_HALF_M, silhouetteVertices } from '../src/markers.mjs';

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

test('silhouetteVertices skips null (unconstrained) elevation bins', () => {
  const perPos = {
    pos0: {
      ridge_az: [0, 90, 180],
      ridge_elev: [45, null, 30],
    },
  };
  const detectors = [{ id: 'pos0', x: 1, y: 2, z: 3 }];
  const v = silhouetteVertices(perPos, detectors, 10);
  // 2 finite bins -> 2 segments -> 2 * 6 floats
  assert.equal(v.length, 2 * 6);
});

test('silhouetteVertices only draws positions present in both perPos and detectors', () => {
  const perPos = {
    pos0: { ridge_az: [0], ridge_elev: [45] },
    pos1: { ridge_az: [0], ridge_elev: [45] },
  };
  const detectors = [{ id: 'pos0', x: 0, y: 0, z: 0 }]; // pos1 missing
  const v = silhouetteVertices(perPos, detectors, 10);
  assert.equal(v.length, 6); // one segment only
});

test('silhouetteVertices: zenith-ish bin points up (+z) from the detector', () => {
  const perPos = { pos0: { ridge_az: [0], ridge_elev: [89] } };
  const detectors = [{ id: 'pos0', x: 5, y: 5, z: 5 }];
  const v = silhouetteVertices(perPos, detectors, 10);
  const [x0, y0, z0, x1, y1, z1] = v;
  assert.ok(Math.abs(x0 - 5) < 1e-9 && Math.abs(y0 - 5) < 1e-9 && Math.abs(z0 - 5) < 1e-9);
  assert.ok(z1 - z0 > 9, 'near-zenith ray must rise steeply in z');
  assert.ok(Math.abs(x1 - x0) < 2, 'near-zenith ray must have small horizontal displacement');
});

test('silhouetteVertices returns empty array for empty perPos', () => {
  const v = silhouetteVertices({}, [{ id: 'pos0', x: 0, y: 0, z: 0 }], 10);
  assert.equal(v.length, 0);
});
