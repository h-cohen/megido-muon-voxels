import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  markerVertices, MARKER_HALF_M, silhouetteVertices,
  dedupeDetectors, projectToScreen,
} from '../src/markers.mjs';
import { identity, multiply, perspective, lookAt } from '../src/mat4.mjs';

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

test('dedupeDetectors collapses same-position exposures into one label with joined ids', () => {
  const detectors = [
    { id: 'P0', x: 0, y: 0, z: 0 },
    { id: 'T20a', x: 0, y: 0, z: 0 },
    { id: 'T20b', x: 0, y: 0, z: 0 },
    { id: 'P1', x: 2.2, y: 0, z: 0 },
  ];
  const out = dedupeDetectors(detectors);
  assert.equal(out.length, 2);
  assert.equal(out[0].label, 'P0/T20a/T20b');
  assert.deepEqual([out[0].x, out[0].y, out[0].z], [0, 0, 0]);
  assert.equal(out[1].label, 'P1');
  assert.deepEqual([out[1].x, out[1].y, out[1].z], [2.2, 0, 0]);
});

test('dedupeDetectors keeps distinct positions separate even with one exposure each', () => {
  const detectors = [
    { id: 'A', x: 0, y: 0, z: 0 },
    { id: 'B', x: 1, y: 0, z: 0 },
    { id: 'C', x: 0, y: 1, z: 0 },
  ];
  const out = dedupeDetectors(detectors);
  assert.equal(out.length, 3);
});

test('dedupeDetectors returns an empty array for no detectors', () => {
  assert.deepEqual(dedupeDetectors([]), []);
  assert.deepEqual(dedupeDetectors(undefined), []);
});

test('projectToScreen maps the world origin to canvas center under an identity viewProj', () => {
  const p = projectToScreen([0, 0, 0], identity(), 800, 600);
  assert.ok(p);
  assert.ok(Math.abs(p.x - 400) < 1e-6);
  assert.ok(Math.abs(p.y - 300) < 1e-6);
});

test('projectToScreen returns null for a point behind the camera (w <= 0)', () => {
  // A degenerate matrix whose w-row makes clip.w <= 0 for the given point.
  const m = identity();
  m[15] = -1; // w = -1*1(homogeneous) + 0 = -1 for world (0,0,0)
  const p = projectToScreen([0, 0, 0], m, 800, 600);
  assert.equal(p, null);
});

test('projectToScreen agrees with a real perspective camera: point straight ahead lands near center', () => {
  const eye = [0, 0, 5];
  const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
  const proj = perspective(Math.PI / 4, 800 / 600, 0.1, 100);
  const viewProj = multiply(proj, view);
  const p = projectToScreen([0, 0, 0], viewProj, 800, 600);
  assert.ok(p);
  assert.ok(Math.abs(p.x - 400) < 1e-3);
  assert.ok(Math.abs(p.y - 300) < 1e-3);
});
