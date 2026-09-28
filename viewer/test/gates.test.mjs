import { test } from 'node:test';
import assert from 'node:assert/strict';
import { voxelGates, spatialGates } from '../src/gates.mjs';

const OFF = { sigmaGateEnabled: false, sigmaGateValue: 1, coverageGateEnabled: false, minRays: 2, snrGateEnabled: false, minSnr: 3 };
const ALL = { ...OFF, sigmaGateEnabled: true, coverageGateEnabled: true, snrGateEnabled: true };
// index:                0    1    2    3    4    5
const sigma = Float32Array.from([0.5, 2.0, NaN, 0.5, 0.5, 0.5]);
const rays  = Float32Array.from([5,   5,   5,   1,   NaN, 5]);
const snr   = Float32Array.from([4,   4,   4,   4,   4,   NaN]);
const L = { sigma, rays, snr };

test('all gates off keeps every voxel', () => {
  const g = voxelGates(OFF, L);
  for (let n = 0; n < 6; n++) assert.equal(g.keep(n), true);
  assert.equal(g.windowActive, false);
});

test('keep truth table with every gate on (NaN semantics)', () => {
  const g = voxelGates(ALL, L);
  assert.deepEqual([0, 1, 2, 3, 4, 5].map(g.keep), [
    true,   // passes all
    false,  // sigma 2 > 1
    true,   // sigma NaN is kept
    false,  // rays 1 < 2
    true,   // rays NaN is kept (R1: shader rule)
    false,  // snr NaN is hidden
  ]);
});

test('keepForWindow ignores the sigma gate on purpose', () => {
  const g = voxelGates(ALL, L);
  assert.equal(g.keepForWindow(1), true);    // hidden by sigma, still windows
  assert.equal(g.keepForWindow(3), false);   // coverage still applies
  assert.equal(g.keepForWindow(5), false);   // snr still applies
  assert.equal(g.windowActive, true);
});

test('a gate whose layer is absent is inert and reports disabled', () => {
  const g = voxelGates(ALL, {});
  for (let n = 0; n < 6; n++) assert.equal(g.keep(n), true);
  assert.deepEqual(g.uniforms, { sigmaEnabled: false, sigmaValue: 1, coverageEnabled: false, minRays: 2, snrEnabled: false, minSnr: 3 });
  assert.equal(g.windowActive, false);
});

test('uniforms report enabled only when gate on AND layer present', () => {
  const g = voxelGates({ ...OFF, snrGateEnabled: true }, L);
  assert.equal(g.uniforms.snrEnabled, true);
  assert.equal(g.uniforms.sigmaEnabled, false);
  assert.equal(g.uniforms.coverageEnabled, false);
});

test('key changes with every input that can change keep()', () => {
  const base = voxelGates(ALL, L).key;
  for (const patch of [{ sigmaGateEnabled: false }, { sigmaGateValue: 1.5 }, { coverageGateEnabled: false },
    { minRays: 3 }, { snrGateEnabled: false }, { minSnr: 2 }]) {
    assert.notEqual(voxelGates({ ...ALL, ...patch }, L).key, base, JSON.stringify(patch));
  }
  assert.notEqual(voxelGates(ALL, { rays, snr }).key, base);   // sigma layer gone
  assert.equal(voxelGates({ ...ALL }, L).key, base);
});

const CLIP = { clipMin: [0, 0, 0], clipMax: [1, 1, 1], clipPlaneEnabled: false, clipPlaneNormal: [0, 0, 1], clipPlaneD: 0, surfClip: false };

test('spatial: clip box, clip plane and surface clip each hide', () => {
  const mid = [0.5, 0.5, 0.5], w = [0, 0, 5];
  assert.equal(spatialGates(CLIP, null).keep(mid, w), true);
  assert.equal(spatialGates({ ...CLIP, clipMax: [1, 1, 0.4] }, null).keep(mid, w), false);
  assert.equal(spatialGates({ ...CLIP, clipPlaneEnabled: true, clipPlaneD: 0.2 }, null).keep(mid, w), false);
  const surface = { H: Float32Array.from([4, 4, 4, 4]), gx: [-1, 1], gy: [-1, 1] };
  assert.equal(spatialGates({ ...CLIP, surfClip: true }, surface).keep(mid, w), false);   // z 5 above H 4
  assert.equal(spatialGates({ ...CLIP, surfClip: true }, surface).keep(mid, [0, 0, 3]), true);
  assert.equal(spatialGates({ ...CLIP, surfClip: false }, surface).keep(mid, w), true);
  assert.equal(spatialGates({ ...CLIP, surfClip: true }, null).keep(mid, w), true);       // no surface: inert
});
