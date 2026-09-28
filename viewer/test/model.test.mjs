import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createState, EFFECTS, EFFECT_ORDER, effectsFor } from '../src/model.mjs';

test('every committable field exists in the default state', () => {
  const s = createState();
  for (const field of Object.keys(EFFECTS)) assert.ok(field in s, field);
});

test('every effect named in the table is a known effect', () => {
  for (const [field, effects] of Object.entries(EFFECTS)) {
    for (const e of effects) assert.ok(EFFECT_ORDER.includes(e), `${field}: ${e}`);
  }
});

test('gate thresholds re-window; the sigma gate does not (on purpose)', () => {
  assert.deepEqual(effectsFor({ minSnr: 2 }), ['window', 'render']);
  assert.deepEqual(effectsFor({ coverageGateEnabled: false }), ['window', 'render']);
  assert.deepEqual(effectsFor({ sigmaGateValue: 0.3 }), ['render']);
});

test('union across a patch, in the fixed order', () => {
  assert.deepEqual(effectsFor({ hillColourMode: 'sigma', smoothSampling: false }),
    ['filter', 'surfaceMesh', 'legend', 'render']);
});

test('an undeclared field throws', () => {
  assert.throws(() => effectsFor({ notAField: 1 }), /no effects declared for state field: notAField/);
});

test('defaults match the viewer today', () => {
  const s = createState();
  assert.equal(s.renderMode, 'fog');
  assert.equal(s.opacity, 1);
  assert.equal(s.coverageGateEnabled, true);
  assert.equal(s.minRays, 2);
  assert.equal(s.snrGateEnabled, true);
  assert.equal(s.minSnr, 3);
  assert.equal(s.sigmaGateEnabled, false);
  assert.equal(s.hillColourMode, 'flat');
  assert.deepEqual(s.clipMax, [1, 1, 1]);
  assert.equal(s.window, null);
});
