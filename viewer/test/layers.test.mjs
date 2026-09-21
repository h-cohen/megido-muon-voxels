import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KNOWN_LAYERS, availableLayers } from '../src/layers.mjs';

test('KNOWN_LAYERS covers the Phase 3 export names', () => {
  for (const key of ['volume', 'sigma', 'snr', 'views', 'systematic', 'backprojection']) {
    assert.ok(key in KNOWN_LAYERS, key);
  }
});

test('availableLayers filters to present names, in meta order', () => {
  const result = availableLayers(['volume', 'sigma']);
  assert.deepEqual(result.map((l) => l.key), ['volume', 'sigma']);
  assert.equal(result[0].label, KNOWN_LAYERS.volume.label);
});

test('availableLayers falls back gracefully for an unknown layer name', () => {
  const result = availableLayers(['mystery_layer']);
  assert.equal(result.length, 1);
  assert.equal(result[0].key, 'mystery_layer');
  assert.equal(result[0].label, 'mystery_layer');
  assert.equal(result[0].kind, 'scalar');
});
