import { test } from 'node:test';
import assert from 'node:assert/strict';
import { COLORMAP_NAMES, colormapStops } from '../src/colormap.mjs';

test('exposes the four presets', () => {
  for (const n of ['viridis', 'inferno', 'magma', 'grayscale']) {
    assert.ok(COLORMAP_NAMES.includes(n), n);
  }
});

test('stops span t=0..1 with a rising alpha ramp', () => {
  const s = colormapStops('viridis');
  assert.equal(s[0].t, 0);
  assert.equal(s[s.length - 1].t, 1);
  assert.ok(s[0].a <= s[s.length - 1].a);          // alpha rises with density
  for (const st of s) {
    for (const c of ['r', 'g', 'b', 'a']) {
      assert.ok(st[c] >= 0 && st[c] <= 255, `${c}=${st[c]}`);
    }
  }
});

test('grayscale is neutral (r==g==b at every stop)', () => {
  for (const st of colormapStops('grayscale')) {
    assert.equal(st.r, st.g);
    assert.equal(st.g, st.b);
  }
});

test('unknown name falls back to viridis', () => {
  assert.deepEqual(colormapStops('nope'), colormapStops('viridis'));
});
