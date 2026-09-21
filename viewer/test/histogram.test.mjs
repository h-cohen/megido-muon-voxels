import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeHistogram } from '../src/histogram.mjs';

test('computeHistogram counts values within the window', () => {
  const data = new Float32Array([0, 0.1, 0.5, 0.9, 1.0, 2.0]);
  const hist = computeHistogram(data, 0, 1, 4);
  assert.equal(hist.length, 4);
  assert.equal(hist.reduce((a, b) => a + b, 0), 5); // 2.0 is outside [0,1]
});

test('computeHistogram ignores NaN', () => {
  const data = new Float32Array([0.1, NaN, 0.4]);
  const hist = computeHistogram(data, 0, 1, 2);
  assert.equal(hist.reduce((a, b) => a + b, 0), 2);
});

test('computeHistogram clamps the top edge into the last bin', () => {
  const data = new Float32Array([1.0]);
  const hist = computeHistogram(data, 0, 1, 4);
  assert.equal(hist[3], 1);
});
