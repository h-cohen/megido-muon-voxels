import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeHistogram, robustWindow } from '../src/histogram.mjs';

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

test('robustWindow ignores a single outlier and windows to the bulk', () => {
  // 999 values clustered near 0.01, one outlier at 10 (mirrors the real
  // campaign: median 0.0025, p95 0.12, value_range[1] 2.369 outlier).
  const data = new Float32Array(1000);
  for (let i = 0; i < 999; i++) data[i] = 0.01 + (i % 10) * 0.001;
  data[999] = 10.0;
  const [lo, hi] = robustWindow(data, 0.99);
  assert.equal(lo, 0);
  assert.ok(hi < 1, `expected hi far below the outlier, got ${hi}`);
});

test('robustWindow keeps integer-like layers (e.g. views 0/1/2) sensible', () => {
  const data = new Float32Array([0, 1, 1, 2, 2, 2]);
  const [lo, hi] = robustWindow(data, 0.99);
  assert.equal(lo, 0);
  assert.ok(hi >= 1.9 && hi <= 2.01, `expected hi ~= 2, got ${hi}`);
});

test('robustWindow falls back sanely on degenerate all-equal data', () => {
  const data = new Float32Array([5, 5, 5, 5]);
  const [lo, hi] = robustWindow(data, 0.99);
  assert.equal(lo, 5);
  assert.equal(hi, 5);
});

test('robustWindow falls back on all-zero data', () => {
  const data = new Float32Array([0, 0, 0]);
  const [lo, hi] = robustWindow(data, 0.99);
  assert.equal(lo, 0);
  assert.equal(hi, 0);
});
