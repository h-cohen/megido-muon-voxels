import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeDelta, deltaVerdict } from '../src/delta.mjs';

test('computeDelta subtracts elementwise (b - a)', () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([1, 5, 2]);
  const d = computeDelta(a, b);
  assert.deepEqual([...d], [0, 3, -1]);
});

test('computeDelta rejects mismatched lengths', () => {
  assert.throws(() => computeDelta(new Float32Array(3), new Float32Array(4)), /length/);
});

test('deltaVerdict matches volexport.compare_volumes thresholds', () => {
  assert.equal(deltaVerdict(0.0001, 1.0), 'unchanged');
  assert.equal(deltaVerdict(0.05, 1.0), 'shifted slightly');
  assert.equal(deltaVerdict(0.5, 1.0), 'shifted substantially');
});
