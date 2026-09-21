import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SHORTCUTS, keyToAction } from '../src/shortcuts.mjs';

test('digits map to layer actions', () => {
  assert.equal(keyToAction({ key: '1' }), 'layer1');
  assert.equal(keyToAction({ key: '5' }), 'layer5');
});
test('r frames all, ? toggles help, Escape closes', () => {
  assert.equal(keyToAction({ key: 'r' }), 'frame-all');
  assert.equal(keyToAction({ key: '?' }), 'toggle-help');
  assert.equal(keyToAction({ key: 'Escape' }), 'close-overlay');
});
test('unmapped key returns null', () => {
  assert.equal(keyToAction({ key: 'q' }), null);
});
test('SHORTCUTS is a non-empty list with labels', () => {
  assert.ok(SHORTCUTS.length > 0);
  for (const s of SHORTCUTS) assert.ok(s.label && s.action);
});
