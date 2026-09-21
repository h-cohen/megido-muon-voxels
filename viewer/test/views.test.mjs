import { test } from 'node:test';
import assert from 'node:assert/strict';
import { captureView, serializeViews, deserializeViews } from '../src/views.mjs';

test('captureView copies camera/window/layer, not references', () => {
  const state = { camera: { yaw: 1, pitch: 2, distance: 3, target: [1, 2, 3] }, window: [0.1, 0.4], activeLayer: 'volume' };
  const v = captureView(state, 'A');
  assert.equal(v.name, 'A');
  assert.deepEqual(v.window, [0.1, 0.4]);
  state.camera.yaw = 99; state.window[0] = 99;
  assert.equal(v.camera.yaw, 1);   // deep copy
  assert.equal(v.window[0], 0.1);
});

test('serialize/deserialize round-trips', () => {
  const list = [captureView({ camera: { yaw: 0, pitch: 0, distance: 1, target: [0,0,0] }, window: [0,1], activeLayer: 'sigma' }, 'v1')];
  assert.deepEqual(deserializeViews(serializeViews(list)), list);
});

test('deserializeViews returns [] on garbage', () => {
  assert.deepEqual(deserializeViews('not json'), []);
  assert.deepEqual(deserializeViews('{"x":1}'), []); // not an array
});
