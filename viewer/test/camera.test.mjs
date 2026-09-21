import { test } from 'node:test';
import assert from 'node:assert/strict';
import { orbitToEye, CAMERA_PRESETS } from '../src/camera.mjs';

function close(a, b, eps = 1e-5) { return Math.abs(a - b) < eps; }

test('orbitToEye at yaw=0, pitch=0 sits on +z from target', () => {
  const eye = orbitToEye([0, 0, 0], 0, 0, 5);
  assert.ok(close(eye[0], 0) && close(eye[1], 0) && close(eye[2], 5));
});

test('orbitToEye respects the target offset', () => {
  const eye = orbitToEye([1, 2, 3], 0, 0, 5);
  assert.ok(close(eye[0], 1) && close(eye[1], 2) && close(eye[2], 8));
});

test('orbitToEye stays at constant distance from target', () => {
  const target = [0, 0, 0];
  for (const [yaw, pitch] of [[0.3, 0.7], [2.1, -0.4], [5.9, 1.0]]) {
    const eye = orbitToEye(target, yaw, pitch, 4);
    const d = Math.hypot(eye[0], eye[1], eye[2]);
    assert.ok(close(d, 4, 1e-4));
  }
});

test('CAMERA_PRESETS defines top, front, side, iso', () => {
  for (const key of ['top', 'front', 'side', 'iso']) {
    assert.ok(key in CAMERA_PRESETS);
    assert.ok(typeof CAMERA_PRESETS[key].yaw === 'number');
    assert.ok(typeof CAMERA_PRESETS[key].pitch === 'number');
  }
});
