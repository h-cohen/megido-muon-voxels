import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseNpy } from '../src/npy.mjs';

function makeNpy(shape, values) {
  // Minimal NPY v1.0 writer: magic + version + header + little-endian f4 data.
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]; // \x93NUMPY, v1.0
  const preLen = magic.length + 2; // + 2-byte header-length field
  let padded = header;
  while ((preLen + padded.length + 1) % 16 !== 0) padded += ' ';
  padded += '\n';
  const headerBytes = new TextEncoder().encode(padded);
  const buf = new ArrayBuffer(preLen + headerBytes.length + values.length * 4);
  const view = new DataView(buf);
  magic.forEach((b, i) => view.setUint8(i, b));
  view.setUint16(8, headerBytes.length, true);
  new Uint8Array(buf, preLen, headerBytes.length).set(headerBytes);
  const dataStart = preLen + headerBytes.length;
  values.forEach((v, i) => view.setFloat32(dataStart + i * 4, v, true));
  return buf;
}

test('parseNpy recovers shape and float32 data', () => {
  const buf = makeNpy([2, 3], [1, 2, 3, 4, 5, 6]);
  const { shape, dtype, data } = parseNpy(buf);
  assert.deepEqual(shape, [2, 3]);
  assert.equal(dtype, '<f4');
  assert.equal(data.length, 6);
  assert.equal(data[0], 1);
  assert.equal(data[5], 6);
  assert.ok(data instanceof Float32Array);
});

test('parseNpy rejects fortran-ordered arrays', () => {
  const buf = makeNpy([2, 2], [1, 2, 3, 4]);
  const bytes = new Uint8Array(buf);
  // Patch the ASCII header in place, byte-for-byte, instead of round-tripping
  // the whole buffer through UTF-8: the magic bytes and the float32 data
  // contain byte values that are not valid UTF-8 on their own, so a decode
  // + replace + re-encode round-trip corrupts them (each invalid byte becomes
  // a 3-byte replacement character), shifting every offset after it and
  // breaking the magic-number check before the fortran_order check is ever
  // reached.
  const falseBytes = new TextEncoder().encode('False');
  const trueBytes = new TextEncoder().encode('True ');
  let idx = -1;
  outer: for (let i = 0; i <= bytes.length - falseBytes.length; i++) {
    for (let j = 0; j < falseBytes.length; j++) {
      if (bytes[i + j] !== falseBytes[j]) continue outer;
    }
    idx = i;
    break;
  }
  assert.ok(idx >= 0, 'expected to find fortran_order False in header');
  bytes.set(trueBytes, idx);
  assert.throws(() => parseNpy(buf), /fortran_order/);
});
