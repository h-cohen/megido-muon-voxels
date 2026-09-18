export function parseNpy(buffer) {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 6);
  const expected = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < 6; i++) {
    if (magic[i] !== expected[i]) throw new Error('not an NPY file: bad magic');
  }
  const major = view.getUint8(6);
  const headerLenBytes = major >= 2 ? 4 : 2;
  const headerLen = major >= 2
    ? view.getUint32(8, true)
    : view.getUint16(8, true);
  const headerStart = 8 + headerLenBytes;
  const headerText = new TextDecoder().decode(
    new Uint8Array(buffer, headerStart, headerLen)
  );

  const descrMatch = headerText.match(/'descr':\s*'([^']+)'/);
  const orderMatch = headerText.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !orderMatch || !shapeMatch) {
    throw new Error(`malformed NPY header: ${headerText}`);
  }
  const dtype = descrMatch[1];
  const fortranOrder = orderMatch[1] === 'True';
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);

  if (dtype !== '<f4' || fortranOrder) {
    throw new Error(
      `unsupported npy dtype/order: descr=${dtype} fortran_order=${fortranOrder} ` +
      `(this viewer only reads little-endian float32, C-order arrays)`
    );
  }

  const dataStart = headerStart + headerLen;
  const count = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(buffer.slice(dataStart, dataStart + count * 4));
  return { shape, dtype, data };
}
