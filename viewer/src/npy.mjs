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

  // Every supported dtype is decoded into its native typed array, then
  // converted to Float32Array: makeVolumeTexture always uploads R32F/FLOAT,
  // so there is no reason to keep the original width past this point. The
  // exporter (megido/volexport.py) writes plain counts (e.g. per-voxel view
  // counts) as smaller integer dtypes for size, not because the viewer wants
  // integer precision.
  const TYPED_CTORS = {
    '<f4': [Float32Array, 4],
    '<f8': [Float64Array, 8],
    '<i2': [Int16Array, 2],
    '<i4': [Int32Array, 4],
    '|u1': [Uint8Array, 1],
    '|i1': [Int8Array, 1],
  };

  if (fortranOrder) {
    throw new Error(
      `unsupported npy dtype/order: descr=${dtype} fortran_order=${fortranOrder} ` +
      `(this viewer only reads little-endian, C-order arrays)`
    );
  }
  const entry = TYPED_CTORS[dtype];
  if (!entry) {
    throw new Error(
      `unsupported npy dtype/order: descr=${dtype} fortran_order=${fortranOrder} ` +
      `(this viewer reads: ${Object.keys(TYPED_CTORS).join(', ')})`
    );
  }
  const [Ctor, itemSize] = entry;

  const dataStart = headerStart + headerLen;
  const count = shape.reduce((a, b) => a * b, 1);
  const raw = new Ctor(buffer.slice(dataStart, dataStart + count * itemSize));
  const data = dtype === '<f4' ? raw : Float32Array.from(raw);
  return { shape, dtype, data };
}
