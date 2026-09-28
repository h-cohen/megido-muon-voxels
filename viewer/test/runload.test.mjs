import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readRun, readCompareVolume } from '../src/runload.mjs';

function npyBuffer(values, shape) {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  let h = header;
  while ((10 + h.length + 1) % 64 !== 0) h += ' ';
  h += '\n';
  const buf = new ArrayBuffer(10 + h.length + values.length * 4);
  const u8 = new Uint8Array(buf);
  u8.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(buf).setUint16(8, h.length, true);
  for (let i = 0; i < h.length; i++) u8[10 + i] = h.charCodeAt(i);
  new Float32Array(buf, 10 + h.length).set(values);
  return buf;
}
const jsonFile = (name, obj) => ({ name, text: async () => JSON.stringify(obj), arrayBuffer: async () => { throw new Error('json'); } });
const npyFile = (name, values, shape) => ({ name, text: async () => { throw new Error('npy'); }, arrayBuffer: async () => npyBuffer(values, shape) });

const META = { shape: [2, 2, 2], spacing_m: 0.5, origin_m: [0, 0, 0], layers: ['volume', 'sigma', 'backprojection'], suggested_iso: [0.3, 0.6], value_range: [0, 1] };
const vol = [0, 1, 2, 3, 4, 5, 6, 7];

test('missing meta.json throws the same error as before', async () => {
  await assert.rejects(readRun([npyFile('volume.npy', vol, [2, 2, 2])]), /no meta\.json/);
});

test('only full-grid layers load; sigmaMax and cubeThreshold derived', async () => {
  const r = await readRun([
    jsonFile('meta.json', META),
    npyFile('volume.npy', vol, [2, 2, 2]),
    npyFile('sigma.npy', [1, 9, 1, 1, 1, 1, 1, 1], [2, 2, 2]),
    npyFile('backprojection.npy', [1, 2, 3, 4], [2, 2]),       // 2D: skipped
  ]);
  assert.deepEqual([...r.layers.keys()], ['volume', 'sigma']);
  assert.equal(r.sigmaMax, 9);
  assert.equal(r.cubeThreshold, 0.3);
  assert.equal(r.silhouette, null);
  assert.equal(r.hillSurface, null);
});

test('cubeThreshold falls back to mid value_range; sigmaMax 1 without sigma', async () => {
  const r = await readRun([jsonFile('meta.json', { ...META, layers: ['volume'], suggested_iso: null, value_range: [0, 2] }), npyFile('volume.npy', vol, [2, 2, 2])]);
  assert.equal(r.cubeThreshold, 1);
  assert.equal(r.sigmaMax, 1);
});

test('hill surface needs both files; mismatched sigma dropped; residualLim fallback', async () => {
  const hm = { gx: [0, 1], gy: [0, 1], residual_grid_lim: null };
  const base = [jsonFile('meta.json', META), npyFile('volume.npy', vol, [2, 2, 2])];
  const noMeta = await readRun([...base, npyFile('hill_surface.npy', [1, 2, 3, 4], [2, 2])]);
  assert.equal(noMeta.hillSurface, null);
  const r = await readRun([...base,
    npyFile('hill_surface.npy', [1, 2, 3, 4], [2, 2]),
    jsonFile('hill_surface_meta.json', hm),
    npyFile('hill_surface_sigma.npy', [1, 2, 3], [3]),                 // wrong length: dropped
    npyFile('hill_residual_grid.npy', [0.1, -0.2, 0.3, -0.4], [2, 2]),
  ]);
  assert.equal(r.hillSurface.sigma, null);
  assert.equal(r.hillSurface.residual.length, 4);
  assert.ok(Number.isFinite(r.hillSurface.residualLim) && r.hillSurface.residualLim > 0);
  assert.deepEqual(r.hillSurface.gx, [0, 1]);
});

test('silhouette json is read when present', async () => {
  const r = await readRun([jsonFile('meta.json', META), npyFile('volume.npy', vol, [2, 2, 2]), jsonFile('hill_silhouette.json', { fans: [] })]);
  assert.deepEqual(r.silhouette, { fans: [] });
});

test('readCompareVolume: no meta, grid mismatch, match', async () => {
  assert.equal(await readCompareVolume([npyFile('volume.npy', vol, [2, 2, 2])], META), null);
  const moved = await readCompareVolume([jsonFile('meta.json', { ...META, origin_m: [1, 0, 0] }), npyFile('volume.npy', vol, [2, 2, 2])], META);
  assert.deepEqual(moved, { mismatch: true });
  const ok = await readCompareVolume([jsonFile('meta.json', META), npyFile('volume.npy', vol, [2, 2, 2])], META);
  assert.equal(ok.volume.length, 8);
  await assert.rejects(readCompareVolume([jsonFile('meta.json', META)], META), /no volume\.npy/);
});
