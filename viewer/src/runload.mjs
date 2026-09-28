import { parseNpy } from './npy.mjs';
import { symmetricLimit } from './surfacemesh.mjs';

// Reads a run directory (File-like objects: name, text(), arrayBuffer()) into
// one dataset record. Pure apart from the file reads; app.mjs applies the
// record to GL, DOM and state. Every optional artifact is null when absent,
// never an error: older and synthetic runs lack them.
export async function readRun(files) {
  const byName = new Map();
  for (const f of files) byName.set(f.name, f);
  const metaFile = byName.get('meta.json');
  if (!metaFile) throw new Error('selected directory has no meta.json');
  const meta = JSON.parse(await metaFile.text());
  const npyData = async (f) => parseNpy(await f.arrayBuffer()).data;

  const silhouetteFile = byName.get('hill_silhouette.json');
  const silhouette = silhouetteFile ? JSON.parse(await silhouetteFile.text()) : null;

  // The fitted hillside surface needs BOTH its heights and its meta. Sigma
  // and residual grids are dropped unless they match the surface's size.
  let hillSurface = null;
  const surfFile = byName.get('hill_surface.npy');
  const surfMetaFile = byName.get('hill_surface_meta.json');
  if (surfFile && surfMetaFile) {
    const H = await npyData(surfFile);
    const hillMeta = JSON.parse(await surfMetaFile.text());
    const sameSize = async (name) => {
      const f = byName.get(name);
      if (!f) return null;
      const d = await npyData(f);
      return d.length === H.length ? d : null;
    };
    const sigma = await sameSize('hill_surface_sigma.npy');
    const residual = await sameSize('hill_residual_grid.npy');
    // The CLI records residual_grid_lim; when it is null (no finite residual
    // at write time, or an older run) derive the same 98th-percentile |r|
    // scale from the data rather than a meaningless default.
    const residualLim = Number.isFinite(hillMeta.residual_grid_lim)
      ? hillMeta.residual_grid_lim
      : (residual ? symmetricLimit(residual) : NaN);
    hillSurface = { H, gx: hillMeta.gx, gy: hillMeta.gy, meta: hillMeta, sigma, residual, residualLim };
  }

  // Only layers with one value per voxel are raymarch-able; lower-dimensional
  // diagnostics (e.g. a 2D backprojection plane) share meta.layers and are
  // skipped here, correctly, not reported as errors.
  const voxelCount = meta.shape[0] * meta.shape[1] * meta.shape[2];
  const layers = new Map();
  for (const name of meta.layers) {
    const f = byName.get(`${name}.npy`);
    if (!f) continue;
    const data = await npyData(f);
    if (data.length === voxelCount) layers.set(name, data);
  }

  // Plain loop, not Math.max(...sig): a spread blows the call stack on the
  // real campaign's 675,840-element sigma array.
  let sigmaMax = 1;
  if (layers.has('sigma')) {
    const sig = layers.get('sigma');
    sigmaMax = 0;
    for (let i = 0; i < sig.length; i++) if (sig[i] > sigmaMax) sigmaMax = sig[i];
  }

  const iso = Array.isArray(meta.suggested_iso) ? meta.suggested_iso[0] : null;
  const vr = Array.isArray(meta.value_range) ? meta.value_range : [0, 1];
  const cubeThreshold = Number.isFinite(iso) ? iso : 0.5 * (vr[0] + vr[1]);

  return { meta, silhouette, hillSurface, layers, sigmaMax, cubeThreshold };
}

// "Load compare": null when the directory has no meta.json; {mismatch: true}
// when the grid differs (shape + spacing + origin, as compare_volumes in
// megido/volexport.py keys it); otherwise {volume}.
export async function readCompareVolume(files, primaryMeta) {
  const byName = new Map(files.map((f) => [f.name, f]));
  const metaFile = byName.get('meta.json');
  if (!metaFile) return null;
  const second = JSON.parse(await metaFile.text());
  const mismatch =
    JSON.stringify(second.shape) !== JSON.stringify(primaryMeta.shape) ||
    second.spacing_m !== primaryMeta.spacing_m ||
    JSON.stringify(second.origin_m) !== JSON.stringify(primaryMeta.origin_m);
  if (mismatch) return { mismatch: true };
  const volFile = byName.get('volume.npy');
  if (!volFile) throw new Error('compare run has no volume.npy');
  return { volume: parseNpy(await volFile.arrayBuffer()).data };
}
