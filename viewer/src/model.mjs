import { CAMERA_PRESETS } from './camera.mjs';
import { colormapStops } from './colormap.mjs';

// The viewer's display state: defaults, and which follow-ups each field needs
// when it changes. app.mjs owns the GL/DOM side; commit() there runs the
// effects this table names, in EFFECT_ORDER.
export const EFFECT_ORDER = ['filter', 'surfaceMesh', 'legend', 'window', 'histogram', 'lut', 'render'];

export function createState() {
  return {
    meta: null,
    layerData: new Map(),
    activeLayer: null,
    // Default view is the "observation" preset (see camera.mjs
    // CAMERA_PRESETS.observation): z-up, detectors (z=0) low in frame,
    // reconstructed rock rising above them.
    camera: {
      yaw: CAMERA_PRESETS.observation.yaw,
      pitch: CAMERA_PRESETS.observation.pitch,
      distance: 3,
      target: [0, 0, 0],
    },
    transferStops: colormapStops('viridis'),
    clipMin: [0, 0, 0],
    clipMax: [1, 1, 1],
    clipPlaneEnabled: false,
    clipPlaneNormal: [0, 0, 1],
    clipPlaneD: 0,
    sigmaGateEnabled: false,
    sigmaGateValue: 1e9,
    // Coverage gate (display-only): hide voxels crossed by fewer than minRays
    // measured directions. On by default, but only live when the run ships a
    // `rays` layer (hasRays) -- the dummy texture reads 0 and would hide all.
    coverageGateEnabled: true,
    minRays: 2,
    hasRays: false,
    // SNR gate (display-only): hide voxels whose bootstrap SNR < minSnr. On
    // by default, live only when the run ships an `snr` layer.
    snrGateEnabled: true,
    minSnr: 3,
    hasSnr: false,
    // 'fog' (density raymarch, default) or 'cubes' (opaque voxel blocks >=
    // cubeThreshold; display-only). cubeThreshold defaults to the run's
    // meta.suggested_iso[0] on load.
    renderMode: 'fog',
    cubeThreshold: 0,
    opacity: 1, // display-only multiplier on voxel opacity (fog alpha / cube translucency)
    // Cube size: b x b x b voxels merged into one display cube (1 = the solved
    // voxels). Built lazily by ensureCubeGrid(), keyed on everything it reads.
    cubeBlock: 1,
    cubeGrid: null,
    cubeGridKey: '',
    loadSeq: 0,
    smoothSampling: true,
    shading: true,
    adaptiveQuality: true, // fast preview (fewer steps, no shading) while interacting
    interacting: false,
    minStepVoxels: 0.5, // test hook: minimum march step, in voxels (uMinStepVoxels)
    detectors: [],
    detectorLabels: [],
    showDetectors: false,
    markerVertexCount: 0,
    silhouette: null,
    showSilhouette: false,
    silhouetteRanges: [],
    hillSurface: null,
    showHillSurface: false,
    hillSurfaceIndexCount: 0,
    hillSurfaceSmooth: 0, // display-only smoothing pass count; 0 = raw fit
    hillColourMode: 'flat', // 'sigma' | 'residual' | 'flat' - reset by loadRun
    hillSigmaRange: null, // [lo, hi] robust range of sigma, set by loadRun
    surfClip: false, // display-only: skip volume samples above the fitted surface
    hillDisplayH: null, // the DISPLAYED (possibly smoothed) height field the clip must match
    surfTex: null,
    surfMin: [0, 0],
    surfStep: [1, 1],
    surfSize: [0, 0],
    window: null,
  };
}

const R = ['render'];
export const EFFECTS = {
  camera: R,
  clipMin: R, clipMax: R, clipPlaneEnabled: R, clipPlaneD: R,
  // The sigma gate is a continuous slider: it never re-windows (gates.mjs).
  sigmaGateEnabled: R, sigmaGateValue: R,
  coverageGateEnabled: ['window', 'render'], minRays: ['window', 'render'],
  snrGateEnabled: ['window', 'render'], minSnr: ['window', 'render'],
  renderMode: R, cubeThreshold: R, cubeBlock: R, opacity: R,
  smoothSampling: ['filter', 'render'],
  shading: R, adaptiveQuality: R,
  showDetectors: R, showSilhouette: R, showHillSurface: R, surfClip: R,
  hillColourMode: ['legend', 'surfaceMesh', 'render'],
  hillSurfaceSmooth: ['surfaceMesh', 'render'],
  window: ['histogram', 'render'],
  transferStops: ['lut', 'render'],
};

export function effectsFor(patch) {
  const wanted = new Set();
  for (const field of Object.keys(patch)) {
    const effects = EFFECTS[field];
    if (!effects) throw new Error(`no effects declared for state field: ${field}`);
    for (const e of effects) wanted.add(e);
  }
  return EFFECT_ORDER.filter((e) => wanted.has(e));
}
