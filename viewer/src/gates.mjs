import { insideClipBox, insideClipPlane } from './clip.mjs';
import { surfaceHeightAt } from './surfacemesh.mjs';

// The voxel gate set: the ONE reference for which voxels the viewer hides.
// Every CPU consumer (hover picking, the cube-size merge, the auto window)
// asks this module, and render() uploads `uniforms` for the two GLSL copies
// (voxelGated and the fog march in app.mjs), which must follow the same rules.
// tests/viewer/test_gate_parity.py checks shader against this, gate by gate.
//
// NaN semantics, defined once:
//   sigma NaN -> kept   (no bootstrap sigma is not "too uncertain")
//   rays  NaN -> kept   (the shader's rule; rays.npy is counts, never NaN)
//   snr   NaN -> hidden (the bootstrap said nothing: not significant)
// A gate is inert when it is off OR its layer is not loaded.
export function voxelGates(settings, layers) {
  const sigma = settings.sigmaGateEnabled && layers.sigma ? layers.sigma : null;
  const rays = settings.coverageGateEnabled && layers.rays ? layers.rays : null;
  const snr = settings.snrGateEnabled && layers.snr ? layers.snr : null;
  const sigmaValue = settings.sigmaGateValue, minRays = settings.minRays, minSnr = settings.minSnr;
  // The auto window deliberately ignores the sigma gate: sigma is a
  // continuous slider, and re-windowing on every drag would shift the colours
  // under the user's hand. Coverage and SNR hide the noise shell whose
  // inflated values would otherwise set the colour scale.
  const keepForWindow = (n) => (!rays || !(rays[n] < minRays)) && (!snr || snr[n] >= minSnr);
  const keep = (n) => (!sigma || !(sigma[n] > sigmaValue)) && keepForWindow(n);
  return {
    keep,
    keepForWindow,
    windowActive: !!(rays || snr),
    uniforms: {
      sigmaEnabled: !!sigma, sigmaValue,
      coverageEnabled: !!rays, minRays,
      snrEnabled: !!snr, minSnr,
    },
    key: [!!sigma, sigmaValue, !!rays, minRays, !!snr, minSnr].join('|'),
  };
}

// Geometric gates at a sample: clip box, clip plane, then surface clip, the
// shader's order. `tex` is the [0,1]^3 volume coordinate, `world` metres.
// `surface` is the DISPLAYED (possibly smoothed) height field {H, gx, gy}.
export function spatialGates(settings, surface) {
  const surf = settings.surfClip && surface && surface.H ? surface : null;
  return {
    keep(tex, world) {
      if (!insideClipBox(tex, settings.clipMin, settings.clipMax)) return false;
      if (settings.clipPlaneEnabled && !insideClipPlane(tex, settings.clipPlaneNormal, settings.clipPlaneD)) return false;
      if (surf) {
        const h = surfaceHeightAt(surf.H, surf.gx, surf.gy, world[0], world[1]);
        if (Number.isFinite(h) && world[2] > h) return false;
      }
      return true;
    },
  };
}
