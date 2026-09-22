// World Z is the physical vertical (up). The orbit is parameterized with
// pitch = elevation above the horizontal xy-plane and yaw = azimuth around
// z, so the camera sits on a sphere around `target` with z as the pole.
export function orbitToEye(target, yaw, pitch, distance) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    target[0] + distance * cp * cy,
    target[1] + distance * cp * sy,
    target[2] + distance * sp,
  ];
}

// pitch is kept just off +-PI/2 for top/bottom-like views so the degenerate
// straight-down/up case is handled by render()'s up-vector fallback
// (see PITCH_GIMBAL_LIMIT in app.mjs), not by relying on an exact pole value.
export const CAMERA_PRESETS = {
  // Map view: looking straight down -z (the up-vector fallback in render()
  // handles the near-pole gimbal here).
  top: { yaw: 0, pitch: Math.PI / 2 - 0.001 },
  // Elevation view: eye on +y, looking toward -y, z up.
  front: { yaw: Math.PI / 2, pitch: 0 },
  // Elevation view: eye on +x, looking toward -x, z up.
  side: { yaw: 0, pitch: 0 },
  // Angled view.
  iso: { yaw: 0.7, pitch: 0.5 },
  // Default view: detectors (z=0) low in frame, reconstructed rock above.
  observation: { yaw: 0.6, pitch: 0.30 },
};
