export function orbitToEye(target, yaw, pitch, distance) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [
    target[0] + distance * cp * sy,
    target[1] + distance * sp,
    target[2] + distance * cp * cy,
  ];
}

// pitch is kept just off +-PI/2 for top/bottom-like views so lookAt's
// up vector (0,1,0) never goes parallel to the eye-to-target axis.
export const CAMERA_PRESETS = {
  top: { yaw: 0, pitch: Math.PI / 2 - 0.001 },
  front: { yaw: 0, pitch: 0 },
  side: { yaw: Math.PI / 2, pitch: 0 },
  iso: { yaw: Math.PI / 4, pitch: Math.PI / 5 },
};
