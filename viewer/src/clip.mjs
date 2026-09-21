export function insideClipBox(p, boxMin, boxMax) {
  return p[0] >= boxMin[0] && p[0] <= boxMax[0]
      && p[1] >= boxMin[1] && p[1] <= boxMax[1]
      && p[2] >= boxMin[2] && p[2] <= boxMax[2];
}

// Matches the shader's convention (see app.mjs FRAGMENT_SRC):
//   float d = dot(tex - vec3(0.5), uClipPlaneNormal) - uClipPlaneD;
//   clipped = clipped || d < 0.0;
// i.e. the plane passes through the box center (0.5,0.5,0.5) offset by
// planeD along normal, in [0,1]^3 texture space, and "inside" is d >= 0.
export function clipPlaneDistance(p, normal, planeD) {
  const rel = [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
  const dot = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
  return dot - planeD;
}

export function insideClipPlane(p, normal, planeD) {
  return clipPlaneDistance(p, normal, planeD) >= 0;
}
