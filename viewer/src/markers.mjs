// Detector position markers: a small 3-axis cross per detector, in world
// space (metres), so results can be referenced back to the instrument.
// Pure geometry only - no GL here; app.mjs owns the buffer upload and draw.

export const MARKER_HALF_M = 0.6;

// markerVertices(detectors) -> Float32Array
// detectors: [{x, y, z}, ...] (meta.json's `detectors` list; extra fields
// such as `id` are ignored).
// Returns a flat XYZ line-list: for each detector, 3 axes x 2 endpoints x 3
// coords = 18 floats, suitable for gl.LINES.
export function markerVertices(detectors) {
  const out = new Float32Array(detectors.length * 6 * 3);
  let o = 0;
  for (const d of detectors) {
    const { x, y, z } = d;
    // x-arm
    out[o++] = x - MARKER_HALF_M; out[o++] = y; out[o++] = z;
    out[o++] = x + MARKER_HALF_M; out[o++] = y; out[o++] = z;
    // y-arm
    out[o++] = x; out[o++] = y - MARKER_HALF_M; out[o++] = z;
    out[o++] = x; out[o++] = y + MARKER_HALF_M; out[o++] = z;
    // z-arm
    out[o++] = x; out[o++] = y; out[o++] = z - MARKER_HALF_M;
    out[o++] = x; out[o++] = y; out[o++] = z + MARKER_HALF_M;
  }
  return out;
}
