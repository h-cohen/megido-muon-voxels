const KEY = 'megido-viewer:views';

export function captureView(state, name) {
  const c = state.camera;
  return {
    name: name || 'view',
    camera: { yaw: c.yaw, pitch: c.pitch, distance: c.distance, target: [...c.target] },
    window: [state.window[0], state.window[1]],
    activeLayer: state.activeLayer,
  };
}

export function serializeViews(list) {
  return JSON.stringify(list);
}

export function deserializeViews(str) {
  try {
    const v = JSON.parse(str);
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

export function loadViews() {
  try { return deserializeViews(localStorage.getItem(KEY) || '[]'); } catch { return []; }
}

export function saveViews(list) {
  try { localStorage.setItem(KEY, serializeViews(list)); } catch { /* ignore */ }
}
