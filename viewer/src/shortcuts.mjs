export const SHORTCUTS = [
  { keys: '1–5', action: 'layerN', label: 'Switch layer' },
  { keys: 't/f/s/i', action: 'preset', label: 'Camera preset (top/front/side/iso)' },
  { keys: 'r', action: 'frame-all', label: 'Frame all' },
  { keys: '?', action: 'toggle-help', label: 'Toggle this help' },
  { keys: 'Esc', action: 'close-overlay', label: 'Close overlay' },
];

export function keyToAction(ev) {
  const k = ev.key;
  if (k >= '1' && k <= '5') return 'layer' + k;
  if (k === 't') return 'preset-top';
  if (k === 'f') return 'preset-front';
  if (k === 's') return 'preset-side';
  if (k === 'i') return 'preset-iso';
  if (k === 'r') return 'frame-all';
  if (k === '?') return 'toggle-help';
  if (k === 'Escape') return 'close-overlay';
  return null;
}
