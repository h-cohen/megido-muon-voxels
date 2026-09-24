export const KNOWN_LAYERS = {
  volume: { label: 'Combined solve', kind: 'scalar' },
  sigma: { label: 'Uncertainty (sigma)', kind: 'scalar' },
  snr: { label: 'SNR', kind: 'scalar' },
  views: { label: 'View count', kind: 'scalar' },
  rays: { label: 'Ray count', kind: 'scalar' },
  systematic: { label: 'Gauge systematic', kind: 'signed' },
  backprojection: { label: 'Backprojection', kind: 'scalar' },
  volume_holdout_pos0: { label: 'Holdout: pos0 removed', kind: 'scalar' },
  volume_holdout_pos1: { label: 'Holdout: pos1 removed', kind: 'scalar' },
  phantom: { label: 'Phantom truth', kind: 'scalar' },
  delta: { label: 'Run delta (B minus A)', kind: 'signed' },
};

export function availableLayers(layerNames) {
  return layerNames.map((key) => {
    const known = KNOWN_LAYERS[key];
    return known ? { key, label: known.label, kind: known.kind }
                 : { key, label: key, kind: 'scalar' };
  });
}
