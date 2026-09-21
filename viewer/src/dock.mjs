const NS = 'megido-viewer:section:';

export function sectionStorageKey(id) {
  return NS + id;
}

export function loadSectionState(id, fallbackOpen = true) {
  try {
    const raw = localStorage.getItem(sectionStorageKey(id));
    if (raw === null) return fallbackOpen;
    return raw === '1';
  } catch {
    return fallbackOpen;
  }
}

export function saveSectionState(id, isOpen) {
  try {
    localStorage.setItem(sectionStorageKey(id), isOpen ? '1' : '0');
  } catch {
    /* per-viewer convenience only; ignore storage failures */
  }
}

export function initDock(root) {
  const sections = root.querySelectorAll('[data-section]');
  for (const sec of sections) {
    const id = sec.getAttribute('data-section');
    const header = sec.querySelector('.section-header');
    const open = loadSectionState(id, sec.getAttribute('data-default-open') !== 'false');
    sec.setAttribute('data-open', open ? 'true' : 'false');
    if (header) {
      header.addEventListener('click', () => {
        const next = sec.getAttribute('data-open') !== 'true';
        sec.setAttribute('data-open', next ? 'true' : 'false');
        saveSectionState(id, next);
      });
    }
  }
}
