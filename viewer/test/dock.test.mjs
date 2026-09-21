import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sectionStorageKey, loadSectionState, saveSectionState } from '../src/dock.mjs';

// Minimal localStorage shim for Node.
function withStorage(fn) {
  const store = new Map();
  globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  try { fn(); } finally { delete globalThis.localStorage; }
}

test('sectionStorageKey namespaces the id', () => {
  assert.match(sectionStorageKey('sec-layers'), /sec-layers/);
});

test('loadSectionState returns the fallback when nothing stored', () => {
  withStorage(() => {
    assert.equal(loadSectionState('sec-x', true), true);
    assert.equal(loadSectionState('sec-x', false), false);
  });
});

test('saveSectionState round-trips', () => {
  withStorage(() => {
    saveSectionState('sec-y', false);
    assert.equal(loadSectionState('sec-y', true), false);
    saveSectionState('sec-y', true);
    assert.equal(loadSectionState('sec-y', true), true);
  });
});

test('loadSectionState survives a throwing localStorage', () => {
  const prev = globalThis.localStorage;
  globalThis.localStorage = { getItem() { throw new Error('blocked'); } };
  try {
    assert.equal(loadSectionState('sec-z', true), true); // fallback, no throw
  } finally {
    if (prev === undefined) delete globalThis.localStorage; else globalThis.localStorage = prev;
  }
});
