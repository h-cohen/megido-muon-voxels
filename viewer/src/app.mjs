import { parseNpy } from './npy.mjs';
import { identity, multiply, perspective, lookAt, invert } from './mat4.mjs';
import { worldToVoxel, sampleNearest } from './grid.mjs';
import { insideClipBox, insideClipPlane } from './clip.mjs';
import { buildTransferLUT } from './transfer.mjs';
import { orbitToEye, CAMERA_PRESETS } from './camera.mjs';
import { computeHistogram, robustWindow, windowToBandPx, bandPxToWindow } from './histogram.mjs';
import { availableLayers } from './layers.mjs';
import { computeDelta, deltaVerdict } from './delta.mjs';
import { initDock } from './dock.mjs';
import { COLORMAP_NAMES, colormapStops } from './colormap.mjs';
import { captureView, loadViews, saveViews } from './views.mjs';

const VERTEX_SRC = `#version 300 es
out vec2 vUv;
void main() {
  vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = pos;
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;

uniform mat4 uInvViewProj;
uniform vec3 uCameraPos;
uniform sampler3D uVolume;
uniform sampler3D uSigmaTex;
uniform sampler2D uTransferLUT;
uniform vec2 uWindow;        // [lo, hi] density remap before LUT lookup
uniform vec3 uClipMin;
uniform vec3 uClipMax;
uniform vec3 uWorldMin;
uniform vec3 uWorldExtent;
uniform bool uClipPlaneEnabled;
uniform vec3 uClipPlaneNormal;
uniform float uClipPlaneD;
uniform bool uSigmaGateEnabled;
uniform float uSigmaGateValue;
uniform int uSteps;

vec3 unproject(vec2 ndc, float z) {
  vec4 clip = vec4(ndc, z, 1.0);
  vec4 world = uInvViewProj * clip;
  return world.xyz / world.w;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 nearP = unproject(ndc, -1.0);
  vec3 farP = unproject(ndc, 1.0);
  vec3 dir = normalize(farP - nearP);

  float stepLen = length(farP - nearP) / float(uSteps);
  vec3 pos = nearP;
  vec4 accum = vec4(0.0);

  for (int i = 0; i < 512; i++) {
    if (i >= uSteps || accum.a > 0.98) break;
    vec3 tex = (pos - uWorldMin) / uWorldExtent;
    if (tex.x >= 0.0 && tex.x <= 1.0 && tex.y >= 0.0 && tex.y <= 1.0 && tex.z >= 0.0 && tex.z <= 1.0) {
      bool clipped = any(lessThan(tex, uClipMin)) || any(greaterThan(tex, uClipMax));
      if (uClipPlaneEnabled) {
        float d = dot(tex - vec3(0.5), uClipPlaneNormal) - uClipPlaneD;
        clipped = clipped || d < 0.0;
      }
      if (uSigmaGateEnabled) {
        float sigma = texture(uSigmaTex, tex).r;
        clipped = clipped || sigma > uSigmaGateValue;
      }
      if (!clipped) {
        float density = texture(uVolume, tex).r;
        float t = clamp((density - uWindow.x) / max(uWindow.y - uWindow.x, 1e-6), 0.0, 1.0);
        vec4 c = texture(uTransferLUT, vec2(t, 0.5));
        c.rgb *= c.a;
        accum += (1.0 - accum.a) * c;
      }
    }
    pos += dir * stepLen;
  }

  outColor = accum;
}`;

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile error: ${log}`);
  }
  return sh;
}

function linkProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`program link error: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

function makeVolumeTexture(gl, shape, data) {
  const [nx, ny, nz] = shape;
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  // NEAREST filtering avoids depending on OES_texture_float_linear, which
  // is not guaranteed on every WebGL2 implementation.
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, nx, ny, nz, 0, gl.RED, gl.FLOAT, data);
  return tex;
}

function makeLutTexture(gl, lutBytes) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lutBytes.length / 4, 1, 0,
                gl.RGBA, gl.UNSIGNED_BYTE, lutBytes);
  return tex;
}

export function initViewer(root) {
  const canvas = root.querySelector('#gl-canvas');
  const fileInput = root.querySelector('#load-run-input');
  // preserveDrawingBuffer: true so toDataURL()/screenshot readback after a
  // draw call sees the frame just rendered, not a backbuffer the browser
  // has already cleared for the next composite.
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 is required');
  gl.getExtension('EXT_color_buffer_float');

  const program = linkProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
  const vao = gl.createVertexArray();

  const uniforms = {};
  for (const name of [
    'uInvViewProj', 'uCameraPos', 'uVolume', 'uSigmaTex', 'uTransferLUT',
    'uWindow', 'uClipMin', 'uClipMax', 'uClipPlaneEnabled', 'uClipPlaneNormal',
    'uClipPlaneD', 'uSigmaGateEnabled', 'uSigmaGateValue', 'uSteps',
    'uWorldMin', 'uWorldExtent',
  ]) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }

  const dummyVolume = makeVolumeTexture(gl, [1, 1, 1], new Float32Array([0]));
  const state = {
    meta: null,
    layerData: new Map(),
    activeLayer: null,
    gl, program, uniforms,
    camera: { yaw: 0.6, pitch: 0.5, distance: 3, target: [0, 0, 0] },
    transferStops: colormapStops('viridis'),
    clipMin: [0, 0, 0],
    clipMax: [1, 1, 1],
    clipPlaneEnabled: false,
    clipPlaneNormal: [0, 0, 1],
    clipPlaneD: 0,
    sigmaGateEnabled: false,
    sigmaGateValue: 1e9,
    volumeTex: dummyVolume,
    sigmaTex: dummyVolume,
    lutTex: makeLutTexture(gl, buildTransferLUT(colormapStops('viridis'))),
  };

  function worldBounds() {
    if (!state.meta) return { min: [0, 0, 0], extent: [1, 1, 1] };
    const { shape, spacing_m, origin_m } = state.meta;
    const extent = shape.map((n) => n * spacing_m);
    return { min: origin_m, extent };
  }

  function render() {
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * (window.devicePixelRatio || 1)));
    canvas.height = Math.max(1, Math.round(height * (window.devicePixelRatio || 1)));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.07, 0.07, 0.09, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { yaw, pitch, distance, target } = state.camera;
    const eye = orbitToEye(target, yaw, pitch, distance);
    const view = lookAt(eye, target, [0, 1, 0]);
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 100);
    const viewProj = multiply(proj, view);
    const invViewProj = invert(viewProj) || identity();

    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniformMatrix4fv(uniforms.uInvViewProj, false, invViewProj);
    gl.uniform3fv(uniforms.uCameraPos, eye);

    const { min, extent } = worldBounds();
    gl.uniform3fv(uniforms.uWorldMin, min);
    gl.uniform3fv(uniforms.uWorldExtent, extent);
    gl.uniform2fv(uniforms.uWindow, state.window || [0, 1]);
    gl.uniform3fv(uniforms.uClipMin, state.clipMin);
    gl.uniform3fv(uniforms.uClipMax, state.clipMax);
    gl.uniform1i(uniforms.uClipPlaneEnabled, state.clipPlaneEnabled ? 1 : 0);
    gl.uniform3fv(uniforms.uClipPlaneNormal, state.clipPlaneNormal);
    gl.uniform1f(uniforms.uClipPlaneD, state.clipPlaneD);
    gl.uniform1i(uniforms.uSigmaGateEnabled, state.sigmaGateEnabled ? 1 : 0);
    gl.uniform1f(uniforms.uSigmaGateValue, state.sigmaGateValue);
    gl.uniform1i(uniforms.uSteps, 200);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, state.volumeTex);
    gl.uniform1i(uniforms.uVolume, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, state.sigmaTex);
    gl.uniform1i(uniforms.uSigmaTex, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, state.lutTex);
    gl.uniform1i(uniforms.uTransferLUT, 2);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    drawGizmo();
  }
  state.render = render;

  // Axis-orientation gizmo (bottom-left): projects the three world axes
  // into view space using the SAME yaw/pitch as render()'s camera, so the
  // triad always matches what's on screen. Distance/target don't affect
  // direction, so we reuse orbitToEye/lookAt with target=[0,0,0],
  // distance=1 purely to get the rotation basis.
  function drawGizmo() {
    const gizmoCanvas = root.querySelector('#gizmo-canvas');
    if (!gizmoCanvas) return;
    const ctx = gizmoCanvas.getContext('2d');
    const w = gizmoCanvas.width, h = gizmoCanvas.height;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.32;

    const { yaw, pitch } = state.camera;
    const eye = orbitToEye([0, 0, 0], yaw, pitch, 1);
    const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    // view[0..2] = view-space coords of world X axis, view[4..6] of world
    // Y, view[8..10] of world Z (see mat4.mjs lookAt column layout).
    const axes = [
      { label: 'X', color: '#e5484d', dx: view[0], dy: view[1] },
      { label: 'Y', color: '#3fb950', dx: view[4], dy: view[5] },
      { label: 'Z', color: '#4d9de5', dx: view[8], dy: view[9] },
    ];
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const axis of axes) {
      // Canvas y grows downward; view-space y grows upward.
      const ex = cx + axis.dx * r, ey = cy - axis.dy * r;
      ctx.strokeStyle = axis.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.fillStyle = axis.color;
      ctx.fillText(axis.label, cx + axis.dx * (r + 10), cy - axis.dy * (r + 10));
    }
  }
  state.drawGizmo = drawGizmo;

  function rebuildLut() {
    gl.deleteTexture(state.lutTex);
    state.lutTex = makeLutTexture(gl, buildTransferLUT(state.transferStops));
    render();
  }

  function drawHistogram() {
    const canvas = root.querySelector('#histogram-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const data = state.layerData.get(state.activeLayer);
    const readout = root.querySelector('#window-readout');
    if (!data || !state.window) return;
    const [lo, hi] = state.window;
    const hist = computeHistogram(data, lo, hi, 64);
    const max = Math.max(...hist, 1);
    const barW = canvas.width / hist.length;
    ctx.fillStyle = '#8cf';
    for (let i = 0; i < hist.length; i++) {
      const h = (hist[i] / max) * canvas.height;
      ctx.fillRect(i * barW, canvas.height - h, barW - 1, h);
    }

    // The window band IS the control: draw it as a translucent accent
    // rectangle with two edge handles over the bars, in the same pixel
    // space #histogram-canvas pointer events are converted into below.
    const { loPx, hiPx } = windowToBandPx(state.window, state.layerMax, canvas.width);
    ctx.fillStyle = 'rgba(140, 204, 255, 0.18)';
    ctx.fillRect(loPx, 0, hiPx - loPx, canvas.height);
    ctx.fillStyle = 'rgba(140, 204, 255, 0.9)';
    ctx.fillRect(loPx - 1.5, 0, 3, canvas.height);
    ctx.fillRect(hiPx - 1.5, 0, 3, canvas.height);

    if (readout) {
      readout.textContent = `${state.window[0].toFixed(3)} – ${state.window[1].toFixed(3)} 1/m`;
    }
    drawLegend();
  }

  // Colorbar legend (bottom-right): fills #legend-canvas with the current
  // transfer LUT and labels #legend-ticks with the active window's lo/mid/hi.
  // Called from drawHistogram()/drawXferEditor() so it stays in sync with
  // every place the LUT or window changes (colormap, window drag, layer
  // switch, initial load).
  function drawLegend() {
    const canvas = root.querySelector('#legend-canvas');
    const ticksEl = root.querySelector('#legend-ticks');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const lut = buildTransferLUT(state.transferStops, canvas.width);
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let x = 0; x < canvas.width; x++) {
      for (let y = 0; y < canvas.height; y++) {
        const idx = (y * canvas.width + x) * 4;
        img.data[idx] = lut[x * 4]; img.data[idx + 1] = lut[x * 4 + 1];
        img.data[idx + 2] = lut[x * 4 + 2]; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    if (ticksEl) {
      const w = state.window || [0, 1];
      const mid = (w[0] + w[1]) / 2;
      const unit = state.activeLayer === 'volume' ? ' 1/m' : '';
      const fmt = (v) => v.toFixed(3) + unit;
      ticksEl.textContent = `${fmt(w[0])}  ${fmt(mid)}  ${fmt(w[1])}`;
    }
  }
  state.drawLegend = drawLegend;

  function drawXferEditor() {
    const canvas = root.querySelector('#xfer-canvas');
    const ctx = canvas.getContext('2d');
    const lut = buildTransferLUT(state.transferStops, canvas.width);
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let x = 0; x < canvas.width; x++) {
      for (let y = 0; y < canvas.height; y++) {
        const idx = (y * canvas.width + x) * 4;
        img.data[idx] = lut[x * 4]; img.data[idx + 1] = lut[x * 4 + 1];
        img.data[idx + 2] = lut[x * 4 + 2]; img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.fillStyle = '#fff';
    for (const s of state.transferStops) {
      ctx.fillRect(s.t * canvas.width - 2, 0, 4, canvas.height);
    }
    drawLegend();
  }
  state.drawHistogram = drawHistogram;
  state.drawXferEditor = drawXferEditor;

  async function readFile(file) {
    return file.arrayBuffer();
  }

  // Frame the camera on the grid's world-space center, sized to the volume's
  // diagonal. Shared by loadRun (on every fresh run) and #frame-all-btn (to
  // recover the view after the user has panned/zoomed away), so the framing
  // math lives in exactly one place.
  function frameAll() {
    if (!state.meta) return;
    const { shape, spacing_m, origin_m } = state.meta;
    state.camera.target = [
      origin_m[0] + 0.5 * shape[0] * spacing_m,
      origin_m[1] + 0.5 * shape[1] * spacing_m,
      origin_m[2] + 0.5 * shape[2] * spacing_m,
    ];
    const diag = Math.hypot(shape[0] * spacing_m, shape[1] * spacing_m, shape[2] * spacing_m);
    state.camera.distance = 1.3 * diag;
  }

  async function loadRun(files) {
    state.ready = false;
    const byName = new Map();
    for (const f of files) byName.set(f.name, f);

    const metaFile = byName.get('meta.json');
    if (!metaFile) throw new Error('selected directory has no meta.json');
    const meta = JSON.parse(await metaFile.text());
    state.meta = meta;

    const banner = root.querySelector('#resolution-banner');
    const res = meta.resolution || {};
    banner.textContent = res.verdict || '';
    banner.classList.toggle('not-resolved', !res.depth_resolved);
    const runNameEl = root.querySelector('#run-name');
    if (runNameEl) runNameEl.textContent = meta.run || '';

    // Only layers whose element count matches the volume grid (nx*ny*nz) are
    // raymarch-able. The exporter also writes lower-dimensional diagnostic
    // layers (e.g. a 2D backprojection plane) under the same meta.layers
    // list; those are silently skipped here, not offered as a radio, and
    // never treated as an error — they are correctly not volume layers.
    const voxelCount = meta.shape[0] * meta.shape[1] * meta.shape[2];
    state.layerData.clear();
    for (const name of meta.layers) {
      const file = byName.get(`${name}.npy`);
      if (!file) continue;
      const { data } = parseNpy(await readFile(file));
      if (data.length !== voxelCount) continue;
      state.layerData.set(name, data);
    }

    // Reset any stale delta from a previously loaded compare run: a new
    // primary load makes the old 'B minus A' comparison meaningless.
    state.layerData.delete('delta');
    const staleDeltaRadio = root.querySelector('#layer-delta');
    if (staleDeltaRadio) staleDeltaRadio.closest('label')?.remove();
    const deltaVerdictEl = root.querySelector('#delta-verdict');
    if (deltaVerdictEl) deltaVerdictEl.textContent = '';

    // Frame the camera on the grid's world-space center, sized to the
    // volume's diagonal, instead of the fixed target=[0,0,0]/distance=3
    // defaults, which orphan the (typically off-origin, many-metre) real
    // campaign volume off-screen or reduced to a speck.
    frameAll();

    state.activeLayer = 'volume';
    state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('volume'));
    if (state.layerData.has('sigma')) {
      const sig = state.layerData.get('sigma');
      state.sigmaTex = makeVolumeTexture(gl, meta.shape, sig);
      // Plain loop, not Math.max(...sig): a spread blows the call stack on
      // the real campaign's 675,840-element sigma array.
      let smax = 0;
      for (let i = 0; i < sig.length; i++) if (sig[i] > smax) smax = sig[i];
      state.sigmaMax = smax;
    } else {
      state.sigmaMax = 1;
    }

    // Window each layer to ITS OWN robust range (see histogram.mjs
    // robustWindow), not meta.value_range: value_range[1] is a single
    // outlier voxel on the real campaign (median 0.0025, p95 0.12,
    // value_range[1] 2.369), so windowing to [min,max] renders near-black.
    function syncWindowSliders() {
      // The histogram canvas's window band is the control now (see
      // drawHistogram); there are no sliders left to sync.
      drawHistogram();
    }
    function applyWindowForLayer(key) {
      const data = state.layerData.get(key);
      let max = 0;
      if (data) {
        for (let i = 0; i < data.length; i++) {
          const v = data[i];
          if (!Number.isNaN(v) && v > max) max = v;
        }
      }
      state.layerMax = max;
      state.window = data ? robustWindow(data) : [0, 1];
      syncWindowSliders();
    }
    state.applyWindowForLayer = applyWindowForLayer;

    function setActiveLayer(key) {
      state.activeLayer = key;
      gl.deleteTexture(state.volumeTex);
      state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get(key));
      applyWindowForLayer(key);
      drawHistogram();
      render();
    }
    state.setActiveLayer = setActiveLayer;

    const panel = root.querySelector('#layer-panel');
    panel.innerHTML = '';
    for (const layer of availableLayers([...state.layerData.keys()])) {
      const id = `layer-${layer.key}`;
      const label = document.createElement('label');
      label.style.display = 'block';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'active-layer';
      radio.id = id;
      radio.checked = layer.key === 'volume';
      radio.addEventListener('change', () => setActiveLayer(layer.key));
      label.appendChild(radio);
      label.appendChild(document.createTextNode(' ' + layer.label));
      panel.appendChild(label);
    }

    applyWindowForLayer('volume');
    drawHistogram();
    drawXferEditor();
    render();
    state.ready = true;
  }

  fileInput.addEventListener('change', (ev) => {
    loadRun(Array.from(ev.target.files)).catch((err) => {
      console.error(err);
      window.__viewerError = String(err);
    });
  });

  root.querySelector('#load-second-run-input').addEventListener('change', async (ev) => {
    const files = Array.from(ev.target.files);
    const byName = new Map(files.map((f) => [f.name, f]));
    const metaFile = byName.get('meta.json');
    if (!metaFile) return;
    if (!state.meta) {
      root.querySelector('#delta-verdict').textContent = 'load a primary run first';
      return;
    }
    const secondMeta = JSON.parse(await metaFile.text());
    // Full grid match, mirroring megido/volexport.py's compare_volumes,
    // which keys the grid on shape + spacing + origin, not shape alone.
    const gridMismatch =
      JSON.stringify(secondMeta.shape) !== JSON.stringify(state.meta.shape) ||
      secondMeta.spacing_m !== state.meta.spacing_m ||
      JSON.stringify(secondMeta.origin_m) !== JSON.stringify(state.meta.origin_m);
    if (gridMismatch) {
      root.querySelector('#delta-verdict').textContent = 'grid mismatch: cannot diff';
      return;
    }
    const volFile = byName.get('volume.npy');
    const { data: secondVolume } = parseNpy(await readFile(volFile));
    const primary = state.layerData.get('volume');
    const delta = computeDelta(primary, secondVolume);
    state.layerData.set('delta', delta);

    // Percentile scale, not Math.max(...primary): a call-arg spread of the
    // full 675,840-element real volume overflows the call stack.
    const abs = new Array(primary.length);
    for (let i = 0; i < primary.length; i++) abs[i] = Math.abs(primary[i]);
    abs.sort((a, b) => a - b);
    const scale = abs[Math.floor(abs.length * 0.95)] || 1e-12;
    let sumSq = 0;
    for (let i = 0; i < delta.length; i++) sumSq += delta[i] * delta[i];
    const rms = Math.sqrt(sumSq / delta.length);
    root.querySelector('#delta-verdict').textContent =
      `delta: ${deltaVerdict(rms, scale)} (rms ${rms.toFixed(4)})`;

    const deltaOption = document.createElement('label');
    deltaOption.style.display = 'block';
    const radio = document.createElement('input');
    radio.type = 'radio'; radio.name = 'active-layer'; radio.id = 'layer-delta';
    radio.addEventListener('change', () => state.setActiveLayer('delta'));
    deltaOption.appendChild(radio);
    deltaOption.appendChild(document.createTextNode(' Run delta (B minus A)'));
    root.querySelector('#layer-panel').appendChild(deltaOption);
  });

  root.querySelector('#export-png-btn').addEventListener('click', () => {
    render();
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'megiddo-voxel-view.png';
      a.click();
      URL.revokeObjectURL(url);
    });
  });

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', (ev) => {
    dragging = true; lastX = ev.clientX; lastY = ev.clientY;
  });
  window.addEventListener('pointerup', () => { dragging = false; });
  window.addEventListener('pointermove', (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
    lastX = ev.clientX; lastY = ev.clientY;
    state.camera.yaw += dx * 0.01;
    state.camera.pitch = Math.max(-1.5, Math.min(1.5, state.camera.pitch + dy * 0.01));
    render();
  });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    state.camera.distance = Math.max(0.5, state.camera.distance * (1 + ev.deltaY * 0.001));
    render();
  }, { passive: false });

  for (const key of Object.keys(CAMERA_PRESETS)) {
    const btn = root.querySelector(`#camera-preset-${key}`);
    btn.addEventListener('click', () => {
      state.camera.yaw = CAMERA_PRESETS[key].yaw;
      state.camera.pitch = CAMERA_PRESETS[key].pitch;
      render();
    });
  }

  root.querySelector('#frame-all-btn').addEventListener('click', () => {
    frameAll();
    render();
  });

  function renderViewList() {
    const list = loadViews();
    const ul = root.querySelector('#saved-views');
    ul.innerHTML = '';
    list.forEach((view, i) => {
      const li = document.createElement('li');
      li.textContent = view.name + ' ';
      const applyBtn = document.createElement('button');
      applyBtn.textContent = 'Apply';
      applyBtn.addEventListener('click', () => {
        if (!state.meta) return;
        state.camera = {
          yaw: view.camera.yaw,
          pitch: view.camera.pitch,
          distance: view.camera.distance,
          target: [...view.camera.target],
        };
        state.window = [view.window[0], view.window[1]];
        if (view.activeLayer && state.setActiveLayer) {
          state.setActiveLayer(view.activeLayer);
        } else {
          render();
        }
      });
      const deleteBtn = document.createElement('button');
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => {
        const current = loadViews();
        current.splice(i, 1);
        saveViews(current);
        renderViewList();
      });
      li.appendChild(applyBtn);
      li.appendChild(deleteBtn);
      ul.appendChild(li);
    });
  }

  root.querySelector('#save-view-btn').addEventListener('click', () => {
    if (!state.meta || !state.setActiveLayer) return;
    const nameInput = root.querySelector('#view-name');
    const list = loadViews();
    list.push(captureView(state, nameInput.value || 'view ' + (list.length + 1)));
    saveViews(list);
    renderViewList();
  });

  renderViewList();

  const histCanvas = root.querySelector('#histogram-canvas');
  let draggingWindowEdge = null; // 'lo' | 'hi' | null, local to this control
  function histCanvasPx(ev) {
    const rect = histCanvas.getBoundingClientRect();
    return (ev.clientX - rect.left) * (histCanvas.width / rect.width);
  }
  histCanvas.addEventListener('pointerdown', (ev) => {
    const px = histCanvasPx(ev);
    const { loPx, hiPx } = windowToBandPx(state.window, state.layerMax, histCanvas.width);
    draggingWindowEdge = Math.abs(px - loPx) <= Math.abs(px - hiPx) ? 'lo' : 'hi';
  });
  window.addEventListener('pointerup', () => { draggingWindowEdge = null; });
  window.addEventListener('pointermove', (ev) => {
    if (!draggingWindowEdge) return;
    const px = histCanvasPx(ev);
    const value = bandPxToWindow(px, state.layerMax, histCanvas.width);
    if (draggingWindowEdge === 'lo') {
      state.window[0] = Math.min(value, state.window[1]);
    } else {
      state.window[1] = Math.max(value, state.window[0]);
    }
    drawHistogram(); render();
  });

  const xferCanvas = root.querySelector('#xfer-canvas');
  let draggingStop = null;
  xferCanvas.addEventListener('pointerdown', (ev) => {
    const rect = xferCanvas.getBoundingClientRect();
    const t = (ev.clientX - rect.left) / rect.width;
    draggingStop = state.transferStops.reduce((best, s) =>
      Math.abs(s.t - t) < Math.abs(best.t - t) ? s : best);
  });
  window.addEventListener('pointerup', () => { draggingStop = null; });
  window.addEventListener('pointermove', (ev) => {
    if (!draggingStop) return;
    const rect = xferCanvas.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    draggingStop.t = t;
    rebuildLut();
    drawXferEditor();
  });

  const axes = { x: 0, y: 1, z: 2 };
  for (const axis of Object.keys(axes)) {
    const minInput = root.querySelector(`#clip-${axis}-min`);
    const minVal = root.querySelector(`#clip-${axis}-min-val`);
    minVal.textContent = minInput.value;
    minInput.addEventListener('input', (ev) => {
      state.clipMin[axes[axis]] = parseFloat(ev.target.value);
      minVal.textContent = ev.target.value;
      render();
    });
    const maxInput = root.querySelector(`#clip-${axis}-max`);
    const maxVal = root.querySelector(`#clip-${axis}-max-val`);
    maxVal.textContent = maxInput.value;
    maxInput.addEventListener('input', (ev) => {
      state.clipMax[axes[axis]] = parseFloat(ev.target.value);
      maxVal.textContent = ev.target.value;
      render();
    });
  }

  root.querySelector('#slice-axis').addEventListener('change', updateSlice);
  root.querySelector('#slice-pos').addEventListener('input', updateSlice);
  function updateSlice() {
    const axis = root.querySelector('#slice-axis').value;
    const pos = parseFloat(root.querySelector('#slice-pos').value);
    if (axis === 'none') {
      state.clipMin = [0, 0, 0];
      state.clipMax = [1, 1, 1];
    } else {
      const idx = axes[axis];
      const half = 0.02;
      state.clipMin = [0, 0, 0]; state.clipMax = [1, 1, 1];
      state.clipMin[idx] = Math.max(0, pos - half);
      state.clipMax[idx] = Math.min(1, pos + half);
    }
    render();
  }

  root.querySelector('#clip-plane-enabled').addEventListener('change', (ev) => {
    state.clipPlaneEnabled = ev.target.checked;
    render();
  });
  const clipPlaneDInput = root.querySelector('#clip-plane-d');
  const clipPlaneDVal = root.querySelector('#clip-plane-d-val');
  clipPlaneDVal.textContent = parseFloat(clipPlaneDInput.value).toFixed(2);
  clipPlaneDInput.addEventListener('input', (ev) => {
    state.clipPlaneD = parseFloat(ev.target.value);
    clipPlaneDVal.textContent = state.clipPlaneD.toFixed(2);
    render();
  });

  root.querySelector('#sigma-gate-enabled').addEventListener('change', (ev) => {
    state.sigmaGateEnabled = ev.target.checked;
    render();
  });
  root.querySelector('#sigma-gate-value').addEventListener('input', (ev) => {
    const frac = parseFloat(ev.target.value);
    const max = state.sigmaMax || 1;
    state.sigmaGateValue = frac * max;
    render();
  });

  function castHoverRay(clientX, clientY) {
    if (!state.meta) return null;
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -(((clientY - rect.top) / rect.height) * 2 - 1);

    const { yaw, pitch, distance, target } = state.camera;
    const eye = orbitToEye(target, yaw, pitch, distance);
    const view = lookAt(eye, target, [0, 1, 0]);
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, 0.05, 100);
    const invViewProj = invert(multiply(proj, view));
    if (!invViewProj) return null;

    function unproject(z) {
      const clip = [ndcX, ndcY, z, 1];
      const m = invViewProj;
      const w = m[3] * clip[0] + m[7] * clip[1] + m[11] * clip[2] + m[15] * clip[3];
      return [
        (m[0] * clip[0] + m[4] * clip[1] + m[8] * clip[2] + m[12] * clip[3]) / w,
        (m[1] * clip[0] + m[5] * clip[1] + m[9] * clip[2] + m[13] * clip[3]) / w,
        (m[2] * clip[0] + m[6] * clip[1] + m[10] * clip[2] + m[14] * clip[3]) / w,
      ];
    }
    const nearP = unproject(-1), farP = unproject(1);
    const dir = [farP[0] - nearP[0], farP[1] - nearP[1], farP[2] - nearP[2]];
    const len = Math.hypot(...dir);
    const step = [dir[0] / len, dir[1] / len, dir[2] / len];
    const data = state.layerData.get(state.activeLayer);
    if (!data) return null;

    const steps = 200;
    const stepLen = len / steps;
    const { min, extent } = worldBounds();
    for (let s = 0; s < steps; s++) {
      const world = [nearP[0] + step[0] * stepLen * s,
                     nearP[1] + step[1] * stepLen * s,
                     nearP[2] + step[2] * stepLen * s];
      const tex = [
        (world[0] - min[0]) / extent[0],
        (world[1] - min[1]) / extent[1],
        (world[2] - min[2]) / extent[2],
      ];
      if (!insideClipBox(tex, state.clipMin, state.clipMax)) continue;
      if (state.clipPlaneEnabled && !insideClipPlane(tex, state.clipPlaneNormal, state.clipPlaneD)) continue;
      const [i, j, k] = worldToVoxel(world, state.meta);
      const value = sampleNearest(data, state.meta.shape, i, j, k);
      if (!Number.isNaN(value) && value > (state.window ? state.window[0] : 0)) {
        return { i: Math.round(i), j: Math.round(j), k: Math.round(k), value };
      }
    }
    return null;
  }

  canvas.addEventListener('pointermove', (ev) => {
    const hit = castHoverRay(ev.clientX, ev.clientY);
    const el = root.querySelector('#hover-readout');
    const wrapRect = root.querySelector('#canvas-wrap').getBoundingClientRect();
    el.style.left = (ev.clientX - wrapRect.left) + 'px';
    el.style.top = (ev.clientY - wrapRect.top) + 'px';
    el.hidden = !hit;
    el.textContent = hit
      ? `voxel (${hit.i}, ${hit.j}, ${hit.k})  value ${hit.value.toFixed(4)}`
      : '';
  });

  render();
  initDock(root);

  const cmapSel = root.querySelector('#colormap-select');
  for (const name of COLORMAP_NAMES) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    cmapSel.appendChild(opt);
  }
  let lastCmap = 'viridis';
  try { lastCmap = localStorage.getItem('megido-viewer:colormap') || 'viridis'; } catch { /* ignore */ }
  cmapSel.value = COLORMAP_NAMES.includes(lastCmap) ? lastCmap : 'viridis';
  function applyColormap(name) {
    state.transferStops = colormapStops(name);
    try { localStorage.setItem('megido-viewer:colormap', name); } catch { /* ignore */ }
    if (state.drawXferEditor) state.drawXferEditor();
    gl.deleteTexture(state.lutTex);
    state.lutTex = makeLutTexture(gl, buildTransferLUT(state.transferStops));
    render();
  }
  cmapSel.addEventListener('change', (ev) => applyColormap(ev.target.value));
  applyColormap(cmapSel.value);

  window.__viewerState = state; // inspected by Playwright tests
}
