import { parseNpy } from './npy.mjs';
import { identity, multiply, perspective, lookAt, invert } from './mat4.mjs';
import { modelMatrixFromMeta } from './grid.mjs';
import { buildTransferLUT, defaultStops } from './transfer.mjs';
import { orbitToEye, CAMERA_PRESETS } from './camera.mjs';
import { computeHistogram } from './histogram.mjs';
import { availableLayers } from './layers.mjs';

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
    transferStops: defaultStops(),
    clipMin: [0, 0, 0],
    clipMax: [1, 1, 1],
    clipPlaneEnabled: false,
    clipPlaneNormal: [0, 0, 1],
    clipPlaneD: 0,
    sigmaGateEnabled: false,
    sigmaGateValue: 1e9,
    secondRun: null,
    volumeTex: dummyVolume,
    sigmaTex: dummyVolume,
    lutTex: makeLutTexture(gl, buildTransferLUT(defaultStops())),
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
  }
  state.render = render;

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
  }

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
  }
  state.drawHistogram = drawHistogram;
  state.drawXferEditor = drawXferEditor;

  async function readFile(file) {
    return file.arrayBuffer();
  }

  async function loadRun(files) {
    const byName = new Map();
    for (const f of files) byName.set(f.name, f);

    const metaFile = byName.get('meta.json');
    if (!metaFile) throw new Error('selected directory has no meta.json');
    const meta = JSON.parse(await metaFile.text());
    state.meta = meta;

    state.layerData.clear();
    for (const name of meta.layers) {
      const file = byName.get(`${name}.npy`);
      if (!file) continue;
      const { data } = parseNpy(await readFile(file));
      state.layerData.set(name, data);
    }

    state.activeLayer = 'volume';
    state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('volume'));
    if (state.layerData.has('sigma')) {
      state.sigmaTex = makeVolumeTexture(gl, meta.shape, state.layerData.get('sigma'));
    }
    function setActiveLayer(key) {
      state.activeLayer = key;
      state.volumeTex = makeVolumeTexture(gl, meta.shape, state.layerData.get(key));
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

    const [lo, hi] = meta.value_range;
    state.window = [lo, hi];
    drawHistogram();
    drawXferEditor();
    render();
  }

  fileInput.addEventListener('change', (ev) => {
    loadRun(Array.from(ev.target.files)).catch((err) => {
      console.error(err);
      window.__viewerError = String(err);
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

  root.querySelector('#window-lo').addEventListener('input', (ev) => {
    state.window[0] = parseFloat(ev.target.value) * (state.meta ? state.meta.value_range[1] : 1);
    drawHistogram(); render();
  });
  root.querySelector('#window-hi').addEventListener('input', (ev) => {
    state.window[1] = parseFloat(ev.target.value) * (state.meta ? state.meta.value_range[1] : 1);
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

  render();
  window.__viewerState = state; // inspected by Playwright tests
}
