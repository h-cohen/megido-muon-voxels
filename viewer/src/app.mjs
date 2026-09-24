import { parseNpy } from './npy.mjs';
import { identity, multiply, perspective, lookAt, invert } from './mat4.mjs';
import { worldToVoxel, sampleNearest, reorderForTexture } from './grid.mjs';
import { insideClipBox, insideClipPlane } from './clip.mjs';
import { buildTransferLUT } from './transfer.mjs';
import { orbitToEye, CAMERA_PRESETS } from './camera.mjs';
import { computeHistogram, robustWindow, windowToBandPx, bandPxToWindow } from './histogram.mjs';
import { availableLayers } from './layers.mjs';
import { computeDelta, deltaVerdict } from './delta.mjs';
import { initDock } from './dock.mjs';
import { COLORMAP_NAMES, colormapStops } from './colormap.mjs';
import { captureView, loadViews, saveViews } from './views.mjs';
import { SHORTCUTS, keyToAction } from './shortcuts.mjs';
import { markerVertices, silhouetteVertices, SILHOUETTE_RAYLEN_M, dedupeDetectors, projectToScreen } from './markers.mjs';
import { surfaceMesh, smoothHeightfield, surfaceVertexColors, robustRange, surfaceHeightAt, surfaceTextureData } from './surfacemesh.mjs';

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
uniform bool uManualTrilinear;
uniform bool uShading;
uniform vec3 uVolSize;    // float(nx, ny, nz)
uniform float uRefStep;   // path length the transfer-function alpha is defined per
uniform bool uSurfClip;
uniform sampler2D uSurfTex;
uniform vec2 uSurfMin;     // (gx[0], gy[0])
uniform vec2 uSurfStep;    // (gx[1]-gx[0], gy[1]-gy[0])  (the fit grid is uniform)
uniform ivec2 uSurfSize;   // (nx, ny)
uniform float uMinStepVoxels;  // minimum step length, in voxels (test-driven; default 0.5)

bool aboveSurface(vec3 world) {
  vec2 q = (world.xy - uSurfMin) / uSurfStep;
  if (q.x < 0.0 || q.y < 0.0 || q.x > float(uSurfSize.x - 1) || q.y > float(uSurfSize.y - 1))
    return false;
  ivec2 i0 = min(ivec2(floor(q)), uSurfSize - 2);
  vec2 f = q - vec2(i0);
  float h00 = texelFetch(uSurfTex, i0, 0).r;
  float h10 = texelFetch(uSurfTex, i0 + ivec2(1, 0), 0).r;
  float h01 = texelFetch(uSurfTex, i0 + ivec2(0, 1), 0).r;
  float h11 = texelFetch(uSurfTex, i0 + ivec2(1, 1), 0).r;
  if (max(max(h00, h10), max(h01, h11)) > 1.0e5) return false;   // unknown ground: never clip
  float h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
  return world.z > h;
}

vec3 unproject(vec2 ndc, float z) {
  vec4 clip = vec4(ndc, z, 1.0);
  vec4 world = uInvViewProj * clip;
  return world.xyz / world.w;
}

float sampleDensity(vec3 tex) {
  if (!uManualTrilinear) return texture(uVolume, tex).r;
  vec3 p = tex * uVolSize - 0.5;
  vec3 f = fract(p);
  ivec3 mx = ivec3(uVolSize) - 1;
  ivec3 a = clamp(ivec3(floor(p)), ivec3(0), mx);
  ivec3 b = clamp(ivec3(floor(p)) + 1, ivec3(0), mx);
  float c000 = texelFetch(uVolume, ivec3(a.x, a.y, a.z), 0).r;
  float c100 = texelFetch(uVolume, ivec3(b.x, a.y, a.z), 0).r;
  float c010 = texelFetch(uVolume, ivec3(a.x, b.y, a.z), 0).r;
  float c110 = texelFetch(uVolume, ivec3(b.x, b.y, a.z), 0).r;
  float c001 = texelFetch(uVolume, ivec3(a.x, a.y, b.z), 0).r;
  float c101 = texelFetch(uVolume, ivec3(b.x, a.y, b.z), 0).r;
  float c011 = texelFetch(uVolume, ivec3(a.x, b.y, b.z), 0).r;
  float c111 = texelFetch(uVolume, ivec3(b.x, b.y, b.z), 0).r;
  return mix(mix(mix(c000, c100, f.x), mix(c010, c110, f.x), f.y),
             mix(mix(c001, c101, f.x), mix(c011, c111, f.x), f.y), f.z);
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 nearP = unproject(ndc, -1.0);
  vec3 farP = unproject(ndc, 1.0);
  vec3 dir = normalize(farP - nearP);

  // March ONLY inside the volume box (slab intersection). A ray that misses
  // the box costs nothing, and all samples land in the volume at a fixed
  // fraction of a voxel -- the old loop spread uSteps over the whole
  // near..far frustum (~100 m), sampling the 0.25 m campaign grid only every
  // other voxel while most steps fell in empty space.
  // Guard the slab division: GLSL ES 3.00 leaves 1.0/0.0 undefined, and while
  // this stack's IEEE ±inf happens to resolve parallel rays correctly, a
  // sign-preserving epsilon keeps invDir finite everywhere without changing
  // the result (measure-zero exact-axis-aligned rays only).
  vec3 sgn = vec3(greaterThanEqual(dir, vec3(0.0))) * 2.0 - 1.0;
  vec3 safeDir = mix(dir, sgn * 1e-8, lessThan(abs(dir), vec3(1e-8)));
  vec3 invDir = 1.0 / safeDir;
  vec3 t0s = (uWorldMin - nearP) * invDir;
  vec3 t1s = (uWorldMin + uWorldExtent - nearP) * invDir;
  vec3 tsm = min(t0s, t1s), tbg = max(t0s, t1s);
  float tEnter = max(max(max(tsm.x, tsm.y), tsm.z), 0.0);
  float tExit = min(min(tbg.x, tbg.y), tbg.z);
  if (tExit <= tEnter) { outColor = vec4(0.0); return; }

  float voxel = uWorldExtent.x / uVolSize.x;       // cubic voxels (single spacing)
  float stepLen = max(uMinStepVoxels * voxel, (tExit - tEnter) / float(uSteps));
  // Opacity correction: the transfer function's alpha is defined per
  // uRefStep of path (the legacy sampling length), so the calibrated look is
  // independent of how finely we now sample.
  float alphaExp = stepLen / uRefStep;
  vec4 accum = vec4(0.0);

  for (int i = 0; i < 1024; i++) {
    float tt = tEnter + (float(i) + 0.5) * stepLen;
    if (tt > tExit || accum.a > 0.98) break;
    vec3 pos = nearP + dir * tt;
    vec3 tex = clamp((pos - uWorldMin) / uWorldExtent, 0.0, 1.0);
    bool clipped = any(lessThan(tex, uClipMin)) || any(greaterThan(tex, uClipMax));
    if (uClipPlaneEnabled) {
      float d = dot(tex - vec3(0.5), uClipPlaneNormal) - uClipPlaneD;
      clipped = clipped || d < 0.0;
    }
    if (uSigmaGateEnabled) {
      float sigma = texture(uSigmaTex, tex).r;
      clipped = clipped || sigma > uSigmaGateValue;
    }
    if (uSurfClip && !clipped) clipped = aboveSurface(pos);
    if (!clipped) {
      float density = sampleDensity(tex);
      float t = clamp((density - uWindow.x) / max(uWindow.y - uWindow.x, 1e-6), 0.0, 1.0);
      vec4 c = texture(uTransferLUT, vec2(t, 0.5));
      if (uShading && c.a > 0.004) {   // gradient only where the sample shows
        vec3 h = 1.0 / uVolSize;   // one voxel per axis; voxels are cubic, so the
                                   // tex-space difference is proportional to the world gradient
        vec3 g = vec3(
          sampleDensity(tex + vec3(h.x, 0.0, 0.0)) - sampleDensity(tex - vec3(h.x, 0.0, 0.0)),
          sampleDensity(tex + vec3(0.0, h.y, 0.0)) - sampleDensity(tex - vec3(0.0, h.y, 0.0)),
          sampleDensity(tex + vec3(0.0, 0.0, h.z)) - sampleDensity(tex - vec3(0.0, 0.0, h.z)));
        float gm = length(g);
        if (gm > 1e-6) {
          float lambert = abs(dot(-g / gm, -dir));
          c.rgb *= 0.35 + 0.65 * lambert;
        }
      }
      c.a = 1.0 - pow(1.0 - clamp(c.a, 0.0, 0.999), alphaExp);
      c.rgb *= c.a;
      accum += (1.0 - accum.a) * c;
    }
  }

  outColor = accum;
}`;

// Detector position markers use a SEPARATE minimal GL program from the
// raymarch shader above: flat-colored world-space lines, drawn with gl.LINES
// after the raymarch fullscreen triangle. This keeps FRAGMENT_SRC/VERTEX_SRC
// (the raymarch shader) untouched.
const MARKER_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMarkerViewProj;
void main() {
  gl_Position = uMarkerViewProj * vec4(aPos, 1.0);
}`;

const MARKER_FRAGMENT_SRC = `#version 300 es
precision highp float;
uniform vec3 uMarkerColor;
out vec4 outColor;
void main() {
  outColor = vec4(uMarkerColor, 1.0);
}`;

// Hillside SURFACE mesh: translucent triangles with a per-vertex colour
// (aColor, location 1) - sigma-coloured, or a flat colour when that mode is
// off. Own program so the alpha/blend logic stays out of the opaque marker
// program and nowhere near the raymarch FRAGMENT_SRC above.
const MESH_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMarkerViewProj;
out vec3 vColor;
void main() { vColor = aColor; gl_Position = uMarkerViewProj * vec4(aPos, 1.0); }`;
const MESH_FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec3 vColor;
uniform float uAlpha;
out vec4 outColor;
void main() { outColor = vec4(vColor, uAlpha); }`;

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
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, nx, ny, nz, 0, gl.RED, gl.FLOAT, reorderForTexture(data, shape));
  return tex;
}

function makeSurfaceTexture(gl) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // NEAREST + texelFetch in the shader (no OES_texture_float_linear
  // dependency); the shader does its own bilinear from the four texel
  // fetches, mirroring surfaceHeightAt on the CPU side.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
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
  const detectorLabelsEl = root.querySelector('#detector-labels');
  // preserveDrawingBuffer: true so toDataURL()/screenshot readback after a
  // draw call sees the frame just rendered, not a backbuffer the browser
  // has already cleared for the next composite.
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 is required');
  gl.getExtension('EXT_color_buffer_float');
  const floatLinear = !!gl.getExtension('OES_texture_float_linear');

  const program = linkProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
  const vao = gl.createVertexArray();

  const markerProgram = linkProgram(gl, MARKER_VERTEX_SRC, MARKER_FRAGMENT_SRC);
  const markerUniforms = {
    uMarkerViewProj: gl.getUniformLocation(markerProgram, 'uMarkerViewProj'),
    uMarkerColor: gl.getUniformLocation(markerProgram, 'uMarkerColor'),
  };
  const markerVao = gl.createVertexArray();
  const markerBuffer = gl.createBuffer();
  gl.bindVertexArray(markerVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, markerBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  // Hillside silhouette fan: SAME markerProgram/markerUniforms as the
  // detector crosses above (Task S5), a second VAO/buffer because the
  // geometry is unrelated. Distinct colors per detector position, drawn as
  // separate ranges within one buffer so each range gets its own
  // uMarkerColor.
  const SILHOUETTE_COLORS = [
    [0.15, 0.9, 0.85],  // teal/cyan
    [0.95, 0.35, 0.85],  // magenta
  ];
  const silhouetteVao = gl.createVertexArray();
  const silhouetteBuffer = gl.createBuffer();
  gl.bindVertexArray(silhouetteVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, silhouetteBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  // Hillside SURFACE mesh (Task 4, primary hillside display; Task 3
  // sigma-colouring): a filled height-field "lid" over the fitted
  // overburden, drawn with meshProgram/MESH_*_SRC above (per-vertex colour,
  // needed both for the flat amber fill and the sigma ramp) so it doesn't
  // touch the opaque marker/silhouette color path. Own VAO/position+index+
  // color buffers since it is indexed triangles, not a flat gl.LINES vertex
  // list.
  const meshProgram = linkProgram(gl, MESH_VERTEX_SRC, MESH_FRAGMENT_SRC);
  const meshUniforms = {
    uMarkerViewProj: gl.getUniformLocation(meshProgram, 'uMarkerViewProj'),
    uAlpha: gl.getUniformLocation(meshProgram, 'uAlpha'),
  };
  const hillSurfaceVao = gl.createVertexArray();
  const hillSurfacePositionBuffer = gl.createBuffer();
  const hillSurfaceColorBuffer = gl.createBuffer();
  const hillSurfaceIndexBuffer = gl.createBuffer();
  gl.bindVertexArray(hillSurfaceVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, hillSurfacePositionBuffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, hillSurfaceColorBuffer);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, hillSurfaceIndexBuffer);
  gl.bindVertexArray(null);
  // Warm amber, distinct from the teal/magenta silhouette fan.
  const CAMERA_NEAR = 0.05, CAMERA_FAR = 100;
  const RAY_STEPS = 200; // max samples per ray INSIDE the volume box
  const HILL_SURFACE_COLOR = [0.95, 0.6, 0.15, 0.35];

  const uniforms = {};
  for (const name of [
    'uInvViewProj', 'uCameraPos', 'uVolume', 'uSigmaTex', 'uTransferLUT',
    'uWindow', 'uClipMin', 'uClipMax', 'uClipPlaneEnabled', 'uClipPlaneNormal',
    'uClipPlaneD', 'uSigmaGateEnabled', 'uSigmaGateValue', 'uSteps',
    'uWorldMin', 'uWorldExtent', 'uManualTrilinear', 'uShading', 'uVolSize', 'uRefStep',
    'uSurfClip', 'uSurfTex', 'uSurfMin', 'uSurfStep', 'uSurfSize', 'uMinStepVoxels',
  ]) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }

  const dummyVolume = makeVolumeTexture(gl, [1, 1, 1], new Float32Array([0]));
  const state = {
    meta: null,
    layerData: new Map(),
    activeLayer: null,
    gl, program, uniforms,
    // Default view is the "observation" preset (see camera.mjs
    // CAMERA_PRESETS.observation): z-up, detectors (z=0) low in frame,
    // reconstructed rock rising above them.
    camera: {
      yaw: CAMERA_PRESETS.observation.yaw,
      pitch: CAMERA_PRESETS.observation.pitch,
      distance: 3,
      target: [0, 0, 0],
    },
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
    floatLinear,
    smoothSampling: true,
    shading: true,
    minStepVoxels: 0.5, // test hook: minimum march step, in voxels (uMinStepVoxels)
    lutTex: makeLutTexture(gl, buildTransferLUT(colormapStops('viridis'))),
    detectors: [],
    detectorLabels: [],
    showDetectors: false,
    markerVertexCount: 0,
    silhouette: null,
    showSilhouette: false,
    silhouetteRanges: [],
    hillSurface: null,
    showHillSurface: false,
    hillSurfaceIndexCount: 0,
    hillSurfaceSmooth: 0, // display-only smoothing pass count; 0 = raw fit
    hillSigmaColor: false, // colour the surface by GP posterior sigma instead of flat amber
    hillSigmaRange: null, // [lo, hi] robust range of sigma, set by loadRun
    surfClip: false, // display-only: skip volume samples above the fitted surface
    hillDisplayH: null, // the DISPLAYED (possibly smoothed) height field the clip must match
    surfTex: null,
    surfMin: [0, 0],
    surfStep: [1, 1],
    surfSize: [0, 0],
  };

  function worldBounds() {
    if (!state.meta) return { min: [0, 0, 0], extent: [1, 1, 1] };
    const { shape, spacing_m, origin_m } = state.meta;
    const extent = shape.map((n) => n * spacing_m);
    return { min: origin_m, extent };
  }

  // World Z is the physical vertical (detectors at z=0, rock above). Near
  // straight-down/up (|pitch| > PITCH_GIMBAL_LIMIT) the z-up vector goes
  // parallel to the eye-to-target axis and lookAt degenerates, so we fall
  // back to y-up there.
  const PITCH_GIMBAL_LIMIT = 1.4;

  // Single source of truth for eye/view/proj/viewProj/invViewProj, shared by
  // render() and castHoverRay() so the two can never drift out of sync (see
  // the cross-ref comment that used to live on castHoverRay).
  function cameraMatrices() {
    const { yaw, pitch, distance, target } = state.camera;
    const eye = orbitToEye(target, yaw, pitch, distance);
    const up = Math.abs(pitch) > PITCH_GIMBAL_LIMIT ? [0, 1, 0] : [0, 0, 1];
    const view = lookAt(eye, target, up);
    const proj = perspective(Math.PI / 4, canvas.width / canvas.height, CAMERA_NEAR, CAMERA_FAR);
    const viewProj = multiply(proj, view);
    const invViewProj = invert(viewProj) || identity();
    return { eye, view, proj, viewProj, invViewProj };
  }

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  // Upgrades the live 3D textures' filter mode from the NEAREST that
  // makeVolumeTexture always creates them with. Hardware LINEAR is only used
  // when OES_texture_float_linear is present (R32F textures otherwise cannot
  // be linearly filtered on WebGL2); the shader's manual 8-tap trilinear
  // (uManualTrilinear) covers smooth sampling everywhere else. Must be
  // called after every makeVolumeTexture assignment, since a fresh texture
  // always starts NEAREST.
  function applyVolumeFilter() {
    const filter = (state.smoothSampling && state.floatLinear) ? gl.LINEAR : gl.NEAREST;
    for (const tex of [state.volumeTex, state.sigmaTex]) {
      if (!tex) continue;
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
    }
  }
  state.applyVolumeFilter = applyVolumeFilter;

  function render() {
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(width * (window.devicePixelRatio || 1)));
    canvas.height = Math.max(1, Math.round(height * (window.devicePixelRatio || 1)));
    gl.viewport(0, 0, canvas.width, canvas.height);
    if (currentTheme() === 'light') {
      gl.clearColor(0.90, 0.92, 0.94, 1);
    } else {
      gl.clearColor(0.04, 0.05, 0.06, 1);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);

    const { eye, viewProj, invViewProj } = cameraMatrices();

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
    gl.uniform1i(uniforms.uSteps, RAY_STEPS);
    // Legacy per-sample path: uSteps spread over the whole near..far span. The
    // LUT/window were tuned against it, so alpha stays defined per this length.
    // The old step was actually (FAR-NEAR)/(uSteps*cos(theta)) per pixel, theta
    // the off-axis angle from unprojecting a frustum instead of a box, so
    // corner pixels had a longer, less-dense step than the centre; this single
    // uniform reference matches the old CENTRE look exactly (edges are now
    // slightly denser than the legacy render, not less).
    gl.uniform1f(uniforms.uRefStep, (CAMERA_FAR - CAMERA_NEAR) / RAY_STEPS);
    gl.uniform1f(uniforms.uMinStepVoxels, state.minStepVoxels != null ? state.minStepVoxels : 0.5);
    gl.uniform1i(uniforms.uManualTrilinear, (state.smoothSampling && !state.floatLinear) ? 1 : 0);
    gl.uniform1i(uniforms.uShading, state.shading ? 1 : 0);
    const volShape = state.meta ? state.meta.shape : [1, 1, 1];
    gl.uniform3fv(uniforms.uVolSize, [volShape[0], volShape[1], volShape[2]]);
    gl.uniform1i(uniforms.uSurfClip, (state.surfClip && !!state.surfTex) ? 1 : 0);
    gl.uniform2fv(uniforms.uSurfMin, state.surfMin);
    gl.uniform2fv(uniforms.uSurfStep, state.surfStep);
    gl.uniform2i(uniforms.uSurfSize, state.surfSize[0], state.surfSize[1]);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, state.volumeTex);
    gl.uniform1i(uniforms.uVolume, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, state.sigmaTex);
    gl.uniform1i(uniforms.uSigmaTex, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, state.lutTex);
    gl.uniform1i(uniforms.uTransferLUT, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, state.surfTex);
    gl.uniform1i(uniforms.uSurfTex, 3);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (state.showDetectors && state.meta && state.markerVertexCount > 0) {
      drawMarkers(viewProj);
    }
    if (state.showSilhouette && state.silhouette && state.silhouetteRanges.length > 0) {
      drawSilhouette(viewProj);
    }
    if (state.showHillSurface && state.hillSurface && state.hillSurfaceIndexCount > 0) {
      drawHillSurface(viewProj);
    }
    drawGizmo();
    updateDetectorLabels();
  }
  state.render = render;

  // Detector position markers: drawn with their own tiny GL program (see
  // MARKER_VERTEX_SRC/MARKER_FRAGMENT_SRC above), using the SAME viewProj
  // render() just computed so the crosses sit correctly in the scene.
  function drawMarkers(viewProj) {
    gl.useProgram(markerProgram);
    gl.bindVertexArray(markerVao);
    gl.uniformMatrix4fv(markerUniforms.uMarkerViewProj, false, viewProj);
    gl.uniform3fv(markerUniforms.uMarkerColor, [1.0, 0.75, 0.1]);
    gl.drawArrays(gl.LINES, 0, state.markerVertexCount);
    gl.bindVertexArray(null);
  }

  function rebuildMarkerBuffer() {
    const detectors = state.detectors || [];
    const verts = markerVertices(detectors);
    state.markerVertexCount = verts.length / 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, markerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    state.detectorLabels = dedupeDetectors(detectors);
    rebuildDetectorLabelEls();
  }

  // HTML overlay labels ("P0", "P1/T20a", ...) for detector positions -
  // crisp, theme-aware text the GL raymarch/marker programs can't give us
  // cheaply. One <div> per distinct world position (see dedupeDetectors in
  // markers.mjs); rebuilt only when the detector set changes, repositioned
  // every render() via the SAME viewProj the scene draws with.
  let detectorLabelEls = [];
  function rebuildDetectorLabelEls() {
    if (!detectorLabelsEl) return;
    detectorLabelsEl.innerHTML = '';
    detectorLabelEls = state.detectorLabels.map((d) => {
      const el = document.createElement('div');
      el.className = 'detector-label';
      el.textContent = d.label;
      el.hidden = true;
      detectorLabelsEl.appendChild(el);
      return el;
    });
  }

  function updateDetectorLabels() {
    if (!detectorLabelsEl) return;
    if (!state.showDetectors || !state.meta || detectorLabelEls.length === 0) {
      for (const el of detectorLabelEls) el.hidden = true;
      return;
    }
    const { viewProj } = cameraMatrices();
    const { width, height } = root.querySelector('#canvas-wrap').getBoundingClientRect();
    state.detectorLabels.forEach((d, idx) => {
      const el = detectorLabelEls[idx];
      const p = projectToScreen([d.x, d.y, d.z], viewProj, width, height);
      if (!p || p.x < 0 || p.x > width || p.y < 0 || p.y > height) {
        el.hidden = true;
        return;
      }
      el.hidden = false;
      el.style.left = `${p.x}px`;
      el.style.top = `${p.y}px`;
    });
  }

  // Hillside silhouette fan: drawn with the SAME markerProgram used for
  // detector crosses (see drawMarkers above), one draw call per detector
  // position so each gets its own color from SILHOUETTE_COLORS. The rays
  // are a display convention length (SILHOUETTE_RAYLEN_M) - this observable
  // is angular only, so no draw call here implies a measured distance.
  function drawSilhouette(viewProj) {
    gl.useProgram(markerProgram);
    gl.bindVertexArray(silhouetteVao);
    gl.uniformMatrix4fv(markerUniforms.uMarkerViewProj, false, viewProj);
    for (let i = 0; i < state.silhouetteRanges.length; i++) {
      const { start, count } = state.silhouetteRanges[i];
      gl.uniform3fv(markerUniforms.uMarkerColor, SILHOUETTE_COLORS[i % SILHOUETTE_COLORS.length]);
      gl.drawArrays(gl.LINES, start, count);
    }
    gl.bindVertexArray(null);
  }

  function rebuildSilhouetteBuffer() {
    const silhouette = state.silhouette;
    // hill_silhouette.json carries its OWN `detectors` list, keyed by the
    // same pos0/pos1 ids as `per_pos` (see megido/hillside.py) - these are
    // NOT the same ids as meta.json's detectors (P0/T20a/T20b/P1), so the
    // silhouette's own list must be used here, not state.detectors.
    const detectors = (silhouette && silhouette.detectors) || [];
    if (!silhouette || !silhouette.per_pos) {
      state.silhouetteRanges = [];
      return;
    }
    const chunks = [];
    const ranges = [];
    let vertOffset = 0;
    for (const posId of Object.keys(silhouette.per_pos)) {
      const single = { [posId]: silhouette.per_pos[posId] };
      const verts = silhouetteVertices(single, detectors, SILHOUETTE_RAYLEN_M);
      if (verts.length === 0) continue;
      chunks.push(verts);
      const count = verts.length / 3;
      ranges.push({ start: vertOffset, count });
      vertOffset += count;
    }
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const combined = new Float32Array(total);
    let o = 0;
    for (const c of chunks) { combined.set(c, o); o += c.length; }
    state.silhouetteRanges = ranges;
    gl.bindBuffer(gl.ARRAY_BUFFER, silhouetteBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, combined, gl.STATIC_DRAW);
  }

  // Hillside SURFACE mesh (Task 4): filled translucent triangles from the
  // `hillside` CLI's regularized height-field fit. Own meshProgram/VAO (see
  // setup above) drawn with gl.TRIANGLES over indices built once by
  // rebuildHillSurfaceBuffer, not per frame. Blending is enabled only for
  // this draw call and disabled again immediately after, so it never leaks
  // into the raymarch pass (which runs earlier in this same render() call
  // on the NEXT frame, by which point blend is already off, but restoring
  // here keeps state deterministic regardless of draw order).
  function drawHillSurface(viewProj) {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(meshProgram);
    gl.bindVertexArray(hillSurfaceVao);
    gl.uniformMatrix4fv(meshUniforms.uMarkerViewProj, false, viewProj);
    gl.uniform1f(meshUniforms.uAlpha, HILL_SURFACE_COLOR[3]);
    gl.drawElements(gl.TRIANGLES, state.hillSurfaceIndexCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  // Rebuilds the hill-surface GL buffers from the RAW fitted height field,
  // re-applying state.hillSurfaceSmooth passes of display-only smoothing
  // (smoothHeightfield) each time. Smoothing always starts from the raw
  // surf.H (never cumulatively from a previous smoothed result), so moving
  // the slider back to 0 exactly restores the raw fit. Also rebuilds the
  // per-vertex colour buffer: the GP posterior sigma ramp when
  // state.hillSigmaColor is on and sigma data is loaded, otherwise a flat
  // amber matching HILL_SURFACE_COLOR (so the flat-fill look is unchanged
  // when sigma coloring is off or unavailable).
  function rebuildHillSurfaceBuffer() {
    const surf = state.hillSurface;
    if (!surf) {
      state.hillSurfaceIndexCount = 0;
      return;
    }
    const iterations = state.hillSurfaceSmooth || 0;
    const H = iterations > 0
      ? smoothHeightfield(surf.H, surf.gx.length, surf.gy.length, iterations)
      : surf.H;
    state.hillDisplayH = H;
    // Clip texture always mirrors the DISPLAYED (possibly smoothed) field, so
    // "clip above surface" never disagrees with the surface actually drawn.
    // A degenerate grid (fewer than 2 nodes on an axis) has no cell to
    // interpolate uSurfStep from, so the clip is disabled entirely rather
    // than guessing a step.
    const nx = surf.gx.length, ny = surf.gy.length;
    if (nx >= 2 && ny >= 2) {
      if (!state.surfTex) state.surfTex = makeSurfaceTexture(gl);
      gl.bindTexture(gl.TEXTURE_2D, state.surfTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, nx, ny, 0, gl.RED, gl.FLOAT, surfaceTextureData(H, nx, ny));
      state.surfMin = [surf.gx[0], surf.gy[0]];
      state.surfStep = [surf.gx[1] - surf.gx[0], surf.gy[1] - surf.gy[0]];
      state.surfSize = [nx, ny];
    } else {
      if (state.surfTex) { gl.deleteTexture(state.surfTex); state.surfTex = null; }
      state.surfClip = false;
      state.surfSize = [0, 0];
      const toggleSurfClipEl = root.querySelector('#toggle-surf-clip');
      if (toggleSurfClipEl) { toggleSurfClipEl.checked = false; toggleSurfClipEl.disabled = true; }
    }
    const { positions, indices } = surfaceMesh(H, surf.gx, surf.gy);
    const vertexCount = positions.length / 3;
    let colors;
    if (state.hillSigmaColor && surf.sigma) {
      const [lo, hi] = state.hillSigmaRange || robustRange(surf.sigma);
      colors = surfaceVertexColors(surf.sigma, lo, hi);
    } else {
      colors = new Float32Array(vertexCount * 3);
      for (let k = 0; k < vertexCount; k++) {
        colors[k * 3] = HILL_SURFACE_COLOR[0];
        colors[k * 3 + 1] = HILL_SURFACE_COLOR[1];
        colors[k * 3 + 2] = HILL_SURFACE_COLOR[2];
      }
    }
    gl.bindVertexArray(hillSurfaceVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, hillSurfacePositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, hillSurfaceColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, hillSurfaceIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    state.hillSurfaceIndexCount = indices.length;
  }

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
    const up = Math.abs(pitch) > PITCH_GIMBAL_LIMIT ? [0, 1, 0] : [0, 0, 1];
    const view = lookAt(eye, [0, 0, 0], up);
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
    const FRAME_MARGIN = 1.6;
    state.camera.distance = FRAME_MARGIN * diag;
  }

  async function loadRun(files) {
    state.ready = false;
    const onboardingEl = root.querySelector('#overlay-onboarding');
    if (onboardingEl) onboardingEl.hidden = true;
    const byName = new Map();
    for (const f of files) byName.set(f.name, f);

    const metaFile = byName.get('meta.json');
    if (!metaFile) throw new Error('selected directory has no meta.json');
    const meta = JSON.parse(await metaFile.text());
    state.meta = meta;
    state.detectors = meta.detectors || [];
    rebuildMarkerBuffer();

    // hill_silhouette.json (S6): an optional ridgeline fan, written by the
    // `hillside` CLI subcommand. Absent for older/synthetic runs - a run
    // without it must not error, and its toggle stays disabled.
    const silhouetteFile = byName.get('hill_silhouette.json');
    state.silhouette = silhouetteFile ? JSON.parse(await silhouetteFile.text()) : null;
    rebuildSilhouetteBuffer();
    const toggleSilhouetteEl = root.querySelector('#toggle-silhouette');
    if (toggleSilhouetteEl) {
      toggleSilhouetteEl.disabled = !state.silhouette;
      if (!state.silhouette) {
        toggleSilhouetteEl.checked = false;
        state.showSilhouette = false;
      }
    }

    // hill_surface.npy + hill_surface_meta.json (Phase 5b Task 4): the fitted
    // overburden height-field, written by the `hillside` CLI subcommand.
    // BOTH files must be present - a run missing either (older/synthetic
    // runs) must not error, and the toggle stays disabled and unchecked.
    const hillSurfaceFile = byName.get('hill_surface.npy');
    const hillSurfaceMetaFile = byName.get('hill_surface_meta.json');
    const hillSurfaceSigmaFile = byName.get('hill_surface_sigma.npy');
    const toggleHillSurfaceEl = root.querySelector('#toggle-hill-surface');
    const hillSurfaceCaveatEl = root.querySelector('#hill-surface-caveat');
    const hillSurfaceSmoothEl = root.querySelector('#hill-surface-smooth');
    const hillSurfaceSmoothReadoutEl = root.querySelector('#hill-surface-smooth-readout');
    const toggleHillSigmaEl = root.querySelector('#toggle-hill-sigma');
    const hillSigmaLegendEl = root.querySelector('#hill-sigma-legend');
    const hillSigmaRangeEl = root.querySelector('#hill-sigma-range');
    const toggleSurfClipEl = root.querySelector('#toggle-surf-clip');
    // Display-only clip is always OFF by default on a fresh load, even when
    // a surface is present - it is opt-in every time, never carried over
    // from a previous run.
    state.surfClip = false;
    if (toggleSurfClipEl) toggleSurfClipEl.checked = false;
    // A fresh run always starts unsmoothed, whatever a previous run's slider
    // was left at - display-only smoothing must never carry across loads.
    state.hillSurfaceSmooth = 0;
    if (hillSurfaceSmoothEl) hillSurfaceSmoothEl.value = '0';
    if (hillSurfaceSmoothReadoutEl) hillSurfaceSmoothReadoutEl.textContent = '0';
    // Sigma coloring is reset on every load too - it must never carry a
    // stale range or a stale "on" state from a previous run into one that
    // has no sigma data.
    state.hillSigmaColor = false;
    state.hillSigmaRange = null;
    if (hillSurfaceFile && hillSurfaceMetaFile) {
      const { data: H } = parseNpy(await readFile(hillSurfaceFile));
      const hillMeta = JSON.parse(await hillSurfaceMetaFile.text());
      let sigma = null;
      if (hillSurfaceSigmaFile) {
        const { data } = parseNpy(await readFile(hillSurfaceSigmaFile));
        if (data.length === H.length) sigma = data;
      }
      state.hillSurface = { H, gx: hillMeta.gx, gy: hillMeta.gy, meta: hillMeta, sigma };
      if (sigma) {
        state.hillSigmaRange = robustRange(sigma);
        state.hillSigmaColor = true;
      }
      rebuildHillSurfaceBuffer();
      if (toggleHillSurfaceEl) {
        toggleHillSurfaceEl.disabled = false;
        toggleHillSurfaceEl.checked = true; // primary hillside display: on by default when present
      }
      if (hillSurfaceSmoothEl) hillSurfaceSmoothEl.disabled = false;
      state.showHillSurface = true;
      // rebuildHillSurfaceBuffer leaves surfSize at [0, 0] (and surfTex null)
      // for a degenerate grid (fewer than 2 nodes on an axis) - the clip
      // toggle must stay disabled in that case, not just default-off.
      if (toggleSurfClipEl) toggleSurfClipEl.disabled = state.surfSize[0] === 0 || state.surfSize[1] === 0;
      if (toggleHillSigmaEl) {
        toggleHillSigmaEl.disabled = !sigma;
        toggleHillSigmaEl.checked = !!sigma;
      }
      if (hillSigmaLegendEl) hillSigmaLegendEl.hidden = !sigma;
      if (sigma && hillSigmaRangeEl) {
        const [lo, hi] = state.hillSigmaRange;
        hillSigmaRangeEl.textContent =
          `σ ${lo.toFixed(2)} – ${hi.toFixed(2)} m · raw posterior std, assumed scale`;
      }
      if (hillSurfaceCaveatEl) {
        const pctCell = Math.round((hillMeta.variance_explained || 0) * 100);
        const rayVe = hillMeta.ray_ve;
        // The per-ray VE depends strongly on the assumed inverse-density
        // scale `a` (see Honest limits in CLAUDE.md), so it must never be
        // shown without naming that scale.
        const rayText = Number.isFinite(rayVe)
          ? ` · ${Math.round(rayVe * 100)}% per ray (a=${hillMeta.a})`
          : '';
        hillSurfaceCaveatEl.textContent =
          `Hillside surface — assumed scale, ${pctCell}% VE per cell${rayText}`;
        hillSurfaceCaveatEl.hidden = false;
      }
    } else {
      state.hillSurface = null;
      state.hillSurfaceIndexCount = 0;
      state.showHillSurface = false;
      if (toggleHillSurfaceEl) {
        toggleHillSurfaceEl.disabled = true;
        toggleHillSurfaceEl.checked = false;
      }
      if (hillSurfaceSmoothEl) hillSurfaceSmoothEl.disabled = true;
      if (hillSurfaceCaveatEl) hillSurfaceCaveatEl.hidden = true;
      if (toggleHillSigmaEl) {
        toggleHillSigmaEl.disabled = true;
        toggleHillSigmaEl.checked = false;
      }
      if (hillSigmaLegendEl) hillSigmaLegendEl.hidden = true;
      if (toggleSurfClipEl) {
        toggleSurfClipEl.disabled = true;
        toggleSurfClipEl.checked = false;
      }
      state.surfClip = false;
      state.hillDisplayH = null;
    }

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
    applyVolumeFilter();

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
      applyVolumeFilter();
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

  const toggleDetectorsEl = root.querySelector('#toggle-detectors');
  if (toggleDetectorsEl) {
    toggleDetectorsEl.addEventListener('change', (ev) => {
      state.showDetectors = ev.target.checked;
      render();
    });
  }

  const toggleSmoothEl = root.querySelector('#toggle-smooth');
  if (toggleSmoothEl) {
    toggleSmoothEl.addEventListener('change', (ev) => {
      state.smoothSampling = ev.target.checked;
      applyVolumeFilter();
      render();
    });
  }

  const toggleShadingEl = root.querySelector('#toggle-shading');
  if (toggleShadingEl) {
    toggleShadingEl.addEventListener('change', (ev) => {
      state.shading = ev.target.checked;
      render();
    });
  }

  const toggleSilhouetteEl = root.querySelector('#toggle-silhouette');
  if (toggleSilhouetteEl) {
    toggleSilhouetteEl.disabled = true; // enabled by loadRun once a run is present
    toggleSilhouetteEl.addEventListener('change', (ev) => {
      state.showSilhouette = ev.target.checked;
      render();
    });
  }

  const toggleHillSurfaceEl = root.querySelector('#toggle-hill-surface');
  if (toggleHillSurfaceEl) {
    toggleHillSurfaceEl.disabled = true; // enabled by loadRun once the artifact is present
    toggleHillSurfaceEl.addEventListener('change', (ev) => {
      state.showHillSurface = ev.target.checked;
      render();
    });
  }

  const toggleHillSigmaEl = root.querySelector('#toggle-hill-sigma');
  if (toggleHillSigmaEl) {
    toggleHillSigmaEl.disabled = true; // enabled by loadRun once sigma data is present
    toggleHillSigmaEl.addEventListener('change', (ev) => {
      state.hillSigmaColor = ev.target.checked;
      rebuildHillSurfaceBuffer();
      render();
    });
  }

  // Display-only volume clip above the fitted (possibly smoothed) surface -
  // never touches the reconstruction, just what's drawn. Off by default,
  // enabled by loadRun once a surface is present.
  const toggleSurfClipEl = root.querySelector('#toggle-surf-clip');
  if (toggleSurfClipEl) {
    toggleSurfClipEl.disabled = true;
    toggleSurfClipEl.addEventListener('change', (ev) => {
      state.surfClip = ev.target.checked;
      render();
    });
  }

  // Display-only smoothing slider for the hillside surface mesh: always
  // rebuilds from the RAW fitted state.hillSurface.H (see
  // rebuildHillSurfaceBuffer), so it adds no information and changing the
  // slider never accumulates smoothing passes. No-op while no run/surface
  // is loaded.
  const hillSurfaceSmoothEl = root.querySelector('#hill-surface-smooth');
  const hillSurfaceSmoothReadoutEl = root.querySelector('#hill-surface-smooth-readout');
  if (hillSurfaceSmoothEl) {
    hillSurfaceSmoothEl.disabled = true; // enabled by loadRun once the artifact is present
    hillSurfaceSmoothEl.addEventListener('input', (ev) => {
      if (!state.hillSurface) return;
      state.hillSurfaceSmooth = Number(ev.target.value) || 0;
      if (hillSurfaceSmoothReadoutEl) {
        hillSurfaceSmoothReadoutEl.textContent = String(state.hillSurfaceSmooth);
      }
      rebuildHillSurfaceBuffer();
      render();
    });
  }

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
        // setActiveLayer rebuilds the layer texture AND resets state.window
        // to that layer's own robust default as a side effect, so it must
        // run BEFORE the saved window is applied, not after — otherwise the
        // saved window is silently clobbered by the robust default.
        if (view.activeLayer && state.setActiveLayer) {
          state.setActiveLayer(view.activeLayer);
          state.window = [view.window[0], view.window[1]];
          if (state.drawHistogram) state.drawHistogram();
          render();
        } else {
          state.window = [view.window[0], view.window[1]];
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
    minVal.textContent = Number(minInput.value).toFixed(2);
    minInput.addEventListener('input', (ev) => {
      state.clipMin[axes[axis]] = parseFloat(ev.target.value);
      minVal.textContent = Number(ev.target.value).toFixed(2);
      render();
    });
    const maxInput = root.querySelector(`#clip-${axis}-max`);
    const maxVal = root.querySelector(`#clip-${axis}-max-val`);
    maxVal.textContent = Number(maxInput.value).toFixed(2);
    maxInput.addEventListener('input', (ev) => {
      state.clipMax[axes[axis]] = parseFloat(ev.target.value);
      maxVal.textContent = Number(ev.target.value).toFixed(2);
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

  // Shares cameraMatrices() with render() (mirrored by the GPU-side
  // unproject() in FRAGMENT_SRC), so hover picking always agrees with what
  // was actually drawn. If the projection convention changes, update
  // cameraMatrices() and the shader together.
  function castHoverRay(clientX, clientY) {
    if (!state.meta) return null;
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -(((clientY - rect.top) / rect.height) * 2 - 1);

    const { invViewProj } = cameraMatrices();
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
      if (state.surfClip && state.hillSurface && state.hillDisplayH) {
        const hs = surfaceHeightAt(state.hillDisplayH, state.hillSurface.gx, state.hillSurface.gy, world[0], world[1]);
        if (Number.isFinite(hs) && world[2] > hs) continue;
      }
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

  // ---- theme toggle (dark default, persisted in localStorage) ----
  const THEME_KEY = 'megido-viewer:theme';
  const themeToggleBtn = root.querySelector('#theme-toggle');
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme === 'light' ? 'light' : 'dark');
    if (themeToggleBtn) themeToggleBtn.textContent = theme === 'light' ? 'Light' : 'Dark';
  }
  let startTheme = 'dark';
  try { startTheme = localStorage.getItem(THEME_KEY) || 'dark'; } catch { /* ignore */ }
  applyTheme(startTheme);
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const next = currentTheme() === 'light' ? 'dark' : 'light';
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
      render();
    });
  }

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

  // ---- shortcuts cheatsheet + onboarding card (Task 7) ----
  const shortcutsTable = root.querySelector('#shortcuts-table');
  if (shortcutsTable) {
    for (const s of SHORTCUTS) {
      const tr = document.createElement('tr');
      const keysTd = document.createElement('td');
      keysTd.className = 'keys';
      keysTd.textContent = s.keys;
      const labelTd = document.createElement('td');
      labelTd.textContent = s.label;
      tr.appendChild(keysTd);
      tr.appendChild(labelTd);
      shortcutsTable.appendChild(tr);
    }
  }

  const ONBOARDED_KEY = 'megido-viewer:onboarded';
  const onboardingOverlay = root.querySelector('#overlay-onboarding');
  const shortcutsOverlay = root.querySelector('#overlay-shortcuts');
  let onboarded = false;
  try { onboarded = localStorage.getItem(ONBOARDED_KEY) === '1'; } catch { /* ignore */ }
  if (onboardingOverlay) onboardingOverlay.hidden = onboarded;

  const onboardingDismissBtn = root.querySelector('#onboarding-dismiss-btn');
  if (onboardingDismissBtn) {
    onboardingDismissBtn.addEventListener('click', () => {
      try { localStorage.setItem(ONBOARDED_KEY, '1'); } catch { /* ignore */ }
      if (onboardingOverlay) onboardingOverlay.hidden = true;
    });
  }

  window.addEventListener('keydown', (ev) => {
    const targetTag = ev.target && ev.target.tagName;
    if (targetTag === 'INPUT' || targetTag === 'SELECT' || targetTag === 'TEXTAREA') return;
    const action = keyToAction(ev);
    if (!action) return;
    if (action.startsWith('layer')) {
      const n = Number(action.slice('layer'.length));
      const radios = root.querySelectorAll('#layer-panel input[type="radio"]');
      const radio = radios[n - 1];
      if (radio) radio.click();
      return;
    }
    if (action.startsWith('preset-')) {
      const key = action.slice('preset-'.length);
      const btn = root.querySelector(`#camera-preset-${key}`);
      if (btn) btn.click();
      return;
    }
    if (action === 'frame-all') {
      frameAll();
      render();
      return;
    }
    if (action === 'toggle-help') {
      if (shortcutsOverlay) shortcutsOverlay.hidden = !shortcutsOverlay.hidden;
      return;
    }
    if (action === 'close-overlay') {
      if (shortcutsOverlay) shortcutsOverlay.hidden = true;
      if (onboardingOverlay) onboardingOverlay.hidden = true;
      return;
    }
  });

  // Test hook: lets Playwright force a code path (e.g. floatLinear=false to
  // exercise the manual trilinear fallback) and redraw without a UI event.
  state.render = render;
  window.__viewerState = state; // inspected by Playwright tests
}
