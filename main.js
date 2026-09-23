import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

/* =====================================================================
   utilities
   ===================================================================== */
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const TAU = Math.PI * 2;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- 2D simplex noise
const perm = new Uint8Array(512);
{
  const p = Array.from({ length: 256 }, (_, i) => i);
  const r = mulberry32(1337);
  for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
}
const G2D = [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]];
const F2 = 0.5 * (Math.sqrt(3) - 1), G2 = (3 - Math.sqrt(3)) / 6;
function noise2(xin, yin) {
  const s = (xin + yin) * F2;
  const i = Math.floor(xin + s), j = Math.floor(yin + s);
  const t = (i + j) * G2;
  const x0 = xin - (i - t), y0 = yin - (j - t);
  const i1 = x0 > y0 ? 1 : 0, j1 = x0 > y0 ? 0 : 1;
  const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2, x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
  const ii = i & 255, jj = j & 255;
  let n = 0, tt, g;
  tt = 0.5 - x0 * x0 - y0 * y0; if (tt > 0) { g = G2D[perm[ii + perm[jj]] & 7]; tt *= tt; n += tt * tt * (g[0] * x0 + g[1] * y0); }
  tt = 0.5 - x1 * x1 - y1 * y1; if (tt > 0) { g = G2D[perm[ii + i1 + perm[jj + j1]] & 7]; tt *= tt; n += tt * tt * (g[0] * x1 + g[1] * y1); }
  tt = 0.5 - x2 * x2 - y2 * y2; if (tt > 0) { g = G2D[perm[ii + 1 + perm[jj + 1]] & 7]; tt *= tt; n += tt * tt * (g[0] * x2 + g[1] * y2); }
  return 70 * n;
}
function fbm(x, y, oct = 5) {
  let a = 0.5, f = 1, s = 0, n = 0;
  for (let i = 0; i < oct; i++) { s += a * noise2(x * f, y * f); n += a; a *= 0.5; f *= 2.03; }
  return s / n;
}
function ridged(x, y, oct = 6) {
  let a = 0.5, f = 1, s = 0, w = 1;
  for (let i = 0; i < oct; i++) {
    let n = 1 - Math.abs(noise2(x * f, y * f));
    n *= n; n *= w; w = clamp(n * 2, 0, 1);
    s += n * a; a *= 0.5; f *= 2.07;
  }
  return s;
}
const noise3 = (x, y, z) => (noise2(x + z * 0.71, y * 1.3) + noise2(y - 3.3, z + x * 0.37) + noise2(z * 1.1 + 7.1, x - y * 0.5)) / 3;

const srgb = (hex) => new THREE.Color(hex); // Color() converts sRGB hex -> linear working space

/* =====================================================================
   renderer / scene
   ===================================================================== */
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
const DPR = Math.min(window.devicePixelRatio, 1.6);
renderer.setPixelRatio(DPR);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
const MAX_ANISO = renderer.capabilities.getMaxAnisotropy();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.5, 9000);
camera.position.set(54, 22, 72);

const U = { uTime: { value: 0 }, uWind: { value: 0.35 } };
// season / weather driven uniforms shared by many materials
const ENV = { uSnow: { value: 0 }, uWet: { value: 0 }, uGreen: { value: 0 }, uLeafAmt: { value: 1 } };

const setProgress = (p) => { document.getElementById('ldbar').style.width = `${Math.round(p * 100)}%`; };
const nextFrame = () => new Promise((r) => setTimeout(r, 16));

/* =====================================================================
   procedural textures
   ===================================================================== */
function makeCanvas(w, h = w) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }
function toTex(c, color = true, repeat = true) {
  const t = new THREE.CanvasTexture(c);
  if (color) t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = MAX_ANISO;
  return t;
}
function addNoise(g, S, amt, r) {
  const id = g.getImageData(0, 0, S, S), d = id.data;
  for (let i = 0; i < d.length; i += 4) { const n = (r() - 0.5) * amt; d[i] += n; d[i + 1] += n; d[i + 2] += n; }
  g.putImageData(id, 0, 0);
}
// draw coursed blocks that tile seamlessly horizontally
function coursedRows(S, rowHeights, minW, maxW, r, draw) {
  let y = 0;
  for (const rh of rowHeights) {
    const ws = []; let sum = 0;
    while (sum < S) { const w = minW + r() * (maxW - minW); ws.push(w); sum += w; }
    const k = S / sum; let x = r() * S;
    for (const w0 of ws) {
      const w = w0 * k;
      for (const ox of [0, -S]) { const xx = x + ox; if (xx + w < 0 || xx > S) continue; draw(xx, y, w, rh); }
      x += w; if (x > S) x -= S;
    }
    y += rh;
  }
}

function makeStoneTextures() {
  const S = 1024, r = mulberry32(11);
  const c = makeCanvas(S), g = c.getContext('2d');
  const cb = makeCanvas(S), gb = cb.getContext('2d');
  g.fillStyle = '#675e4c'; g.fillRect(0, 0, S, S);
  gb.fillStyle = '#2a2a2a'; gb.fillRect(0, 0, S, S);
  const rows = new Array(14).fill(S / 14);
  coursedRows(S, rows, 70, 210, r, (x, y, w, h) => {
    const hue = 36 + r() * 12, sat = 12 + r() * 8, L = 50 + r() * 11;
    g.fillStyle = `hsl(${hue},${sat}%,${L}%)`;
    g.fillRect(x + 3, y + 3, w - 6, h - 6);
    const gr = g.createLinearGradient(0, y, 0, y + h);
    gr.addColorStop(0, 'rgba(255,240,210,0.13)'); gr.addColorStop(0.45, 'rgba(0,0,0,0)'); gr.addColorStop(1, 'rgba(0,0,0,0.2)');
    g.fillStyle = gr; g.fillRect(x + 3, y + 3, w - 6, h - 6);
    const bv = 150 + r() * 90;
    gb.fillStyle = `rgb(${bv},${bv},${bv})`; gb.fillRect(x + 4, y + 4, w - 8, h - 8);
  });
  for (let i = 0; i < 70; i++) {
    const x = r() * S, y = r() * S, rad = 30 + r() * 130;
    const gr = g.createRadialGradient(x, y, 0, x, y, rad);
    const green = r() < 0.55;
    gr.addColorStop(0, green ? 'rgba(78,96,38,0.26)' : 'rgba(28,22,14,0.2)');
    gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
  for (let i = 0; i < 45; i++) {
    g.fillStyle = `rgba(22,18,10,${0.04 + r() * 0.08})`;
    g.fillRect(r() * S, r() * S, 4 + r() * 16, 80 + r() * 320);
  }
  addNoise(g, S, 24, r); addNoise(gb, S, 40, r);
  return [toTex(c), toTex(cb, false)];
}

function makeRoofTexture() {
  const S = 512, r = mulberry32(21);
  const c = makeCanvas(S), g = c.getContext('2d');
  g.fillStyle = '#1b201d'; g.fillRect(0, 0, S, S);
  const rows = 16, rh = S / rows;
  // bottom row first so upper rows overlap lower ones
  for (let row = rows - 1; row >= 0; row--) {
    const y = row * rh;
    const ws = []; let sum = 0;
    while (sum < S) { const w = 20 + r() * 18; ws.push(w); sum += w; }
    const k = S / sum; let x = r() * S;
    for (const w0 of ws) {
      const w = w0 * k;
      const hue = 140 + r() * 70, sat = 5 + r() * 12, L = 22 + r() * 16;
      for (const ox of [0, -S]) {
        const xx = x + ox; if (xx + w < 0 || xx > S) continue;
        g.fillStyle = `hsl(${hue},${sat}%,${L}%)`;
        g.beginPath();
        g.moveTo(xx + 1, y); g.lineTo(xx + w - 1, y); g.lineTo(xx + w - 1, y + rh * 0.8);
        g.quadraticCurveTo(xx + w / 2, y + rh * 1.2, xx + 1, y + rh * 0.8);
        g.closePath(); g.fill();
        g.strokeStyle = 'rgba(0,0,0,0.35)'; g.lineWidth = 1.2; g.stroke();
      }
      x += w; if (x > S) x -= S;
    }
  }
  for (let i = 0; i < 160; i++) {
    const x = r() * S, y = r() * S, rad = 6 + r() * 40;
    const gr = g.createRadialGradient(x, y, 0, x, y, rad);
    gr.addColorStop(0, r() < 0.7 ? 'rgba(112,128,52,0.35)' : 'rgba(170,160,90,0.3)');
    gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
  addNoise(g, S, 16, r);
  return toTex(c);
}

function makeGlassTextures() {
  const S = 256, r = mulberry32(31);
  const c = makeCanvas(S), g = c.getContext('2d');
  const ce = makeCanvas(S), ge = ce.getContext('2d');
  for (let i = 0; i < 40; i++) {
    const x = r() * S, y = r() * S, w = 30 + r() * 80;
    g.fillStyle = `hsl(${200 + r() * 30},${14 + r() * 14}%,${20 + r() * 14}%)`; g.fillRect(x - w / 2, y - w / 2, w, w);
    ge.fillStyle = `hsl(${30 + r() * 12},${80 + r() * 15}%,${45 + r() * 22}%)`; ge.fillRect(x - w / 2, y - w / 2, w, w);
  }
  const step = S / 4;
  for (const gg of [g, ge]) {
    gg.strokeStyle = '#100e0b'; gg.lineWidth = 5;
    for (let i = -8; i <= 8; i++) {
      gg.beginPath(); gg.moveTo(i * step, 0); gg.lineTo(i * step + S, S); gg.stroke();
      gg.beginPath(); gg.moveTo(i * step, 0); gg.lineTo(i * step - S, S); gg.stroke();
    }
  }
  return [toTex(c), toTex(ce)];
}

function makePaveTexture() {
  const S = 1024, r = mulberry32(41);
  const c = makeCanvas(S), g = c.getContext('2d');
  g.fillStyle = '#5f594d'; g.fillRect(0, 0, S, S);
  const hs = []; let sum = 0;
  while (sum < S) { const h = 70 + r() * 90; hs.push(h); sum += h; }
  coursedRows(S, hs.map((h) => (h * S) / sum), 90, 260, r, (x, y, w, h) => {
    g.fillStyle = `hsl(${32 + r() * 14},${6 + r() * 8}%,${50 + r() * 16}%)`;
    g.fillRect(x + 4, y + 4, w - 8, h - 8);
    const gr = g.createLinearGradient(x, y, x + w, y + h);
    gr.addColorStop(0, 'rgba(255,255,255,0.06)'); gr.addColorStop(1, 'rgba(0,0,0,0.14)');
    g.fillStyle = gr; g.fillRect(x + 4, y + 4, w - 8, h - 8);
  });
  for (let i = 0; i < 90; i++) {
    const x = r() * S, y = r() * S, rad = 20 + r() * 90;
    const gr = g.createRadialGradient(x, y, 0, x, y, rad);
    gr.addColorStop(0, r() < 0.5 ? 'rgba(60,70,30,0.18)' : 'rgba(30,26,20,0.16)');
    gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
  // a few fallen leaves on stone
  for (let i = 0; i < 160; i++) {
    g.fillStyle = `hsla(${14 + r() * 32},${70 + r() * 20}%,${36 + r() * 18}%,0.9)`;
    g.beginPath(); g.ellipse(r() * S, r() * S, 3 + r() * 5, 2 + r() * 3, r() * 3, 0, TAU); g.fill();
  }
  addNoise(g, S, 20, r);
  return toTex(c);
}

function makeGrassTexture(green = false) {
  const S = 512, r = mulberry32(51);
  const c = makeCanvas(S), g = c.getContext('2d');
  g.fillStyle = green ? '#4a6a26' : '#58602f'; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 40; i++) {
    const x = r() * S, y = r() * S, rad = 30 + r() * 90;
    const gr = g.createRadialGradient(x, y, 0, x, y, rad);
    gr.addColorStop(0, r() < 0.5 ? (green ? 'rgba(110,140,50,0.35)' : 'rgba(140,120,50,0.35)') : 'rgba(40,60,20,0.35)');
    gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
  g.lineWidth = 1.2;
  for (let i = 0; i < 9000; i++) {
    const x = r() * S, y = r() * S, l = 3 + r() * 8, a = -Math.PI / 2 + (r() - 0.5) * 1.2;
    g.strokeStyle = green ? `hsl(${78 + r() * 34},${38 + r() * 25}%,${18 + r() * 26}%)` : `hsl(${55 + r() * 40},${30 + r() * 25}%,${18 + r() * 26}%)`;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); g.stroke();
  }
  for (let i = 0; i < (green ? 160 : 520); i++) {
    g.fillStyle = green ? (r() < 0.5 ? 'rgba(250,250,240,0.9)' : 'rgba(245,215,80,0.9)') : `hsla(${10 + r() * 36},${70 + r() * 25}%,${34 + r() * 22}%,0.95)`;
    g.beginPath(); g.ellipse(r() * S, r() * S, green ? 1.6 : 2 + r() * 3.5, green ? 1.6 : 1.2 + r() * 2, r() * 3, 0, TAU); g.fill();
  }
  addNoise(g, S, 18, r);
  return toTex(c);
}

function makeBarkTexture() {
  const c = makeCanvas(128, 256), g = c.getContext('2d'), r = mulberry32(61);
  g.fillStyle = '#ddd7ca'; g.fillRect(0, 0, 128, 256);
  for (let i = 0; i < 40; i++) { g.fillStyle = `rgba(150,140,120,${r() * 0.25})`; g.fillRect(r() * 128, r() * 256, 10 + r() * 40, 4 + r() * 20); }
  for (let i = 0; i < 90; i++) { g.fillStyle = `rgba(30,26,22,${0.5 + r() * 0.5})`; g.fillRect(r() * 128, r() * 256, 4 + r() * 22, 1 + r() * 3.5); }
  return toTex(c);
}

function makeLeafClusterTexture() {
  const S = 256, r = mulberry32(71);
  const c = makeCanvas(S), g = c.getContext('2d');
  g.strokeStyle = 'rgba(60,45,30,0.9)'; g.lineWidth = 2;
  for (let i = 0; i < 6; i++) {
    g.beginPath(); g.moveTo(S / 2, S / 2);
    g.lineTo(S / 2 + (r() - 0.5) * S * 0.7, S / 2 + (r() - 0.5) * S * 0.7); g.stroke();
  }
  for (let i = 0; i < 120; i++) {
    const ang = r() * TAU, dist = Math.pow(r(), 0.6) * (S * 0.42);
    const x = S / 2 + Math.cos(ang) * dist, y = S / 2 + Math.sin(ang) * dist;
    const len = 16 + r() * 18, wid = len * 0.6;
    const L = Math.floor(165 + r() * 90);
    g.save(); g.translate(x, y); g.rotate(r() * TAU);
    g.fillStyle = `rgb(${L},${L},${Math.floor(L * 0.96)})`;
    g.beginPath(); g.moveTo(0, -len / 2);
    g.quadraticCurveTo(wid, -len * 0.05, 0, len / 2);
    g.quadraticCurveTo(-wid, -len * 0.05, 0, -len / 2);
    g.fill();
    g.strokeStyle = 'rgba(0,0,0,0.22)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, -len / 2); g.lineTo(0, len / 2); g.stroke();
    g.restore();
  }
  return toTex(c, true, false);
}

function makeSingleLeafTexture() {
  const S = 64, c = makeCanvas(S), g = c.getContext('2d');
  g.translate(S / 2, S / 2 + 4);
  g.fillStyle = '#fff';
  g.beginPath();
  for (let k = 0; k <= 60; k++) {
    const a = -Math.PI / 2 + (k / 60) * TAU;
    const lobes = Math.abs(Math.cos(2.5 * (a + Math.PI / 2)));
    const rad = 24 * (0.5 + 0.5 * Math.pow(lobes, 0.6)) * (a > 0.2 && a < 2.9 ? 0.7 : 1);
    const x = Math.cos(a) * rad, y = Math.sin(a) * rad;
    k === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
  }
  g.fill();
  g.strokeStyle = 'rgba(0,0,0,0.25)'; g.lineWidth = 1.5;
  for (let k = 0; k < 5; k++) { const a = -Math.PI / 2 + (k - 2) * 0.75; g.beginPath(); g.moveTo(0, 0); g.lineTo(Math.cos(a) * 18, Math.sin(a) * 18); g.stroke(); }
  g.beginPath(); g.moveTo(0, 0); g.lineTo(0, 26); g.lineWidth = 2; g.stroke();
  return toTex(c, true, false);
}

function makeGlowTexture(inner, outer, stretch = 1) {
  const c = makeCanvas(64, 64 * stretch), g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32 * stretch * 0.62, 0, 32, 32 * stretch * 0.62, 32);
  gr.addColorStop(0, inner); gr.addColorStop(0.35, outer); gr.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = gr;
  g.save(); g.scale(1, stretch); g.fillRect(0, 0, 64, 64); g.restore();
  return toTex(c, true, false);
}

function makeMistTexture() {
  const S = 256, c = makeCanvas(S), g = c.getContext('2d'), r = mulberry32(81);
  for (let i = 0; i < 26; i++) {
    const x = S / 2 + (r() - 0.5) * S * 0.5, y = S / 2 + (r() - 0.5) * S * 0.25, rad = 40 + r() * 60;
    const gr = g.createRadialGradient(x, y, 0, x, y, rad);
    gr.addColorStop(0, 'rgba(255,255,255,0.18)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, S, S);
  }
  return toTex(c, true, false);
}

function makeFlagTexture() {
  const c = makeCanvas(256, 160), g = c.getContext('2d');
  g.fillStyle = '#7d1b1b'; g.fillRect(0, 0, 256, 160);
  g.strokeStyle = '#d8aa4c'; g.lineWidth = 8; g.strokeRect(10, 10, 236, 140);
  g.fillStyle = '#d8aa4c';
  g.beginPath(); g.moveTo(128, 34); g.lineTo(160, 80); g.lineTo(128, 126); g.lineTo(96, 80); g.closePath(); g.fill();
  g.fillStyle = '#7d1b1b'; g.beginPath(); g.arc(128, 80, 13, 0, TAU); g.fill();
  return toTex(c, true, false);
}

/* =====================================================================
   geometry helpers
   ===================================================================== */
const _q = new THREE.Quaternion(), _e = new THREE.Euler(), _v = new THREE.Vector3(), _s = new THREE.Vector3();
function M(x, y, z, ry = 0, sx = 1, sy = 1, sz = 1, rx = 0, rz = 0) {
  _e.set(rx, ry, rz, 'YXZ'); _q.setFromEuler(_e);
  return new THREE.Matrix4().compose(_v.set(x, y, z), _q, _s.set(sx, sy, sz));
}
const T = (x, y, z) => new THREE.Matrix4().makeTranslation(x, y, z);

function scaleUV(g, su, sv) {
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  return g;
}
function worldUV(g, s) {
  const p = g.attributes.position, n = g.attributes.normal;
  let uv = g.attributes.uv;
  if (!uv) { uv = new THREE.BufferAttribute(new Float32Array(p.count * 2), 2); g.setAttribute('uv', uv); }
  for (let i = 0; i < p.count; i += 3) {
    const ax = Math.abs(n.getX(i) + n.getX(i + 1) + n.getX(i + 2));
    const ay = Math.abs(n.getY(i) + n.getY(i + 1) + n.getY(i + 2));
    const az = Math.abs(n.getZ(i) + n.getZ(i + 1) + n.getZ(i + 2));
    for (let k = 0; k < 3; k++) {
      const j = i + k, x = p.getX(j), y = p.getY(j), z = p.getZ(j);
      let u, v;
      if (ay >= ax && ay >= az) { u = x; v = z; } else if (ax >= az) { u = z; v = y; } else { u = x; v = y; }
      uv.setXY(j, u * s, v * s);
    }
  }
}

// merges lots of small parts into one mesh per material
class Batcher {
  constructor() { this.parts = new Map(); }
  add(geo, mat, matrix, uvMode = 'world', uvArg = 1 / 6) {
    let g = geo.index ? geo.toNonIndexed() : geo.clone();
    g.applyMatrix4(matrix);
    for (const k of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(k)) g.deleteAttribute(k);
    if (uvMode === 'world') worldUV(g, uvArg);
    else if (uvMode === 'scale') scaleUV(g, uvArg[0], uvArg[1]);
    if (!g.attributes.uv) g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
    g.clearGroups();
    if (!this.parts.has(mat)) this.parts.set(mat, []);
    this.parts.get(mat).push(g);
    geo.dispose();
  }
  build(parent, shadows = true) {
    for (const [mat, geos] of this.parts) {
      const mesh = new THREE.Mesh(mergeGeometries(geos, false), mat);
      mesh.castShadow = shadows; mesh.receiveShadow = true;
      parent.add(mesh);
    }
    this.parts.clear();
  }
}

// pointed (equilateral gothic) arch, base centred at (ox, oy)
function archShape(w, h, ox = 0, oy = 0) {
  const s = new THREE.Shape(), hw = w / 2, spring = oy + h - w * 0.866;
  s.moveTo(ox - hw, oy); s.lineTo(ox + hw, oy); s.lineTo(ox + hw, spring);
  s.absarc(ox - hw, spring, w, 0, Math.PI / 3, false);
  s.absarc(ox + hw, spring, w, (2 * Math.PI) / 3, Math.PI, false);
  s.lineTo(ox - hw, oy);
  return s;
}
function roundArchShape(w, h, ox = 0, oy = 0) {
  const s = new THREE.Shape(), hw = w / 2;
  s.moveTo(ox - hw, oy); s.lineTo(ox + hw, oy); s.lineTo(ox + hw, oy + h - hw);
  s.absarc(ox, oy + h - hw, hw, 0, Math.PI, false);
  s.lineTo(ox - hw, oy);
  return s;
}

/* =====================================================================
   main build (async so the loader can paint)
   ===================================================================== */
let sunLight, moonLight, hemi, skyU, skyScene, pmrem, envRT;
let glassMat, lampMat, ghGlassMat, fogColor = new THREE.Color();
const pointLights = []; const flames = []; const birds = []; const mists = []; const flags = [];
let fallingLeaves;
let interior = null;
const leafSpawnPoints = [];
let materials = {};
const TX = {}; // textures shared with the interior
// collision data for first-person walking outside
const WALK = { outlineR: null, rails: [], circles: [], trunks: [] };

// castle layout constants
const CX = -8, CZ = -12, PRX = 58, PRZ = 44; // plateau
const OV = new THREE.Vector3(42, 0, 54); // overlook terrace centre
const VS = new THREE.Vector3(8, 0, 22); // viaduct start
const VDIR = OV.clone().sub(VS).setY(0).normalize();
const VPERP = new THREE.Vector3(-VDIR.z, 0, VDIR.x);
const VE = OV.clone().addScaledVector(VDIR, -9.6);
const GH = { x: 92, z: -70 }; // greenhouse in the valley

async function build() {
  setProgress(0.05); await nextFrame();

  /* ---------- textures & materials ---------- */
  const [stoneTex, stoneBump] = makeStoneTextures();
  const roofTex = makeRoofTexture();
  const [glassTex, glassEmis] = makeGlassTextures();
  const paveTex = makePaveTexture();
  const grassTex = makeGrassTexture();
  const barkTex = makeBarkTexture();
  const leafTex = makeLeafClusterTexture();
  const oneLeafTex = makeSingleLeafTexture();
  Object.assign(TX, { stone: stoneTex, bump: stoneBump, pave: paveTex, leaf: leafTex, bark: barkTex });
  setProgress(0.2); await nextFrame();

  const stoneMat = new THREE.MeshStandardMaterial({ map: stoneTex, bumpMap: stoneBump, bumpScale: 0.6, roughness: 0.92, color: 0xf4e6c8 });
  const trimMat = new THREE.MeshStandardMaterial({ map: stoneTex, bumpMap: stoneBump, bumpScale: 0.8, roughness: 0.9, color: 0xcfc0a2 });
  const roofMat = new THREE.MeshStandardMaterial({ map: roofTex, roughness: 0.7, metalness: 0.05, side: THREE.DoubleSide });
  glassMat = new THREE.MeshStandardMaterial({ map: glassTex, emissiveMap: glassEmis, emissive: 0xffffff, emissiveIntensity: 0.05, roughness: 0.12, metalness: 0.75 });
  const woodMat = new THREE.MeshStandardMaterial({ color: 0x5a3a22, roughness: 0.8 });
  const ironMat = new THREE.MeshStandardMaterial({ color: 0x24211d, roughness: 0.45, metalness: 0.7 });
  lampMat = new THREE.MeshStandardMaterial({ color: 0xfff0d0, emissive: 0xffb455, emissiveIntensity: 0.4, roughness: 0.3 });
  const paveMat = new THREE.MeshStandardMaterial({ map: paveTex, roughness: 0.85, color: 0xf0e6d4 });
  const grassMat = new THREE.MeshStandardMaterial({ map: grassTex, roughness: 1 });
  const copperMat = new THREE.MeshStandardMaterial({ color: 0x5d8c78, roughness: 0.45, metalness: 0.55 });
  ghGlassMat = new THREE.MeshStandardMaterial({ color: 0xa8d4c4, roughness: 0.05, metalness: 0.2, transparent: true, opacity: 0.38, depthWrite: false, emissive: 0xffc070, emissiveIntensity: 0, side: THREE.DoubleSide });
  materials = { stoneMat, trimMat, roofMat };
  patchMat(stoneMat, { wetK: 0.8 }); patchMat(trimMat, { wetK: 0.8 }); patchMat(roofMat, { snowK: 1.15 });
  patchMat(paveMat, {}); patchMat(grassMat, { map2: makeGrassTexture(true) });

  /* ---------- sky & lights ---------- */
  buildSky();
  hemi = new THREE.HemisphereLight(0xbfd4ff, 0x5a4a30, 0.6);
  scene.add(hemi);
  sunLight = new THREE.DirectionalLight(0xfff0d8, 3);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(4096, 4096);
  const sc = sunLight.shadow.camera;
  sc.left = -120; sc.right = 120; sc.top = 120; sc.bottom = -120; sc.near = 10; sc.far = 900;
  sunLight.shadow.bias = -0.0004; sunLight.shadow.normalBias = 0.06;
  sunLight.target.position.set(5, 0, 5);
  scene.add(sunLight, sunLight.target);
  moonLight = new THREE.DirectionalLight(0x9fb6ff, 0);
  moonLight.position.set(-300, 420, -500);
  scene.add(moonLight);
  scene.fog = new THREE.FogExp2(0xcfd6d8, 0.00085);

  setProgress(0.28); await nextFrame();

  /* ---------- terrain & rock ---------- */
  buildTerrain();
  setProgress(0.4); await nextFrame();
  const plateau = buildRock({ cx: CX, cz: CZ, rx: PRX, rz: PRZ, H: 95, seed: 3, flare: 0.28 });
  const pillar = buildRock({ cx: OV.x, cz: OV.z, rx: 11.5, rz: 11.5, H: 95, seed: 9, flare: 0.5 });
  WALK.outlineR = plateau.outlineR;
  // grass on top of plateau
  {
    const shape = new THREE.Shape(plateau.outline.map((p) => new THREE.Vector2(p.x, -p.z)));
    const g = new THREE.ShapeGeometry(shape, 1); g.rotateX(-Math.PI / 2); scaleUV(g, 1 / 5, 1 / 5);
    const m = new THREE.Mesh(g, grassMat); m.receiveShadow = true; m.position.y = 0.01; scene.add(m);
  }
  setProgress(0.5); await nextFrame();

  /* ---------- castle ---------- */
  const castle = new THREE.Group(); scene.add(castle);
  const b = new Batcher();
  const TEX = 1 / 6, RTEX = 1 / 4;
  const box = (mat, w, h, d, x, y, z, ry = 0) => b.add(new THREE.BoxGeometry(w, h, d), mat, M(x, y + h / 2, z, ry), 'world', mat === roofMat ? RTEX : TEX);
  const cyl = (mat, rt, rb, h, x, y, z, seg = 40) => b.add(new THREE.CylinderGeometry(rt, rb, h, seg, 1), mat, M(x, y + h / 2, z), 'scale', [TAU * rb * TEX, h * TEX]);
  const prism = (mat, w, h, len, m) => {
    const s = new THREE.Shape(); s.moveTo(-w / 2, 0); s.lineTo(w / 2, 0); s.lineTo(0, h); s.closePath();
    b.add(new THREE.ExtrudeGeometry(s, { depth: len, bevelEnabled: false }), mat, m, 'world', mat === roofMat ? RTEX : TEX);
  };
  const finial = (x, y, z, h = 2.4) => {
    b.add(new THREE.CylinderGeometry(0.06, 0.08, h, 6), ironMat, M(x, y + h / 2, z));
    b.add(new THREE.SphereGeometry(0.2, 10, 8), ironMat, M(x, y + h * 0.55, z));
  };
  const bellRoof = (x, y, z, R, H, seg = 40) => {
    const pts = [new THREE.Vector2(R + 0.9, -0.5), new THREE.Vector2(R + 0.35, 0.25)];
    const n = 14;
    for (let i = 1; i <= n; i++) { const s = i / n; pts.push(new THREE.Vector2(Math.max(R * Math.pow(1 - s, 1.18), 0.02), s * H)); }
    const slant = Math.hypot(R, H);
    b.add(new THREE.LatheGeometry(pts, seg), roofMat, M(x, y, z), 'scale', [TAU * R * RTEX, slant * RTEX]);
    finial(x, y + H - 0.2, z, 3.2);
  };
  const addWindow = (x, y, z, ry, w, h, opts = {}) => {
    const W = M(x, y, z, ry);
    const depth = opts.depth ?? 0.38;
    const outer = archShape(w + 0.7, h + 0.7, 0, -0.35);
    outer.holes.push(archShape(w, h, 0, 0));
    b.add(new THREE.ExtrudeGeometry(outer, { depth, bevelEnabled: false, curveSegments: 10 }), trimMat, W, 'world', TEX);
    b.add(new THREE.ShapeGeometry(archShape(w, h), 10), glassMat, W.clone().multiply(T(0, 0, 0.07)), 'scale', [1.25, 1.25]);
    b.add(new THREE.BoxGeometry(w + 1.0, 0.24, 0.6), trimMat, W.clone().multiply(T(0, -0.45, 0.22)));
    if (w >= 1.9) {
      const mh = h - w * 0.8;
      b.add(new THREE.BoxGeometry(0.14, mh, 0.14), trimMat, W.clone().multiply(T(0, mh / 2, 0.12)));
      b.add(new THREE.BoxGeometry(w, 0.12, 0.14), trimMat, W.clone().multiply(T(0, mh * 0.55, 0.12)));
    }
  };
  const towerWindows = (x, z, r, y, count, w, h, a0 = 0) => {
    for (let i = 0; i < count; i++) {
      const a = a0 + (i / count) * TAU;
      addWindow(x + Math.sin(a) * (r - 0.04), y, z + Math.cos(a) * (r - 0.04), a, w, h);
    }
  };
  const tower = (o) => {
    const { x, z, r, h } = o;
    cyl(stoneMat, r, r * 1.03, h, x, 0, z, 44);
    cyl(trimMat, r * 1.07, r * 1.1, 2.2, x, 0, z, 44);
    for (const by of o.bands ?? []) cyl(trimMat, r + 0.22, r + 0.22, 0.55, x, by, z, 44);
    // machicolation
    const ringR = r + 0.85;
    cyl(trimMat, ringR, ringR, 2.1, x, h - 0.3, z, 44);
    const nc = Math.floor((TAU * r) / 1.15);
    for (let i = 0; i < nc; i++) {
      const a = (i / nc) * TAU;
      b.add(new THREE.BoxGeometry(0.5, 1.3, 0.9), trimMat, M(x + Math.sin(a) * (r + 0.4), h - 1.5 + 0.65, z + Math.cos(a) * (r + 0.4), a));
    }
    const top = h + 1.8;
    if (o.crenel) {
      const nm = Math.floor((TAU * ringR) / 1.7);
      for (let i = 0; i < nm; i++) {
        const a = (i / nm) * TAU;
        b.add(new THREE.BoxGeometry(0.95, 1.3, 0.6), trimMat, M(x + Math.sin(a) * (ringR - 0.3), top + 0.65, z + Math.cos(a) * (ringR - 0.3), a));
      }
    }
    if (o.roofH) bellRoof(x, top - 0.05, z, r + 0.95, o.roofH);
    for (const wdef of o.windows ?? []) towerWindows(x, z, r, wdef[0], wdef[1], wdef[2], wdef[3], wdef[4] ?? 0);
    return top;
  };
  const pinnacle = (x, y, z, s = 1, h = 3) => {
    box(trimMat, 0.9 * s, h, 0.9 * s, x, y, z);
    const c = new THREE.ConeGeometry(0.78 * s, 3 * s, 4); c.rotateY(Math.PI / 4);
    b.add(c, roofMat, M(x, y + h + 1.5 * s, z), 'scale', [1, 1]);
    b.add(new THREE.SphereGeometry(0.16 * s, 8, 6), trimMat, M(x, y + h + 3 * s, z));
  };
  const buttress = (x, zFace, h, dir = 1) => {
    box(trimMat, 1.4, h * 0.55, 2.3, x, 0, zFace + dir * 1.15);
    box(trimMat, 1.2, h * 0.3, 1.5, x, h * 0.55, zFace + dir * 0.75);
    pinnacle(x, h * 0.85, zFace + dir * 0.6, 0.85, 3.2);
  };
  const merlonLine = (x0, z0, x1, z1, y, thick) => {
    const L = Math.hypot(x1 - x0, z1 - z0), n = Math.floor(L / 1.8), ry = Math.atan2(x1 - x0, z1 - z0);
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      box(trimMat, thick, 1.3, 0.95, lerp(x0, x1, t), y, lerp(z0, z1, t), ry);
    }
  };
  const wallSeg = (x0, z0, x1, z1, h, thick) => {
    const L = Math.hypot(x1 - x0, z1 - z0), ry = Math.atan2(x1 - x0, z1 - z0);
    box(stoneMat, thick, h, L, (x0 + x1) / 2, 0, (z0 + z1) / 2, ry);
    box(trimMat, thick + 0.5, 0.6, L, (x0 + x1) / 2, h - 0.3, (z0 + z1) / 2, ry);
    merlonLine(x0, z0, x1, z1, h + 0.3, thick * 0.35);
  };

  // --- main hall (front at z=4) ---
  const hallX0 = -28, hallX1 = 14, hallZ0 = -10, hallZ1 = 4, hallH = 22;
  const hallCX = (hallX0 + hallX1) / 2, hallCZ = (hallZ0 + hallZ1) / 2, hallW = hallX1 - hallX0, hallD = hallZ1 - hallZ0;
  box(stoneMat, hallW, hallH, hallD, hallCX, 0, hallCZ);
  box(trimMat, hallW + 0.6, 1.6, hallD + 0.6, hallCX, 0, hallCZ);
  box(trimMat, hallW + 0.4, 0.5, hallD + 0.4, hallCX, 11.4, hallCZ);
  box(trimMat, hallW + 1.0, 0.9, hallD + 1.0, hallCX, hallH - 0.5, hallCZ);
  for (let x = hallX0 + 0.8; x < hallX1; x += 1.3) {
    b.add(new THREE.BoxGeometry(0.42, 0.6, 0.5), trimMat, M(x, hallH - 0.9, hallZ1 + 0.3));
    b.add(new THREE.BoxGeometry(0.42, 0.6, 0.5), trimMat, M(x, hallH - 0.9, hallZ0 - 0.3));
  }
  prism(roofMat, hallD + 1.8, 11, hallW + 1.0, M(hallX0 - 0.5, hallH + 0.35, hallCZ, Math.PI / 2));
  box(ironMat, hallW, 0.25, 0.3, hallCX, hallH + 11.2, hallCZ);
  for (let x = hallX0 + 4; x < hallX1; x += 8) finial(x, hallH + 11.3, hallCZ, 1.6);
  // front buttresses & windows
  for (const bx of [-20.5, -13.4, 7.1]) buttress(bx, hallZ1, hallH);
  for (const wx of [-23.5, -17, 4.2, 10.2]) {
    addWindow(wx, 3.2, hallZ1, 0, 2.3, 7.4);
    addWindow(wx, 13.6, hallZ1, 0, 2.0, 5.6);
  }
  // back windows
  for (let wx = -22; wx <= 10; wx += 5.4) addWindow(wx, 13.6, hallZ0, Math.PI, 1.9, 5.4);
  // dormers on front roof slope
  for (const dx of [-23.5, -17, 4.2, 10.2]) {
    box(stoneMat, 2.3, 2.8, 3.2, dx, 24.2, 1.9);
    addWindow(dx, 24.6, 3.5, 0, 1.1, 1.9, { depth: 0.2 });
    prism(roofMat, 3.0, 1.6, 3.6, M(dx, 27.0, 0.1));
  }

  // --- central gothic bay & portal (front at z=7.5) ---
  const bayX = -6, bayZf = 7.5;
  box(stoneMat, 14, 28, 5, bayX, 0, 5);
  box(trimMat, 14.6, 1.8, 5.6, bayX, 0, 5);
  box(trimMat, 14.5, 0.7, 5.5, bayX, 27.3, 5);
  prism(stoneMat, 14.4, 8, 1.2, M(bayX, 28, bayZf - 1.2));
  prism(roofMat, 15.6, 8.8, 4.6, M(bayX, 27.8, 1.9));
  finial(bayX, 35.8, bayZf - 0.6, 2.6);
  for (const px of [bayX - 7.25, bayX + 7.25]) {
    cyl(trimMat, 0.95, 1.05, 31, px, 0, bayZf - 0.2, 8);
    b.add(new THREE.ConeGeometry(1.05, 7.5, 8), roofMat, M(px, 31 + 3.75, bayZf - 0.2), 'scale', [2, 2]);
    b.add(new THREE.SphereGeometry(0.22, 8, 6), trimMat, M(px, 38.8, bayZf - 0.2));
  }
  for (const [wx, ww] of [[bayX - 2.8, 1.9], [bayX, 2.1], [bayX + 2.8, 1.9]]) addWindow(wx, 11.5, bayZf, 0, ww, wx === bayX ? 12.2 : 10.8);
  // rose window
  {
    const ring = new THREE.Shape(); ring.absarc(0, 0, 2.5, 0, TAU, false);
    const hole = new THREE.Path(); hole.absarc(0, 0, 1.95, 0, TAU, true); ring.holes.push(hole);
    b.add(new THREE.ExtrudeGeometry(ring, { depth: 0.45, bevelEnabled: false, curveSegments: 40 }), trimMat, M(bayX, 31.2, bayZf - 1.2 + 1.2), 'world', TEX);
    b.add(new THREE.CircleGeometry(1.95, 40), glassMat, M(bayX, 31.2, bayZf + 0.05), 'scale', [5, 5]);
    for (let i = 0; i < 8; i++) b.add(new THREE.BoxGeometry(0.12, 3.9, 0.12), trimMat, M(bayX, 31.2, bayZf + 0.15, 0, 1, 1, 1, 0, (i / 8) * Math.PI));
  }
  // recessed portal: stepped archivolts
  for (let k = 0; k < 4; k++) {
    const w = 4.2 + k * 0.9, h = 7.0 + k * 0.75;
    const s = archShape(w + 0.9, h + 0.6, 0, 0); s.holes.push(archShape(w, h, 0, 0));
    b.add(new THREE.ExtrudeGeometry(s, { depth: 0.35 + (3 - k) * 0.25, bevelEnabled: false, curveSegments: 10 }), trimMat, M(bayX, 0.8, bayZf), 'world', TEX);
  }
  b.add(new THREE.ShapeGeometry(archShape(4.2, 7.0), 10), woodMat, M(bayX, 0.8, bayZf + 0.02));
  for (let i = 0; i < 6; i++) b.add(new THREE.BoxGeometry(0.1, 5.0, 0.06), ironMat, M(bayX - 1.7 + i * 0.68, 3.3, bayZf + 0.06));
  for (let s = 0; s < 3; s++) box(trimMat, 8 - s * 0.8, 0.4, 1.0, bayX, 0.8 - s * 0.4 - 0.4, bayZf + 0.5 + s * 1.0);

  // --- great left tower ---
  tower({ x: -34, z: -3, r: 10, h: 36, roofH: 22, bands: [12, 24], windows: [[5, 7, 1.9, 6.2, 0.35], [15, 7, 1.7, 6, 0.8], [27, 8, 1.5, 5.2, 0.2]] });
  // --- right tower & turret ---
  tower({ x: 18, z: -1, r: 6.5, h: 40, roofH: 17, bands: [14, 28], windows: [[6, 4, 1.4, 4.6, 0.3], [18, 5, 1.4, 5.2, 0.0], [30, 5, 1.3, 5.0, 0.6]] });
  tower({ x: 23.6, z: -6.5, r: 3.4, h: 47, roofH: 11.5, windows: [[40, 3, 0.9, 2.6, 0.5]] });
  // --- keep ---
  {
    const kx = -8, kz = -22, kw = 18, kd = 14, kh = 50;
    box(stoneMat, kw, kh, kd, kx, 0, kz);
    for (const by of [20, 34]) box(trimMat, kw + 0.5, 0.6, kd + 0.5, kx, by, kz);
    box(trimMat, kw + 1.6, 2.0, kd + 1.6, kx, kh - 0.4, kz);
    for (let x = kx - kw / 2 + 0.6; x < kx + kw / 2; x += 1.2) {
      b.add(new THREE.BoxGeometry(0.45, 1.2, 0.8), trimMat, M(x, kh - 1.0, kz + kd / 2 + 0.4));
      b.add(new THREE.BoxGeometry(0.45, 1.2, 0.8), trimMat, M(x, kh - 1.0, kz - kd / 2 - 0.4));
    }
    const pr = new THREE.ConeGeometry((kw + 1.6) / Math.SQRT2, 22, 4, 1); pr.rotateY(Math.PI / 4);
    b.add(pr, roofMat, M(kx, kh + 1.6 + 11, kz, 0, 1, 1, (kd + 1.6) / (kw + 1.6)), 'scale', [16, 5.6]);
    finial(kx, kh + 1.6 + 21.5, kz, 4);
    for (const wx of [kx - 4.5, kx, kx + 4.5]) {
      addWindow(wx, 37, kz + kd / 2, 0, 2.1, 8);
      addWindow(wx, 22.5, kz + kd / 2, 0, 1.8, 6.5);
      addWindow(wx, 37, kz - kd / 2, Math.PI, 2.1, 8);
      addWindow(wx, 22.5, kz - kd / 2, Math.PI, 1.8, 6.5);
    }
    for (const wz of [kz - 3.5, kz + 3.5]) {
      addWindow(kx - kw / 2, 37, wz, -Math.PI / 2, 2.0, 8);
      addWindow(kx + kw / 2, 37, wz, Math.PI / 2, 2.0, 8);
      addWindow(kx - kw / 2, 23, wz, -Math.PI / 2, 1.7, 6);
      addWindow(kx + kw / 2, 23, wz, Math.PI / 2, 1.7, 6);
    }
    for (const [cxo, czo] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const tx = kx + cxo * (kw / 2 + 0.3), tz = kz + czo * (kd / 2 + 0.3);
      cyl(stoneMat, 1.8, 1.4, 12, tx, 40, tz, 20);
      cyl(trimMat, 2.2, 2.2, 1.0, tx, 51.5, tz, 20);
      bellRoof(tx, 52.5, tz, 2.2, 8.5, 20);
    }
  }
  // --- slender astronomy tower ---
  tower({ x: -22.5, z: -32.5, r: 4.2, h: 58, roofH: 17, bands: [20, 40], windows: [[10, 3, 1.1, 3.6, 0.2], [30, 3, 1.1, 3.6, 1.2], [50, 4, 1.2, 3.8, 0.4]] });
  // --- back hall ---
  box(stoneMat, 40, 18, 12, -10, 0, -36);
  box(trimMat, 41, 0.8, 13, -10, 17.4, -36);
  prism(roofMat, 13.8, 9, 41, M(-30.5, 18.2, -36, Math.PI / 2));
  for (let wx = -26; wx <= 6; wx += 5.3) { addWindow(wx, 9, -42, Math.PI, 1.8, 5.8); addWindow(wx, 1.8, -42, Math.PI, 1.6, 4.6); }
  // --- rear towers ---
  tower({ x: -38.5, z: -40, r: 7, h: 30, roofH: 15, bands: [14], windows: [[8, 5, 1.4, 4.6, 0.4], [20, 6, 1.4, 4.8, 0]] });
  tower({ x: 16, z: -38, r: 6, h: 33, roofH: 14, bands: [16], windows: [[8, 5, 1.3, 4.4, 0.2], [22, 5, 1.3, 4.6, 0.7]] });
  // --- curtain walls ---
  wallSeg(-42.5, -10, -44, -34, 15, 3);
  wallSeg(21.5, -9, 20.5, -33, 17, 3);

  setProgress(0.6); await nextFrame();

  /* ---------- forecourt, viaduct, overlook ---------- */
  const decor = new Batcher();
  const Lp = VS.clone().addScaledVector(VPERP, 3.45), Rp = VS.clone().addScaledVector(VPERP, -3.45);
  {
    const pts = [[-20.5, 7.5], [Rp.x, 7.5], [Rp.x, Rp.z], [Lp.x, Lp.z], [-20.5, Lp.z]];
    const shape = new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, -z)));
    const g = new THREE.ShapeGeometry(shape); g.rotateX(-Math.PI / 2); scaleUV(g, 1 / 8, 1 / 8);
    const m = new THREE.Mesh(g, paveMat); m.position.y = 0.04; m.receiveShadow = true; scene.add(m);
    // kerb
    for (let i = 0; i < pts.length; i++) {
      const [x0, z0] = pts[i], [x1, z1] = pts[(i + 1) % pts.length];
      if (i === 2) continue; // viaduct opening
      const L = Math.hypot(x1 - x0, z1 - z0);
      decor.add(new THREE.BoxGeometry(0.35, 0.22, L), trimMat, M((x0 + x1) / 2, 0.11, (z0 + z1) / 2, Math.atan2(x1 - x0, z1 - z0)));
    }
  }
  const vLen = VE.distanceTo(VS);
  const vRy = Math.atan2(VDIR.x, VDIR.z);
  const vMid = VS.clone().add(VE).multiplyScalar(0.5);
  decor.add(new THREE.BoxGeometry(7.2, 1.3, vLen + 1.2), trimMat, M(vMid.x, -0.62, vMid.z, vRy));
  decor.add(new THREE.BoxGeometry(6.6, 0.1, vLen + 1.2), paveMat, M(vMid.x, 0.05, vMid.z, vRy), 'world', 1 / 8);
  decor.add(new THREE.BoxGeometry(7.6, 0.45, vLen + 1.2), trimMat, M(vMid.x, -1.45, vMid.z, vRy));
  {
    // aqueduct-like arch wall under the deck
    const L = vLen, top = -1.6, bot = -95;
    const s = new THREE.Shape();
    s.moveTo(0, bot); s.lineTo(L, bot); s.lineTo(L, top); s.lineTo(0, top); s.lineTo(0, bot);
    const bay = L / 6;
    for (let i = 0; i < 6; i++) {
      const cx = bay * (i + 0.5);
      s.holes.push(roundArchShape(bay * 0.66, 8.5, cx, -12));
      s.holes.push(roundArchShape(bay * 0.62, 34, cx, -52));
    }
    const g = new THREE.ExtrudeGeometry(s, { depth: 5.2, bevelEnabled: false, curveSegments: 14 });
    const th = Math.atan2(-VDIR.z, VDIR.x);
    const origin = VS.clone().addScaledVector(VPERP, -2.6);
    decor.add(g, stoneMat, M(origin.x, 0, origin.z, th), 'world', TEX);
  }
  // overlook paving + rim
  decor.add(new THREE.CylinderGeometry(10.1, 10.1, 0.12, 64), paveMat, M(OV.x, 0.0, OV.z), 'world', 1 / 8);
  decor.add(new THREE.CylinderGeometry(10.5, 10.9, 1.9, 64, 1, true), trimMat, M(OV.x, -0.9, OV.z), 'scale', [TAU * 10.5 * TEX, 1.9 * TEX]);
  decor.add(new THREE.TorusGeometry(10.55, 0.28, 6, 80), trimMat, M(OV.x, -1.9, OV.z, 0, 1, 1, 1, Math.PI / 2));

  // balustrades
  const balusters = [];
  const lampSpots = [];
  const railSeg = (a, c) => {
    WALK.rails.push([a.x, a.z, c.x, c.z]);
    const L = a.distanceTo(c), ry = Math.atan2(c.x - a.x, c.z - a.z);
    const mx = (a.x + c.x) / 2, mz = (a.z + c.z) / 2;
    decor.add(new THREE.BoxGeometry(0.36, 0.16, L), trimMat, M(mx, 0.86 + a.y, mz, ry));
    decor.add(new THREE.BoxGeometry(0.3, 0.14, L), trimMat, M(mx, 0.07 + a.y, mz, ry));
    const n = Math.max(1, Math.floor(L / 0.42));
    for (let i = 0; i < n; i++) { const t = (i + 0.5) / n; balusters.push(new THREE.Vector3(lerp(a.x, c.x, t), a.y + 0.14, lerp(a.z, c.z, t))); }
  };
  const pedestal = (p, withLamp = false, urn = false) => {
    decor.add(new THREE.BoxGeometry(0.62, 1.1, 0.62), trimMat, M(p.x, p.y + 0.55, p.z));
    decor.add(new THREE.BoxGeometry(0.78, 0.14, 0.78), trimMat, M(p.x, p.y + 1.13, p.z));
    if (urn) {
      const prof = [[0.08, 0], [0.22, 0.05], [0.12, 0.18], [0.3, 0.42], [0.34, 0.55], [0.26, 0.62]].map(([x, y]) => new THREE.Vector2(x, y));
      decor.add(new THREE.LatheGeometry(prof, 14), trimMat, M(p.x, p.y + 1.2, p.z), 'scale', [1, 1]);
    }
    if (withLamp) lampSpots.push(new THREE.Vector3(p.x, p.y + 1.2, p.z));
  };
  const balLine = (pts, lampEvery = 0) => {
    let dist = 0, lastPed = -1e9, k = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], c = pts[i + 1], L = a.distanceTo(c);
      const n = Math.max(1, Math.round(L / 4.6));
      for (let j = 0; j < n; j++) {
        const p0 = a.clone().lerp(c, j / n), p1 = a.clone().lerp(c, (j + 1) / n);
        const d0 = p0.clone().lerp(p1, 0.07), d1 = p1.clone().lerp(p0, 0.07);
        railSeg(d0, d1);
        if (dist - lastPed > 0.5) { pedestal(p0, lampEvery && k % lampEvery === 1, !lampEvery && k % 2 === 0); lastPed = dist; k++; }
        dist += L / n;
      }
    }
    pedestal(pts[pts.length - 1], false, true);
  };
  const y0 = new THREE.Vector3(0, 0.05, 0);
  const V = (x, z) => new THREE.Vector3(x, 0.05, z);
  balLine([V(-20.5, Lp.z), V(Lp.x, Lp.z)]);
  balLine([Lp.clone().add(y0), VE.clone().addScaledVector(VPERP, 3.45).add(y0)], 2);
  balLine([V(Rp.x, 7.8), V(Rp.x, Rp.z), Rp.clone().add(y0), VE.clone().addScaledVector(VPERP, -3.45).add(y0)], 2);
  {
    const entry = Math.atan2(-VDIR.z, -VDIR.x), open = 0.37, R = 9.75;
    const pts = [];
    const n = 16;
    for (let i = 0; i <= n; i++) {
      const a = entry + open + (i / n) * (TAU - 2 * open);
      pts.push(new THREE.Vector3(OV.x + Math.cos(a) * R, 0.05, OV.z + Math.sin(a) * R));
    }
    balLine(pts);
    lampSpots.push(pts[4].clone().setY(1.25), pts[12].clone().setY(1.25));
  }
  {
    const prof = [[0.13, 0], [0.13, 0.07], [0.08, 0.11], [0.07, 0.18], [0.12, 0.34], [0.135, 0.42], [0.08, 0.56], [0.06, 0.62], [0.1, 0.66], [0.1, 0.72]].map(([x, y]) => new THREE.Vector2(x, y));
    const geo = new THREE.LatheGeometry(prof, 10);
    const im = new THREE.InstancedMesh(geo, trimMat, balusters.length);
    const d = new THREE.Object3D();
    balusters.forEach((p, i) => { d.position.copy(p); d.updateMatrix(); im.setMatrixAt(i, d.matrix); });
    im.castShadow = true; im.receiveShadow = true; scene.add(im);
  }
  // standalone lamp posts in the forecourt
  for (const [lx, lz] of [[-16.5, 21.5], [4.5, 12.5]]) {
    decor.add(new THREE.CylinderGeometry(0.35, 0.45, 0.5, 10), trimMat, M(lx, 0.25, lz));
    lampSpots.push(new THREE.Vector3(lx, 0.5, lz));
  }
  // lanterns
  lampSpots.forEach((p, i) => {
    WALK.circles.push([p.x, p.z, 0.45]);
    const hPost = 2.3;
    decor.add(new THREE.CylinderGeometry(0.06, 0.09, hPost, 8), ironMat, M(p.x, p.y + hPost / 2, p.z));
    decor.add(new THREE.BoxGeometry(0.4, 0.5, 0.4), lampMat, M(p.x, p.y + hPost + 0.25, p.z));
    const cap = new THREE.ConeGeometry(0.36, 0.35, 4); cap.rotateY(Math.PI / 4);
    decor.add(cap, ironMat, M(p.x, p.y + hPost + 0.68, p.z));
    decor.add(new THREE.BoxGeometry(0.5, 0.06, 0.5), ironMat, M(p.x, p.y + hPost, p.z));
    if (i % 2 === 0 && pointLights.length < 3) {
      const pl = new THREE.PointLight(0xffa850, 0, 0, 2); pl.position.set(p.x, p.y + hPost + 0.3, p.z);
      pl.userData.kind = 'lamp'; scene.add(pl); pointLights.push(pl);
    }
  });
  // braziers flanking the portal
  const flameTex = makeGlowTexture('rgba(255,245,210,1)', 'rgba(255,140,40,0.75)', 1.6);
  TX.flame = flameTex;
  for (const bx of [bayX - 5.2, bayX + 5.2]) {
    const bz = bayZf + 2.2;
    WALK.circles.push([bx, bz, 0.8]);
    decor.add(new THREE.BoxGeometry(0.9, 1.2, 0.9), trimMat, M(bx, 0.6, bz));
    const bowl = [[0.1, 0], [0.45, 0.15], [0.62, 0.45], [0.66, 0.55]].map(([x, y]) => new THREE.Vector2(x, y));
    decor.add(new THREE.LatheGeometry(bowl, 16), ironMat, M(bx, 1.2, bz), 'scale', [1, 1]);
    const grp = new THREE.Group(); grp.position.set(bx, 1.75, bz);
    for (let k = 0; k < 3; k++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: flameTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, color: new THREE.Color(4, 3, 2) }));
      sp.userData.base = [0.9 - k * 0.2, 1.5 - k * 0.3, k * 1.7];
      grp.add(sp);
    }
    scene.add(grp); flames.push(grp);
    const pl = new THREE.PointLight(0xff8a30, 10, 0, 2); pl.position.set(bx, 2.5, bz); pl.userData.kind = 'fire';
    scene.add(pl); pointLights.push(pl);
  }

  // flags
  const flagTex = makeFlagTexture();
  TX.flag = flagTex;
  const flagMat = new THREE.MeshStandardMaterial({ map: flagTex, side: THREE.DoubleSide, roughness: 0.85 });
  flagMat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = U.uTime; sh.uniforms.uWind = U.uWind;
    sh.vertexShader = 'uniform float uTime; uniform float uWind;\n' + sh.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
      float fx = position.x / 3.2;
      float amp = (0.25 + uWind * 0.55) * fx;
      transformed.z += sin(position.x * 2.1 - uTime * (4.0 + uWind * 5.0)) * amp;
      transformed.y += sin(position.x * 1.3 - uTime * 3.0) * amp * 0.25 - (1.0 - uWind) * fx * 0.35;`);
  };
  for (const [fx, fy, fz] of [[-8, 50 + 1.6 + 23.5, -22], [-34, 36 + 1.8 + 24, -3]]) {
    const g = new THREE.PlaneGeometry(3.2, 1.9, 24, 6); g.translate(1.6, -0.8, 0);
    const f = new THREE.Mesh(g, flagMat); f.position.set(fx, fy, fz); f.castShadow = true;
    scene.add(f); flags.push(f);
    b.add(new THREE.CylinderGeometry(0.05, 0.05, 3.4, 6), ironMat, M(fx, fy - 1.4, fz));
  }

  b.build(castle);
  decor.build(scene);

  setProgress(0.7); await nextFrame();

  /* ---------- greenhouse in the valley ---------- */
  buildGreenhouse(copperMat, trimMat);

  /* ---------- trees ---------- */
  buildTrees(barkTex, leafTex, plateau, pillar);
  setProgress(0.88); await nextFrame();

  /* ---------- atmosphere ---------- */
  buildFallingLeaves(oneLeafTex);
  buildBirds();
  buildMist();
  buildPrecip();

  setProgress(1);
}

/* =====================================================================
   sky
   ===================================================================== */
function buildSky() {
  skyU = {
    topColor: { value: new THREE.Color() }, horizonColor: { value: new THREE.Color() }, groundColor: { value: new THREE.Color() },
    sunColor: { value: new THREE.Color() }, sunDir: { value: new THREE.Vector3(0, 1, 0) }, moonDir: { value: new THREE.Vector3(-0.35, 0.5, -0.8).normalize() },
    uTime: U.uTime, night: { value: 0 }, uCloud: { value: 0.15 }, uOver: { value: 0 }, uFlash: { value: 0 },
  };
  const skyMat = new THREE.ShaderMaterial({
    uniforms: skyU, side: THREE.BackSide, depthWrite: false, fog: false,
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main(){
        vDir = position;
        vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = p.xyww;
      }`,
    fragmentShader: /* glsl */`
      varying vec3 vDir;
      uniform vec3 topColor, horizonColor, groundColor, sunColor, sunDir, moonDir;
      uniform float uTime, night, uCloud, uOver, uFlash;
      float hash3(vec3 p){ p = fract(p*0.3183099 + 0.1); p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }
      float h2(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
        return mix(mix(h2(i), h2(i+vec2(1,0)), f.x), mix(h2(i+vec2(0,1)), h2(i+vec2(1,1)), f.x), f.y); }
      float fbm(vec2 p){ float s = 0.0, a = 0.5; for(int i=0;i<5;i++){ s += a*vn(p); p = p*2.03 + 1.7; a *= 0.5; } return s; }
      void main(){
        vec3 d = normalize(vDir);
        float h = d.y;
        vec3 col = mix(horizonColor, topColor, pow(clamp(h, 0.0, 1.0), 0.5));
        col = mix(col, groundColor, smoothstep(0.0, -0.3, h));
        float sd = max(dot(d, sunDir), 0.0);
        float vis = smoothstep(-0.12, 0.03, sunDir.y);
        vis *= 1.0 - uOver * 0.97;
        col += sunColor * (pow(sd, 1800.0) * 40.0 + pow(sd, 90.0) * 0.5 + pow(sd, 7.0) * 0.28) * vis;
        // hazy warm band along horizon on the sun side
        col += sunColor * pow(1.0 - abs(h), 12.0) * pow(sd, 2.0) * 0.35 * vis;
        if (h > -0.02) {
          vec2 uv = d.xz / (h + 0.14);
          uv = uv * 0.8 + vec2(uTime * 0.006, uTime * 0.002);
          float c = fbm(uv * 1.4);
          c = smoothstep(mix(0.6, 0.2, uCloud), mix(0.92, 0.62, uCloud), c) * smoothstep(0.0, 0.2, h);
          vec3 lit = mix(horizonColor, vec3(1.0, 0.98, 0.95), 0.45) + sunColor * pow(sd, 5.0) * 0.9;
          vec3 cc = mix(lit, topColor * 0.4 + horizonColor * 0.5, 0.35) * (1.0 - night * 0.9);
          cc *= mix(1.0, 0.62 + 0.25 * fbm(uv * 3.1), uOver);
          col = mix(col, cc, c * mix(0.7, 0.96, uCloud));
          col += vec3(0.75, 0.8, 1.0) * uFlash * (0.5 + 0.9 * c);
          vec3 sp = d * 520.0;
          vec3 cell = floor(sp);
          float st = hash3(cell);
          float star = step(0.9965, st) * smoothstep(0.42, 0.0, length(fract(sp) - 0.5));
          star *= 0.6 + 0.4 * sin(uTime * 2.5 + st * 800.0);
          col += vec3(0.9, 0.93, 1.0) * star * night * (1.0 - c) * (1.0 - uOver) * smoothstep(0.03, 0.3, h) * 2.5;
          float md = max(dot(d, moonDir), 0.0);
          col += vec3(0.95, 0.97, 1.05) * (smoothstep(0.99985, 0.99992, md) * 3.0 + pow(md, 120.0) * 0.12) * night * (1.0 - uOver);
        }
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(4000, 48, 24), skyMat);
  sky.frustumCulled = false; sky.renderOrder = -1;
  scene.add(sky);
  scene.userData.sky = sky;
  skyScene = new THREE.Scene();
  skyScene.add(new THREE.Mesh(new THREE.SphereGeometry(1000, 32, 16), skyMat));
  pmrem = new THREE.PMREMGenerator(renderer);
}

/* =====================================================================
   terrain
   ===================================================================== */
function terrainH(x, z) {
  const d = Math.hypot(x - CX, z - CZ);
  let h = -60 + fbm(x * 0.0032 + 3.1, z * 0.0032 - 7.2, 4) * 26 + fbm(x * 0.018, z * 0.018, 3) * 3.5;
  const m = smooth(300, 1500, d);
  h += m * (ridged(x * 0.0009 + 11.3, z * 0.0009 + 5.1, 5) * 520 + (fbm(x * 0.0005 - 4, z * 0.0005 + 2, 3) * 0.5 + 0.5) * 380 * m);
  h = lerp(h, -62, smooth(150, 70, d) * 0.6);
  const gd = Math.hypot(x - GH.x, z - GH.z);
  h = lerp(h, -58, smooth(55, 22, gd));
  return h;
}

function buildTerrain() {
  const SIZE = 6400, SEG = 380;
  const g = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
  g.rotateX(-Math.PI / 2);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) p.setY(i, terrainH(p.getX(i), p.getZ(i)));
  g.computeVertexNormals();
  const n = g.attributes.normal;
  const col = new Float32Array(p.count * 3), col2 = new Float32Array(p.count * 3);
  const c = new THREE.Color(), c2 = new THREE.Color(), tmp = new THREE.Color();
  const greenA = srgb(0x4f6e2a), greenB = srgb(0x74873a);
  const grassA = srgb(0x6b7236), grassB = srgb(0x8c8440), autumn = srgb(0x9a5a26), autumn2 = srgb(0x7a3a1c), rock = srgb(0x7a7166), rockDark = srgb(0x4f4a44), snow = srgb(0xeef1f5);
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), ny = n.getY(i);
    const gk = smooth(-0.3, 0.5, fbm(x * 0.01, z * 0.01, 3));
    c.copy(grassA).lerp(grassB, gk);
    c2.copy(greenA).lerp(greenB, gk);
    const f = fbm(x * 0.006 + 40, z * 0.006 - 20, 4);
    c.lerp(tmp.copy(autumn).lerp(autumn2, smooth(0, 0.6, noise2(x * 0.03, z * 0.03))), smooth(-0.05, 0.3, f) * 0.85);
    c2.lerp(tmp.copy(greenA).multiplyScalar(0.7), smooth(-0.05, 0.3, f) * 0.7);
    const slope = 1 - ny;
    const rk = smooth(0.18, 0.45, slope);
    tmp.copy(rock).lerp(rockDark, smooth(0, 1, noise2(x * 0.02, z * 0.02) * 0.5 + 0.5));
    c.lerp(tmp, rk); c2.lerp(tmp, rk);
    const snowLine = 230 + fbm(x * 0.004, z * 0.004, 3) * 70;
    const sk = smooth(snowLine, snowLine + 60, y) * smooth(0.75, 0.45, slope);
    c.lerp(snow, sk); c2.lerp(snow, sk);
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    col2[i * 3] = c2.r; col2[i * 3 + 1] = c2.g; col2[i * 3 + 2] = c2.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setAttribute('color2', new THREE.BufferAttribute(col2, 3));
  // subtle detail texture
  const dc = makeCanvas(256), dg = dc.getContext('2d'), r = mulberry32(5);
  dg.fillStyle = '#bbb'; dg.fillRect(0, 0, 256, 256); addNoise(dg, 256, 60, r);
  for (let i = 0; i < 300; i++) { dg.fillStyle = `rgba(${r() < 0.5 ? '255,255,255' : '0,0,0'},0.08)`; dg.beginPath(); dg.arc(r() * 256, r() * 256, 2 + r() * 10, 0, TAU); dg.fill(); }
  const dt = toTex(dc, false);
  scaleUV(g, SIZE / 12, SIZE / 12);
  const mat = patchMat(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, map: dt }), { green2: true, wetK: 0.6 });
  const mesh = new THREE.Mesh(g, mat);
  mesh.receiveShadow = true;
  scene.add(mesh);
}

/* =====================================================================
   rock mesa
   ===================================================================== */
function buildRock({ cx, cz, rx, rz, H, seed, flare }) {
  const s = seed * 17.13;
  const outlineR = (a) => {
    const ca = Math.cos(a), sa = Math.sin(a);
    const re = (rx * rz) / Math.sqrt((rz * ca) ** 2 + (rx * sa) ** 2);
    return re * (1 + 0.05 * noise3(ca * 1.6 + s, 0, sa * 1.6) + 0.025 * noise3(ca * 5 + s, 1, sa * 5));
  };
  const g0 = new THREE.CylinderGeometry(1, 1, H, 220, 90, true);
  g0.translate(0, -H / 2, 0);
  const p = g0.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const a = Math.atan2(z, x), ca = Math.cos(a), sa = Math.sin(a);
    const base = outlineR(a);
    const depth = -y / H;
    const amp = smooth(0, 7, -y);
    let r = base * (1 + flare * Math.pow(depth, 1.5));
    r += amp * (noise3(ca * 3 + s, y * 0.045, sa * 3) * base * 0.1 + noise3(ca * 10 + s, y * 0.14, sa * 10) * 2.2 + Math.sin(y * 0.85 + noise3(ca * 2, 0.3, sa * 2) * 5) * 0.55);
    const yy = y + amp * noise3(ca * 7 + s, y * 0.1, sa * 7) * 1.6;
    p.setXYZ(i, cx + ca * r, yy, cz + sa * r);
  }
  const g = g0.toNonIndexed();
  g.computeVertexNormals();
  const n = g.attributes.normal, pp = g.attributes.position;
  const col = new Float32Array(pp.count * 3);
  const c = new THREE.Color(), rockA = srgb(0x7f7466), rockB = srgb(0x544b41), moss = srgb(0x5d6a2c), mossB = srgb(0x7d7a36);
  for (let i = 0; i < pp.count; i += 3) {
    const y = (pp.getY(i) + pp.getY(i + 1) + pp.getY(i + 2)) / 3;
    const x = pp.getX(i), z = pp.getZ(i);
    const ny = (n.getY(i) + n.getY(i + 1) + n.getY(i + 2)) / 3;
    const strata = 0.5 + 0.5 * Math.sin(y * 0.55 + noise2(x * 0.05, z * 0.05) * 3);
    c.copy(rockA).lerp(rockB, strata * 0.7 + 0.15 * noise2(x * 0.2 + y * 0.1, z * 0.2));
    c.lerp(moss.clone().lerp(mossB, noise2(x * 0.1, z * 0.1) * 0.5 + 0.5), smooth(0.25, 0.6, ny));
    c.multiplyScalar(0.72 + 0.28 * smooth(-H, -5, y));
    for (let k = 0; k < 3; k++) { col[(i + k) * 3] = c.r; col[(i + k) * 3 + 1] = c.g; col[(i + k) * 3 + 2] = c.b; }
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mesh = new THREE.Mesh(g, patchMat(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, flatShading: true }), { snowK: 0.9 }));
  mesh.castShadow = true; mesh.receiveShadow = true;
  scene.add(mesh);
  const outline = [];
  for (let i = 0; i < 160; i++) { const a = (i / 160) * TAU; const r = outlineR(a); outline.push(new THREE.Vector3(cx + Math.cos(a) * r, 0, cz + Math.sin(a) * r)); }
  // ledges for shrubs
  const ledges = [];
  for (let i = 0; i < pp.count; i += 3) {
    const ny = (n.getY(i) + n.getY(i + 1) + n.getY(i + 2)) / 3;
    const y = pp.getY(i);
    if (ny > 0.72 && y < -3 && y > -70) ledges.push(new THREE.Vector3(pp.getX(i), y, pp.getZ(i)));
  }
  return { mesh, outline, ledges, outlineR, cx, cz };
}

/* =====================================================================
   greenhouse
   ===================================================================== */
function buildGreenhouse(copperMat, trimMat) {
  const gb = new Batcher();
  const x = GH.x, z = GH.z, y = terrainH(x, z) - 0.2;
  const R = 8;
  gb.add(new THREE.CylinderGeometry(R + 0.6, R + 0.9, 1.4, 8), trimMat, M(x, y + 0.7, z), 'world', 1 / 6);
  gb.add(new THREE.CylinderGeometry(R, R, 7, 8, 1, true), ghGlassMat, M(x, y + 1.4 + 3.5, z), 'scale', [1, 1]);
  gb.add(new THREE.SphereGeometry(R, 24, 10, 0, TAU, 0, Math.PI / 2), ghGlassMat, M(x, y + 8.4, z), 'scale', [1, 1]);
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * TAU + Math.PI / 8;
    gb.add(new THREE.BoxGeometry(0.22, 7, 0.22), copperMat, M(x + Math.sin(a) * R * 0.99, y + 4.9, z + Math.cos(a) * R * 0.99, a));
  }
  for (const hy of [1.5, 5, 8.4]) gb.add(new THREE.TorusGeometry(R, 0.1, 5, 8), copperMat, M(x, y + hy, z, Math.PI / 8, 1, 1, 1, Math.PI / 2));
  for (let i = 0; i < 8; i++) gb.add(new THREE.TorusGeometry(R, 0.09, 5, 24, Math.PI), copperMat, M(x, y + 8.4, z, (i / 8) * Math.PI));
  gb.add(new THREE.CylinderGeometry(1.2, 1.2, 1.8, 8), copperMat, M(x, y + 8.4 + R + 0.5, z));
  gb.add(new THREE.ConeGeometry(1.5, 2.2, 8), copperMat, M(x, y + 8.4 + R + 2.5, z));
  // wings
  for (const sgn of [-1, 1]) {
    const wx = x + sgn * (R + 7);
    gb.add(new THREE.BoxGeometry(14, 1.2, 8), trimMat, M(wx, y + 0.6, z), 'world', 1 / 6);
    gb.add(new THREE.BoxGeometry(14, 4.8, 7.4), ghGlassMat, M(wx, y + 1.2 + 2.4, z), 'scale', [1, 1]);
    const s = new THREE.Shape(); s.moveTo(-3.9, 0); s.lineTo(3.9, 0); s.lineTo(0, 3); s.closePath();
    gb.add(new THREE.ExtrudeGeometry(s, { depth: 14, bevelEnabled: false }), ghGlassMat, M(wx - 7, y + 6, z, Math.PI / 2), 'scale', [0.1, 0.1]);
    for (let k = 0; k <= 7; k++) gb.add(new THREE.BoxGeometry(0.16, 4.8, 7.6), copperMat, M(wx - 7 + k * 2, y + 3.6, z));
    gb.add(new THREE.BoxGeometry(14.2, 0.2, 0.2), copperMat, M(wx, y + 9, z));
  }
  gb.build(scene, false);
}

/* =====================================================================
   trees
   ===================================================================== */
// Injects wind sway, snow cover, wetness, seasonal ground colour and leaf thinning into a standard material.
function patchMat(mat, o = {}) {
  const snowK = (o.snowK ?? 1).toFixed(2), wetK = (o.wetK ?? 1).toFixed(2);
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { uTime: U.uTime, uWind: U.uWind, uSnow: ENV.uSnow, uWet: ENV.uWet, uGreen: ENV.uGreen, uLeafAmt: ENV.uLeafAmt });
    if (o.map2) sh.uniforms.uMap2 = { value: o.map2 };
    let vs = sh.vertexShader, fs = sh.fragmentShader;
    vs = 'uniform float uTime; uniform float uWind; uniform float uGreen;\nvarying vec3 vWN; varying vec3 vWP;\n'
      + (o.leafAmt ? 'attribute float aRand; varying float vRand;\n' : '')
      + (o.green2 ? 'attribute vec3 color2;\n' : '') + vs;
    if (o.wind) vs = vs.replace('#include <begin_vertex>', `#include <begin_vertex>
      #ifdef USE_INSTANCING
        vec3 ip = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
      #else
        vec3 ip = vec3(0.0);
      #endif
      float ph = ip.x * 0.13 + ip.z * 0.17;
      float hgt = max(position.y, 0.0);
      float sw = (sin(uTime * 1.2 + ph) * 0.6 + sin(uTime * 2.6 + ph * 1.7) * 0.25) * (0.25 + uWind * 1.2);
      transformed.x += sw * hgt * hgt * 0.0025;
      transformed.z += sw * hgt * hgt * 0.0016;
      ${o.flutter ? 'transformed += normal * sin(uTime * (5.0 + uWind * 6.0) + position.x * 3.0 + position.z * 2.0 + ph) * 0.06 * (0.2 + uWind);' : ''}`);
    if (o.green2) vs = vs.replace('#include <color_vertex>', '#include <color_vertex>\n vColor.rgb = mix(vColor.rgb, color2, uGreen);');
    vs = vs.replace('#include <worldpos_vertex>', `#include <worldpos_vertex>
      vWN = normalize((vec4(transformedNormal, 0.0) * viewMatrix).xyz);
      vec4 eWP = vec4(transformed, 1.0);
      #ifdef USE_INSTANCING
        eWP = instanceMatrix * eWP;
      #endif
      vWP = (modelMatrix * eWP).xyz;
      ${o.leafAmt ? 'vRand = aRand;' : ''}`);
    fs = `uniform float uSnow; uniform float uWet; uniform float uGreen; uniform float uLeafAmt;
      varying vec3 vWN; varying vec3 vWP;
      ${o.leafAmt ? 'varying float vRand;' : ''}
      ${o.map2 ? 'uniform sampler2D uMap2;' : ''}
      float eH(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
      float eVN(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
        return mix(mix(eH(i), eH(i+vec2(1,0)), f.x), mix(eH(i+vec2(0,1)), eH(i+vec2(1,1)), f.x), f.y); }
    ` + fs;
    if (o.map2) fs = fs.replace('#include <map_fragment>', `#ifdef USE_MAP
        diffuseColor *= mix(texture2D(map, vMapUv), texture2D(uMap2, vMapUv), uGreen);
      #endif`);
    if (o.leafAmt) fs = fs.replace('#include <alphatest_fragment>', '#include <alphatest_fragment>\n if (vRand > uLeafAmt) discard;');
    fs = fs.replace('#include <color_fragment>', `#include <color_fragment>
      float eUp = clamp(vWN.y, 0.0, 1.0);
      float eN = eVN(vWP.xz * 1.3) * 0.6 + eVN(vWP.xz * 5.0) * 0.4;
      float eSnow = smoothstep(0.62 - uSnow * 0.45, 0.8 - uSnow * 0.35, eUp + (eN - 0.5) * 0.3) * smoothstep(0.0, 0.3, uSnow) * ${snowK};
      diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.88, 0.91, 0.97), clamp(eSnow, 0.0, 1.0));
      float eWet = uWet * ${wetK} * (1.0 - clamp(eSnow, 0.0, 1.0));
      diffuseColor.rgb *= 1.0 - eWet * 0.4;`);
    fs = fs.replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
      roughnessFactor = mix(roughnessFactor, 0.45, eWet * 0.6);
      roughnessFactor = mix(roughnessFactor, 0.1, eWet * smoothstep(0.55, 0.95, eUp) * smoothstep(0.35, 0.65, eN));
      roughnessFactor = mix(roughnessFactor, 0.8, clamp(eSnow, 0.0, 1.0));`);
    if (o.flutter) fs = fs.replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\n totalEmissiveRadiance += diffuseColor.rgb * 0.05;');
    sh.vertexShader = vs; sh.fragmentShader = fs;
  };
  const key = 'env' + JSON.stringify(o, (k, v) => (v && v.isTexture ? 'tex' : v));
  mat.customProgramCacheKey = () => key;
  return mat;
}
// leaf thinning must also apply to the shadow pass
function patchLeafDepth(mat) {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uLeafAmt = ENV.uLeafAmt;
    sh.vertexShader = 'attribute float aRand; varying float vRand;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\n vRand = aRand;');
    sh.fragmentShader = 'uniform float uLeafAmt; varying float vRand;\n' + sh.fragmentShader.replace('#include <alphatest_fragment>', '#include <alphatest_fragment>\n if (vRand > uLeafAmt) discard;');
  };
  mat.customProgramCacheKey = () => 'leafdepth';
  return mat;
}

function buildTreeTemplate(seed, o) {
  const r = mulberry32(seed);
  const H = o.height, CR = o.crownR, CH = o.crownH, N = o.cards, CS = o.cardSize;
  let trunkGeo = null;
  if (o.trunkR) {
    const parts = [];
    const tH = H * 0.9;
    const t = new THREE.CylinderGeometry(o.trunkR * 0.3, o.trunkR, tH, 8, 6);
    t.translate(0, tH / 2, 0);
    const tp = t.attributes.position, bendA = r() * TAU, bend = 0.2 + r() * 0.4;
    for (let i = 0; i < tp.count; i++) { const k = (tp.getY(i) / tH) ** 2 * bend; tp.setX(i, tp.getX(i) + Math.cos(bendA) * k); tp.setZ(i, tp.getZ(i) + Math.sin(bendA) * k); }
    t.computeVertexNormals(); scaleUV(t, 1, tH / 2.5);
    parts.push(t.toNonIndexed());
    for (let i = 0; i < (o.branches ?? 6); i++) {
      const len = CR * (0.7 + r() * 0.6), y0 = H - CH * (0.3 + r() * 0.6);
      const bg = new THREE.CylinderGeometry(0.03, o.trunkR * 0.3, len, 5, 1, true);
      bg.translate(0, len / 2, 0); bg.rotateZ(-(0.5 + r() * 0.6)); bg.rotateY(r() * TAU); bg.translate(0, y0, 0);
      scaleUV(bg, 1, len / 2.5); parts.push(bg.toNonIndexed());
    }
    trunkGeo = mergeGeometries(parts);
  }
  const pos = [], nor = [], uv = [], col = [], idx = [], rnd = [];
  const cy = H - CH * 0.5;
  const q = new THREE.Quaternion(), e = new THREE.Euler(), va = new THREE.Vector3(), vb = new THREE.Vector3(), p = new THREE.Vector3(), nn = new THREE.Vector3();
  for (let i = 0; i < N; i++) {
    const u = r() * 2 - 1, th = r() * TAU, s = Math.sqrt(1 - u * u);
    const rad = Math.pow(r(), 0.4);
    const taper = 1 - 0.35 * Math.max(0, u);
    p.set(Math.cos(th) * s * CR * rad * taper, u * CH * 0.5 * rad + cy, Math.sin(th) * s * CR * rad * taper);
    nn.set(p.x, (p.y - cy) * 0.6, p.z).normalize(); nn.y += 0.4; nn.normalize();
    e.set(r() * TAU, r() * TAU, r() * TAU); q.setFromEuler(e);
    const sz = CS * (0.7 + r() * 0.6);
    va.set(sz / 2, 0, 0).applyQuaternion(q); vb.set(0, sz / 2, 0).applyQuaternion(q);
    const base = pos.length / 3;
    pos.push(p.x - va.x - vb.x, p.y - va.y - vb.y, p.z - va.z - vb.z);
    pos.push(p.x + va.x - vb.x, p.y + va.y - vb.y, p.z + va.z - vb.z);
    pos.push(p.x + va.x + vb.x, p.y + va.y + vb.y, p.z + va.z + vb.z);
    pos.push(p.x - va.x + vb.x, p.y - va.y + vb.y, p.z - va.z + vb.z);
    for (let k = 0; k < 4; k++) nor.push(nn.x, nn.y, nn.z);
    uv.push(0, 0, 1, 0, 1, 1, 0, 1);
    const br = 0.72 + r() * 0.4, ao = 0.5 + 0.5 * rad;
    const cr = br * (0.93 + r() * 0.14) * ao, cg = br * (0.84 + r() * 0.22) * ao, cb = br * (0.88 + r() * 0.15) * ao;
    for (let k = 0; k < 4; k++) col.push(cr, cg, cb);
    const rv = r(); rnd.push(rv, rv, rv, rv);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('aRand', new THREE.Float32BufferAttribute(rnd, 1));
  lg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  lg.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  lg.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  lg.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  lg.setIndex(idx);
  return { trunk: trunkGeo, leaves: lg, crown: { cy, CR, CH } };
}

function buildConiferTemplate(seed, H, R) {
  const r = mulberry32(seed);
  const parts = [];
  const layers = 8;
  const green = [srgb(0x2c4526), srgb(0x3a5530), srgb(0x243a22)];
  const addColored = (g, c) => {
    const ng = g.toNonIndexed(); const n = ng.attributes.position.count; const a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const v = 0.85 + r() * 0.3; a[i * 3] = c.r * v; a[i * 3 + 1] = c.g * v; a[i * 3 + 2] = c.b * v; }
    ng.setAttribute('color', new THREE.BufferAttribute(a, 3)); ng.deleteAttribute('uv'); parts.push(ng);
  };
  const trunk = new THREE.CylinderGeometry(0.12, 0.3, H * 0.4, 6); trunk.translate(0, H * 0.2, 0);
  addColored(trunk, srgb(0x4a3526));
  for (let l = 0; l < layers; l++) {
    const t = l / layers, y = H * 0.14 + t * H * 0.74, rr = R * (1 - t) * 0.95 + 0.35, h = H * 0.26;
    const cg = new THREE.ConeGeometry(rr, h, 9, 2, true);
    const cp = cg.attributes.position;
    for (let i = 0; i < cp.count; i++) {
      const k = 1 + (r() - 0.5) * 0.35;
      cp.setX(i, cp.getX(i) * k); cp.setZ(i, cp.getZ(i) * k);
      if (cp.getY(i) < 0) cp.setY(i, cp.getY(i) - r() * 0.4);
    }
    cg.rotateY(r() * TAU); cg.translate(0, y + h / 2, 0); cg.computeVertexNormals();
    addColored(cg, green[l % 3]);
  }
  return mergeGeometries(parts);
}

function buildConiferCards(seed, H, R) {
  const r = mulberry32(seed);
  const t = new THREE.CylinderGeometry(0.1, 0.38, H * 0.95, 7, 4); t.translate(0, H * 0.475, 0); scaleUV(t, 1, H / 3);
  const pos = [], nor = [], uv = [], col = [], idx = [];
  const q = new THREE.Quaternion(), e = new THREE.Euler(), va = new THREE.Vector3(), vb = new THREE.Vector3();
  for (let i = 0; i < 320; i++) {
    const k = Math.pow(r(), 0.8), y = H * 0.12 + k * H * 0.85;
    const rr = R * (1 - k) * (0.35 + 0.65 * Math.sqrt(r())) + 0.2;
    const a = r() * TAU;
    const px = Math.cos(a) * rr, pz = Math.sin(a) * rr, py = y - rr * 0.25;
    const n = new THREE.Vector3(px, 0.6, pz).normalize();
    e.set(-0.9 + r() * 0.5, a + (r() - 0.5), r() * TAU); q.setFromEuler(e);
    const sz = 1.3 + r() * 0.8;
    va.set(sz / 2, 0, 0).applyQuaternion(q); vb.set(0, sz / 2, 0).applyQuaternion(q);
    const base = pos.length / 3;
    pos.push(px - va.x - vb.x, py - va.y - vb.y, pz - va.z - vb.z, px + va.x - vb.x, py + va.y - vb.y, pz + va.z - vb.z,
      px + va.x + vb.x, py + va.y + vb.y, pz + va.z + vb.z, px - va.x + vb.x, py - va.y + vb.y, pz - va.z + vb.z);
    for (let j = 0; j < 4; j++) nor.push(n.x, n.y, n.z);
    uv.push(0, 0, 1, 0, 1, 1, 0, 1);
    const ao = 0.55 + 0.45 * (rr / (R * (1 - k) + 0.2)), br = (0.75 + r() * 0.35) * ao;
    for (let j = 0; j < 4; j++) col.push(br * 0.9, br, br * 0.85);
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  lg.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  lg.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  lg.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  lg.setIndex(idx);
  return { trunk: t, leaves: lg };
}

function buildTrees(barkTex, leafTex, plateau, pillar) {
  const leafMat = new THREE.MeshStandardMaterial({ map: leafTex, alphaTest: 0.45, side: THREE.DoubleSide, vertexColors: true, roughness: 0.82 });
  patchMat(leafMat, { wind: true, flutter: true, leafAmt: true, snowK: 0.5, wetK: 0.3 });
  const leafDepth = patchLeafDepth(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: leafTex, alphaTest: 0.45 }));
  const conLeafMat = patchMat(new THREE.MeshStandardMaterial({ map: leafTex, alphaTest: 0.45, side: THREE.DoubleSide, vertexColors: true, roughness: 0.82 }), { wind: true, flutter: true, snowK: 1.1, wetK: 0.3 });
  const conDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: leafTex, alphaTest: 0.45 });
  const barkMat = new THREE.MeshStandardMaterial({ map: barkTex, roughness: 0.8 });
  patchMat(barkMat, { wind: true, snowK: 0.6, wetK: 0.7 });
  const conMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide, flatShading: true });
  patchMat(conMat, { wind: true, snowK: 1.1, wetK: 0.3 });

  const nearT = [
    buildTreeTemplate(1, { height: 15, crownR: 2.8, crownH: 10, cards: 230, cardSize: 1.7, trunkR: 0.3 }),
    buildTreeTemplate(2, { height: 11, crownR: 3.6, crownH: 7.5, cards: 220, cardSize: 1.8, trunkR: 0.28 }),
    buildTreeTemplate(3, { height: 18, crownR: 2.5, crownH: 12.5, cards: 250, cardSize: 1.7, trunkR: 0.34 }),
    buildTreeTemplate(4, { height: 2.2, crownR: 1.6, crownH: 1.8, cards: 60, cardSize: 1.2 }), // shrub
  ];
  const farT = [
    buildTreeTemplate(11, { height: 14, crownR: 3.4, crownH: 9, cards: 34, cardSize: 3.6, trunkR: 0.3, branches: 0 }),
    buildTreeTemplate(12, { height: 11, crownR: 4, crownH: 7, cards: 30, cardSize: 3.8, trunkR: 0.3, branches: 0 }),
  ];
  const conGeo = buildConiferTemplate(21, 20, 3.6);
  const conNear = buildConiferCards(22, 19, 3.4);

  const autumn = [0xe79a2a, 0xf1bb35, 0xdb6a26, 0xc2401f, 0xe9d24a, 0xb7c03c, 0xd9862f, 0xa5321a].map(srgb);
  const shrubCols = [0xb3321d, 0xc9532a, 0x9b2a1b, 0xd7822e, 0x7c7a2c].map(srgb);
  const r = mulberry32(99);
  const pickCol = (list) => list[Math.floor(r() * list.length)].clone().multiplyScalar(0.85 + r() * 0.3);

  const near = [], far = [], cons = [];
  const addNear = (x, z, t, s, y = 0, col) => near.push({ t, x, y, z, s, ry: r() * TAU, col: col ?? pickCol(t === 3 ? shrubCols : autumn) });

  // hand-placed trees framing the castle
  const spots = [
    [-24.8, 7.0, 2, 1.0], [-17.6, 7.6, 0, 0.95], [11.5, 7.2, 1, 0.9], [13.2, 10.8, 0, 1.0],
    [-29, 13, 2, 1.15], [-35, 9.5, 0, 1.05], [-40.5, 16, 2, 1.2], [-27.5, 21.5, 1, 1.0], [-33.5, 25.5, 0, 1.1],
    [-46, 5, 1, 1.0], [-47, -8, 0, 1.1], [16.5, 14.5, 1, 0.9], [22.5, 9, 2, 0.95], [27.5, 3, 0, 1.0],
    [-48, -22, 2, 1.05], [-50, -33, 0, 1.0], [28, -18, 1, 1.0], [30, -28, 2, 1.1], [26, -46, 0, 1.0],
    [-8, -47, 2, 1.05], [-26, -48, 1, 1.0], [5, -47, 0, 0.95], [-52, -12, 2, 1.1], [33, -10, 1, 0.9],
  ];
  for (const [x, z, t, s] of spots) addNear(x, z, t, s * (0.9 + r() * 0.2));
  // shrubs along the facade and forecourt
  for (let i = 0; i < 26; i++) { const sx = lerp(-26, 12, r()), sz = lerp(4.8, 6.8, r()); if (Math.abs(sx + 6) > 8.2) addNear(sx, sz, 3, 0.7 + r() * 0.5); }
  for (let i = 0; i < 14; i++) addNear(lerp(-20, 4, r()), 25.6 + r() * 1.2, 3, 0.8 + r() * 0.4);

  const blocked = (x, z) => {
    if (x > -31 && x < 17 && z > -44 && z < 9) return true;
    for (const [bx, bz, br] of [[-34, -3, 12], [18, -1, 8.5], [23.6, -6.5, 5], [-38.5, -40, 9], [16, -38, 8], [-22.5, -32.5, 6]]) if (Math.hypot(x - bx, z - bz) < br) return true;
    if (x > -22 && x < 12 && z > 6 && z < 27) return true;
    if (x > -46 && x < -40 && z > -36 && z < -8) return true;
    if (x > 19 && x < 23 && z > -35 && z < -7) return true;
    const t = clamp((x - VS.x) * VDIR.x + (z - VS.z) * VDIR.z, 0, 99);
    if (Math.hypot(x - (VS.x + VDIR.x * t), z - (VS.z + VDIR.z * t)) < 6) return true;
    return false;
  };
  for (let i = 0, placed = 0; i < 400 && placed < 34; i++) {
    const a = r() * TAU, rr = Math.sqrt(r()) * 0.9;
    const R = plateau.outlineR(a) * rr;
    const x = CX + Math.cos(a) * R, z = CZ + Math.sin(a) * R;
    if (blocked(x, z)) continue;
    const isShrub = r() < 0.3;
    addNear(x, z, isShrub ? 3 : Math.floor(r() * 3), isShrub ? 0.8 + r() * 0.5 : 0.85 + r() * 0.35);
    placed++;
  }
  // a few dark conifers for contrast
  for (const [x, z, s] of [[34, 14, 1.0], [-50, 14, 0.9], [36, -38, 1.1], [-44, -44, 0.95], [30, 22, 0.8]]) cons.push({ x, y: 0, z, s, ry: r() * TAU, col: new THREE.Color(1, 1, 1) });
  // shrubs & small trees on rock ledges
  for (const rock of [plateau, pillar]) {
    const L = rock.ledges;
    for (let i = 0; i < Math.min(60, L.length); i++) {
      const p = L[Math.floor(r() * L.length)];
      if (r() < 0.6) addNear(p.x, p.z, 3, 0.8 + r() * 0.6, p.y - 0.2);
      else addNear(p.x, p.z, Math.floor(r() * 3), 0.5 + r() * 0.3, p.y - 0.3);
    }
  }
  // valley forest
  for (let i = 0; i < 26000 && (far.length < 4200 || cons.length < 2600); i++) {
    const x = CX + (r() - 0.5) * 2400, z = CZ + (r() - 0.5) * 2400;
    const d = Math.hypot(x - CX, z - CZ);
    if (d < 70) continue;
    if (Math.hypot(x - OV.x, z - OV.z) < 20 || Math.hypot(x - GH.x, z - GH.z) < 30) continue;
    const h = terrainH(x, z);
    if (h > 190) continue;
    const dx = terrainH(x + 3, z) - h, dz = terrainH(x, z + 3) - h;
    if (Math.hypot(dx, dz) / 3 > 0.75) continue;
    const dens = fbm(x * 0.005 + 9, z * 0.005 - 3, 4);
    if (dens < -0.12 + r() * 0.2) continue;
    const conifer = h > 20 || noise2(x * 0.008, z * 0.008) > 0.25;
    if (conifer) { if (cons.length < 2600) cons.push({ x, y: h - 0.5, z, s: 0.8 + r() * 0.7, ry: r() * TAU, col: new THREE.Color(1, 1, 1).multiplyScalar(0.8 + r() * 0.35) }); }
    else if (far.length < 4200) far.push({ t: Math.floor(r() * 2), x, y: h - 0.5, z, s: 0.8 + r() * 0.6, ry: r() * TAU, col: pickCol(autumn) });
  }

  const d = new THREE.Object3D();
  const instance = (geo, mat, items, shadow, depthMat) => {
    if (!items.length) return null;
    const im = new THREE.InstancedMesh(geo, mat, items.length);
    items.forEach((o, i) => {
      d.position.set(o.x, o.y, o.z); d.rotation.set(0, o.ry, 0); d.scale.set(o.s, o.s * (0.92 + ((i * 0.618) % 1) * 0.2), o.s); d.updateMatrix();
      im.setMatrixAt(i, d.matrix);
      if (o.col) im.setColorAt(i, o.col);
    });
    im.castShadow = shadow; im.receiveShadow = true;
    if (depthMat) im.customDepthMaterial = depthMat;
    im.computeBoundingSphere();
    scene.add(im); return im;
  };
  for (const o of [...near.filter((o) => o.t < 3), ...cons.slice(0, 5)]) if (o.y > -1) WALK.trunks.push([o.x, o.z, 0.45 * o.s]);
  nearT.forEach((tpl, ti) => {
    const items = near.filter((o) => o.t === ti);
    registerSeasonal(instance(tpl.leaves, leafMat, items, true, leafDepth), ti === 3);
    if (tpl.trunk) instance(tpl.trunk, barkMat, items.map((o) => ({ ...o, col: null })), true);
    // falling-leaf spawn points = crowns of near trees
    if (ti < 3) for (const o of items) if (o.y > -2) leafSpawnPoints.push({ x: o.x, y: o.y + tpl.crown.cy * o.s, z: o.z, r: tpl.crown.CR * o.s, h: tpl.crown.CH * o.s * 0.5, col: o.col });
  });
  farT.forEach((tpl, ti) => {
    const items = far.filter((o) => o.t === ti);
    registerSeasonal(instance(tpl.leaves, leafMat, items, false), false);
    instance(tpl.trunk, barkMat, items.map((o) => ({ ...o, col: null })), false);
  });
  const nearCons = cons.slice(0, 5), farCons = cons.slice(5);
  instance(conNear.leaves, conLeafMat, nearCons.map((o) => ({ ...o, col: srgb(0x3d5a2e).multiplyScalar(0.8 + r() * 0.3) })), true, conDepth);
  instance(conNear.trunk, barkMat, nearCons, true);
  instance(conGeo, conMat, farCons, false);
}

/* =====================================================================
   seasons
   ===================================================================== */
const PAL = {
  sakura: [0xf6c3d0, 0xf9d6df, 0xeea6ba, 0xfbe4ea].map(srgb),
  fresh: [0x9cc653, 0x86b845, 0xb3d466, 0x77a83c].map(srgb),
  summer: [0x4d7a2a, 0x5c8a30, 0x3f6b24, 0x6e9636, 0x557f2c].map(srgb),
  azalea: [0xd9477a, 0xe86a9a, 0xc23a6a, 0xf29ab8].map(srgb),
};
const SEASONS = {
  spring: { name: '春', en: 'Spring', leafAmt: 0.92, green: 1, snow: 0, fall: 0.6, fallSize: 0.55, fallCols: PAL.sakura },
  summer: { name: '夏', en: 'Summer', leafAmt: 1, green: 1, snow: 0, fall: 0.08, fallSize: 1, fallCols: PAL.summer },
  autumn: { name: '秋', en: 'Autumn', leafAmt: 1, green: 0, snow: 0, fall: 1, fallSize: 1, fallCols: null },
  winter: { name: '冬', en: 'Winter', leafAmt: 0.05, green: 0.35, snow: 0.95, fall: 0, fallSize: 1, fallCols: null },
};
const seasonal = [];
const SS = { key: 'autumn', k: 1, from: null, leafFrom: 1, greenFrom: 0, fallFrac: 1 };
function registerSeasonal(im, shrub) {
  if (!im) return;
  const r = mulberry32(im.count * 7 + (shrub ? 3 : 1));
  const n = im.count, autumn = im.instanceColor.array.slice();
  const spring = new Float32Array(n * 3), summer = new Float32Array(n * 3), winter = new Float32Array(n * 3);
  const pick = (l) => l[Math.floor(r() * l.length)];
  for (let i = 0; i < n; i++) {
    const k = 0.85 + r() * 0.3;
    const cs = shrub ? (r() < 0.55 ? pick(PAL.azalea) : pick(PAL.fresh)) : (r() < 0.6 ? pick(PAL.sakura) : pick(PAL.fresh));
    const cu = pick(PAL.summer);
    spring[i * 3] = cs.r * k; spring[i * 3 + 1] = cs.g * k; spring[i * 3 + 2] = cs.b * k;
    summer[i * 3] = cu.r * k; summer[i * 3 + 1] = cu.g * k; summer[i * 3 + 2] = cu.b * k;
    for (let j = 0; j < 3; j++) winter[i * 3 + j] = autumn[i * 3 + j] * 0.45;
  }
  seasonal.push({ im, pal: { spring, summer, autumn, winter } });
}
function setSeason(key, instant = false) {
  SS.key = key; SS.k = instant ? 1 : 0;
  SS.from = seasonal.map((m) => m.im.instanceColor.array.slice());
  SS.leafFrom = ENV.uLeafAmt.value; SS.greenFrom = ENV.uGreen.value;
  if (instant) stepSeason(0);
  const S = SEASONS[key];
  document.querySelector('#title h1').textContent = `${S.name}ノ古城`;
  document.querySelector('#title p').textContent = `The Castle in ${S.en}`;
  if (fallingLeaves) fallingLeaves.material.map = key === 'spring' ? petalTex : leafTexOne;
  if (interior) interior.setDecor(key);
  document.querySelectorAll('#seasons button').forEach((b) => b.classList.toggle('on', b.dataset.season === key));
}
function stepSeason(dt) {
  if (SS.k >= 1 && SS.from === null) return;
  SS.k = Math.min(1, SS.k + dt / 2.8);
  const e = SS.k * SS.k * (3 - 2 * SS.k), S = SEASONS[SS.key];
  seasonal.forEach((m, mi) => {
    const a = m.im.instanceColor.array, f = SS.from[mi], to = m.pal[SS.key];
    for (let i = 0; i < a.length; i++) a[i] = f[i] + (to[i] - f[i]) * e;
    m.im.instanceColor.needsUpdate = true;
  });
  ENV.uLeafAmt.value = lerp(SS.leafFrom, S.leafAmt, e);
  ENV.uGreen.value = lerp(SS.greenFrom, S.green, e);
  if (SS.k >= 1) SS.from = null;
}

/* =====================================================================
   weather
   ===================================================================== */
const WEATHERS = {
  clear: { cloud: 0.15, over: 0, rain: 0, snow: 0, fog: 0, storm: 0 },
  cloudy: { cloud: 0.85, over: 0.6, rain: 0, snow: 0, fog: 0.1, storm: 0 },
  rain: { cloud: 1, over: 0.8, rain: 1, snow: 0, fog: 0.35, storm: 0 },
  storm: { cloud: 1, over: 0.95, rain: 1.5, snow: 0, fog: 0.4, storm: 1 },
  fog: { cloud: 0.5, over: 0.5, rain: 0, snow: 0, fog: 1, storm: 0 },
  snow: { cloud: 0.95, over: 0.7, rain: 0, snow: 1, fog: 0.4, storm: 0 },
};
const W = { key: 'clear', cloud: 0.15, over: 0, rain: 0, snow: 0, fog: 0, storm: 0, flash: 0, boltT: 4 };
const greyDay = srgb(0x9aa2aa), greyNight = srgb(0x0b0e13), greyC = new THREE.Color();
let rainFx, snowFx;
function setWeather(key) {
  W.key = key;
  document.querySelectorAll('#weathers button').forEach((b) => b.classList.toggle('on', b.dataset.weather === key));
}
const approach = (v, t, rate) => (v < t ? Math.min(t, v + rate) : Math.max(t, v - rate));
function stepWeather(dt) {
  const T = WEATHERS[W.key];
  for (const k of ['cloud', 'over', 'rain', 'snow', 'fog', 'storm']) W[k] = approach(W[k], T[k], dt * 0.3);
  ENV.uWet.value = approach(ENV.uWet.value, Math.min(1, W.rain), dt * (W.rain > 0.3 ? 0.12 : 0.035));
  const snowTarget = Math.max(SEASONS[SS.key].snow, W.snow * 0.9);
  ENV.uSnow.value = approach(ENV.uSnow.value, snowTarget, dt * (SS.k < 1 ? 0.45 : W.snow > 0.3 ? 0.04 : 0.06));
  // lightning
  W.flash = Math.max(0, W.flash - dt * 3.5);
  if (W.storm > 0.6) {
    W.boltT -= dt;
    if (W.boltT <= 0) {
      W.flash = 1; W.boltT = 4 + Math.random() * 9;
      setTimeout(() => { W.flash = Math.max(W.flash, 0.7); }, 120 + Math.random() * 120);
      audio.thunder(0.4 + Math.random() * 1.6);
    }
  }
}
function buildPrecip() {
  const box = new THREE.Vector3(64, 44, 64);
  // rain: short line streaks, world-anchored and wrapped around the camera
  const RN = 12000, rp = new Float32Array(RN * 6), re = new Float32Array(RN * 2);
  for (let i = 0; i < RN; i++) {
    const x = Math.random(), y = Math.random(), z = Math.random();
    rp.set([x, y, z, x, y, z], i * 6); re[i * 2 + 1] = 1;
  }
  const rg = new THREE.BufferGeometry();
  rg.setAttribute('position', new THREE.BufferAttribute(rp, 3));
  rg.setAttribute('aEnd', new THREE.BufferAttribute(re, 1));
  const common = { uTime: U.uTime, uCam: { value: camera.position }, uBox: { value: box }, uWind: { value: new THREE.Vector2() }, uAmt: { value: 0 }, uColor: { value: new THREE.Color() } };
  const wrap = `
      vec3 w;
      w.xz = uCam.xz + mod(p.xz - uCam.xz, uBox.xz) - uBox.xz * 0.5;
      w.y = uCam.y + mod(p.y - uCam.y, uBox.y) - uBox.y * 0.5;`;
  rainFx = new THREE.LineSegments(rg, new THREE.ShaderMaterial({
    uniforms: { ...common }, transparent: true, depthWrite: false,
    vertexShader: `attribute float aEnd; uniform float uTime, uAmt; uniform vec3 uCam, uBox; uniform vec2 uWind; varying float vA;
      void main(){
        vec3 s = position;
        float speed = 22.0 + s.x * 8.0;
        vec3 p = s * uBox;
        p.y -= uTime * speed; p.x += uWind.x * uTime; p.z += uWind.y * uTime;
        ${wrap}
        vec3 dir = normalize(vec3(uWind.x, -speed, uWind.y));
        w -= dir * aEnd * (0.9 + s.z * 0.7);
        float d = length(w - uCam);
        vA = step(fract(s.x * 7.31 + s.z * 3.7), uAmt) * smoothstep(32.0, 10.0, d) * smoothstep(1.0, 3.0, d);
        gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
      }`,
    fragmentShader: `uniform vec3 uColor; varying float vA;
      void main(){ if (vA < 0.01) discard; gl_FragColor = vec4(uColor, vA * 0.5); }`,
  }));
  // snow: soft point sprites drifting with the wind
  const SN = 16000, sp = new Float32Array(SN * 3);
  for (let i = 0; i < SN * 3; i++) sp[i] = Math.random();
  const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  snowFx = new THREE.Points(sg, new THREE.ShaderMaterial({
    uniforms: { ...common, uWind: { value: new THREE.Vector2() }, uAmt: { value: 0 }, uColor: { value: new THREE.Color() }, uSize: { value: 70 * DPR } }, transparent: true, depthWrite: false,
    vertexShader: `uniform float uTime, uAmt, uSize; uniform vec3 uCam, uBox; uniform vec2 uWind; varying float vA;
      void main(){
        vec3 s = position;
        vec3 p = s * uBox;
        p.y -= uTime * (1.1 + s.x * 0.9);
        p.x += uWind.x * uTime * 0.5 + sin(uTime * 0.8 + s.z * 40.0) * 1.3;
        p.z += uWind.y * uTime * 0.5 + cos(uTime * 0.7 + s.x * 40.0) * 1.3;
        ${wrap}
        float d = length(w - uCam);
        vA = step(fract(s.y * 13.7 + s.x * 5.1), uAmt) * smoothstep(32.0, 14.0, d);
        vec4 mv = viewMatrix * vec4(w, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = uSize * (0.5 + s.z * 0.8) / max(-mv.z, 0.5);
      }`,
    fragmentShader: `uniform vec3 uColor; varying float vA;
      void main(){ float a = smoothstep(0.5, 0.12, length(gl_PointCoord - 0.5)) * vA; if (a < 0.01) discard; gl_FragColor = vec4(uColor, a * 0.9); }`,
  }));
  for (const fx of [rainFx, snowFx]) { fx.frustumCulled = false; fx.renderOrder = 5; scene.add(fx); }
}
function updatePrecip(wind) {
  const lightK = 0.2 + 0.8 * state.dayK + W.flash * 1.5;
  rainFx.material.uniforms.uAmt.value = Math.min(1, W.rain) * 0.95;
  rainFx.material.uniforms.uWind.value.set(wind.x, wind.z);
  rainFx.material.uniforms.uColor.value.copy(fogColor).multiplyScalar(1.1).addScalar(0.08 * lightK);
  rainFx.visible = W.rain > 0.01;
  snowFx.material.uniforms.uAmt.value = W.snow;
  snowFx.material.uniforms.uWind.value.set(wind.x, wind.z);
  snowFx.material.uniforms.uColor.value.setRGB(0.95, 0.97, 1.0).multiplyScalar(0.25 + 0.75 * lightK);
  snowFx.visible = W.snow > 0.01;
}

/* =====================================================================
   falling leaves
   ===================================================================== */
const LEAF_N = 900;
const LF = {
  pos: new Float32Array(LEAF_N * 3), vel: new Float32Array(LEAF_N * 3), rot: new Float32Array(LEAF_N * 3), spin: new Float32Array(LEAF_N * 3),
  rest: new Float32Array(LEAF_N), phase: new Float32Array(LEAF_N), size: new Float32Array(LEAF_N),
};
const lr = mulberry32(777);
function groundAt(x, z) {
  const ex = (x - CX) / PRX, ez = (z - CZ) / PRZ;
  if (ex * ex + ez * ez < 0.93) return 0.05;
  if (Math.hypot(x - OV.x, z - OV.z) < 10) return 0.12;
  const t = clamp((x - VS.x) * VDIR.x + (z - VS.z) * VDIR.z, 0, VS.distanceTo(VE));
  if (Math.hypot(x - (VS.x + VDIR.x * t), z - (VS.z + VDIR.z * t)) < 3.4) return 0.12;
  return -40;
}
function spawnLeaf(i, burstAt = null) {
  const o = i * 3;
  if (burstAt) {
    LF.pos[o] = burstAt.x + (lr() - 0.5) * 6; LF.pos[o + 1] = burstAt.y + 0.3 + lr() * 1.5; LF.pos[o + 2] = burstAt.z + (lr() - 0.5) * 6;
    LF.vel[o] = (lr() - 0.5) * 5; LF.vel[o + 1] = 4 + lr() * 6; LF.vel[o + 2] = (lr() - 0.5) * 5;
  } else {
    const s = leafSpawnPoints[Math.floor(lr() * leafSpawnPoints.length)];
    if (!s) return;
    LF.pos[o] = s.x + (lr() - 0.5) * s.r * 1.6; LF.pos[o + 1] = s.y + (lr() - 0.3) * s.h; LF.pos[o + 2] = s.z + (lr() - 0.5) * s.r * 1.6;
    LF.vel[o] = 0; LF.vel[o + 1] = -0.3; LF.vel[o + 2] = 0;
    const fc = SEASONS[SS.key].fallCols;
    const col = fc ? fc[Math.floor(lr() * fc.length)] : s.col;
    if (fallingLeaves && col) fallingLeaves.setColorAt(i, col.clone().multiplyScalar(0.8 + lr() * 0.4));
  }
  LF.rot[o] = lr() * TAU; LF.rot[o + 1] = lr() * TAU; LF.rot[o + 2] = lr() * TAU;
  LF.spin[o] = (lr() - 0.5) * 6; LF.spin[o + 1] = (lr() - 0.5) * 4; LF.spin[o + 2] = (lr() - 0.5) * 6;
  LF.rest[i] = 0; LF.phase[i] = lr() * TAU;
}
let petalTex, leafTexOne;
function makePetalTexture() {
  const c = makeCanvas(64), g = c.getContext('2d');
  g.translate(32, 32); g.fillStyle = '#fff';
  g.beginPath(); g.moveTo(0, 26);
  g.bezierCurveTo(24, 14, 22, -18, 6, -24); g.lineTo(0, -16); g.lineTo(-6, -24);
  g.bezierCurveTo(-22, -18, -24, 14, 0, 26); g.fill();
  return toTex(c, true, false);
}
function buildFallingLeaves(tex) {
  leafTexOne = tex; petalTex = makePetalTexture();
  const geo = new THREE.PlaneGeometry(0.34, 0.34);
  const mat = new THREE.MeshStandardMaterial({ map: tex, alphaTest: 0.4, side: THREE.DoubleSide, roughness: 0.75 });
  fallingLeaves = new THREE.InstancedMesh(geo, mat, LEAF_N);
  fallingLeaves.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  fallingLeaves.frustumCulled = false;
  fallingLeaves.castShadow = false;
  const autumn = [0xe79a2a, 0xf1bb35, 0xdb6a26, 0xc2401f, 0xe9d24a].map(srgb);
  for (let i = 0; i < LEAF_N; i++) {
    fallingLeaves.setColorAt(i, autumn[i % autumn.length]);
    LF.size[i] = 0.7 + lr() * 0.7;
    spawnLeaf(i);
    // scatter initial heights so they don't fall in sync
    LF.pos[i * 3 + 1] -= lr() * 12;
    if (lr() < 0.35) { const g = groundAt(LF.pos[i * 3], LF.pos[i * 3 + 2]); if (g > -1) { LF.pos[i * 3 + 1] = g; LF.rest[i] = lr() * 10; } }
  }
  scene.add(fallingLeaves);
}
const _d = new THREE.Object3D();
const windDir = new THREE.Vector3(0.8, 0, 0.35).normalize();
function updateLeaves(dt, t, windStrength) {
  const wx = windDir.x * windStrength, wz = windDir.z * windStrength;
  const S = SEASONS[SS.key];
  SS.fallFrac = approach(SS.fallFrac, S.fall, dt * 0.3);
  const active = Math.floor(LEAF_N * SS.fallFrac), sizeK = S.fallSize;
  for (let i = 0; i < LEAF_N; i++) {
    const o = i * 3;
    if (i >= active) { _d.scale.setScalar(0); _d.updateMatrix(); fallingLeaves.setMatrixAt(i, _d.matrix); continue; }
    if (LF.rest[i] > 0) {
      LF.rest[i] -= dt;
      // gusts can lift resting leaves
      if (windStrength > 4 && lr() < dt * 0.8) { LF.rest[i] = 0; LF.vel[o + 1] = 2 + lr() * 3; }
      if (LF.rest[i] <= 0 && LF.vel[o + 1] <= 0) spawnLeaf(i);
    } else {
      const ph = LF.phase[i];
      const flutX = Math.sin(t * 2.3 + ph) * 0.9, flutZ = Math.cos(t * 1.9 + ph * 1.3) * 0.9;
      LF.vel[o] += (wx + flutX - LF.vel[o]) * dt * 1.2;
      LF.vel[o + 2] += (wz + flutZ - LF.vel[o + 2]) * dt * 1.2;
      LF.vel[o + 1] += (-1.0 + Math.sin(t * 3.1 + ph) * 0.5 - LF.vel[o + 1]) * dt * 1.5;
      LF.pos[o] += LF.vel[o] * dt; LF.pos[o + 1] += LF.vel[o + 1] * dt; LF.pos[o + 2] += LF.vel[o + 2] * dt;
      LF.rot[o] += LF.spin[o] * dt; LF.rot[o + 1] += LF.spin[o + 1] * dt; LF.rot[o + 2] += LF.spin[o + 2] * dt;
      const g = groundAt(LF.pos[o], LF.pos[o + 2]);
      if (LF.pos[o + 1] < g) {
        if (g > -1) { LF.pos[o + 1] = g + 0.02; LF.rest[i] = 4 + lr() * 10; LF.rot[o] = -Math.PI / 2 + (lr() - 0.5) * 0.3; LF.rot[o + 2] = 0; LF.vel[o + 1] = 0; }
        else spawnLeaf(i);
      }
      if (Math.abs(LF.pos[o] - CX) > 220 || Math.abs(LF.pos[o + 2] - CZ) > 220 || LF.pos[o + 1] > 90) spawnLeaf(i);
    }
    _d.position.set(LF.pos[o], LF.pos[o + 1], LF.pos[o + 2]);
    _d.rotation.set(LF.rot[o], LF.rot[o + 1], LF.rot[o + 2]);
    _d.scale.setScalar(LF.size[i] * sizeK);
    _d.updateMatrix();
    fallingLeaves.setMatrixAt(i, _d.matrix);
  }
  fallingLeaves.instanceMatrix.needsUpdate = true;
  if (fallingLeaves.instanceColor) fallingLeaves.instanceColor.needsUpdate = true;
}

/* =====================================================================
   birds & mist
   ===================================================================== */
function buildBirds() {
  const mat = new THREE.MeshBasicMaterial({ color: 0x1e1a18, side: THREE.DoubleSide });
  const wing = new THREE.BufferGeometry();
  wing.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0.3, 0, 0, -0.3, 1.1, 0.05, -0.15], 3));
  wing.computeVertexNormals();
  const r = mulberry32(55);
  for (let i = 0; i < 9; i++) {
    const g = new THREE.Group();
    const wl = new THREE.Mesh(wing, mat), wr = new THREE.Mesh(wing, mat);
    wl.scale.x = -1;
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.14, 0.7), mat);
    g.add(wl, wr, body); g.scale.setScalar(2.2);
    g.userData = { wl, wr, c: new THREE.Vector3(150 + r() * 40, 45 + r() * 30, -60 + r() * 40), R: 40 + r() * 50, sp: (0.06 + r() * 0.05) * (r() < 0.5 ? 1 : -1), ph: r() * TAU, fl: 7 + r() * 3 };
    scene.add(g); birds.push(g);
  }
}
function buildMist() {
  const tex = makeMistTexture();
  const r = mulberry32(66);
  for (let i = 0; i < 22; i++) {
    const a = r() * TAU, d = 240 + r() * 600;
    const x = CX + Math.cos(a) * d, z = CZ + Math.sin(a) * d;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0.16 + r() * 0.16, fog: true }));
    sp.position.set(x, Math.min(terrainH(x, z), -20) + 8 + r() * 14, z);
    sp.scale.set(260 + r() * 220, 60 + r() * 40, 1);
    sp.userData.drift = (r() - 0.5) * 2;
    sp.userData.op = sp.material.opacity;
    scene.add(sp); mists.push(sp);
  }
}

/* =====================================================================
   time of day
   ===================================================================== */
const P = {
  topNight: srgb(0x050a1a), topDusk: srgb(0x3d4f78), topDay: srgb(0x5b8fd0),
  horNight: srgb(0x0e1628), horDusk: srgb(0xf09a5c), horDay: srgb(0xc6d3da),
  sunLow: srgb(0xff5e1a), sunMid: srgb(0xffb877), sunHigh: srgb(0xffe8c8),
  gDay: srgb(0x6f6a52), gNight: srgb(0x06080d),
};
const sunDir = new THREE.Vector3();
const state = { t: 0.577, nightK: 0, dayK: 1, envT: -1, envOv: 0, hour: 15.6, label: '' };
const tmpC = new THREE.Color();
function applyTime(t, force = false) {
  state.t = t;
  const hour = 5.5 + t * 17.5; state.hour = hour;
  const df = (hour - 6.3) / 13.0;
  let el = Math.sin(Math.PI * df) * 44;
  if (df < 0 || df > 1) el *= 0.7;
  const az = THREE.MathUtils.degToRad(lerp(115, -115, df));
  const er = THREE.MathUtils.degToRad(el);
  sunDir.set(Math.sin(az) * Math.cos(er), Math.sin(er), Math.cos(az) * Math.cos(er));
  const dayK = smooth(-2, 22, el), nightK = 1 - smooth(-9, 1, el), duskK = smooth(-11, 0, el);
  state.dayK = dayK; state.nightK = nightK;

  skyU.topColor.value.copy(P.topNight).lerp(P.topDusk, duskK).lerp(P.topDay, dayK);
  skyU.horizonColor.value.copy(P.horNight).lerp(P.horDusk, duskK).lerp(P.horDay, smooth(4, 26, el));
  skyU.groundColor.value.copy(P.gNight).lerp(P.gDay, duskK);
  skyU.sunColor.value.copy(P.sunLow).lerp(P.sunMid, smooth(0, 12, el)).lerp(P.sunHigh, smooth(14, 40, el));
  skyU.sunDir.value.copy(sunDir);
  skyU.night.value = nightK;
  // overcast: pull the sky toward grey
  const ov = W.over;
  greyC.copy(greyNight).lerp(greyDay, duskK * (0.35 + 0.65 * dayK));
  skyU.topColor.value.lerp(tmpC.copy(greyC).multiplyScalar(0.85), ov * 0.85);
  skyU.horizonColor.value.lerp(greyC, ov * 0.8);
  skyU.uCloud.value = W.cloud; skyU.uOver.value = ov; skyU.uFlash.value = W.flash;

  sunLight.color.copy(skyU.sunColor.value);
  sunLight.intensity = 3.4 * smooth(-1.5, 7, el) * (1 - 0.88 * ov);
  sunLight.castShadow = sunLight.intensity > 0.25;
  sunLight.position.copy(sunLight.target.position).addScaledVector(sunDir, 450);
  moonLight.intensity = 1.0 * nightK * (1 - 0.7 * ov);
  hemi.color.copy(skyU.topColor.value).lerp(skyU.horizonColor.value, 0.4);
  hemi.groundColor.copy(P.gDay).multiplyScalar(0.2 + 0.8 * duskK);
  hemi.intensity = lerp(0.5, 0.95, smooth(-10, 12, el)) * (1 + 0.45 * ov) + W.flash * 4;

  fogColor.copy(skyU.horizonColor.value).lerp(skyU.topColor.value, 0.28);
  fogColor.lerp(greyC, Math.max(ov, W.fog) * 0.6);
  if (W.flash > 0) fogColor.lerp(tmpC.setRGB(0.7, 0.75, 0.9), W.flash * 0.4);
  scene.fog.color.copy(fogColor);
  scene.fog.density = lerp(0.0007, 0.0009, 1 - dayK) * (1 + ov * 0.7 + W.fog * 9 + Math.min(1, W.rain) * 2.2 + W.snow * 3.5);
  renderer.toneMappingExposure = lerp(1.0, 1.35, nightK);
  scene.environmentIntensity = lerp(0.55, 0.25, nightK);

  glassMat.emissiveIntensity = 0.04 + (1 - dayK) * 0.8 + nightK * 2.6;
  lampMat.emissiveIntensity = 0.5 + (1 - dayK) * 2 + nightK * 6;
  ghGlassMat.emissiveIntensity = nightK * 0.35;
  for (const m of mists) {
    m.material.color.copy(fogColor).multiplyScalar(1.05 + (1 - dayK) * 0.2);
    m.material.opacity = m.userData.op * (1 + W.fog * 2.4 + Math.min(1, W.rain) * 0.8);
  }

  if (force || Math.abs(t - state.envT) > 0.015 || Math.abs(ov - state.envOv) > 0.05) {
    state.envT = t; state.envOv = ov;
    if (envRT) envRT.dispose();
    envRT = pmrem.fromScene(skyScene, 0, 1, 3000);
    scene.environment = envRT.texture;
  }
  const hh = Math.floor(hour) % 24, mm = Math.floor((hour % 1) * 60);
  const label = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
  if (label !== state.label) { state.label = label; document.getElementById('timeLabel').textContent = label; }
}

/* =====================================================================
   interior: the Great Hall (separate scene, entered through the portal)
   ===================================================================== */
function makeWoodTexture() {
  const S = 512, r = mulberry32(91), c = makeCanvas(S), g = c.getContext('2d');
  g.fillStyle = '#3a2616'; g.fillRect(0, 0, S, S);
  for (let x = 0; x < S; x += 64) {
    g.fillStyle = `hsl(${24 + r() * 8},${34 + r() * 12}%,${20 + r() * 9}%)`;
    g.fillRect(x + 1, 0, 62, S);
    for (let k = 0; k < 14; k++) {
      g.strokeStyle = `rgba(20,10,4,${0.15 + r() * 0.25})`; g.lineWidth = 1 + r() * 1.5;
      g.beginPath(); const gx = x + 4 + r() * 56; g.moveTo(gx, 0);
      for (let y = 0; y <= S; y += 32) g.lineTo(gx + Math.sin(y * 0.02 + k) * 2.5, y);
      g.stroke();
    }
  }
  addNoise(g, S, 14, r);
  return toTex(c);
}
function makeStainedTexture(seed, rose = false) {
  const W = rose ? 512 : 256, H = rose ? 512 : 768, r = mulberry32(seed);
  const c = makeCanvas(W, H), g = c.getContext('2d');
  const jewels = ['#1d3f9a', '#2757c9', '#9c1c2a', '#c93a2a', '#d99a1c', '#f0c040', '#1f7a45', '#5a2a86', '#2a9aa8'];
  g.fillStyle = '#111'; g.fillRect(0, 0, W, H);
  if (rose) {
    const cx = W / 2, cy = H / 2;
    for (let ring = 0; ring < 5; ring++) {
      const r0 = ring * 52, r1 = r0 + 52, n = 8 + ring * 8;
      for (let i = 0; i < n; i++) {
        const a0 = (i / n) * TAU, a1 = ((i + 1) / n) * TAU;
        g.fillStyle = jewels[(i + ring * 3) % jewels.length];
        g.beginPath(); g.arc(cx, cy, r1, a0, a1); g.arc(cx, cy, r0, a1, a0, true); g.closePath(); g.fill();
      }
    }
    g.strokeStyle = '#0c0a08'; g.lineWidth = 5;
    for (let ring = 1; ring <= 5; ring++) { g.beginPath(); g.arc(cx, cy, ring * 52, 0, TAU); g.stroke(); }
    for (let i = 0; i < 16; i++) { const a = (i / 16) * TAU; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * 260, cy + Math.sin(a) * 260); g.stroke(); }
  } else {
    // jittered grid of glass pieces + a central medallion
    const cols = 4, rows = 12, cw = W / cols, ch = H / rows;
    const pts = [];
    for (let j = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) {
      const edge = i === 0 || j === 0 || i === cols || j === rows;
      pts.push([i * cw + (edge ? 0 : (r() - 0.5) * cw * 0.5), j * ch + (edge ? 0 : (r() - 0.5) * ch * 0.5)]);
    }
    const P = (i, j) => pts[j * (cols + 1) + i];
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      g.fillStyle = jewels[Math.floor(r() * jewels.length)];
      g.beginPath(); g.moveTo(...P(i, j)); g.lineTo(...P(i + 1, j)); g.lineTo(...P(i + 1, j + 1)); g.lineTo(...P(i, j + 1)); g.closePath(); g.fill();
      g.strokeStyle = '#0c0a08'; g.lineWidth = 4; g.stroke();
    }
    g.fillStyle = '#e8c35a'; g.beginPath(); g.arc(W / 2, H * 0.42, W * 0.3, 0, TAU); g.fill();
    g.fillStyle = seed % 2 ? '#9c1c2a' : '#1d3f9a'; g.beginPath(); g.arc(W / 2, H * 0.42, W * 0.22, 0, TAU); g.fill();
    g.fillStyle = '#f3e3b0';
    g.beginPath(); g.moveTo(W / 2, H * 0.42 - W * 0.16); g.lineTo(W / 2 + W * 0.1, H * 0.42); g.lineTo(W / 2, H * 0.42 + W * 0.16); g.lineTo(W / 2 - W * 0.1, H * 0.42); g.closePath(); g.fill();
    g.strokeStyle = '#0c0a08'; g.lineWidth = 5;
    g.beginPath(); g.arc(W / 2, H * 0.42, W * 0.3, 0, TAU); g.stroke();
    g.beginPath(); g.arc(W / 2, H * 0.42, W * 0.22, 0, TAU); g.stroke();
  }
  for (let i = 0; i < 400; i++) { g.fillStyle = `rgba(255,255,255,${r() * 0.08})`; g.fillRect(r() * W, r() * H, 3 + r() * 10, 3 + r() * 10); }
  const t = toTex(c);
  if (!rose) { t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping; t.offset.set(0.5, 0); }
  return t;
}
function makeBannerTexture(bg, fg, emblem) {
  const c = makeCanvas(128, 384), g = c.getContext('2d');
  g.fillStyle = bg; g.fillRect(0, 0, 128, 384);
  g.fillStyle = fg; g.fillRect(0, 0, 128, 14); g.fillRect(8, 20, 4, 330); g.fillRect(116, 20, 4, 330);
  // swallow-tail bottom
  g.clearRect(0, 330, 128, 54); g.fillStyle = bg;
  g.beginPath(); g.moveTo(0, 330); g.lineTo(128, 330); g.lineTo(128, 384); g.lineTo(64, 350); g.lineTo(0, 384); g.closePath(); g.fill();
  g.fillStyle = fg; g.save(); g.translate(64, 170);
  g.beginPath();
  if (emblem === 'leaf') { g.moveTo(0, -46); g.quadraticCurveTo(40, -10, 0, 46); g.quadraticCurveTo(-40, -10, 0, -46); }
  else if (emblem === 'tower') { g.rect(-18, -20, 36, 60); g.rect(-24, -34, 12, 16); g.rect(-6, -34, 12, 16); g.rect(12, -34, 12, 16); }
  else if (emblem === 'star') { for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + (k / 10) * TAU, rr = k % 2 ? 18 : 44; g.lineTo(Math.cos(a) * rr, Math.sin(a) * rr); } }
  else { g.arc(0, 0, 40, 0.6, TAU - 0.6); g.arc(16, 0, 30, TAU - 0.9, 0.9, true); }
  g.closePath(); g.fill(); g.restore();
  return toTex(c, true, false);
}

function buildInterior() {
  const S = new THREE.Scene();
  S.fog = new THREE.FogExp2(0x2a1c10, 0.006);
  const HW = 9, L = 42, WH = 16, RIDGE = 25, TH = 0.85;
  const woodTex = makeWoodTexture();
  const inStone = new THREE.MeshStandardMaterial({ map: TX.stone, bumpMap: TX.bump, bumpScale: 0.6, roughness: 0.9, color: 0xdac9a6 });
  const inTrim = new THREE.MeshStandardMaterial({ map: TX.stone, bumpMap: TX.bump, bumpScale: 0.5, roughness: 0.85, color: 0xb3a183 });
  const floorMat = new THREE.MeshStandardMaterial({ map: TX.pave, roughness: 0.5, color: 0xc2b59e });
  const woodDark = new THREE.MeshStandardMaterial({ map: woodTex, color: 0x9a7658, roughness: 0.75 });
  const woodTable = new THREE.MeshStandardMaterial({ map: woodTex, color: 0xd8aa7a, roughness: 0.55 });
  const gold = new THREE.MeshStandardMaterial({ color: 0xe0b35a, metalness: 0.9, roughness: 0.28 });
  const iron = new THREE.MeshStandardMaterial({ color: 0x1c1a18, metalness: 0.6, roughness: 0.5 });
  const soot = new THREE.MeshStandardMaterial({ color: 0x0d0a08, roughness: 1 });
  const glassTex = makeStainedTexture(3), glassTex2 = makeStainedTexture(4), roseTex = makeStainedTexture(5, true);
  const glassMats = [glassTex, glassTex2, roseTex].map((t) => new THREE.MeshStandardMaterial({ map: t, emissiveMap: t, emissive: 0xffffff, emissiveIntensity: 1, roughness: 0.3 }));
  const b = new Batcher(), gb = new Batcher();

  // floor, dais
  b.add(new THREE.BoxGeometry(2 * HW + 2 * TH, 0.2, L + 2 * TH), floorMat, M(0, -0.1, -L / 2), 'world', 1 / 6);
  b.add(new THREE.BoxGeometry(2 * HW, 0.6, 7), floorMat, M(0, 0.3, -L + 3.5), 'world', 1 / 6);
  b.add(new THREE.BoxGeometry(2 * HW, 0.3, 0.8), inTrim, M(0, 0.15, -L + 7.4), 'world', 1 / 6);

  // long walls with lancet windows
  const winU = []; for (let i = 0; i < 8; i++) winU.push(4.2 + i * 4.8);
  const winW = 2.4, winH = 8.6, winY = 3.4;
  const glassSpots = [];
  for (const side of [-1, 1]) {
    const sh = new THREE.Shape(); sh.moveTo(0, 0); sh.lineTo(L, 0); sh.lineTo(L, WH); sh.lineTo(0, WH); sh.lineTo(0, 0);
    for (const u of winU) sh.holes.push(archShape(winW, winH, u, winY));
    b.add(new THREE.ExtrudeGeometry(sh, { depth: TH, bevelEnabled: false, curveSegments: 10 }), inStone, M(side < 0 ? -HW - TH : HW, 0, 0, Math.PI / 2), 'world', 1 / 6);
    winU.forEach((u, i) => {
      const ry = side < 0 ? Math.PI / 2 : -Math.PI / 2, gx = side * (HW + TH - 0.12);
      gb.add(new THREE.ShapeGeometry(archShape(winW, winH), 10), glassMats[i % 2], M(gx, winY, -u, ry), 'scale', [1 / winW, 1 / winH]);
      glassSpots.push({ x: gx, y: winY, z: -u, ry });
      b.add(new THREE.BoxGeometry(0.7, 0.3, winW + 0.5), inTrim, M(side * (HW - 0.2), winY - 0.3, -u));
    });
    // engaged columns + hammerbeam trusses
    for (let i = 0; i <= 8; i++) {
      const u = 1.8 + i * 4.8, z = -u, xw = side * HW;
      b.add(new THREE.CylinderGeometry(0.42, 0.5, 14.6, 16), inTrim, M(side * (HW - 0.2), 7.3, z), 'scale', [0.5, 2.4]);
      b.add(new THREE.BoxGeometry(1.2, 0.6, 1.2), inTrim, M(side * (HW - 0.3), 14.9, z));
      b.add(new THREE.BoxGeometry(2.8, 0.45, 0.45), woodDark, M(side * (HW - 1.4), 15.4, z));
      b.add(new THREE.BoxGeometry(3.6, 0.32, 0.32), woodDark, M(xw - side * 1.3, 13.9, z, 0, 1, 1, 1, 0, side < 0 ? Math.PI / 4 : -Math.PI / 4));
      b.add(new THREE.BoxGeometry(0.34, 3.8, 0.34), woodDark, M(side * (HW - 2.6), 17.2, z));
      b.add(new THREE.BoxGeometry(3.2, 0.3, 0.3), woodDark, M(side * (HW - 4.1), 20.0, z, 0, 1, 1, 1, 0, side < 0 ? 0.7 : -0.7));
      if (side > 0) b.add(new THREE.BoxGeometry(8, 0.4, 0.4), woodDark, M(0, 21.5, z));
    }
    // purlins
    for (const ax of [8.2, 5.6, 3.0]) b.add(new THREE.BoxGeometry(0.3, 0.3, L), woodDark, M(side * ax, 16 + (9 * (10.2 - ax)) / 10.2 - 0.35, -L / 2));
  }
  // roof slabs (casting shadows so sunlight only enters through windows)
  const th = Math.atan2(9, 10.2), slabW = Math.hypot(10.2, 9);
  for (const side of [-1, 1]) {
    const cx = side * 5.1 + side * Math.sin(th) * 0.22, cy = 20.5 + Math.cos(th) * 0.22;
    b.add(new THREE.BoxGeometry(slabW + 0.6, 0.44, L + 2.6), woodDark, M(cx, cy, -L / 2, 0, 1, 1, 1, 0, -side * th), 'world', 1 / 3);
  }
  b.add(new THREE.BoxGeometry(0.5, 0.5, L), woodDark, M(0, 24.6, -L / 2));
  // gable end walls
  const gable = () => {
    const s = new THREE.Shape();
    s.moveTo(-HW - TH, 0); s.lineTo(HW + TH, 0); s.lineTo(HW + TH, WH); s.lineTo(0, RIDGE + 0.5); s.lineTo(-HW - TH, WH); s.lineTo(-HW - TH, 0);
    return s;
  };
  {
    const s = gable(); s.holes.push(archShape(3.2, 6.6, 0, 9.4));
    b.add(new THREE.ExtrudeGeometry(s, { depth: TH, bevelEnabled: false, curveSegments: 10 }), inStone, M(0, 0, 0), 'world', 1 / 6);
    gb.add(new THREE.ShapeGeometry(archShape(3.2, 6.6), 10), glassMats[0], M(0, 9.4, TH - 0.12, Math.PI), 'scale', [1 / 3.2, 1 / 6.6]);
    glassSpots.push({ x: 0, y: 9.4, z: TH - 0.12, ry: Math.PI, w: 3.2, h: 6.6 });
    // entrance doors (closed; walking into them leads outside)
    for (let k = 0; k < 4; k++) {
      const w = 3.8 + k * 0.7, h = 6.6 + k * 0.55;
      const sh = archShape(w + 0.7, h + 0.5); sh.holes.push(archShape(w, h));
      b.add(new THREE.ExtrudeGeometry(sh, { depth: 0.3 + k * 0.2, bevelEnabled: false, curveSegments: 10 }), inTrim, M(0, 0, 0, Math.PI), 'world', 1 / 6);
    }
    b.add(new THREE.ShapeGeometry(archShape(3.8, 6.6), 10), woodDark, M(0, 0, -0.02, Math.PI), 'scale', [0.3, 0.3]);
    b.add(new THREE.BoxGeometry(0.08, 6.0, 0.08), iron, M(0, 3, -0.08));
    for (const hy of [1.4, 4.2]) b.add(new THREE.BoxGeometry(3.4, 0.14, 0.06), iron, M(0, hy, -0.07));
  }
  {
    const s = gable();
    const rose = new THREE.Path(); rose.absarc(0, 13.3, 3.3, 0, TAU, true); s.holes.push(rose);
    for (const lx of [-5.8, 5.8]) s.holes.push(archShape(1.8, 6.4, lx, 5.2));
    b.add(new THREE.ExtrudeGeometry(s, { depth: TH, bevelEnabled: false, curveSegments: 24 }), inStone, M(0, 0, -L - TH), 'world', 1 / 6);
    gb.add(new THREE.CircleGeometry(3.3, 48), glassMats[2], M(0, 13.3, -L - TH + 0.12), 'scale', [1, 1]);
    const ringS = new THREE.Shape(); ringS.absarc(0, 0, 3.8, 0, TAU, false);
    const ringH = new THREE.Path(); ringH.absarc(0, 0, 3.3, 0, TAU, true); ringS.holes.push(ringH);
    b.add(new THREE.ExtrudeGeometry(ringS, { depth: 0.4, bevelEnabled: false, curveSegments: 40 }), inTrim, M(0, 13.3, -L), 'world', 1 / 6);
    for (const lx of [-5.8, 5.8]) {
      gb.add(new THREE.ShapeGeometry(archShape(1.8, 6.4), 10), glassMats[1], M(lx, 5.2, -L - TH + 0.12), 'scale', [1 / 1.8, 1 / 6.4]);
      glassSpots.push({ x: lx, y: 5.2, z: -L - TH + 0.12, ry: 0, w: 1.8, h: 6.4 });
    }
    // fireplace
    for (const px of [-2.5, 2.5]) b.add(new THREE.BoxGeometry(1.0, 3.4, 1.4), inTrim, M(px, 0.6 + 1.7, -L + 0.7));
    b.add(new THREE.BoxGeometry(6.2, 1.0, 1.6), inTrim, M(0, 0.6 + 3.9, -L + 0.8));
    b.add(new THREE.BoxGeometry(5.2, 4.2, 1.0), inStone, M(0, 0.6 + 6.5, -L + 0.5), 'world', 1 / 6);
    b.add(new THREE.BoxGeometry(4.0, 3.4, 0.2), soot, M(0, 0.6 + 1.7, -L + 0.1));
    b.add(new THREE.BoxGeometry(4.0, 0.1, 1.2), soot, M(0, 0.62, -L + 0.7));
    for (let k = 0; k < 3; k++) b.add(new THREE.CylinderGeometry(0.16, 0.16, 1.8, 8), woodDark, M(-0.1 + k * 0.1, 0.8 + (k === 2 ? 0.25 : 0), -L + 0.8 - k * 0.25 + (k === 2 ? 0.25 : 0), 0.3 * k, 1, 1, 1, 0, Math.PI / 2));
  }

  // high table & chairs on the dais
  const dz = -L + 4;
  b.add(new THREE.BoxGeometry(13, 0.12, 1.4), woodTable, M(0, 0.6 + 0.8, dz), 'world', 1 / 3);
  b.add(new THREE.BoxGeometry(12.6, 0.72, 0.1), woodDark, M(0, 0.6 + 0.38, dz + 0.6), 'world', 1 / 3);
  for (let i = 0; i < 7; i++) {
    const cx = -5.4 + i * 1.8, big = i === 3;
    b.add(new THREE.BoxGeometry(0.7, 0.1, 0.7), woodDark, M(cx, 0.6 + 0.5, dz - 1.1));
    b.add(new THREE.BoxGeometry(0.7, big ? 2.4 : 1.5, 0.1), big ? gold : woodDark, M(cx, 0.6 + (big ? 1.7 : 1.25), dz - 1.45));
  }
  // long tables & benches
  const tables = [-5.6, -2.0, 2.0, 5.6], tz0 = -6, tz1 = -32, tLen = tz0 - tz1, tzc = (tz0 + tz1) / 2;
  for (const tx of tables) {
    b.add(new THREE.BoxGeometry(1.05, 0.09, tLen), woodTable, M(tx, 0.78, tzc), 'world', 1 / 3);
    for (const bs of [-0.8, 0.8]) {
      b.add(new THREE.BoxGeometry(0.38, 0.07, tLen), woodDark, M(tx + bs, 0.45, tzc), 'world', 1 / 3);
    }
    for (let z = tz1 + 0.6; z <= tz0 - 0.6; z += 3.2) {
      b.add(new THREE.BoxGeometry(0.85, 0.72, 0.12), woodDark, M(tx, 0.37, z));
      for (const bs of [-0.8, 0.8]) b.add(new THREE.BoxGeometry(0.3, 0.42, 0.1), woodDark, M(tx + bs, 0.21, z));
    }
  }
  // banners on the entrance wall
  [['#7d1b1b', '#d8aa4c', 'leaf'], ['#1f4d2e', '#c9ccd2', 'tower'], ['#1d2f66', '#c89a5a', 'star'], ['#b98a24', '#1a1612', 'moon']].forEach(([bg, fg, em], i) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 5.4), new THREE.MeshStandardMaterial({ map: makeBannerTexture(bg, fg, em), side: THREE.DoubleSide, transparent: true, alphaTest: 0.5, roughness: 0.9 }));
    m.position.set([-7.1, -4.2, 4.2, 7.1][i], 10.2, -0.1); m.rotation.y = Math.PI;
    S.add(m);
  });

  b.build(S, true);
  gb.build(S, false);

  // place settings (instanced)
  const plates = [], cups = [];
  for (const tx of tables) for (const sd of [-0.3, 0.3]) for (let z = tz1 + 0.8; z <= tz0 - 0.8; z += 0.9) { plates.push([tx + sd, 0.835, z]); cups.push([tx + sd * 0.55, 0.83, z + 0.25]); }
  for (let i = 0; i < 7; i++) { plates.push([-5.4 + i * 1.8, 1.47, dz - 0.3]); cups.push([-5.1 + i * 1.8, 1.465, dz - 0.1]); }
  const inst = (geo, mat, list) => {
    const im = new THREE.InstancedMesh(geo, mat, list.length), d = new THREE.Object3D();
    list.forEach((p, i) => { d.position.set(...p); d.updateMatrix(); im.setMatrixAt(i, d.matrix); });
    im.castShadow = true; im.receiveShadow = true; S.add(im); return im;
  };
  inst(new THREE.CylinderGeometry(0.15, 0.12, 0.02, 18), gold, plates);
  inst(new THREE.LatheGeometry([[0.03, 0], [0.05, 0.01], [0.012, 0.03], [0.012, 0.1], [0.05, 0.13], [0.055, 0.22]].map(([x, y]) => new THREE.Vector2(x, y)), 12), gold, cups);

  // seasonal table decorations
  const decoGeo = new THREE.SphereGeometry(1, 20, 14);
  { const p = decoGeo.attributes.position; for (let i = 0; i < p.count; i++) { const x = p.getX(i), z = p.getZ(i), k = 1 + 0.07 * Math.cos(8 * Math.atan2(z, x)) * Math.sqrt(x * x + z * z); p.setX(i, x * k); p.setZ(i, z * k); } decoGeo.computeVertexNormals(); }
  const deco = new THREE.InstancedMesh(decoGeo, new THREE.MeshStandardMaterial({ roughness: 0.55 }), 260);
  deco.castShadow = true; S.add(deco);
  const xmas = new THREE.Group(); S.add(xmas);
  {
    const tree = buildConiferCards(33, 6.5, 2.1);
    const mat = new THREE.MeshStandardMaterial({ map: TX.leaf, alphaTest: 0.45, side: THREE.DoubleSide, vertexColors: true, roughness: 0.8, color: 0x3f6a34 });
    const orn = [];
    const ro = mulberry32(8);
    for (const tx of [-7.2, 7.2]) {
      const m = new THREE.Mesh(tree.leaves, mat); m.position.set(tx, 0.6, -L + 7.2); m.castShadow = true; xmas.add(m);
      for (let k = 0; k < 70; k++) {
        const h = ro() * 5.4 + 0.8, rr = 2.1 * (1 - h / 6.6) + 0.1, a = ro() * TAU;
        orn.push(tx + Math.cos(a) * rr, 0.6 + h, -L + 7.2 + Math.sin(a) * rr);
      }
    }
    const og = new THREE.BufferGeometry(); og.setAttribute('position', new THREE.Float32BufferAttribute(orn, 3));
    const oc = []; for (let i = 0; i < orn.length / 3; i++) { const c = [srgb(0xffd27a), srgb(0xff5a4a), srgb(0xfff2d0)][i % 3]; oc.push(c.r, c.g, c.b); }
    og.setAttribute('color', new THREE.Float32BufferAttribute(oc, 3));
    xmas.add(new THREE.Points(og, new THREE.PointsMaterial({ size: 0.16, vertexColors: true, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, map: TX.flame })));
  }
  const setDecor = (key) => {
    const d = new THREE.Object3D(), r = mulberry32(12 + key.length);
    const put = [];
    for (const tx of tables) for (let z = tz1 + 1.6; z <= tz0 - 1.0; z += 3.2) {
      if (key === 'autumn') {
        put.push([tx, 0.95, z, 0.17, 0.13, 0.17, 0xd8741e]);
        put.push([tx + 0.18, 0.88, z + 0.3, 0.07, 0.065, 0.07, 0xa8231a], [tx - 0.15, 0.88, z - 0.32, 0.07, 0.065, 0.07, 0xb8b02a]);
      } else if (key === 'winter') {
        put.push([tx, 0.89, z, 0.28, 0.07, 0.12, 0x2f5a2a]);
        for (let k = 0; k < 4; k++) put.push([tx + (r() - 0.5) * 0.2, 0.93, z + (r() - 0.5) * 0.3, 0.04, 0.04, 0.04, 0xc01818]);
      } else if (key === 'spring') {
        for (let k = 0; k < 6; k++) put.push([tx + (r() - 0.5) * 0.25, 0.95 + r() * 0.12, z + (r() - 0.5) * 0.3, 0.08, 0.08, 0.08, [0xf6c3d0, 0xfbe4ea, 0xe88aa6, 0xfff6d8][k % 4]]);
      } else {
        for (let k = 0; k < 5; k++) put.push([tx + (r() - 0.5) * 0.22, 0.89 + (k > 3 ? 0.08 : 0), z + (r() - 0.5) * 0.25, 0.075, 0.07, 0.075, [0x9ac23a, 0xe8d23a, 0x6fa82a][k % 3]]);
      }
    }
    deco.count = Math.min(put.length, 260);
    put.slice(0, 260).forEach(([x, y, z, sx, sy, sz, c], i) => {
      d.position.set(x, y, z); d.scale.set(sx, sy, sz); d.rotation.set(0, r() * TAU, 0); d.updateMatrix();
      deco.setMatrixAt(i, d.matrix); deco.setColorAt(i, srgb(c));
    });
    deco.instanceMatrix.needsUpdate = true; if (deco.instanceColor) deco.instanceColor.needsUpdate = true;
    xmas.visible = key === 'winter';
  };

  // floating candles
  const CN = 150, cr = mulberry32(77), candles = [];
  for (let i = 0; i < CN; i++) candles.push({ x: (cr() - 0.5) * 15, y: 7 + cr() * 3.2, z: -3.5 - cr() * 33, ph: cr() * TAU, h: 0.28 + cr() * 0.2 });
  const candleMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.045, 0.05, 1, 8).translate(0, -0.5, 0), new THREE.MeshStandardMaterial({ color: 0xf3e9d6, emissive: 0xffd9a0, emissiveIntensity: 0.25, roughness: 0.6 }), CN);
  S.add(candleMesh);
  const flamePos = new Float32Array(CN * 3), flameSeed = new Float32Array(CN);
  candles.forEach((c, i) => { flameSeed[i] = c.ph; });
  const fg = new THREE.BufferGeometry(); fg.setAttribute('position', new THREE.BufferAttribute(flamePos, 3)); fg.setAttribute('aSeed', new THREE.BufferAttribute(flameSeed, 1));
  const flamePts = new THREE.Points(fg, new THREE.ShaderMaterial({
    uniforms: { uTime: U.uTime, uMap: { value: TX.flame }, uSize: { value: 70 * DPR } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float aSeed; uniform float uTime, uSize; varying float vF;
      void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        vF = 0.8 + 0.2 * sin(uTime * 13.0 + aSeed * 9.0) + 0.1 * sin(uTime * 23.0 + aSeed);
        gl_PointSize = uSize * vF / max(-mv.z, 0.3); }`,
    fragmentShader: `uniform sampler2D uMap; varying float vF;
      void main(){ vec2 uv = gl_PointCoord; uv.y = 1.0 - uv.y; uv = (uv - 0.5) * vec2(1.6, 1.0) + 0.5; vec4 t = texture2D(uMap, uv); gl_FragColor = vec4(vec3(1.0, 0.72, 0.38) * 3.0 * vF, t.a); }`,
  }));
  flamePts.frustumCulled = false; S.add(flamePts);

  // fireplace fire
  const fire = new THREE.Group(); fire.position.set(0, 0.95, -L + 0.8); S.add(fire);
  for (let k = 0; k < 6; k++) {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: TX.flame, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, fog: false, color: new THREE.Color(4, 2.6, 1.4) }));
    sp.userData.base = [1.1 - (k % 3) * 0.25, 1.9 - (k % 3) * 0.4, k * 1.3, (k - 2.5) * 0.35];
    fire.add(sp);
  }

  // light shafts through the windows (stretched along the sun direction in the shader)
  const shaftMat = new THREE.ShaderMaterial({
    uniforms: { uTime: U.uTime, uLight: { value: new THREE.Vector3(0, -1, 0) }, uStr: { value: 0 }, uColor: { value: new THREE.Color(1, 0.85, 0.6) }, uLen: { value: 34 } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false,
    vertexShader: `uniform vec3 uLight; uniform float uLen; varying float vZ, vFace; varying vec2 vXY; varying vec3 vW;
      void main(){
        vec4 base = modelMatrix * vec4(position.xy, 0.0, 1.0);
        vec3 inward = normalize((modelMatrix * vec4(0.0, 0.0, 1.0, 0.0)).xyz);
        vFace = max(dot(inward, uLight), 0.0);
        vec3 w = base.xyz + uLight * position.z * uLen;
        vZ = position.z; vXY = vec2(position.x / 1.1, (position.y - 4.0) / 3.9); vW = w;
        gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
      }`,
    fragmentShader: `uniform float uStr, uTime; uniform vec3 uColor; varying float vZ, vFace; varying vec2 vXY; varying vec3 vW;
      void main(){
        float e = 1.0 - smoothstep(0.55, 1.0, min(abs(vXY.x), abs(vXY.y)));
        float n = 0.7 + 0.3 * sin(vW.x * 0.8 + vW.z * 0.6 + uTime * 0.25) * sin(vW.y * 1.1 - uTime * 0.17);
        float a = uStr * smoothstep(0.05, 0.3, vFace) * pow(1.0 - vZ, 1.4) * e * n * smoothstep(0.0, 0.03, vZ);
        gl_FragColor = vec4(uColor, a);
      }`,
  });
  const shaftGeo = new THREE.BoxGeometry(2.2, 7.8, 1, 1, 1, 1).translate(0, 4.0, 0.5);
  for (const g of glassSpots) {
    const m = new THREE.Mesh(shaftGeo, shaftMat);
    m.position.set(g.x, g.y, g.z); m.rotation.y = g.ry;
    if (g.w) m.scale.set(g.w / 2.4, g.h / 8.6, 1);
    m.frustumCulled = false; m.renderOrder = 3; S.add(m);
  }

  // dust motes
  const DN = 1400, dp = new Float32Array(DN * 3), dr = mulberry32(5);
  for (let i = 0; i < DN; i++) { dp[i * 3] = (dr() - 0.5) * 17; dp[i * 3 + 1] = dr() * 14; dp[i * 3 + 2] = -dr() * L; }
  const dg = new THREE.BufferGeometry(); dg.setAttribute('position', new THREE.BufferAttribute(dp, 3));
  const dustMat = new THREE.ShaderMaterial({
    uniforms: { uTime: U.uTime, uAmt: { value: 0.5 }, uSize: { value: 6 * DPR } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `uniform float uTime, uSize; varying float vA;
      void main(){ vec3 p = position; p.x += sin(uTime * 0.13 + p.z) * 0.6; p.y += sin(uTime * 0.09 + p.x * 2.0) * 0.8; p.z += cos(uTime * 0.11 + p.y) * 0.6;
        vec4 mv = modelViewMatrix * vec4(p, 1.0); gl_Position = projectionMatrix * mv;
        vA = 0.5 + 0.5 * sin(uTime * 0.7 + position.x * 13.0); gl_PointSize = uSize / max(-mv.z, 0.5); }`,
    fragmentShader: `uniform float uAmt; varying float vA; void main(){ float a = smoothstep(0.5, 0.0, length(gl_PointCoord - 0.5)) * vA * uAmt; gl_FragColor = vec4(1.0, 0.9, 0.7, a); }`,
  });
  S.add(new THREE.Points(dg, dustMat));

  // lights
  const hemiIn = new THREE.HemisphereLight(0xffe6c8, 0x3a2a1a, 0.3); S.add(hemiIn);
  const sunIn = new THREE.DirectionalLight(0xfff0d8, 3);
  sunIn.castShadow = true; sunIn.shadow.mapSize.set(2048, 2048);
  Object.assign(sunIn.shadow.camera, { left: -32, right: 32, top: 32, bottom: -32, near: 1, far: 220 });
  sunIn.shadow.bias = -0.0005; sunIn.shadow.normalBias = 0.04;
  sunIn.target.position.set(0, 0, -L / 2); S.add(sunIn, sunIn.target);
  const moonIn = new THREE.DirectionalLight(0x9fb6ff, 0); moonIn.position.set(-40, 50, -10); moonIn.target.position.set(0, 0, -L / 2); S.add(moonIn, moonIn.target);
  const warm = [-9, -20, -31].map((z) => { const l = new THREE.PointLight(0xffb466, 20, 0, 1.6); l.position.set(0, 8.6, z); S.add(l); return l; });
  const fireLight = new THREE.PointLight(0xff7a2a, 30, 0, 1.8); fireLight.position.set(0, 2.2, -L + 2.2); S.add(fireLight);
  const pm = new THREE.PMREMGenerator(renderer);
  S.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  S.environmentIntensity = 0.18;

  const cd = new THREE.Object3D();
  const skyTint = new THREE.Color();
  return {
    scene: S,
    setDecor,
    spawn: { x: 0, z: -2.8, yaw: 0 },
    ground: (x, z) => (z < -L + 7 ? 0.6 : z < -L + 7.8 ? 0.3 : 0),
    canStand(x, z) {
      if (Math.abs(x) > HW - 0.95 || z > -0.6 || z < -L + 1.8) return false;
      for (const tx of tables) if (Math.abs(x - tx) < 1.3 && z < tz0 + 0.3 && z > tz1 - 0.3) return false;
      if (Math.abs(x) < 6.9 && z < dz + 1.1 && z > dz - 1.9) return false;
      if (xmas.visible && Math.hypot(Math.abs(x) - 7.2, z - (-L + 7.2)) < 2.3) return false;
      return true;
    },
    isExit: (x, z) => z > -1.3 && Math.abs(x) < 1.9,
    update(dt, t) {
      const sunUp = sunDir.y > 0.02;
      const ov = W.over;
      sunIn.color.copy(sunLight.color);
      sunIn.intensity = sunUp ? sunLight.intensity * 1.2 : 0;
      sunIn.castShadow = sunIn.intensity > 0.2;
      sunIn.position.copy(sunIn.target.position).addScaledVector(sunDir, 90);
      moonIn.intensity = moonLight.intensity * 0.5;
      hemiIn.intensity = 0.14 + 0.38 * state.dayK * (1 - 0.4 * ov) + W.flash * 1.5;
      skyTint.copy(skyU.horizonColor.value).lerp(tmpC.setRGB(1, 1, 1), 0.55);
      const glow = 0.06 + state.dayK * (1.5 - ov * 0.8) + (1 - state.dayK) * (1 - state.nightK) * 0.6 + W.flash * 3;
      for (const m of glassMats) { m.emissiveIntensity = glow; m.emissive.copy(skyTint); }
      shaftMat.uniforms.uLight.value.copy(sunDir).negate();
      shaftMat.uniforms.uStr.value = sunUp ? (sunLight.intensity / 3.4) * 0.09 : 0;
      shaftMat.uniforms.uColor.value.copy(sunLight.color);
      dustMat.uniforms.uAmt.value = 0.25 + 0.5 * state.dayK * (1 - ov);
      const nightBoost = 0.45 + 0.55 * (1 - state.dayK);
      warm.forEach((l, i) => { l.intensity = 26 * nightBoost * (0.92 + 0.08 * Math.sin(t * 7 + i * 2)); });
      fireLight.intensity = (26 + 22 * nightBoost) * (0.82 + 0.12 * Math.sin(t * 17) + 0.08 * Math.sin(t * 29 + 1.3));
      S.fog.color.setRGB(0.05, 0.035, 0.02).lerp(tmpC.setRGB(0.25, 0.2, 0.15), state.dayK * (1 - ov) * 0.6);
      candles.forEach((c, i) => {
        const y = c.y + Math.sin(t * 0.6 + c.ph) * 0.18;
        cd.position.set(c.x, y, c.z); cd.scale.set(1, c.h, 1); cd.updateMatrix(); candleMesh.setMatrixAt(i, cd.matrix);
        flamePos[i * 3] = c.x; flamePos[i * 3 + 1] = y + 0.07; flamePos[i * 3 + 2] = c.z;
      });
      candleMesh.instanceMatrix.needsUpdate = true; fg.attributes.position.needsUpdate = true;
      fire.children.forEach((sp) => {
        const [w, h, ph, ox] = sp.userData.base;
        const fl = 0.8 + 0.2 * Math.sin(t * 15 + ph) + 0.12 * Math.sin(t * 27 + ph * 2);
        sp.scale.set(w * fl, h * (0.85 + 0.25 * Math.sin(t * 9 + ph)), 1);
        sp.position.set(ox + Math.sin(t * 6 + ph) * 0.06, h * 0.32, 0);
      });
      renderer.toneMappingExposure = 1.2;
    },
  };
}

/* =====================================================================
   post processing
   ===================================================================== */
const rt = new THREE.WebGLRenderTarget(window.innerWidth * DPR, window.innerHeight * DPR, { type: THREE.HalfFloatType, samples: 4 });
const composer = new EffectComposer(renderer, rt);
composer.setSize(window.innerWidth, window.innerHeight);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.32, 0.55, 0.92);
composer.addPass(bloom);
composer.addPass(new OutputPass());
const gradePass = new ShaderPass({
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uVig: { value: 1.15 } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse; uniform float uTime; uniform float uVig; varying vec2 vUv;
    void main(){
      vec4 c = texture2D(tDiffuse, vUv);
      c.rgb = mix(c.rgb, c.rgb * vec3(1.045, 1.0, 0.93), 0.7);
      float l = dot(c.rgb, vec3(0.299, 0.587, 0.114));
      c.rgb = mix(vec3(l), c.rgb, 1.06);
      vec2 d = vUv - 0.5;
      c.rgb *= smoothstep(0.0, 1.0, 1.0 - dot(d, d) * uVig);
      float n = fract(sin(dot(vUv * vec2(1733.0, 977.0) + fract(uTime) * 91.7, vec2(12.9898, 78.233))) * 43758.5453);
      c.rgb += (n - 0.5) * 0.022;
      gl_FragColor = c;
    }`,
});
composer.addPass(gradePass);

/* =====================================================================
   controls & interaction
   ===================================================================== */
const controls = new OrbitControls(camera, canvas);
controls.target.set(-5, 18, -8);
// portrait screens: widen the lens and step back so the whole castle fits
if (camera.aspect < 1) {
  camera.fov = 55; camera.updateProjectionMatrix();
  camera.position.sub(controls.target).multiplyScalar(clamp(1.0 / camera.aspect, 1, 2.1)).add(controls.target);
}
const ORBIT_FOV = camera.fov;
const ORBIT_HOME = { pos: camera.position.clone(), target: controls.target.clone() };
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 22;
controls.maxDistance = 340;
controls.maxPolarAngle = Math.PI * 0.53;
controls.autoRotateSpeed = 0.22;
controls.rotateSpeed = 0.6;
controls.zoomSpeed = 0.8;

let lastInteract = performance.now() - 5000;
controls.addEventListener('start', () => { lastInteract = performance.now(); controls.autoRotate = false; });

const ui = {
  time: document.getElementById('time'), wind: document.getElementById('wind'), windLabel: document.getElementById('windLabel'),
  cam: document.getElementById('btnCam'), flow: document.getElementById('btnFlow'), leaves: document.getElementById('btnLeaves'),
  sound: document.getElementById('btnSound'), shot: document.getElementById('btnShot'), walk: document.getElementById('btnWalk'),
};
ui.walk.addEventListener('click', () => (walk.on ? exitWalk() : enterWalk()));
const opts = { autoCam: true, flow: false, leaves: true, sound: false };
let gust = 0;
let timeTween = null;

ui.time.addEventListener('input', () => { timeTween = null; applyTime(parseFloat(ui.time.value)); });
const windNames = ['凪', 'そよ風', '秋風', '木枯らし', '疾風'];
ui.wind.addEventListener('input', () => { ui.windLabel.textContent = windNames[Math.min(4, Math.floor(parseFloat(ui.wind.value) * 4.999))]; });
document.querySelectorAll('#timeChips button').forEach((btn) => btn.addEventListener('click', () => {
  const target = (parseFloat(btn.dataset.hour) - 5.5) / 17.5;
  timeTween = { from: state.t, to: target, k: 0 };
}));
document.querySelectorAll('#seasons button').forEach((btn) => btn.addEventListener('click', () => { if (btn.dataset.season !== SS.key) setSeason(btn.dataset.season); }));
document.querySelectorAll('#weathers button').forEach((btn) => btn.addEventListener('click', () => setWeather(btn.dataset.weather)));
const toggle = (btn, key, fn) => btn.addEventListener('click', () => { opts[key] = !opts[key]; btn.classList.toggle('on', opts[key]); fn?.(opts[key]); });
toggle(ui.cam, 'autoCam', (on) => { if (on) lastInteract = performance.now() - 7000; else controls.autoRotate = false; });
toggle(ui.flow, 'flow');
toggle(ui.leaves, 'leaves', (on) => { fallingLeaves.visible = on; });
toggle(ui.sound, 'sound', (on) => { on ? audio.start() : audio.stop(); });
ui.shot.addEventListener('click', () => {
  composer.render();
  const a = document.createElement('a');
  a.href = renderer.domElement.toDataURL('image/png');
  a.download = `castle-${SS.key}-${W.key}-${Date.now()}.png`;
  a.click();
});
window.addEventListener('keydown', (e) => {
  if (e.key === 'h' || e.key === 'H') document.getElementById('ui').classList.toggle('hidden');
});

// click (not drag) -> gust of wind that whirls leaves up
let downAt = null;
const raycaster = new THREE.Raycaster();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
canvas.addEventListener('pointerdown', (e) => { downAt = walk.on ? null : [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (walk.on || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  gust = 1;
  raycaster.setFromCamera(new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1), camera);
  const hit = new THREE.Vector3();
  if (raycaster.ray.intersectPlane(groundPlane, hit) && groundAt(hit.x, hit.z) > -1 && hit.distanceTo(camera.position) < 260) {
    for (let k = 0; k < 90; k++) spawnLeaf(Math.floor(lr() * LEAF_N), hit);
  }
  audio.gust();
});

/* =====================================================================
   ambient sound (WebAudio synthesis — no files)
   ===================================================================== */
const audio = (() => {
  let ctx, master, windGain, windFilter, rustleGain, rainGain, brownBuf, whiteBuf, running = false, birdTimer = 0;
  function noiseBuffer(brown) {
    const len = ctx.sampleRate * 4, buf = ctx.createBuffer(1, len, ctx.sampleRate), d = buf.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) { const w = Math.random() * 2 - 1; if (brown) { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; } else d[i] = w; }
    return buf;
  }
  function init() {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);
    const w = ctx.createBufferSource(); w.buffer = noiseBuffer(true); w.loop = true;
    windFilter = ctx.createBiquadFilter(); windFilter.type = 'lowpass'; windFilter.frequency.value = 500;
    windGain = ctx.createGain(); windGain.gain.value = 0.3;
    w.connect(windFilter).connect(windGain).connect(master); w.start();
    const rs = ctx.createBufferSource(); rs.buffer = noiseBuffer(false); rs.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 3200; bp.Q.value = 0.6;
    rustleGain = ctx.createGain(); rustleGain.gain.value = 0.02;
    rs.connect(bp).connect(rustleGain).connect(master); rs.start();
    const rn = ctx.createBufferSource(); rn.buffer = noiseBuffer(false); rn.loop = true;
    const rf = ctx.createBiquadFilter(); rf.type = 'bandpass'; rf.frequency.value = 1400; rf.Q.value = 0.35;
    const rh = ctx.createBiquadFilter(); rh.type = 'highshelf'; rh.frequency.value = 4000; rh.gain.value = -8;
    rainGain = ctx.createGain(); rainGain.gain.value = 0;
    rn.connect(rf).connect(rh).connect(rainGain).connect(master); rn.start();
    brownBuf = noiseBuffer(true); whiteBuf = noiseBuffer(false);
  }
  function chirp(night) {
    const t0 = ctx.currentTime;
    const notes = night ? 3 : 2 + Math.floor(Math.random() * 4);
    const f0 = night ? 4200 : 2400 + Math.random() * 2000;
    for (let i = 0; i < notes; i++) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      const st = t0 + i * (night ? 0.07 : 0.11 + Math.random() * 0.05);
      o.type = 'sine';
      o.frequency.setValueAtTime(f0 * (night ? 1 : 1 + Math.random() * 0.2), st);
      if (!night) o.frequency.exponentialRampToValueAtTime(f0 * (0.7 + Math.random() * 0.6), st + 0.09);
      g.gain.setValueAtTime(0, st); g.gain.linearRampToValueAtTime(night ? 0.012 : 0.03, st + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, st + (night ? 0.05 : 0.12));
      o.connect(g).connect(master); o.start(st); o.stop(st + 0.2);
    }
  }
  return {
    start() { if (!ctx) init(); ctx.resume(); running = true; master.gain.setTargetAtTime(0.7, ctx.currentTime, 0.8); },
    stop() { if (!ctx) return; running = false; master.gain.setTargetAtTime(0, ctx.currentTime, 0.4); },
    gust() { if (running) windGain.gain.setTargetAtTime(0.9, ctx.currentTime, 0.05); },
    thunder(delay) {
      if (!running) return;
      const t0 = ctx.currentTime + delay;
      const src = ctx.createBufferSource(); src.buffer = brownBuf;
      const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 220;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(1.4, t0 + 0.08);
      g.gain.setTargetAtTime(0.5, t0 + 0.3, 0.3); g.gain.setTargetAtTime(0, t0 + 1.2, 0.9);
      src.connect(lp).connect(g).connect(master); src.start(t0, Math.random() * 2); src.stop(t0 + 4.5);
    },
    step(kind) {
      if (!running) return;
      const t0 = ctx.currentTime;
      const src = ctx.createBufferSource(); src.buffer = kind === 'stone' ? whiteBuf : brownBuf;
      const f = ctx.createBiquadFilter(); f.type = kind === 'stone' ? 'bandpass' : 'lowpass';
      f.frequency.value = kind === 'stone' ? 900 + Math.random() * 300 : kind === 'snow' ? 1400 : 700; f.Q.value = 0.8;
      const g = ctx.createGain();
      const peak = kind === 'stone' ? 0.16 : kind === 'snow' ? 0.5 : 0.35, len = kind === 'snow' ? 0.18 : 0.11;
      g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(peak, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t0 + len);
      src.connect(f).connect(g).connect(master); src.start(t0, Math.random() * 3); src.stop(t0 + len + 0.05);
    },
    update(dt, t, wind, g, night, w, inside) {
      if (!running) return;
      const now = ctx.currentTime;
      const muffle = inside ? 0.3 : 1;
      windFilter.frequency.setTargetAtTime(inside ? 180 : 500, now, 0.5);
      rainGain.gain.setTargetAtTime(Math.min(1.3, w.rain) * 0.32 * (inside ? 0.45 : 1), now, 0.6);
      if (inside && Math.random() < dt * 5) {
        const t0 = now + Math.random() * 0.05, src = ctx.createBufferSource(); src.buffer = whiteBuf;
        const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 1500 + Math.random() * 2500;
        const cg = ctx.createGain(); cg.gain.setValueAtTime(0.05 + Math.random() * 0.12, t0); cg.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.03 + Math.random() * 0.04);
        src.connect(hp).connect(cg).connect(master); src.start(t0, Math.random() * 3); src.stop(t0 + 0.1);
      }
      windGain.gain.setTargetAtTime((0.12 + wind * 0.35 + g * 0.5 + Math.sin(t * 0.37) * 0.05 * (0.5 + wind)) * muffle, now, 0.4);
      if (!inside) windFilter.frequency.setTargetAtTime(260 + wind * 500 + g * 700 + Math.sin(t * 0.23) * 80, now, 0.4);
      rustleGain.gain.setTargetAtTime((0.008 + wind * 0.035 + g * 0.05) * (inside ? 0 : 1), now, 0.3);
      birdTimer -= dt;
      if (birdTimer <= 0) {
        birdTimer = night > 0.5 ? 0.9 + Math.random() * 1.5 : 2 + Math.random() * 6;
        const quiet = inside || w.rain > 0.3 || w.snow > 0.3 || (SS.key === 'winter' && night > 0.5);
        if (!quiet) chirp(night > 0.5);
      }
    },
  };
})();

/* =====================================================================
   first-person walking
   ===================================================================== */
const isTouch = window.matchMedia('(pointer: coarse)').matches;
const walk = { on: false, place: 'outside', x: 0, z: 0, yaw: 0, pitch: 0, keys: {}, locked: false, bob: 0, busy: false, joy: { x: 0, y: 0, id: null }, look: { id: null, x: 0, y: 0 }, step: 0 };
const hud = {
  fade: document.getElementById('fade'), walkHud: document.getElementById('walkHud'), prompt: document.getElementById('prompt'),
  toast: document.getElementById('placeToast'), joy: document.getElementById('joy'), knob: document.querySelector('#joy i'),
  tip: document.getElementById('walkTip'),
};
const DOOR = { x: -6, z: 8.3 };
function segDist(px, pz, [ax, az, bx, bz]) {
  const dx = bx - ax, dz = bz - az, t = clamp(((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz || 1), 0, 1);
  return Math.hypot(px - (ax + dx * t), pz - (az + dz * t));
}
const OUT_RECTS = [[-28.6, 14.6, -10.6, 4.6], [-13.6, 1.6, 2, 8.0], [-17.8, 1.8, -29.8, -14.2], [-30.6, 10.6, -42.6, -29.4], [-45.8, -40.8, -35, -9], [18.8, 23.2, -34, -8], [-10.2, -1.8, 7.9, 8.6]];
const OUT_CIRCLES = [[-34, -3, 11.3], [18, -1, 7.6], [23.6, -6.5, 4.3], [-22.5, -32.5, 5.1], [-38.5, -40, 8.1], [16, -38, 7.1], [bayX0(-7.25), 7.3, 1.4], [bayX0(7.25), 7.3, 1.4]];
function bayX0(o) { return -6 + o; }
function canStandOut(x, z) {
  let ok = false;
  const ex = x - CX, ez = z - CZ;
  if (Math.hypot(ex, ez) < WALK.outlineR(Math.atan2(ez, ex)) - 1.2) ok = true;
  if (segDist(x, z, [VS.x, VS.z, VE.x, VE.z]) < 3.0) ok = true;
  if (Math.hypot(x - OV.x, z - OV.z) < 9.2) ok = true;
  if (!ok) return false;
  for (const [x0, x1, z0, z1] of OUT_RECTS) if (x > x0 && x < x1 && z > z0 && z < z1) return false;
  for (const [cx, cz, r] of OUT_CIRCLES) if (Math.hypot(x - cx, z - cz) < r) return false;
  for (const [cx, cz, r] of WALK.circles) if (Math.hypot(x - cx, z - cz) < r) return false;
  for (const [cx, cz, r] of WALK.trunks) if (Math.hypot(x - cx, z - cz) < r) return false;
  for (const s of WALK.rails) if (segDist(x, z, s) < 0.38) return false;
  return true;
}
function groundOut(x, z) {
  if (Math.abs(x - DOOR.x) < 4 && z > 7.5 && z < 10.5) return z < 8.5 ? 0.8 : z < 9.5 ? 0.4 : 0.04;
  return 0.04;
}
function fadeTo(fn) {
  if (walk.busy) return Promise.resolve();
  walk.busy = true;
  hud.fade.classList.add('on');
  return new Promise((res) => setTimeout(() => {
    fn();
    setTimeout(() => { hud.fade.classList.remove('on'); walk.busy = false; res(); }, 120);
  }, 520));
}
let toastTimer = 0;
function showToast(jp, en) {
  hud.toast.innerHTML = `<b>${jp}</b><span>${en}</span>`;
  hud.toast.classList.add('on');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => hud.toast.classList.remove('on'), 3200);
}
function setPlace(place) {
  walk.place = place;
  if (place === 'inside') {
    if (!interior) { interior = buildInterior(); interior.setDecor(SS.key); }
    renderPass.scene = interior.scene;
    walk.x = interior.spawn.x; walk.z = interior.spawn.z; walk.yaw = interior.spawn.yaw; walk.pitch = 0.05;
    showToast('大広間', 'The Great Hall');
  } else {
    renderPass.scene = scene;
  }
}
function enterWalk() {
  return fadeTo(() => {
    walk.on = true; controls.enabled = false; controls.autoRotate = false;
    setPlace('outside');
    walk.x = -6; walk.z = 23; walk.yaw = 0; walk.pitch = 0.1;
    camera.fov = isTouch && camera.aspect < 1 ? 75 : 65; camera.near = 0.1; camera.updateProjectionMatrix();
    document.body.classList.add('walking'); ui.walk.classList.add('on');
    if (document.activeElement) document.activeElement.blur();
    showToast('古城の前庭', 'The Forecourt');
  });
}
function exitWalk() {
  if (document.pointerLockElement) document.exitPointerLock();
  return fadeTo(() => {
    walk.on = false; setPlace('outside');
    controls.enabled = true;
    camera.fov = ORBIT_FOV; camera.near = 0.5; camera.updateProjectionMatrix();
    camera.position.copy(ORBIT_HOME.pos); controls.target.copy(ORBIT_HOME.target); controls.update();
    lastInteract = performance.now();
    document.body.classList.remove('walking'); ui.walk.classList.remove('on');
  });
}
function updateWalk(dt) {
  const k = walk.keys;
  let f = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0) + walk.joy.y;
  let r = (k.KeyD || k.ArrowRight ? 1 : 0) - (k.KeyA || k.ArrowLeft ? 1 : 0) + walk.joy.x;
  const len = Math.hypot(f, r); if (len > 1) { f /= len; r /= len; }
  const speed = (k.ShiftLeft || k.ShiftRight ? 6.5 : 3.2) * (walk.busy ? 0 : 1);
  const sy = Math.sin(walk.yaw), cy = Math.cos(walk.yaw);
  const mx = (-sy * f + cy * r) * speed * dt, mz = (-cy * f - sy * r) * speed * dt;
  const inside = walk.place === 'inside';
  const can = inside ? (x, z) => interior.canStand(x, z) : canStandOut;
  const ox = walk.x, oz = walk.z;
  if (can(walk.x + mx, walk.z)) walk.x += mx;
  if (can(walk.x, walk.z + mz)) walk.z += mz;
  const moved = Math.hypot(walk.x - ox, walk.z - oz);
  // doors
  if (!walk.busy) {
    if (!inside && Math.abs((walk.x + mx * 4) - DOOR.x) < 2.1 && walk.z + mz * 4 < DOOR.z + 0.3 && f > 0.2) fadeTo(() => setPlace('inside'));
    else if (inside && interior.isExit(walk.x, walk.z + mz * 4) && f > 0.2) fadeTo(() => { setPlace('outside'); walk.x = DOOR.x; walk.z = 12; walk.yaw = Math.PI; walk.pitch = 0.05; showToast('古城の前庭', 'The Forecourt'); });
  }
  // prompts
  let msg = '';
  if (!inside && Math.hypot(walk.x - DOOR.x, walk.z - 10) < 6) msg = '扉へ進むと城の中へ';
  if (inside && walk.z > -5 && Math.abs(walk.x) < 3) msg = '扉へ進むと外へ';
  if (hud.prompt.textContent !== msg) { hud.prompt.textContent = msg; hud.prompt.classList.toggle('on', !!msg); }
  // camera
  walk.bob += moved * 2.2;
  const g = inside ? interior.ground(walk.x, walk.z) : groundOut(walk.x, walk.z);
  walk.gy = walk.gy === undefined ? g : lerp(walk.gy, g, Math.min(1, dt * 12));
  camera.position.set(walk.x, walk.gy + 1.65 + Math.sin(walk.bob * 2) * 0.035, walk.z);
  camera.rotation.set(walk.pitch, walk.yaw, 0, 'YXZ');
  // footsteps
  walk.step += moved;
  if (walk.step > (speed > 4 ? 0.95 : 0.72)) {
    walk.step = 0;
    const onStone = inside || Math.abs(walk.x - DOOR.x) < 16 && walk.z > 7 && walk.z < 26 || segDist(walk.x, walk.z, [VS.x, VS.z, VE.x, VE.z]) < 3.2 || Math.hypot(walk.x - OV.x, walk.z - OV.z) < 10;
    audio.step(ENV.uSnow.value > 0.4 && !inside ? 'snow' : onStone ? 'stone' : 'grass');
  }
}
// input
window.addEventListener('keydown', (e) => {
  if (!walk.on) return;
  walk.keys[e.code] = true;
  if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
});
window.addEventListener('keyup', (e) => { walk.keys[e.code] = false; });
window.addEventListener('blur', () => { walk.keys = {}; });
document.addEventListener('pointerlockchange', () => {
  walk.locked = document.pointerLockElement === canvas;
  hud.tip.classList.toggle('dim', walk.locked);
});
const lookBy = (dx, dy, k) => { walk.yaw -= dx * k; walk.pitch = clamp(walk.pitch - dy * k, -1.35, 1.35); };
document.addEventListener('mousemove', (e) => { if (walk.on && walk.locked) lookBy(e.movementX, e.movementY, 0.0022); });
canvas.addEventListener('pointerdown', (e) => {
  if (!walk.on) return;
  if (!isTouch && e.pointerType === 'mouse' && !walk.locked && canvas.requestPointerLock) {
    try { const p = canvas.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch (_) { /* fall back to drag-look */ }
  }
  if (walk.look.id === null) { walk.look = { id: e.pointerId, x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); }
});
canvas.addEventListener('pointermove', (e) => {
  if (!walk.on || walk.locked || e.pointerId !== walk.look.id) return;
  lookBy(e.clientX - walk.look.x, e.clientY - walk.look.y, isTouch ? 0.005 : 0.004);
  walk.look.x = e.clientX; walk.look.y = e.clientY;
});
const endLook = (e) => { if (e.pointerId === walk.look.id) walk.look.id = null; };
canvas.addEventListener('pointerup', endLook); canvas.addEventListener('pointercancel', endLook);
// virtual joystick (touch)
const joyMove = (e) => {
  const r = hud.joy.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  let dx = e.clientX - cx, dy = e.clientY - cy; const m = Math.hypot(dx, dy), max = r.width / 2 - 10;
  if (m > max) { dx *= max / m; dy *= max / m; }
  hud.knob.style.transform = `translate(${dx}px, ${dy}px)`;
  walk.joy.x = dx / max; walk.joy.y = -dy / max;
};
hud.joy.addEventListener('pointerdown', (e) => { walk.joy.id = e.pointerId; hud.joy.setPointerCapture(e.pointerId); joyMove(e); e.stopPropagation(); });
hud.joy.addEventListener('pointermove', (e) => { if (e.pointerId === walk.joy.id) joyMove(e); });
const joyEnd = (e) => { if (e.pointerId !== walk.joy.id) return; walk.joy = { x: 0, y: 0, id: null }; hud.knob.style.transform = ''; };
hud.joy.addEventListener('pointerup', joyEnd); hud.joy.addEventListener('pointercancel', joyEnd);
document.getElementById('btnExitWalk').addEventListener('click', () => exitWalk());
if (isTouch) hud.tip.textContent = '左のスティックで移動／画面をドラッグして見回す';

/* =====================================================================
   loop
   ===================================================================== */
const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;
  U.uTime.value = t;
  gradePass.uniforms.uTime.value = t;

  // time of day
  if (timeTween) {
    timeTween.k = Math.min(1, timeTween.k + dt / 2.6);
    const e = timeTween.k < 0.5 ? 4 * timeTween.k ** 3 : 1 - (-2 * timeTween.k + 2) ** 3 / 2;
    const v = lerp(timeTween.from, timeTween.to, e);
    ui.time.value = v; applyTime(v);
    if (timeTween.k >= 1) timeTween = null;
  } else if (opts.flow) {
    let v = state.t + dt / 240; if (v > 1) v -= 1;
    ui.time.value = v; applyTime(v);
  }
  stepSeason(dt);
  stepWeather(dt);
  applyTime(state.t);

  // wind
  gust = Math.max(0, gust - dt * 0.45);
  const windBase = parseFloat(ui.wind.value) + W.storm * 0.55 + Math.min(1, W.rain) * 0.1;
  const breath = 0.5 + 0.5 * Math.sin(t * 0.21) * Math.sin(t * 0.13 + 1.3);
  const windNow = windBase * (0.7 + 0.6 * breath) + gust * 1.3;
  U.uWind.value = lerp(U.uWind.value, clamp(windNow, 0, 1.8), dt * 2);
  const ws = 0.4 + windNow * 5.5;
  if (opts.leaves && fallingLeaves) updateLeaves(dt, t, ws);
  updatePrecip(_v.set(windDir.x * ws, 0, windDir.z * ws));

  // flames & lights
  for (const f of flames) {
    f.children.forEach((sp) => {
      const [w, h, ph] = sp.userData.base;
      const fl = 0.85 + 0.15 * Math.sin(t * 17 + ph) + 0.1 * Math.sin(t * 29 + ph * 2);
      sp.scale.set(w * fl, h * (0.9 + 0.2 * Math.sin(t * 11 + ph)), 1);
      sp.position.set(Math.sin(t * 7 + ph) * 0.05, h * 0.3, 0);
    });
  }
  for (const pl of pointLights) {
    if (pl.userData.kind === 'fire') pl.intensity = (6 + state.nightK * 22) * (0.85 + 0.15 * Math.sin(t * 19 + pl.position.x) + 0.08 * Math.sin(t * 31));
    else pl.intensity = state.nightK * 35 + (1 - state.dayK) * 6;
  }

  // birds (only in daylight)
  for (const bd of birds) {
    const u = bd.userData, a = t * u.sp + u.ph;
    bd.visible = state.dayK > 0.15 && W.rain < 0.3 && W.snow < 0.4 && W.fog < 0.6;
    bd.position.set(u.c.x + Math.cos(a) * u.R, u.c.y + Math.sin(a * 2.3) * 4, u.c.z + Math.sin(a) * u.R);
    bd.rotation.y = Math.atan2(-Math.sin(a) * Math.sign(u.sp), Math.cos(a) * Math.sign(u.sp));
    const flap = Math.sin(t * u.fl + u.ph) * 0.65 * (Math.sin(t * 0.7 + u.ph) > -0.3 ? 1 : 0.1);
    u.wr.rotation.z = flap; u.wl.rotation.z = -flap;
  }
  for (const m of mists) m.position.x += m.userData.drift * dt;

  // camera
  if (walk.on) {
    updateWalk(dt);
  } else {
    if (opts.autoCam && !timeTween && performance.now() - lastInteract > 9000) controls.autoRotate = true;
    controls.update();
    controls.target.x = clamp(controls.target.x, -80, 90);
    controls.target.z = clamp(controls.target.z, -90, 90);
    controls.target.y = clamp(controls.target.y, -20, 60);
    if (camera.position.y < 2) camera.position.y = 2;
  }
  scene.userData.sky.position.copy(camera.position);
  const inside = walk.on && walk.place === 'inside';
  if (inside) interior.update(dt, t);

  audio.update(dt, t, windBase, gust, state.nightK, W, inside);
  composer.render();
  requestAnimationFrame(frame);
}

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
});

/* =====================================================================
   boot
   ===================================================================== */
(async () => {
  await build();
  setSeason('autumn', true);
  setWeather('clear');
  applyTime(parseFloat(ui.time.value), true);
  ui.wind.dispatchEvent(new Event('input'));
  renderer.compile(scene, camera);
  // dev hook: document.dispatchEvent(new CustomEvent('castle-debug', { detail: '{"cam":[x,y,z],"target":[x,y,z],"t":0.5,"season":"winter","walk":"in"}' }))
  document.addEventListener('castle-debug', (e) => {
    const d = JSON.parse(e.detail);
    opts.autoCam = false; controls.autoRotate = false; ui.cam.classList.remove('on');
    if (d.cam) camera.position.set(...d.cam);
    if (d.target) controls.target.set(...d.target);
    if (d.t !== undefined) { ui.time.value = d.t; applyTime(d.t); }
    if (d.season) setSeason(d.season);
    if (d.walk) { walk.on = true; controls.enabled = false; camera.fov = 65; camera.near = 0.1; camera.updateProjectionMatrix(); document.body.classList.add('walking'); setPlace(d.walk === 'in' ? 'inside' : 'outside'); }
    if (d.pos) { walk.x = d.pos[0]; walk.z = d.pos[1]; }
    if (d.yaw !== undefined) walk.yaw = d.yaw;
    if (d.pitch !== undefined) walk.pitch = d.pitch;
    if (d.weather) setWeather(d.weather);
    controls.update();
  });
  clock.start();
  frame();
  setTimeout(() => document.getElementById('loader').classList.add('done'), 250);
  setTimeout(() => document.getElementById('hint').classList.add('fade'), 14000);
})();
