import * as THREE from 'three/webgpu';
import {
  Fn, uniform, vec2, vec3, vec4, float, int,
  abs, max, min, mix, clamp as nodeClamp, smoothstep as nodeSmoothstep, dot, normalize, length, pow, exp, fract, floor, cross, oneMinus,
  If, Loop, Discard, positionLocal, cameraPosition, uv,
} from 'three/tsl';

const canvas = document.querySelector('#scene');
const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
await renderer.init();
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
camera.position.set(0, 0, 5);

// Orbit-ish controls by hand so the sandbox stays dependency-free.
let yaw = 0, pitch = 0, dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => { dragging = false; });
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.01;
  pitch += (e.clientY - lastY) * 0.01;
  pitch = Math.max(-1.4, Math.min(1.4, pitch));
  lastX = e.clientX; lastY = e.clientY;
});

// ---- Noise (procedural, in-shader) -------------------------------------
// Classic iq-style hash + value noise + fbm. We keep it procedural here so
// the sandbox has zero asset dependencies.
const hash3 = Fn(([p]) => {
  const q = fract(p.mul(vec3(0.1031, 0.1130, 0.0973)));
  const r = dot(q, q.yzx.add(33.33));
  return fract(vec3(q.x.add(q.y).mul(q.z).add(r), q.y.add(q.z).mul(q.x).add(r), q.z.add(q.x).mul(q.y).add(r)));
});

const noise3 = Fn(([p]) => {
  const i = floor(p);
  const f = fract(p);
  const u = f.mul(f).mul(vec3(3.0).sub(f.mul(2.0)));
  const n000 = hash3(i.add(vec3(0.0, 0.0, 0.0))).x;
  const n100 = hash3(i.add(vec3(1.0, 0.0, 0.0))).x;
  const n010 = hash3(i.add(vec3(0.0, 1.0, 0.0))).x;
  const n110 = hash3(i.add(vec3(1.0, 1.0, 0.0))).x;
  const n001 = hash3(i.add(vec3(0.0, 0.0, 1.0))).x;
  const n101 = hash3(i.add(vec3(1.0, 0.0, 1.0))).x;
  const n011 = hash3(i.add(vec3(0.0, 1.0, 1.0))).x;
  const n111 = hash3(i.add(vec3(1.0, 1.0, 1.0))).x;
  const x00 = mix(n000, n100, u.x);
  const x10 = mix(n010, n110, u.x);
  const x01 = mix(n001, n101, u.x);
  const x11 = mix(n011, n111, u.x);
  const y0 = mix(x00, x10, u.y);
  const y1 = mix(x01, x11, u.y);
  return mix(y0, y1, u.z).mul(2.0).sub(1.0);
});

const fbm = Fn(([p]) => {
  let f = float(0.0).toVar();
  let scale = float(0.5).toVar();
  let q = p.toVar();
  let factor = float(2.02).toVar();
  Loop({ start: int(0), end: int(6), type: 'int', name: 'i' }, () => {
    f.addAssign(scale.mul(noise3(q)));
    q.mulAssign(factor);
    factor.addAssign(0.21);
    scale.mulAssign(0.5);
  });
  return f;
});

const sdSphere = Fn(([p, r]) => length(p).sub(r));

// Scene density: inside a sphere, add FBM noise so the cloud billows.
const sceneDensity = Fn(([p]) => {
  const d = sdSphere(p, float(1.6));
  const f = fbm(p);
  return d.negate().add(f);
});

const HenyeyGreenstein = Fn(([g, mu]) => {
  const gg = g.mul(g);
  return float(1.0 / (4.0 * Math.PI)).mul(float(1.0).sub(gg)).div(
    pow(float(1.0).add(gg).sub(g.mul(mu).mul(2.0)), float(1.5)),
  );
});

// ---- Constants (article defaults) -------------------------------------
const MAX_STEPS = 64;
const MARCH_SIZE = float(0.08);
const ABSORPTION = float(0.9);
const MAX_STEPS_LIGHT = 6;
const uSunPos = uniform(new THREE.Vector3(1.0, 0.5, 0.0));
const uTime = uniform(0);

const lightmarch = Fn(([pos]) => {
  const lightDir = normalize(uSunPos).toVar();
  let totalDensity = float(0.0).toVar();
  const marchSize = float(0.03);
  const p = pos.toVar();
  Loop({ start: int(0), end: int(MAX_STEPS_LIGHT), type: 'int', name: 'i' }, () => {
    p.addAssign(lightDir.mul(marchSize));
    totalDensity.addAssign(nodeClamp(sceneDensity(p), 0.0, 1.0));
  });
  return exp(totalDensity.negate().mul(ABSORPTION));
});

const raymarch = Fn(([ro, rd, offset]) => {
  let depth = float(0.0).toVar();
  depth.addAssign(MARCH_SIZE.mul(offset));
  const sunDir = normalize(uSunPos);
  const phase = HenyeyGreenstein(float(0.5), dot(rd, sunDir));

  let totalTransmittance = float(1.0).toVar();
  let lightEnergy = vec3(0.0).toVar();

  Loop({ start: int(0), end: int(MAX_STEPS), type: 'int', name: 'i' }, () => {
    const p = ro.add(rd.mul(depth)).toVar();
    const density = sceneDensity(p);
    If(density.greaterThan(0.0), () => {
      const lightTransmittance = lightmarch(p);
      const luminance = float(0.05).add(density.mul(phase));
      const lin = vec3(0.6, 0.6, 0.75).mul(1.1).add(vec3(1.0, 0.6, 0.3).mul(lightTransmittance).mul(2.0));
      const color = vec3(luminance).mul(lin);
      color.mulAssign(totalTransmittance);
      lightEnergy.addAssign(color);
      totalTransmittance.mulAssign(lightTransmittance);
    });
    depth.addAssign(MARCH_SIZE);
  });

  return vec4(lightEnergy, totalTransmittance);
});

// ---- Fullscreen quad material ----------------------------------------
const material = new THREE.MeshBasicNodeMaterial();
material.transparent = true;
material.depthWrite = false;
material.toneMapped = true;

material.fragmentNode = Fn(() => {
  // uv() is 0..1 across the fullscreen quad; reconstruct a simple camera ray
  // exactly like the article/Shadertoy setup.
  const ndc = uv().mul(2.0).sub(1.0).toVar();
  const viewDir = normalize(vec3(ndc.x, ndc.y, -1.5));
  const ro = vec3(0.0, 0.0, 5.0);

  // Per-pixel dither so a low-ish step count hides banding.
  const dither = fract(dot(uv().mul(255.0), vec2(127.1, 311.7)));
  const res = raymarch(ro, viewDir, dither);

  // Background sky gradient where no cloud.
  const sky = mix(vec3(0.2, 0.4, 0.7), vec3(0.6, 0.7, 0.9), nodeSmoothstep(0.0, 0.3, viewDir.y));
  const col = mix(sky, res.rgb, oneMinus(res.a));
  return vec4(col, 1.0);
})();

const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
quad.frustumCulled = false;
// Place it in front of the camera each frame via the scene below instead.
scene.add(quad);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', resize);
resize();

let t = 0;
function animate() {
  requestAnimationFrame(animate);
  t += 0.016;
  uTime.value = t;
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);
}
animate();