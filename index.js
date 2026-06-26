/**
 * metaverse-sky - Three.js WebGPU/TSL sky, sun helpers, and fast 2.5D sky-volume clouds.
 *
 * Peer dependency: the host app must resolve `three` and `three/addons/`.
 */

import * as THREE from 'three/webgpu';
import {
  Fn, uniform, texture, attribute,
  vec2, vec3, vec4, float, int,
  abs, max, min, mix, clamp as nodeClamp, smoothstep as nodeSmoothstep, dot, normalize, length, pow, exp, sin, sqrt, fract, floor, oneMinus,
  If, Loop, Discard, positionLocal, positionWorld, cameraPosition, uv,
} from 'three/tsl';

export const DEFAULT_SKY_SCALE = 450;
export const DEFAULT_TURBIDITY = 8;
export const DEFAULT_RAYLEIGH = 2.2;
export const DEFAULT_MIE_COEFFICIENT = 0.005;
export const DEFAULT_MIE_DIRECTIONAL_G = 0.8;
export const DEFAULT_EXPOSURE = 0.6;
export const DEFAULT_SUN_POSITION = [0.45, 0.86, 0.24];
export const DEFAULT_ENV_INTENSITY_MIN = 1.0;
export const DEFAULT_ENV_INTENSITY_MAX = 2.0;
export const ELEVATION_MIN = -20;
export const ELEVATION_MAX = 90;

const FADE_MS = 280;


const CLOUD_DEFAULTS = {
  enabled: true,
  renderMode: 'volume',
  altitude: 80,
  opacity: 0.95,
  tile: 6,
  drawDistance: 420,
  cloudColor: new THREE.Color(0xf2f6fc),
  autoTint: true,
  coverage: 0.5,
  noiseScale: 0.012,
  detailStrength: 0.45,
  holes: 0.5,
  sharpness: 0.35,
  wispiness: 0.45,
  darkness: 0,
};

export const DEFAULT_CLOUD_SETTINGS = Object.freeze({
  enabled: CLOUD_DEFAULTS.enabled,
  renderMode: CLOUD_DEFAULTS.renderMode,
  altitude: CLOUD_DEFAULTS.altitude,
  opacity: CLOUD_DEFAULTS.opacity,
  tile: CLOUD_DEFAULTS.tile,
  drawDistance: CLOUD_DEFAULTS.drawDistance,
  cloudColor: CLOUD_DEFAULTS.cloudColor.getHex(),
  autoTint: CLOUD_DEFAULTS.autoTint,
  coverage: CLOUD_DEFAULTS.coverage,
  noiseScale: CLOUD_DEFAULTS.noiseScale,
  detailStrength: CLOUD_DEFAULTS.detailStrength,
  holes: CLOUD_DEFAULTS.holes,
  sharpness: CLOUD_DEFAULTS.sharpness,
  wispiness: CLOUD_DEFAULTS.wispiness,
  darkness: CLOUD_DEFAULTS.darkness,
});

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function floorMod(n, m) {
  return ((n % m) + m) % m;
}

function vectorFrom(value, fallback = DEFAULT_SUN_POSITION) {
  if (value?.isVector3) return value.clone();
  if (Array.isArray(value)) return new THREE.Vector3(value[0], value[1], value[2]);
  if (value && Number.isFinite(value.x) && Number.isFinite(value.y) && Number.isFinite(value.z)) {
    return new THREE.Vector3(value.x, value.y, value.z);
  }
  return new THREE.Vector3(fallback[0], fallback[1], fallback[2]);
}

function colorFrom(value, fallback) {
  if (value?.isColor) return value.clone();
  return new THREE.Color(value ?? fallback);
}

function normalizeCloudParams(options = {}) {
  return {
    ...CLOUD_DEFAULTS,
    ...options,
    cloudColor: colorFrom(options.cloudColor, CLOUD_DEFAULTS.cloudColor),
  };
}

function uniformProxy(node) {
  return {
    get value() { return node.value; },
    set value(next) {
      if (node.value?.copy && next?.isColor) node.value.copy(next);
      else if (node.value?.copy && next?.isVector2) node.value.copy(next);
      else if (node.value?.copy && next?.isVector3) node.value.copy(next);
      else node.value = next;
    },
  };
}

function assignUniform(material, name, node) {
  if (!material.userData.uniforms) material.userData.uniforms = {};
  material.uniforms = material.userData.uniforms;
  material.userData.uniforms[name] = uniformProxy(node);
  return node;
}

function createSolidTexture(r = 255, g = 255, b = 255, a = 255) {
  const tex = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}
// --- 3D Perlin-Worley cloud noise texture (baked once on init) ----------
// A 3D texture is far cheaper to sample than procedural noise per ray march
// step, which makes the volumetric cloud shader performant while still using a
// genuine Perlin-Worley density field.

function hash31(x, y, z) {
  // Deterministic [0,1) hash from 3 integers. Works across wrap boundaries
  // because the inputs are pre-wrapped to a tile period by the callers.
  let h = (x | 0) * 374761393 + (y | 0) * 668265263 + (z | 0) * 1274126177;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

// Modular floor so lattice points wrap at the tile period → seamless tiling.
function tmod(n, m) {
  return ((n % m) + m) % m;
}

// Gradient-noise helper: derive a stable 3D gradient direction from a seed
// hash, then dot it with the distance vector. This is what makes Perlin noise
// smooth and blobby instead of blocky (value noise gives square lattice edges).
const _GRAD = [
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
  [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
  [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1],
];

function gradDot(hash, dx, dy, dz) {
  const g = _GRAD[Math.floor(hash * _GRAD.length) % _GRAD.length];
  return g[0] * dx + g[1] * dy + g[2] * dz;
}

function smooth3(t) {
  return t * t * (3 - 2 * t);
}

// Proper gradient (Perlin) 3D noise — smooth and free of value-noise blockiness.
// `tile` is the lattice period; lattice points wrap modulo it so the result is
// seamlessly tileable (matching at x=0 and x=tile).
function perlin3(x, y, z, tile) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = smooth3(xf), v = smooth3(yf), w = smooth3(zf);
  const wx0 = tmod(xi, tile), wy0 = tmod(yi, tile), wz0 = tmod(zi, tile);
  const wx1 = tmod(xi + 1, tile), wy1 = tmod(yi + 1, tile), wz1 = tmod(zi + 1, tile);
  const g000 = hash31(wx0, wy0, wz0), g100 = hash31(wx1, wy0, wz0);
  const g010 = hash31(wx0, wy1, wz0), g110 = hash31(wx1, wy1, wz0);
  const g001 = hash31(wx0, wy0, wz1), g101 = hash31(wx1, wy0, wz1);
  const g011 = hash31(wx0, wy1, wz1), g111 = hash31(wx1, wy1, wz1);
  const d000 = gradDot(g000, xf, yf, zf), d100 = gradDot(g100, xf - 1, yf, zf);
  const d010 = gradDot(g010, xf, yf - 1, zf), d110 = gradDot(g110, xf - 1, yf - 1, zf);
  const d001 = gradDot(g001, xf, yf, zf - 1), d101 = gradDot(g101, xf - 1, yf, zf - 1);
  const d011 = gradDot(g011, xf, yf - 1, zf - 1), d111 = gradDot(g111, xf - 1, yf - 1, zf - 1);
  const x00 = d000 + (d100 - d000) * u, x10 = d010 + (d110 - d010) * u;
  const x01 = d001 + (d101 - d001) * u, x11 = d011 + (d111 - d011) * u;
  const y0 = x00 + (x10 - x00) * v, y1 = x01 + (x11 - x01) * v;
  // Perlin output is in ~[-1,1]; remap to [0,1].
  return y0 + (y1 - y0) * w + 0.5;
}

function worley3(x, y, z, tile) {
  // Inverted Worley: 1 near cell feature points, 0 between. Tileable: feature
  // points live inside wrapped cells so the field matches at boundaries.
  let minD = 1e9;
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = tmod(xi + dx, tile), cy = tmod(yi + dy, tile), cz = tmod(zi + dz, tile);
        const fx = cx + hash31(cx, cy, cz);
        const fy = cy + hash31(cy, cz, cx);
        const fz = cz + hash31(cz, cx, cy);
        // Use unwrapped coordinates for the distance check so crossing the edge
        // still measures true distance to the (wrapped) feature point.
        const ddx = (xi + dx) + hash31(cx, cy, cz) - x;
        const ddy = (yi + dy) + hash31(cy, cz, cx) - y;
        const ddz = (zi + dz) + hash31(cz, cx, cy) - z;
        const d = ddx * ddx + ddy * ddy + ddz * ddz;
        if (d < minD) minD = d;
      }
    }
  }
  return 1 - clamp(Math.sqrt(minD), 0, 1);
}

// FBM over the tileable noise. `tile` is the lattice period; octaves use integer
// frequency multipliers that divide the period so each octave stays tileable.
function fbmPerlin(x, y, z, octaves, tile) {
  let total = 0, amp = 0.5, freq = 1, max = 0;
  for (let i = 0; i < octaves; i++) {
    total += perlin3(x * freq, y * freq, z * freq, tile / freq) * amp;
    max += amp;
    freq *= 2;
    amp *= 0.5;
  }
  return total / max;
}

function fbmWorley(x, y, z, octaves, tile) {
  let total = 0, amp = 0.5, freq = 1, max = 0;
  for (let i = 0; i < octaves; i++) {
    total += worley3(x * freq, y * freq, z * freq, tile / freq) * amp;
    max += amp;
    freq *= 2;
    amp *= 0.5;
  }
  return total / max;
}

const CLOUD_NOISE_SIZE = 128;

let _cloudNoiseTexture = null;

function getCloudNoiseTexture() {
  if (_cloudNoiseTexture) return _cloudNoiseTexture;
  const size = CLOUD_NOISE_SIZE;
  const data = new Uint8Array(size * size * size);
  for (let z = 0; z < size; z++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        // Sample the tileable noise in lattice space (texel coords), so the
        // 3D texture is seamlessly periodic with period `size`.
        const perlin = fbmPerlin(x, y, z, 3, size);
        const worley = fbmWorley(x, y, z, 2, size);
        // Perlin-Worley: Perlin gives the billowy base, Worley carves cells.
        let n = perlin * 0.6 + worley * 0.4;
        // Bias the field so smooth thresholding yields soft gradients.
        n = clamp(n, 0, 1);
        data[x + y * size + z * size * size] = Math.round(n * 255);
      }
    }
  }
  const tex = new THREE.Data3DTexture(data, size, size, size);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.wrapR = THREE.RepeatWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  _cloudNoiseTexture = tex;
  return tex;
}

// Sample the baked 3D Perlin-Worley texture and turn it into a cloud density
// [0,1] given a world-space sample position. Expects shared uniforms u.
// Sample the baked 3D Perlin-Worley texture and turn it into a cloud density
// [0,1]. Density is shaped by a vertical envelope centered on the cloud
// `altitude` (a smooth band of half-height `u.uThickness`), so the layer is
// visible from any view angle, not just rays crossing a slab.
// Sample baked Perlin-Worley density by a view-space coordinate (the march
// position relative to the camera). Because the noise is camera-anchored the
// cloud pattern is fixed to the sky (it follows the viewer like a sky dome),
// so it never swims and is visible from any camera angle.
// Article-style procedural 3D noise: hash -> value noise -> FBM. This removes
// the interim 3D texture path, which was reading as static/noisy in WebGL/WebGPU.
const cloudHash3 = Fn(([p]) => {
  const q = fract(p.mul(vec3(0.1031, 0.1130, 0.0973)));
  const r = dot(q, q.yzx.add(33.33));
  return fract(vec3(q.x.add(q.y).mul(q.z).add(r), q.y.add(q.z).mul(q.x).add(r), q.z.add(q.x).mul(q.y).add(r)));
});

const cloudNoise3 = Fn(([p]) => {
  const i = floor(p);
  const f = fract(p);
  const u = f.mul(f).mul(vec3(3.0).sub(f.mul(2.0)));
  const n000 = cloudHash3(i.add(vec3(0.0, 0.0, 0.0))).x;
  const n100 = cloudHash3(i.add(vec3(1.0, 0.0, 0.0))).x;
  const n010 = cloudHash3(i.add(vec3(0.0, 1.0, 0.0))).x;
  const n110 = cloudHash3(i.add(vec3(1.0, 1.0, 0.0))).x;
  const n001 = cloudHash3(i.add(vec3(0.0, 0.0, 1.0))).x;
  const n101 = cloudHash3(i.add(vec3(1.0, 0.0, 1.0))).x;
  const n011 = cloudHash3(i.add(vec3(0.0, 1.0, 1.0))).x;
  const n111 = cloudHash3(i.add(vec3(1.0, 1.0, 1.0))).x;
  const x00 = mix(n000, n100, u.x);
  const x10 = mix(n010, n110, u.x);
  const x01 = mix(n001, n101, u.x);
  const x11 = mix(n011, n111, u.x);
  const y0 = mix(x00, x10, u.y);
  const y1 = mix(x01, x11, u.y);
  return mix(y0, y1, u.z).mul(2.0).sub(1.0);
});

const cloudFbm = Fn(([p]) => {
  const f = float(0.0).toVar();
  const amp = float(0.5).toVar();
  const q = p.toVar();
  const factor = float(2.02).toVar();
  Loop({ start: int(0), end: int(3), type: 'int', name: 'i' }, () => {
    f.addAssign(amp.mul(cloudNoise3(q)));
    q.mulAssign(factor);
    factor.addAssign(0.21);
    amp.mulAssign(0.5);
  });
  return f;
});

// Density in [0,1]. `p` is camera-anchored so the pattern behaves like a sky
// dome and does not swim with camera translation.
function sampleCloudDensity(u, p, covLow, width, scale) {
  // Low-frequency field = cloud macro shape; medium field = fluffy body.
  const macro = cloudFbm(p.mul(scale.mul(0.22))).mul(0.5).add(0.5);
  const body = cloudFbm(p.mul(scale.mul(1.0))).mul(0.5).add(0.5);
  // One cheap non-FBM high-frequency sample to erode holes/details without
  // paying for another expensive raymarch FBM.
  const detail = cloudNoise3(p.mul(scale.mul(2.8))).mul(0.5).add(0.5);
  const shape = nodeSmoothstep(covLow, covLow.add(width), macro);
  let fill = nodeSmoothstep(0.24, 0.78, body);
  // Detail slider controls small-scale breakup; holes slider controls how much
  // of that breakup actually cuts through to sky.
  fill = fill.add(detail.sub(0.5).mul(mix(0.0, 0.35, u.uDetailStrength)));
  fill = fill.sub(oneMinus(detail).mul(mix(0.04, 0.62, u.uHoles)));
  return nodeClamp(shape.mul(fill).add(0.055), 0.0, 1.0);
}

// Volumetric cloud shader following the standard raymarch model:
//   - constant-step march through the view ray
//   - density from a baked 3D Perlin-Worley texture
//   - directional-derivative single-sample diffuse (no normals)
//   - Beer's law transmittance accumulation
//   - Henyey-Greenstein phase for anisotropic scattering
//   - blue-noise-style per-pixel dither on the march start to hide banding at
//     a low step count
// Camera-anchored sampling keeps the pattern fixed to the sky.
const CLOUD_STEPS = 8;
const CLOUD_ABSORPTION = 7.0;

function createSkyVolumeCloudMaterial(color, opacity, coverage, darkness, detailStrength, holes, sharpness, wispiness) {
  const material = new THREE.MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthWrite = false;
  material.side = THREE.BackSide;
  material.toneMapped = false;

  const u = {
    uColor: assignUniform(material, 'uColor', uniform(color.clone())),
    uOpacity: assignUniform(material, 'uOpacity', uniform(opacity)),
    uCoverage: assignUniform(material, 'uCoverage', uniform(coverage)),
    uDetailStrength: assignUniform(material, 'uDetailStrength', uniform(detailStrength)),
    uHoles: assignUniform(material, 'uHoles', uniform(holes)),
    uSharpness: assignUniform(material, 'uSharpness', uniform(sharpness)),
    uWispiness: assignUniform(material, 'uWispiness', uniform(wispiness)),
    uScale: assignUniform(material, 'uScale', uniform(0.04)),
    uSunDirection: assignUniform(material, 'uSunDirection', uniform(new THREE.Vector3(...DEFAULT_SUN_POSITION).normalize())),
    uSunColor: assignUniform(material, 'uSunColor', uniform(new THREE.Color(0xfff0d2))),
    uShadowColor: assignUniform(material, 'uShadowColor', uniform(new THREE.Color(0xc9d6e8))),
    uFogColor: assignUniform(material, 'uFogColor', uniform(new THREE.Color(0x9fb7d5))),
    uDarkness: assignUniform(material, 'uDarkness', uniform(darkness)),
    uRadius: assignUniform(material, 'uRadius', uniform(420)),
    uThickness: assignUniform(material, 'uThickness', uniform(60)),
    uAltitude: assignUniform(material, 'uAltitude', uniform(80)),
  };

  material.fragmentNode = Fn(() => {
    const px = positionWorld;
    const cam = cameraPosition;
    const viewDir = normalize(px.sub(cam));
    const dist = length(px.sub(cam));

    const covLow = mix(0.55, 0.05, nodeClamp(u.uCoverage, 0.0, 1.0));
    const width = mix(0.50, 0.26, nodeClamp(u.uSharpness, 0.0, 1.0));
    const scale = u.uScale;
    const sunDir = normalize(u.uSunDirection);

    // Horizon fade: clouds fill the sky above the horizon, fade out below it.
    // Broad lower-horizon fade: clouds reach the horizon, but do not wrap down
    // around/under the scene like a full enclosing sphere.
    const horizonFade = nodeSmoothstep(float(0.0), float(0.16), viewDir.y);

    // Henyey-Greenstein phase: brightens light looking toward the sun.
    const mu = dot(viewDir, sunDir);
    const g = float(0.5);
    const gg = g.mul(g);
    const phase = float(0.25).mul(float(1.0).sub(gg)).div(
      pow(float(1.0).add(gg).sub(g.mul(mu).mul(2.0)), float(1.5)),
    ).add(float(0.25));

    // Fixed world-altitude cloud slab. This keeps the noise locked in the sky
    // instead of making the camera fly through arbitrary sphere noise.
    const up = max(viewDir.y, float(0.08));
    const slabStart = max(u.uAltitude.sub(cam.y).div(up), float(0.0));
    const stepLen = u.uThickness.div(float(CLOUD_STEPS)).div(up);
    const transmittance = float(1.0).toVar();
    const lightEnergy = vec3(0.0).toVar();
    const depth = slabStart.add(stepLen.mul(0.5)).toVar();

    Loop({ start: int(0), end: int(CLOUD_STEPS), type: 'int', name: 'i' }, () => {
      // World-anchored sample point in the fixed altitude layer: no wind/time
      // morphing, but camera translation sees the same clouds from new angles.
      const p = cam.add(viewDir.mul(depth));
      const d = sampleCloudDensity(u, p, covLow, width, scale).mul(horizonFade);

      // Only accumulate where there's actually cloud.
      If(d.greaterThan(float(0.005)), () => {
        // Cheap lighting: avoid an extra density sample per ray step. This is a
        // big perf win versus directional-derivative lighting while preserving
        // enough sun/shadow variation for the sky layer.
        const diffuse = nodeClamp(sunDir.y.mul(0.5).add(0.5), 0.35, 1.0);

        // Beer's law style attenuation based on local density.
        const lightTransmittance = exp(d.negate().mul(1.1));
        const luminance = float(0.05).add(d.mul(phase));
        // Cloud color is sun-warmed where lit, shadow-colored where occluded.
        const lit = mix(u.uShadowColor, u.uSunColor, lightTransmittance.mul(diffuse));
        const stepColor = u.uColor.mul(lit).mul(luminance);

        // Front-to-back compositing via Beer's law for the view ray.
        // Normalize optical depth by step count instead of raw world distance;
        // raw stepLen made the volume read like opaque paint.
        const tau = d.div(float(CLOUD_STEPS)).mul(CLOUD_ABSORPTION).mul(u.uOpacity);
        lightEnergy.addAssign(transmittance.mul(stepColor).mul(tau).mul(3.5));
        transmittance.mulAssign(exp(tau.negate()));
      });

      depth.addAssign(stepLen);
    });

    const alpha = nodeClamp(oneMinus(transmittance).mul(u.uOpacity).mul(1.35), 0.0, 1.0);
    If(alpha.lessThan(0.003), () => { Discard(); });

    let col = lightEnergy;
    // Darkness tints the whole cloud cooler/darker for overcast moods.
    col = mix(col, col.mul(vec3(0.34, 0.38, 0.48)), u.uDarkness.mul(0.85));
    // Sun-facing forward scatter glow on the denser parts.
    col = col.add(u.uSunColor.mul(pow(max(mu, 0.0), 8.0)).mul(oneMinus(transmittance)).mul(0.35).mul(oneMinus(u.uDarkness)));
    // Distance fog blends far clouds into the sky.
    col = mix(col, u.uFogColor, nodeSmoothstep(u.uRadius.mul(0.45), u.uRadius, dist).mul(0.5));
    col = nodeClamp(col, vec3(0.0), vec3(3.0));
    return vec4(col, alpha);
  })();

  return material;
}

export class CloudSkyLayer {
  constructor({ scene, camera, sky = null, ...options } = {}) {
    this.scene = scene;
    this.camera = camera;
    this.sky = sky;
    this.params = normalizeCloudParams(options);
    this._mesh = null;
    this._material = null;
    this._autoTintColor = new THREE.Color();
    this._sunColor = new THREE.Color(0xfff0d2);
    this._shadowColor = new THREE.Color(0xc9d6e8);
    this._ready = false;
  }

  init() {
    if (!this.scene || !this.camera) throw new Error('metaverse-sky: CloudSkyLayer requires `scene` and `camera`');
    if (this._ready) return this;
    this._material = createSkyVolumeCloudMaterial(
      this.params.cloudColor,
      this.params.opacity,
      this.params.coverage,
      this.params.darkness,
      this.params.detailStrength,
      this.params.holes,
      this.params.sharpness,
      this.params.wispiness,
    );
    const radius = Math.max(100, this.params.drawDistance);
    const geo = new THREE.SphereGeometry(radius, 32, 16);
    this._mesh = new THREE.Mesh(geo, this._material);
    this._mesh.position.copy(this.camera.position);
    this._mesh.frustumCulled = false;
    this._mesh.renderOrder = 2;
    this.scene.add(this._mesh);
    this._ready = true;
    this.applyAtmosphereSettings({});
    this._syncFog();
    return this;
  }

  _syncVisibility() {
    if (this._mesh) this._mesh.visible = this.params.enabled && this._ready;
  }

  _syncFog() {
    const fog = this.scene?.fog;
    if (fog && this._material) this._material.uniforms.uFogColor.value.copy(fog.color);
  }

  _syncSunLighting() {
    if (!this._material) return;
    const sun = this.sky?.material?.uniforms?.sunPosition?.value ?? new THREE.Vector3(...DEFAULT_SUN_POSITION).normalize();
    const sunHeight = clamp(sun.y, -0.25, 1);
    const day = smoothstep(-0.08, 0.22, sunHeight);
    const warmth = 1 - smoothstep(0.12, 0.62, sunHeight);
    this._sunColor.setHex(0xffffff).lerp(new THREE.Color(0xffb36f), warmth * 0.55);
    this._shadowColor.setHex(0x9fb7d5).lerp(new THREE.Color(0xd6e2f2), day * 0.72);
    const u = this._material.uniforms;
    u.uSunDirection.value.copy(sun).normalize();
    u.uSunColor.value.copy(this._sunColor).multiplyScalar(0.35 + day * 0.65);
    u.uShadowColor.value.copy(this._shadowColor).multiplyScalar(0.75 + day * 0.25);
  }

  setSunDirection(direction) {
    if (!this.sky) return this;
    const d = vectorFrom(direction).normalize();
    this.sky.material.uniforms.sunPosition.value.copy(d);
    this._syncSunLighting();
    return this;
  }

  getAtmosphereSettings() {
    const p = this.params;
    return {
      cloudsEnabled: p.enabled,
      cloudRenderMode: 'volume',
      cloudOpacity: p.opacity,
      cloudAltitude: p.altitude,
      cloudTile: p.tile,
      cloudDrawDistance: p.drawDistance,
      cloudColor: p.cloudColor.getHex(),
      cloudAutoTint: p.autoTint,
      cloudCoverage: p.coverage,
      cloudNoiseScale: p.noiseScale,
      cloudDetailStrength: p.detailStrength,
      cloudHoles: p.holes,
      cloudSharpness: p.sharpness,
      cloudWispiness: p.wispiness,
      cloudDarkness: p.darkness,
    };
  }

  applyAtmosphereSettings(data = {}) {
    const p = this.params;
    if (data.cloudsEnabled != null) p.enabled = !!data.cloudsEnabled;
    if (data.cloudOpacity != null) p.opacity = data.cloudOpacity;
    if (data.cloudAltitude != null) p.altitude = data.cloudAltitude;
    if (data.cloudTile != null) p.tile = data.cloudTile;
    if (data.cloudDrawDistance != null) p.drawDistance = data.cloudDrawDistance;
    if (data.cloudColor != null) p.cloudColor.set(data.cloudColor);
    if (data.cloudAutoTint != null) p.autoTint = !!data.cloudAutoTint;
    if (data.cloudCoverage != null) p.coverage = data.cloudCoverage;
    if (data.cloudNoiseScale != null) p.noiseScale = data.cloudNoiseScale;
    if (data.cloudDetailStrength != null) p.detailStrength = data.cloudDetailStrength;
    if (data.cloudHoles != null) p.holes = data.cloudHoles;
    if (data.cloudSharpness != null) p.sharpness = data.cloudSharpness;
    if (data.cloudWispiness != null) p.wispiness = data.cloudWispiness;
    if (data.cloudDarkness != null) p.darkness = data.cloudDarkness;
    if (this._material) {
      const u = this._material.uniforms;
      u.uOpacity.value = p.opacity;
      u.uCoverage.value = p.coverage;
      u.uScale.value = Math.max(0.0005, p.noiseScale * (p.tile / 6) * 0.45);
      u.uDetailStrength.value = p.detailStrength;
      u.uHoles.value = p.holes;
      u.uSharpness.value = p.sharpness;
      u.uWispiness.value = p.wispiness;
      u.uRadius.value = p.drawDistance;
      u.uAltitude.value = p.altitude;
      u.uThickness.value = Math.max(20, p.drawDistance * 0.18);
      u.uDarkness.value = p.darkness;
      u.uColor.value.copy(p.cloudColor);
    }
    if (this._mesh) {
      const radius = Math.max(100, p.drawDistance);
      this._mesh.geometry.dispose();
      this._mesh.geometry = new THREE.SphereGeometry(radius, 32, 16);
      this._mesh.position.copy(this.camera.position);
    }
    this._syncVisibility();
    this._syncSunLighting();
    return this;
  }

  update(dt) {
    if (!this._ready || !this.params.enabled) return;
    const p = this.params;
    this._mesh.position.copy(this.camera.position);
    const u = this._material.uniforms;
    this._syncFog();
    this._syncSunLighting();

    if (this.sky && p.autoTint) {
      const sunY = this.sky.material.uniforms.sunPosition.value.y;
      const day = smoothstep(-0.06, 0.28, sunY);
      const goldenHour = 1 - smoothstep(0.08, 0.62, Math.max(0, sunY));
      this._autoTintColor.setRGB(
        0.72 + day * 0.22 + goldenHour * 0.06,
        0.78 + day * 0.17 + goldenHour * 0.02,
        0.9 + day * 0.08 - goldenHour * 0.05,
      );
      u.uColor.value.copy(this._autoTintColor);
    }
  }

  dispose() {
    if (this._mesh) {
      this.scene?.remove(this._mesh);
      this._mesh.geometry.dispose();
    }
    this._material?.dispose();
    this._mesh = null;
    this._material = null;
    this._ready = false;
  }
}

export function sunDirectionFromAngles(elevation, azimuth) {
  const phi = THREE.MathUtils.degToRad(90 - elevation);
  const theta = THREE.MathUtils.degToRad(azimuth);
  return new THREE.Vector3().setFromSphericalCoords(1, phi, theta).normalize();
}

export function getSunAnglesFromDirection(direction) {
  const d = vectorFrom(direction).normalize();
  return {
    elevation: THREE.MathUtils.radToDeg(Math.asin(clamp(d.y, -1, 1))),
    azimuth: THREE.MathUtils.radToDeg(Math.atan2(d.x, d.z)),
  };
}

export function createAtmosphereSky({
  scale = DEFAULT_SKY_SCALE,
  turbidity = DEFAULT_TURBIDITY,
  rayleigh = DEFAULT_RAYLEIGH,
  mieCoefficient = DEFAULT_MIE_COEFFICIENT,
  mieDirectionalG = DEFAULT_MIE_DIRECTIONAL_G,
  sunPosition = DEFAULT_SUN_POSITION,
} = {}) {
  const geometry = new THREE.SphereGeometry(1, 64, 32);
  const material = new THREE.MeshBasicNodeMaterial();
  material.side = THREE.BackSide;
  material.depthWrite = false;
  material.toneMapped = false;

  const u = {
    turbidity: assignUniform(material, 'turbidity', uniform(turbidity)),
    rayleigh: assignUniform(material, 'rayleigh', uniform(rayleigh)),
    mieCoefficient: assignUniform(material, 'mieCoefficient', uniform(mieCoefficient)),
    mieDirectionalG: assignUniform(material, 'mieDirectionalG', uniform(mieDirectionalG)),
    sunPosition: assignUniform(material, 'sunPosition', uniform(vectorFrom(sunPosition).normalize())),
    up: assignUniform(material, 'up', uniform(new THREE.Vector3(0, 1, 0))),
  };

  material.fragmentNode = Fn(() => {
    // Fresh single-scattering sky model inspired by O'Neil/Hillaire, with
    // Earth-like constants cross-checked against Bruneton. Units are kilometers.
    // The sky mesh follows the camera, so positionLocal is the view direction.
    const viewDir = normalize(positionLocal);
    const sunDir = normalize(u.sunPosition);
    const sunY = sunDir.y;
    const day = nodeSmoothstep(-0.10, 0.18, sunY);
    const twilight = oneMinus(nodeSmoothstep(0.05, 0.45, max(sunY, 0.0))).mul(nodeSmoothstep(-0.22, 0.08, sunY));

    const planetRadius = float(6360.0);
    const atmosphereRadius = float(6460.0);
    const cameraHeight = float(1.8 / 1000.0); // 1.8m eye height
    const origin = vec3(0.0, planetRadius.add(cameraHeight), 0.0);

    const bOuter = dot(origin, viewDir);
    const cOuter = dot(origin, origin).sub(atmosphereRadius.mul(atmosphereRadius));
    const hOuter = max(bOuter.mul(bOuter).sub(cOuter), 0.0);
    let tMax = bOuter.negate().add(sqrt(hOuter));

    const bGround = dot(origin, viewDir);
    const cGround = dot(origin, origin).sub(planetRadius.mul(planetRadius));
    const hGround = bGround.mul(bGround).sub(cGround);
    const tGround = bGround.negate().sub(sqrt(max(hGround, 0.0)));
    const hitsGround = hGround.greaterThan(0.0).and(tGround.greaterThan(0.0));
    tMax = hitsGround.select(min(tMax, tGround), tMax);

    const viewSamples = float(12.0);
    const lightSamples = float(4.0);
    const segmentLength = tMax.div(viewSamples);
    const betaRayleigh = vec3(5.802, 13.558, 33.100).mul(0.001).mul(u.rayleigh);
    const betaMie = vec3(3.996).mul(0.001).mul(u.mieCoefficient.mul(180.0)).mul(nodeClamp(u.turbidity.div(8.0), 0.25, 2.5));
    const scaleRayleigh = float(8.0);
    const scaleMie = float(1.2);

    const opticalDepthR = float(0.0).toVar();
    const opticalDepthM = float(0.0).toVar();
    const sumR = vec3(0.0).toVar();
    const sumM = vec3(0.0).toVar();
    const t = float(0.0).toVar();

    Loop({ start: int(0), end: int(12), type: 'int', name: 'i' }, () => {
      const sampleT = t.add(segmentLength.mul(0.5));
      const p = origin.add(viewDir.mul(sampleT));
      const height = max(length(p).sub(planetRadius), 0.0);
      const densityR = exp(height.div(scaleRayleigh).negate());
      const densityM = exp(height.div(scaleMie).negate());
      opticalDepthR.addAssign(densityR.mul(segmentLength));
      opticalDepthM.addAssign(densityM.mul(segmentLength));

      const bLight = dot(p, sunDir);
      const cLight = dot(p, p).sub(atmosphereRadius.mul(atmosphereRadius));
      const hLight = max(bLight.mul(bLight).sub(cLight), 0.0);
      const lightLength = bLight.negate().add(sqrt(hLight)).div(lightSamples);
      const lightDepthR = float(0.0).toVar();
      const lightDepthM = float(0.0).toVar();
      const lt = float(0.0).toVar();
      Loop({ start: int(0), end: int(4), type: 'int', name: 'j' }, () => {
        const lp = p.add(sunDir.mul(lt.add(lightLength.mul(0.5))));
        const lh = max(length(lp).sub(planetRadius), 0.0);
        lightDepthR.addAssign(exp(lh.div(scaleRayleigh).negate()).mul(lightLength));
        lightDepthM.addAssign(exp(lh.div(scaleMie).negate()).mul(lightLength));
        lt.addAssign(lightLength);
      });

      const tau = betaRayleigh.mul(opticalDepthR.add(lightDepthR)).add(betaMie.mul(opticalDepthM.add(lightDepthM)));
      const attenuation = exp(tau.negate());
      sumR.addAssign(attenuation.mul(densityR).mul(segmentLength));
      sumM.addAssign(attenuation.mul(densityM).mul(segmentLength));
      t.addAssign(segmentLength);
    });

    const mu = dot(viewDir, sunDir);
    const mu2 = mu.mul(mu);
    const rayleighPhase = float(3.0 / (16.0 * Math.PI)).mul(float(1.0).add(mu2));
    const g = nodeClamp(u.mieDirectionalG, 0.0, 0.95);
    const g2 = g.mul(g);
    const miePhase = float(3.0 / (8.0 * Math.PI))
      .mul(float(1.0).sub(g2))
      .mul(float(1.0).add(mu2))
      .div(float(2.0).add(g2))
      .div(pow(float(1.0).add(g2).sub(g.mul(mu).mul(2.0)), 1.5));

    const sunIntensity = mix(8.0, 18.0, day).mul(nodeSmoothstep(-0.18, 0.05, sunY));
    let scatter = sumR.mul(betaRayleigh).mul(rayleighPhase).add(sumM.mul(betaMie).mul(miePhase)).mul(sunIntensity);

    // A small multiple-scattering-inspired ambient lift keeps the anti-solar sky
    // from going unnaturally black without needing LUTs.
    const horizon = nodeClamp(viewDir.y.mul(0.5).add(0.5), 0.0, 1.0);
    const ambientSky = mix(vec3(0.020, 0.035, 0.075), vec3(0.10, 0.24, 0.55).mul(u.rayleigh.mul(0.35)), day)
      .mul(pow(horizon, 0.35))
      .mul(float(0.15).add(day.mul(0.35)));
    scatter = scatter.add(ambientSky);

    // Sunset warm extinction and a compact solar disc/glow.
    const sunDot = max(mu, 0.0);
    const sunDisc = pow(sunDot, 3500.0).mul(day).mul(3.0);
    const sunGlow = pow(sunDot, mix(20.0, 6.0, nodeClamp(u.turbidity.div(12.0), 0.0, 1.0))).mul(nodeSmoothstep(-0.08, 0.35, sunY)).mul(0.18);
    const warm = mix(vec3(1.0, 0.38, 0.16), vec3(1.0, 0.92, 0.72), day);
    scatter = scatter.add(warm.mul(sunDisc.add(sunGlow)));

    const twilightTint = mix(vec3(0.90, 0.28, 0.18), vec3(0.22, 0.16, 0.42), horizon);
    scatter = mix(scatter, scatter.add(twilightTint.mul(0.22)), twilight);

    const night = mix(vec3(0.008, 0.014, 0.035), vec3(0.025, 0.040, 0.090), pow(horizon, 0.7));
    let col = mix(night, scatter, nodeSmoothstep(-0.18, 0.02, sunY));
    col = oneMinus(exp(col.mul(-1.15)));
    col = nodeClamp(col, vec3(0.0), vec3(8.0));
    return vec4(col, 1.0);
  })();

  const sky = new THREE.Mesh(geometry, material);
  sky.name = 'metaverse-sky-atmosphere-tsl';
  sky.scale.setScalar(scale);
  return sky;
}

export function setSkySun(sky, { elevation, azimuth, light = null, lightDistance = 220 } = {}) {
  const direction = sunDirectionFromAngles(elevation, azimuth);
  sky.material.uniforms.sunPosition.value.copy(direction);
  if (light) light.position.copy(direction).multiplyScalar(lightDistance);
  return direction;
}

export function syncEnvironmentIntensity({
  scene = null,
  materials = [],
  elevation,
  envIntensityMin = DEFAULT_ENV_INTENSITY_MIN,
  envIntensityMax = DEFAULT_ENV_INTENSITY_MAX,
} = {}) {
  const elevationMix = clamp((elevation - ELEVATION_MIN) / (ELEVATION_MAX - ELEVATION_MIN), 0, 1);
  const intensity = THREE.MathUtils.lerp(envIntensityMin, envIntensityMax, elevationMix);
  if (scene && 'environmentIntensity' in scene) scene.environmentIntensity = intensity;
  for (const material of materials) {
    if (material) material.envMapIntensity = intensity;
  }
  return intensity;
}

export function showPanel(panel, { display = 'flex' } = {}) {
  if (!panel) return;
  clearTimeout(panel._fadeTimer);
  clearTimeout(panel._fadeInTimer);
  if (panel._fadeRaf) cancelAnimationFrame(panel._fadeRaf);
  panel._fadeRaf = 0;
  panel.dataset.open = '1';
  panel.classList.remove('is-hiding');
  panel.classList.add('is-visible');
  panel.style.display = display;
  panel.style.transition = 'none';
  panel.style.opacity = '0';
  void panel.offsetHeight;
  panel.style.transition = `opacity ${FADE_MS}ms ease`;
  panel.style.opacity = '1';
  panel._fadeInTimer = setTimeout(() => {
    panel.style.transition = '';
    panel.style.opacity = '';
    panel._fadeInTimer = null;
  }, FADE_MS + 30);
}

export function hidePanel(panel) {
  if (!panel) return;
  if (panel._fadeRaf) cancelAnimationFrame(panel._fadeRaf);
  panel._fadeRaf = 0;
  clearTimeout(panel._fadeTimer);
  clearTimeout(panel._fadeInTimer);
  delete panel.dataset.open;
  if (panel.style.display === 'none') {
    panel.classList.remove('is-visible', 'is-hiding');
    panel.style.transition = '';
    panel.style.opacity = '';
    return;
  }
  panel.classList.remove('is-visible');
  panel.style.transition = `opacity ${FADE_MS}ms ease`;
  panel.style.opacity = '0';
  panel._fadeTimer = setTimeout(() => {
    panel.style.display = 'none';
    panel.style.transition = '';
    panel.style.opacity = '';
    panel.classList.remove('is-hiding');
    panel._fadeTimer = null;
  }, FADE_MS + 30);
}

export function isPanelOpen(panel) {
  return panel?.dataset.open === '1' || panel?.classList.contains('is-visible');
}

const PRECIP_DEFAULTS = {
  type: 'none',
  intensity: 0.7,
  speed: 1,
  size: 1,
  windDrift: 1,
};

const PRECIP_PROFILES = {
  rain: {
    color: 0xaecbe0,
    size: 0.7,
    speed: 60,
    windDrift: 0.6,
    softness: 0.5,
    slant: 1,
    swirl: 0,
  },
  snow: {
    color: 0xf6fbff,
    size: 0.8,
    speed: 8,
    windDrift: 1.4,
    softness: 0.18,
    slant: 0.2,
    swirl: 1,
  },
  hail: {
    color: 0xdce8f2,
    size: 0.5,
    speed: 40,
    windDrift: 0.3,
    softness: 0.08,
    slant: 0.1,
    swirl: 0,
  },
};

const PRECIP_MAX = 4000;
const PRECIP_BOX = { w: 220, h: 140, d: 220 };

export class Precipitation {
  constructor({ scene, camera, sky = null, textures = {} } = {}) {
    if (!scene || !camera) throw new Error('metaverse-sky: Precipitation requires `scene` and `camera`');
    this.scene = scene;
    this.camera = camera;
    this.sky = sky;
    this.params = { ...PRECIP_DEFAULTS };
    this._profile = null;
    this._group = new THREE.Group();
    this._group.visible = false;
    this.scene.add(this._group);

    this._pos = new Float32Array(PRECIP_MAX * 3);
    this._seed = new Float32Array(PRECIP_MAX);
    this._size = new Float32Array(PRECIP_MAX);
    this._active = 0;
    this._envelope = 0;
    this._target = 0;
    this._fadeStart = 0;
    this._fadeFrom = 0;
    this._fadeDur = 0;
    this._time = 0;
    this._windX = 0;
    this._windZ = 0;
    this._textures = {};
    this._textureLoader = null;
    this._ready = false;

    if (textures && Object.keys(textures).length > 0) {
      this.setTextures(textures);
    }
  }

  init() {
    if (this._ready) return this;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this._pos, 3));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(this._seed, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(this._size, 1));
    geo.setDrawRange(0, 0);

    this._defaultPrecipTexture = createSolidTexture(255, 255, 255, 255);
    this._pointMat = new THREE.PointsNodeMaterial();
    this._pointMat.transparent = true;
    this._pointMat.depthWrite = false;
    this._pointMat.sizeAttenuation = true;

    const u = {
      uTexture: assignUniform(this._pointMat, 'uTexture', texture(this._defaultPrecipTexture)),
      uTextureEnabled: assignUniform(this._pointMat, 'uTextureEnabled', uniform(0)),
      uColor: assignUniform(this._pointMat, 'uColor', uniform(new THREE.Color(0xffffff))),
      uSunColor: assignUniform(this._pointMat, 'uSunColor', uniform(new THREE.Color(0xfff0d2))),
      uSunFactor: assignUniform(this._pointMat, 'uSunFactor', uniform(0)),
      uFogColor: assignUniform(this._pointMat, 'uFogColor', uniform(new THREE.Color(0x9fb7d5))),
      uFogNear: assignUniform(this._pointMat, 'uFogNear', uniform(60)),
      uFogFar: assignUniform(this._pointMat, 'uFogFar', uniform(360)),
      uFogEnabled: assignUniform(this._pointMat, 'uFogEnabled', uniform(0)),
      uOpacity: assignUniform(this._pointMat, 'uOpacity', uniform(0)),
      uSizeScale: assignUniform(this._pointMat, 'uSizeScale', uniform(1)),
      uSoftness: assignUniform(this._pointMat, 'uSoftness', uniform(0.18)),
      uFadeNear: assignUniform(this._pointMat, 'uFadeNear', uniform(12)),
      uFadeFar: assignUniform(this._pointMat, 'uFadeFar', uniform(360)),
      uTime: assignUniform(this._pointMat, 'uTime', uniform(0)),
      uTwinkle: assignUniform(this._pointMat, 'uTwinkle', uniform(0)),
    };
    const aSeed = attribute('aSeed', 'float');
    const aSize = attribute('aSize', 'float');
    const dist = length(cameraPosition.sub(positionWorld));
    const fade = oneMinus(nodeSmoothstep(u.uFadeNear, u.uFadeFar, dist));
    this._pointMat.scaleNode = aSize.mul(u.uSizeScale).mul(float(340.0).div(max(dist, 1.0)));
    this._pointMat.fragmentNode = Fn(() => {
      const c = uv().sub(0.5);
      const r = length(c);
      const proceduralAlpha = nodeSmoothstep(0.5, float(0.5).sub(u.uSoftness), r);
      const tex = texture(u.uTexture, uv());
      const shapeAlpha = u.uTextureEnabled.greaterThan(0.5).select(tex.a, proceduralAlpha);
      const texColor = u.uTextureEnabled.greaterThan(0.5).select(tex.rgb, vec3(1.0));
      If(shapeAlpha.lessThan(0.01), () => { Discard(); });
      const twinkle = oneMinus(u.uTwinkle).add(u.uTwinkle.mul(float(0.55).add(sin(u.uTime.mul(3.0).add(aSeed.mul(50.0))).mul(0.45))));
      let col = u.uColor.mul(texColor).add(u.uSunColor.mul(u.uSunFactor).mul(0.35));
      const ff = nodeSmoothstep(u.uFogNear, u.uFogFar, dist).mul(u.uFogEnabled);
      col = mix(col, u.uFogColor, ff.mul(0.7));
      const alpha = shapeAlpha.mul(u.uOpacity).mul(fade).mul(twinkle).mul(oneMinus(ff.mul(0.5)));
      return vec4(col, alpha);
    })();
    this._points = new THREE.Points(geo, this._pointMat);
    this._points.frustumCulled = false;
    this._group.add(this._points);

    this._ready = true;
    return this;
  }

  setWindDirection(direction) {
    const vec = Array.isArray(direction) ? direction : [Number(direction?.x) || 0, Number(direction?.y) || 0];
    const len = Math.hypot(vec[0], vec[1]) || 1;
    this._windX = vec[0] / len;
    this._windZ = vec[1] / len;
    return this;
  }

  setWindSpeed(speed) {
    this._windSpeed = Math.max(0, speed);
    return this;
  }

  setTextures(textures = {}) {
    if (!this._textureLoader) this._textureLoader = new THREE.TextureLoader();
    for (const [type, url] of Object.entries(textures)) {
      if (!url) continue;
      this._textureLoader.load(url, (tex) => {
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.premultiplyAlpha = false;
        this._textures[type] = tex;
      });
    }
    return this;
  }

  setPrecipitation(data = {}) {
    if (data.type != null) this.params.type = data.type;
    if (data.intensity != null) this.params.intensity = data.intensity;
    if (data.speed != null) this.params.speed = data.speed;
    if (data.size != null) this.params.size = data.size;
    if (data.windDrift != null) this.params.windDrift = data.windDrift;
    return this._applyType();
  }

  _applyType() {
    const wantType = this.params.type;
    const profile = wantType !== 'none' ? PRECIP_PROFILES[wantType] : null;
    if (profile) {
      this._profile = profile;
      const want = this.params.intensity;
      this._startFade(want, 2.4);
    } else {
      this._profile = null;
      this._startFade(0, 2.4);
    }
    return this;
  }

  _startFade(target, dur) {
    this._target = target;
    this._fadeFrom = this._envelope;
    this._fadeStart = this._time;
    this._fadeDur = dur;
  }

  _envelopeValue() {
    if (this._fadeDur <= 0) return this._target;
    const t = (this._time - this._fadeStart) / this._fadeDur;
    if (t >= 1) return this._target;
    const e = t * t * (3 - 2 * t);
    return this._fadeFrom + (this._target - this._fadeFrom) * e;
  }

  _spawn(i, atTop, cam) {
    const b = PRECIP_BOX;
    const depthBias = Math.pow(Math.random(), 1.5);
    this._pos[i * 3] = cam.x + (Math.random() - 0.5) * b.w;
    this._pos[i * 3 + 1] = atTop
      ? cam.y + b.h * 0.3 + Math.random() * b.h * 0.5
      : cam.y + (Math.random() - 0.25) * b.h;
    this._pos[i * 3 + 2] = cam.z + (Math.random() - 0.5) * b.d * (0.4 + depthBias);
    this._seed[i] = Math.random();
    this._size[i] = 0.4 + Math.random() * 1.3;
  }

  _syncLighting() {
    const fog = this.scene?.fog;
    const fogOn = fog ? 1 : 0;
    let sunFactor = 0;
    let sunColor = 0xfff0d2;
    if (this.sky) {
      const sun = this.sky.material.uniforms.sunPosition.value;
      const sunY = clamp(sun.y, -0.1, 1);
      sunFactor = sunY * 0.6;
      const warmth = 1 - smoothstep(0.12, 0.62, sunY);
      const c = new THREE.Color(0xffffff).lerp(new THREE.Color(0xffb36f), warmth * 0.55);
      sunColor = c.getHex();
    }
    const pu = this._pointMat.uniforms;
    pu.uSunColor.value.setHex(sunColor);
    pu.uSunFactor.value = sunFactor;
    pu.uFogEnabled.value = fogOn;
    if (fog) {
      pu.uFogColor.value.copy(fog.color);
      pu.uFogNear.value = fog.near;
      pu.uFogFar.value = fog.far;
    }
  }

  _simulate(dt) {
    const env = this._envelope;
    if (env <= 0.001 && this._target <= 0.001) {
      this._group.visible = false;
      return;
    }
    this._group.visible = true;
    const prof = this._profile;

    const wantActive = Math.floor(PRECIP_MAX * env);
    const cam = this.camera.position;
    while (this._active < wantActive) {
      this._spawn(this._active, true, cam);
      this._active += 1;
    }
    if (this._active > wantActive) this._active = wantActive;

    if (!prof) return;

    const b = PRECIP_BOX;
    const fallSpeed = prof.speed * this.params.speed;
    const driftMul = prof.windDrift * this.params.windDrift;
    const windSpeed = this._windSpeed || 0;
    const windMag = windSpeed * 30 * driftMul;
    const wx = this._windX * windMag;
    const wz = this._windZ * windMag;
    const swirl = prof.swirl;
    const t = this._time;

    const pos = this._pos;
    const bx = b.w * 0.5;
    const by = b.h * 0.5;
    const bz = b.d * 0.5;

    for (let i = 0; i < this._active; i += 1) {
      const ix = i * 3;
      const iy = ix + 1;
      const iz = ix + 2;
      const seed = this._seed[i];

      let vx = wx;
      let vz = wz;
      if (swirl > 0) {
        vx += Math.sin(t * 1.2 + seed * 31) * swirl * 4;
        vz += Math.cos(t * 0.9 + seed * 17) * swirl * 4;
      }
      pos[ix] += vx * dt;
      pos[iy] -= fallSpeed * (0.8 + seed * 0.4) * dt;
      pos[iz] += vz * dt;

      const relx = pos[ix] - cam.x;
      const rely = pos[iy] - cam.y;
      const relz = pos[iz] - cam.z;
      if (rely < -by) {
        this._spawn(i, true, cam);
        continue;
      }
      if (relx > bx) pos[ix] = cam.x - bx + (relx - bx);
      else if (relx < -bx) pos[ix] = cam.x + bx + (relx + bx);
      if (relz > bz) pos[iz] = cam.z - bz + (relz - bz);
      else if (relz < -bz) pos[iz] = cam.z + bz + (relz + bz);
    }

    const sizeScale = prof.size * this.params.size;
    const tex = this._textures[this.params.type];
    const opacity = env;
    this._pointMat.uniforms.uColor.value.setHex(prof.color);
    this._pointMat.uniforms.uOpacity.value = opacity;
    this._pointMat.uniforms.uSizeScale.value = sizeScale;
    this._pointMat.uniforms.uSoftness.value = prof.softness;
    this._pointMat.uniforms.uTwinkle.value = this.params.type === 'snow' ? 0.5 : 0;
    this._pointMat.uniforms.uTime.value = t;
    if (tex) {
      this._pointMat.uniforms.uTexture.value = tex;
      this._pointMat.uniforms.uTextureEnabled.value = 1;
    } else {
      this._pointMat.uniforms.uTextureEnabled.value = 0;
    }
    this._points.geometry.setDrawRange(0, this._active);
    this._points.geometry.getAttribute('position').needsUpdate = true;
    this._points.geometry.getAttribute('aSeed').needsUpdate = true;
    this._points.geometry.getAttribute('aSize').needsUpdate = true;
  }

  update(dt) {
    if (!this._ready) return;
    this._time += dt;
    this._envelope = this._envelopeValue();
    this._syncLighting();
    this._simulate(dt);
  }

  dispose() {
    if (!this._ready) return;
    this.scene.remove(this._group);
    this._points.geometry.dispose();
    this._pointMat.dispose();
    this._ready = false;
  }
}

export class MetaverseSky {
  constructor({
    scene,
    camera,
    renderer = null,
    light = null,
    sky = null,
    clouds = true,
    cloudOptions = {},
    precipitation = false,
    precipitationOptions = {},
    skyScale = DEFAULT_SKY_SCALE,
    atmosphere = {},
    envIntensityMin = DEFAULT_ENV_INTENSITY_MIN,
    envIntensityMax = DEFAULT_ENV_INTENSITY_MAX,
    environmentMaterials = [],
    onSunChange = null,
  } = {}) {
    if (!scene) throw new Error('metaverse-sky: MetaverseSky requires `scene`');
    if (!camera) throw new Error('metaverse-sky: MetaverseSky requires `camera`');

    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.light = light;
    this.lightDistance = light?.position.length() || 220;
    this.envIntensityMin = envIntensityMin;
    this.envIntensityMax = envIntensityMax;
    this.environmentMaterials = environmentMaterials;
    this.onSunChange = onSunChange;
    this.sky = sky ?? createAtmosphereSky({ scale: skyScale, ...atmosphere });
    this.scene.add(this.sky);

    if (renderer && atmosphere.exposure != null) renderer.toneMappingExposure = atmosphere.exposure;

    const angles = getSunAnglesFromDirection(this.sky.material.uniforms.sunPosition.value);
    this.elevation = atmosphere.elevation ?? angles.elevation;
    this.azimuth = atmosphere.azimuth ?? angles.azimuth;
    setSkySun(this.sky, { elevation: this.elevation, azimuth: this.azimuth, light: this.light, lightDistance: this.lightDistance });

    this.clouds = clouds
      ? new CloudSkyLayer({ scene, camera, sky: this.sky, ...cloudOptions }).init()
      : null;
    this.precipitation = precipitation
      ? new Precipitation({ scene, camera, sky: this.sky, ...precipitationOptions }).init()
      : null;
    if (this.precipitation && precipitationOptions.type) {
      this.precipitation.setPrecipitation({ type: precipitationOptions.type });
    }
    this.syncEnvironmentLighting();
  }

  setSun(elevation = this.elevation, azimuth = this.azimuth) {
    this.elevation = elevation;
    this.azimuth = azimuth;
    const direction = setSkySun(this.sky, {
      elevation,
      azimuth,
      light: this.light,
      lightDistance: this.lightDistance,
    });
    this.syncEnvironmentLighting();
    this.onSunChange?.(direction);
    return this;
  }

  setSunDirection(direction) {
    const d = vectorFrom(direction).normalize();
    const angles = getSunAnglesFromDirection(d);
    return this.setSun(angles.elevation, angles.azimuth);
  }

  setWindDirection(direction) {
    this.precipitation?.setWindDirection(direction);
    return this;
  }

  setWindSpeed(speed) {
    this.precipitation?.setWindSpeed(speed);
    return this;
  }

  setWind(directionOrAngle, speed) {
    let dir;
    if (Array.isArray(directionOrAngle)) {
      dir = directionOrAngle;
    } else if (directionOrAngle?.isVector2) {
      dir = [directionOrAngle.x, directionOrAngle.y];
    } else {
      const rad = THREE.MathUtils.degToRad(Number(directionOrAngle) || 0);
      dir = [Math.cos(rad), Math.sin(rad)];
    }
    this.setWindDirection(dir);
    if (speed != null) this.setWindSpeed(speed);
    return this;
  }

  setPrecipitation(data = {}) {
    this.precipitation?.setPrecipitation(data);
    return this;
  }

  setPrecipitationTextures(textures = {}) {
    this.precipitation?.setTextures(textures);
    return this;
  }

  setExposure(value) {
    if (this.renderer) this.renderer.toneMappingExposure = value;
    return this;
  }

  getExposure() {
    return this.renderer?.toneMappingExposure ?? null;
  }

  setAtmosphere(data = {}) {
    return this.applyAtmosphereSettings(data);
  }

  getAtmosphere() {
    return this.getAtmosphereSettings();
  }

  setClouds(data = {}) {
    const cloudData = {};
    for (const [key, val] of Object.entries(data)) {
      cloudData[`cloud${key[0].toUpperCase()}${key.slice(1)}`] = val;
    }
    return this.applyAtmosphereSettings(cloudData);
  }

  getClouds() {
    return this.clouds ? this.clouds.getAtmosphereSettings() : null;
  }

  getPrecipSettings() {
    return this.precipitation ? { ...this.precipitation.params } : null;
  }

  syncEnvironmentLighting(materials = this.environmentMaterials) {
    return syncEnvironmentIntensity({
      scene: this.scene,
      materials,
      elevation: this.elevation,
      envIntensityMin: this.envIntensityMin,
      envIntensityMax: this.envIntensityMax,
    });
  }

  addEnvironmentMaterial(material) {
    if (material && !this.environmentMaterials.includes(material)) this.environmentMaterials.push(material);
    return this;
  }

  getAtmosphereSettings() {
    const u = this.sky.material.uniforms;
    const out = {
      elevation: this.elevation,
      azimuth: this.azimuth,
      turbidity: u.turbidity.value,
      rayleigh: u.rayleigh.value,
      mieCoefficient: u.mieCoefficient.value,
      mieDirectionalG: u.mieDirectionalG.value,
      exposure: this.renderer?.toneMappingExposure,
      envIntensityMin: this.envIntensityMin,
      envIntensityMax: this.envIntensityMax,
    };
    if (this.clouds) Object.assign(out, this.clouds.getAtmosphereSettings());
    return out;
  }

  applyAtmosphereSettings(data = {}) {
    const u = this.sky.material.uniforms;
    if (data.turbidity != null) u.turbidity.value = data.turbidity;
    if (data.rayleigh != null) u.rayleigh.value = data.rayleigh;
    if (data.mieCoefficient != null) u.mieCoefficient.value = data.mieCoefficient;
    if (data.mieDirectionalG != null) u.mieDirectionalG.value = data.mieDirectionalG;
    if (data.exposure != null && this.renderer) this.renderer.toneMappingExposure = data.exposure;
    if (data.envIntensityMin != null) this.envIntensityMin = data.envIntensityMin;
    if (data.envIntensityMax != null) this.envIntensityMax = data.envIntensityMax;
    if (data.elevation != null) this.elevation = data.elevation;
    if (data.azimuth != null) this.azimuth = data.azimuth;
    if (data.sunPosition != null) {
      u.sunPosition.value.copy(vectorFrom(data.sunPosition).normalize());
      const angles = getSunAnglesFromDirection(u.sunPosition.value);
      this.elevation = angles.elevation;
      this.azimuth = angles.azimuth;
    }
    setSkySun(this.sky, {
      elevation: this.elevation,
      azimuth: this.azimuth,
      light: this.light,
      lightDistance: this.lightDistance,
    });
    this.clouds?.applyAtmosphereSettings(data);
    this.syncEnvironmentLighting();
    return this;
  }

  update(deltaTime) {
    this.sky.position.copy(this.camera.position);
    this.clouds?.update(deltaTime);
    this.precipitation?.update(deltaTime);
    return this;
  }

  dispose() {
    this.clouds?.dispose();
    this.precipitation?.dispose();
    this.scene.remove(this.sky);
    this.sky.geometry?.dispose?.();
    this.sky.material?.dispose?.();
  }
}

export class SkyEditor {
  constructor({
    sky,
    light = null,
    renderer,
    clouds = null,
    onSunChange = null,
    envIntensityMin = DEFAULT_ENV_INTENSITY_MIN,
    envIntensityMax = DEFAULT_ENV_INTENSITY_MAX,
  }) {
    this.sky = sky;
    this.u = sky.material.uniforms;
    this.light = light;
    this.renderer = renderer;
    this.clouds = clouds;
    this.onSunChange = onSunChange;
    this.envIntensityMin = envIntensityMin;
    this.envIntensityMax = envIntensityMax;
    this.active = false;

    const d = this.u.sunPosition.value;
    const angles = getSunAnglesFromDirection(d);
    this.elevation = angles.elevation;
    this.azimuth = angles.azimuth;
    this.lightDist = light?.position.length() || 220;

    this._build();
  }

  open() { this.active = true; showPanel(this.panel); }
  close() { this.active = false; hidePanel(this.panel); }

  _build() {
    this.panel = document.createElement('div');
    this.panel.className = 'sky-panel';
    this.panel.style.display = 'none';

    const title = document.createElement('div');
    title.className = 'sky-panel-title';
    title.textContent = 'Sky';
    this.panel.appendChild(title);

    this._section('Sun');
    this._slider('Elevation', -20, 90, 0.5, this.elevation, (v) => { this.elevation = v; this._updateSun(); });
    this._slider('Azimuth', -180, 180, 1, this.azimuth, (v) => { this.azimuth = v; this._updateSun(); });

    this._section('Environment');
    this._slider('IBL min', 0, 2, 0.01, this.envIntensityMin, (v) => {
      this.envIntensityMin = v;
      if (this.envIntensityMin > this.envIntensityMax) this.envIntensityMax = this.envIntensityMin;
      this._envMaxSlider?.set(this.envIntensityMax);
      this.onSunChange?.();
    });
    this._envMinSlider = this._lastSlider;
    this._slider('IBL max', 0, 2, 0.01, this.envIntensityMax, (v) => {
      this.envIntensityMax = v;
      if (this.envIntensityMax < this.envIntensityMin) this.envIntensityMin = this.envIntensityMax;
      this._envMinSlider?.set(this.envIntensityMin);
      this.onSunChange?.();
    });
    this._envMaxSlider = this._lastSlider;

    this._section('Atmosphere');
    this._slider('Turbidity', 0, 20, 0.1, this.u.turbidity.value, (v) => { this.u.turbidity.value = v; });
    this._slider('Rayleigh', 0, 4, 0.05, this.u.rayleigh.value, (v) => { this.u.rayleigh.value = v; });
    this._slider('Haze (Mie)', 0, 0.1, 0.001, this.u.mieCoefficient.value, (v) => { this.u.mieCoefficient.value = v; });
    this._slider('Sun glow (Mie-G)', 0, 1, 0.01, this.u.mieDirectionalG.value, (v) => { this.u.mieDirectionalG.value = v; });
    if (this.renderer) {
      this._slider('Exposure', 0, 1, 0.01, this.renderer.toneMappingExposure, (v) => { this.renderer.toneMappingExposure = v; });
    }

    if (this.clouds) this._buildCloudSection();

    const done = document.createElement('button');
    done.className = 'sky-done';
    done.textContent = 'Done';
    done.addEventListener('click', () => this.close());
    this.panel.appendChild(done);

    document.body.appendChild(this.panel);
  }

  _buildCloudSection() {
    const c = this.clouds;
    const p = c.params;
    this._section('Clouds');
    this._checkbox('Enabled', p.enabled, (on) => c.applyAtmosphereSettings({ cloudsEnabled: on }));
    this._slider('Opacity', 0, 1, 0.01, p.opacity, (v) => c.applyAtmosphereSettings({ cloudOpacity: v }));
    this._slider('Altitude (m)', 55, 220, 1, p.altitude, (v) => c.applyAtmosphereSettings({ cloudAltitude: v }));
    this._slider('Draw distance (m)', 120, 900, 10, p.drawDistance, (v) => c.applyAtmosphereSettings({ cloudDrawDistance: v }));
    this._slider('Tiling', 3, 10, 0.5, p.tile, (v) => c.applyAtmosphereSettings({ cloudTile: v }));
    this._slider('Darkness', 0, 1, 0.01, p.darkness, (v) => c.applyAtmosphereSettings({ cloudDarkness: v }));

    this._section('Cloud noise');
    this._slider('Coverage', 0.2, 0.9, 0.01, p.coverage, (v) => c.applyAtmosphereSettings({ cloudCoverage: v }));
    this._slider('Pattern scale', 0.002, 0.04, 0.001, p.noiseScale, (v) => c.applyAtmosphereSettings({ cloudNoiseScale: v }));
    this._slider('Detail', 0, 2, 0.01, p.detailStrength, (v) => c.applyAtmosphereSettings({ cloudDetailStrength: v }));
    this._slider('Holes', 0, 1, 0.01, p.holes, (v) => c.applyAtmosphereSettings({ cloudHoles: v }));
    this._slider('Sharpness', 0, 1, 0.01, p.sharpness, (v) => c.applyAtmosphereSettings({ cloudSharpness: v }));
    this._slider('Wispiness', 0, 1, 0.01, p.wispiness, (v) => c.applyAtmosphereSettings({ cloudWispiness: v }));

    this._section('Cloud color');
    this._checkbox('Tint from sun', p.autoTint, (on) => c.applyAtmosphereSettings({ cloudAutoTint: on }));
    this._color('Color', p.cloudColor.getHex(), (hex) => c.applyAtmosphereSettings({ cloudColor: hex, cloudAutoTint: false }));
  }

  _checkbox(label, checked, onChange) {
    const row = document.createElement('label');
    row.className = 'sky-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = checked;
    const span = document.createElement('span');
    span.textContent = label;
    input.addEventListener('change', () => {
      row.classList.toggle('is-off', !input.checked);
      onChange(input.checked);
    });
    row.classList.toggle('is-off', !checked);
    row.append(input, span);
    this.panel.appendChild(row);
  }

  _section(text) {
    const el = document.createElement('div');
    el.className = 'sky-section';
    el.textContent = text;
    this.panel.appendChild(el);
  }

  _color(label, hex, onInput) {
    const row = document.createElement('label');
    row.className = 'sky-row';
    const head = document.createElement('div');
    head.className = 'sky-row-head';
    const cap = document.createElement('span');
    cap.textContent = label;
    const input = document.createElement('input');
    input.type = 'color';
    input.className = 'sky-color';
    input.value = `#${hex.toString(16).padStart(6, '0')}`;
    input.addEventListener('input', () => onInput(parseInt(input.value.slice(1), 16)));
    head.append(cap);
    row.append(head, input);
    this.panel.appendChild(row);
  }

  _slider(label, min, max, step, value, onInput) {
    const row = document.createElement('label');
    row.className = 'sky-row';
    const head = document.createElement('div');
    head.className = 'sky-row-head';
    const cap = document.createElement('span');
    const val = document.createElement('b');
    const fmt = (v) => (step < 1 ? Number(v).toFixed(step < 0.01 ? 3 : 2) : String(Math.round(v)));
    cap.textContent = label;
    val.textContent = fmt(value);
    const input = document.createElement('input');
    input.type = 'range';
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;
    input.addEventListener('input', () => {
      val.textContent = fmt(input.value);
      onInput(parseFloat(input.value));
    });
    head.append(cap, val);
    row.append(head, input);
    this.panel.appendChild(row);
    this._lastSlider = {
      set: (v) => {
        input.value = v;
        val.textContent = fmt(v);
      },
    };
  }

  _updateSun() {
    setSkySun(this.sky, {
      elevation: this.elevation,
      azimuth: this.azimuth,
      light: this.light,
      lightDistance: this.lightDist,
    });
    this.onSunChange?.();
  }
}
