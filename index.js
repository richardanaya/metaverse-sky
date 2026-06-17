/**
 * metaverse-sky - Three.js Preetham sky, sun helpers, and procedural voxel clouds.
 *
 * Peer dependency: the host app must resolve `three` and `three/addons/`.
 */

import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';

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

const MASK_SIZE = 256;

const CLOUD_LAYERS = [
  { yOff: 0, thickness: 13, opacityMul: 0.4, phaseU: 0, phaseV: 0 },
  { yOff: 4.5, thickness: 10.5, opacityMul: 0.3, phaseU: 64, phaseV: 64 },
  { yOff: 9, thickness: 11, opacityMul: 0.28, phaseU: 128, phaseV: 0 },
];

const CLOUD_DEFAULTS = {
  enabled: true,
  altitude: 80,
  opacity: 0.95,
  windSpeed: 0.045,
  windDirection: 255,
  tile: 6,
  cloudColor: new THREE.Color(0xf2f6fc),
  autoTint: true,
  puffScale: 1,
  layerHeight: 1,
  coverage: 0.5,
  noiseSeed: 42,
  noiseScale: 0.028,
  noiseOctaves: 5,
  noiseJitter: 0.08,
  roundness: 0.16,
  softness: 1,
  darkness: 0,
};

export const DEFAULT_CLOUD_SETTINGS = Object.freeze({
  enabled: CLOUD_DEFAULTS.enabled,
  altitude: CLOUD_DEFAULTS.altitude,
  opacity: CLOUD_DEFAULTS.opacity,
  windSpeed: CLOUD_DEFAULTS.windSpeed,
  windDirection: CLOUD_DEFAULTS.windDirection,
  tile: CLOUD_DEFAULTS.tile,
  cloudColor: CLOUD_DEFAULTS.cloudColor.getHex(),
  autoTint: CLOUD_DEFAULTS.autoTint,
  puffScale: CLOUD_DEFAULTS.puffScale,
  layerHeight: CLOUD_DEFAULTS.layerHeight,
  coverage: CLOUD_DEFAULTS.coverage,
  noiseSeed: CLOUD_DEFAULTS.noiseSeed,
  noiseScale: CLOUD_DEFAULTS.noiseScale,
  noiseOctaves: CLOUD_DEFAULTS.noiseOctaves,
  noiseJitter: CLOUD_DEFAULTS.noiseJitter,
  roundness: CLOUD_DEFAULTS.roundness,
  softness: CLOUD_DEFAULTS.softness,
  darkness: CLOUD_DEFAULTS.darkness,
});

const PUFF_VERT = /* glsl */`
  attribute float instanceSeed;
  attribute float instanceFade;
  attribute float instanceDensity;
  attribute vec4 instanceRand;

  uniform vec3 uSunDirection;

  varying vec3 vLocalPos;
  varying vec3 vWorldPos;
  varying vec3 vInvScale;
  varying vec3 vLocalSunDir;
  varying float vSeed;
  varying float vFade;
  varying float vDensity;
  varying vec4 vRand;

  void main() {
    vLocalPos = position;
    vSeed = instanceSeed;
    vFade = instanceFade;
    vDensity = instanceDensity;
    vRand = instanceRand;

    vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;

    mat3 m = mat3(instanceMatrix);
    vec3 scl = vec3(m[0][0], m[1][1], m[2][2]);
    vInvScale = 1.0 / scl;
    vLocalSunDir = normalize(uSunDirection * vInvScale);

    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const PUFF_FRAG = /* glsl */`
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uRoundness;
  uniform float uSoftness;
  uniform vec3 uFogColor;
  uniform float uFogNear;
  uniform float uFogFar;
  uniform float uFogEnabled;
  uniform vec3 uSunDirection;
  uniform vec3 uSunColor;
  uniform vec3 uShadowColor;
  uniform vec3 uSkyColor;
  uniform vec3 uGroundColor;
  uniform float uTime;
  uniform float uPhaseG;
  uniform float uQuality;
  uniform float uDarkness;

  varying vec3 vLocalPos;
  varying vec3 vWorldPos;
  varying vec3 vInvScale;
  varying vec3 vLocalSunDir;
  varying float vSeed;
  varying float vFade;
  varying float vDensity;
  varying vec4 vRand;

  const float BASE_Y = -0.42;
  const float TOP_Y = 0.48;

  float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
  }

  float vnoise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i);
    float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash13(i + vec3(1.0, 1.0, 1.0));
    return mix(
      mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
      mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
      f.z
    );
  }

  float sdRoundBox(vec3 p, vec3 halfSize, float cornerR) {
    vec3 q = abs(p) - halfSize + cornerR;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - cornerR;
  }

  float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - h * (1.0 - h) * k;
  }

  float mapSDFCore(vec3 p) {
    p.x *= 0.92 + vSeed * 0.14;
    p.z *= 0.92 + (1.0 - vSeed) * 0.14;

    float cornerR = uRoundness + vSeed * 0.04;

    vec3 halfSize = vec3(0.44, 0.32 + vSeed * 0.05, 0.44);
    vec3 bodyCenter = vec3(0.0, 0.05, 0.0);
    float body = sdRoundBox(p - bodyCenter, halfSize, cornerR) - vDensity * 0.03;

    vec3 domeCenter = vec3(0.0, -0.15, 0.0);
    float dome = length(vec3(p.xz * 1.1, p.y - domeCenter.y)) - 0.38;

    float lobeR = max(0.05, cornerR * 0.7);
    vec3 o1 = vec3(0.16 * (vSeed - 0.5), 0.12, 0.14 * (vRand.x - 0.5));
    vec3 o2 = vec3(-0.18 * (vRand.y - 0.5), 0.22, 0.12 * (vSeed - 0.35));
    vec3 o3 = vec3(0.1 * (vRand.z - 0.5), 0.32, -0.15 * (vRand.w - 0.5));
    vec3 o4 = vec3(0.05 * (vRand.x + vRand.z - 1.0) * 0.5, 0.40, 0.07 * (vRand.y + vRand.w - 1.0) * 0.5);
    float l1 = sdRoundBox(p - o1, vec3(0.26, 0.22, 0.26), lobeR);
    float l2 = sdRoundBox(p - o2, vec3(0.23, 0.20, 0.23), lobeR * 0.92);
    float l3 = sdRoundBox(p - o3, vec3(0.20, 0.18, 0.20), lobeR * 0.84);
    float l4 = sdRoundBox(p - o4, vec3(0.16, 0.15, 0.16), lobeR * 0.76);

    float k = 0.10 + uSoftness * 0.08;
    float towers = smin(smin(smin(l1, l2, k), l3, k), l4, k) - vDensity * 0.04;
    return smin(smin(body, dome, k * 1.3), towers, k);
  }

  float mapSDF(vec3 p) {
    float shape = mapSDFCore(p);
    if (uQuality > 0.5) {
      float wob = vnoise3(p * 4.0 + vec3(vSeed * 9.0, vSeed * 5.0, uTime * 0.1));
      shape += (wob - 0.5) * 0.06;
    }
    return shape;
  }

  float densityFast(vec3 p) {
    float d = mapSDFCore(p);
    float edge = uSoftness * 0.16 + 0.04;
    return 1.0 - smoothstep(-edge, edge, d);
  }

  float densityQuality(vec3 p) {
    float d = mapSDF(p);
    float edge = max(fwidth(d) * 1.5, uSoftness * 0.16 + 0.04);
    return 1.0 - smoothstep(-edge, edge, d);
  }

  float shadowOcc(vec3 p) {
    float d = mapSDFCore(p);
    return 1.0 - smoothstep(-0.1, 0.1, d);
  }

  vec3 analyticNormal(vec3 p) {
    return normalize(vec3(p.x * 0.65, p.y * 1.1 + 0.12, p.z * 0.65));
  }

  vec3 tetraNormal(vec3 p) {
    const float e = 0.015;
    return normalize(
      mapSDFCore(p + vec3(e, -e, -e)) * vec3(1.0, -1.0, -1.0) +
      mapSDFCore(p + vec3(-e, e, -e)) * vec3(-1.0, 1.0, -1.0) +
      mapSDFCore(p + vec3(-e, -e, e)) * vec3(-1.0, -1.0, 1.0) +
      mapSDFCore(p + vec3(e, e, e)) * vec3(1.0, 1.0, 1.0)
    );
  }

  float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
  }

  void main() {
    vec3 p = vLocalPos;
    float dens = densityFast(p);
    if (dens < 0.002) discard;

    float volDens = dens;
    if (uQuality > 0.5) {
      vec3 viewDirWorld = normalize(cameraPosition - vWorldPos);
      vec3 viewDirLocal = normalize(viewDirWorld * vInvScale);
      volDens = 0.0;
      const int STEPS = 6;
      for (int i = 0; i < STEPS; i += 1) {
        float t = (float(i) + 0.5) / float(STEPS) - 0.5;
        vec3 sp = p + viewDirLocal * t * 0.8;
        volDens += densityQuality(sp);
      }
      volDens /= float(STEPS);
      volDens = clamp(volDens, 0.0, 1.0);
      dens = volDens;
    }

    vec3 nLocal = uQuality > 0.5 ? tetraNormal(p) : analyticNormal(p);
    vec3 n = normalize(nLocal * vInvScale);

    vec3 sunDir = normalize(uSunDirection);
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float sunFacing = clamp(dot(n, sunDir) * 0.5 + 0.5, 0.0, 1.0);
    float sunLit = smoothstep(0.2, 0.95, sunFacing);

    float cosTheta = dot(viewDir, sunDir);
    float phase = henyeyGreenstein(cosTheta, uPhaseG);
    float rim = phase * smoothstep(0.15, 0.75, dens);

    vec3 localSunDir = normalize(vLocalSunDir);
    float occ;
    if (uQuality > 0.5) {
      occ = 1.0;
      for (int i = 1; i <= 4; i += 1) {
        occ -= shadowOcc(p + localSunDir * float(i) * 0.07) * 0.18;
      }
    } else {
      occ = 1.0 - shadowOcc(p + localSunDir * 0.1) * 0.35;
    }
    occ = clamp(occ, 0.0, 1.0);

    float wisp = hash13(vec3(vWorldPos.x * 0.04, vWorldPos.z * 0.04, vSeed * 9.0 + uTime * 0.05));
    float heightNorm = clamp((p.y - BASE_Y) / (TOP_Y - BASE_Y), 0.0, 1.0);
    float topFade = mix(1.0, 0.65 + wisp * 0.25, smoothstep(0.45, 0.98, heightNorm));
    float bellyShadow = mix(0.78, 1.0, smoothstep(BASE_Y, BASE_Y + 0.22, p.y));

    vec3 ambient = mix(uGroundColor, uSkyColor, heightNorm);
    float selfShadow = mix(0.76, 1.0, heightNorm) * mix(0.9, 1.08, sunLit) * (0.55 + occ * 0.45);

    vec3 shadowTint = uShadowColor * mix(1.15, 0.85, vDensity) * mix(1.0, 0.55, uDarkness);
    vec3 col = uColor * mix(0.82, 1.14, heightNorm) * (0.94 + wisp * 0.06);
    col = mix(col * shadowTint, col, selfShadow);
    col += uSunColor * rim * (0.1 + 0.22 * heightNorm) * mix(1.0, 0.4, uDarkness);
    col += ambient * 0.18 * mix(1.0, 0.5, uDarkness);
    col *= mix(vec3(0.92, 0.95, 1.04), vec3(1.0), heightNorm);
    col += uSunColor * pow(sunFacing, 16.0) * heightNorm * 0.06;
    col = mix(col, col * vec3(0.32, 0.36, 0.44), uDarkness * 0.85);

    float alpha = volDens * (0.88 + wisp * 0.12) * topFade * bellyShadow * uOpacity * vFade;
    alpha *= 0.6 + vDensity * 0.4;

    if (uFogEnabled > 0.5) {
      float fogDepth = length(vWorldPos - cameraPosition);
      float fogFactor = smoothstep(uFogNear, uFogFar, fogDepth);
      col = mix(col, uFogColor, fogFactor);
      alpha *= 1.0 - fogFactor;
    }

    gl_FragColor = vec4(col, alpha);
  }
`;

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

function createPuffMaterial(color, opacity, roundness, softness, quality) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color.clone() },
      uOpacity: { value: opacity },
      uRoundness: { value: roundness },
      uSoftness: { value: softness },
      uFogColor: { value: new THREE.Color(0x9fb7d5) },
      uFogNear: { value: 300 },
      uFogFar: { value: 700 },
      uFogEnabled: { value: 1 },
      uSunDirection: { value: new THREE.Vector3(...DEFAULT_SUN_POSITION).normalize() },
      uSunColor: { value: new THREE.Color(0xfff0d2) },
      uShadowColor: { value: new THREE.Color(0xc9d6e8) },
      uSkyColor: { value: new THREE.Color(0x9fc4ff) },
      uGroundColor: { value: new THREE.Color(0xb8a890) },
      uTime: { value: 0 },
      uPhaseG: { value: 0.6 },
      uQuality: { value: quality ? 1 : 0 },
      uDarkness: { value: 0 },
    },
    vertexShader: PUFF_VERT,
    fragmentShader: PUFF_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.FrontSide,
    extensions: { derivatives: true },
  });
}

function createCloudMask({
  seed = 42,
  coverage = 0.5,
  noiseScale = 0.028,
  noiseOctaves = 5,
  noiseJitter = 0.08,
} = {}) {
  const rng = (() => {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), s | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();

  const lattice = new Float32Array(MASK_SIZE * MASK_SIZE);
  const grad = (ix, iy) => {
    const h = Math.imul(ix ^ iy, 0x45d9f3b) >>> 0;
    return (h & 255) / 255;
  };
  const smooth = (t) => t * t * (3 - 2 * t);

  const vnoise = (x, y) => {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const tx = smooth(x - x0);
    const ty = smooth(y - y0);
    const a = grad(x0, y0);
    const b = grad(x0 + 1, y0);
    const c = grad(x0, y0 + 1);
    const d = grad(x0 + 1, y0 + 1);
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
  };

  for (let y = 0; y < MASK_SIZE; y += 1) {
    for (let x = 0; x < MASK_SIZE; x += 1) {
      let v = 0;
      let amp = 0.55;
      let freq = noiseScale;
      const octaves = Math.max(1, Math.min(7, Math.round(noiseOctaves)));
      for (let o = 0; o < octaves; o += 1) {
        v += amp * vnoise(x * freq + seed * 0.01, y * freq + seed * 0.013);
        freq *= 1.95;
        amp *= 0.52;
      }
      lattice[y * MASK_SIZE + x] = v;
    }
  }

  const data = new Uint8Array(MASK_SIZE * MASK_SIZE);
  for (let i = 0; i < data.length; i += 1) {
    const n = lattice[i] + (rng() - 0.5) * noiseJitter;
    data[i] = n > coverage ? 255 : 0;
  }
  return data;
}

function createMaskDensity(mask) {
  const out = new Uint8Array(MASK_SIZE * MASK_SIZE);
  const weights = [
    [0, 0, 3],
    [1, 0, 2], [-1, 0, 2], [0, 1, 2], [0, -1, 2],
    [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
    [2, 0, 1], [-2, 0, 1], [0, 2, 1], [0, -2, 1],
  ];
  const maxWeight = 15;
  for (let y = 0; y < MASK_SIZE; y += 1) {
    for (let x = 0; x < MASK_SIZE; x += 1) {
      let sum = 0;
      for (const [ox, oy, w] of weights) {
        const sx = floorMod(x + ox, MASK_SIZE);
        const sy = floorMod(y + oy, MASK_SIZE);
        if (mask[sy * MASK_SIZE + sx] >= 24) sum += w;
      }
      out[y * MASK_SIZE + x] = Math.round((sum / maxWeight) * 255);
    }
  }
  return out;
}

function cellHash(u, v, salt = 0) {
  let h = Math.imul((u + salt * 17) ^ (v + salt * 31), 0x45d9f3b) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  return (h & 0xffff) / 0xffff;
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
  const sky = new Sky();
  sky.scale.setScalar(scale);
  const u = sky.material.uniforms;
  u.turbidity.value = turbidity;
  u.rayleigh.value = rayleigh;
  u.mieCoefficient.value = mieCoefficient;
  u.mieDirectionalG.value = mieDirectionalG;
  u.sunPosition.value.copy(vectorFrom(sunPosition).normalize());
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

export class CloudLayer {
  constructor({ scene, camera, sky = null, ...options } = {}) {
    this.scene = scene;
    this.camera = camera;
    this.sky = sky;
    this.params = normalizeCloudParams(options);
    this._mask = null;
    this._maskDensity = null;
    this._scroll = new THREE.Vector2(0, 0);
    this._layers = [];
    this._piece = 12;
    this._radius = 75;
    this._maxInstances = (this._radius * 2 + 1) ** 2;
    this._lastCell = { x: NaN, z: NaN, sx: NaN, sz: NaN };
    this._dummy = new THREE.Object3D();
    this._sunColor = new THREE.Color();
    this._shadowColor = new THREE.Color();
    this._sunWarmColor = new THREE.Color(0xffb36f);
    this._shadowDayColor = new THREE.Color(0xd6e2f2);
    this._autoTintColor = new THREE.Color();
    this._skyColor = new THREE.Color();
    this._groundColor = new THREE.Color();
    this._skyDayColor = new THREE.Color(0x9fc4ff);
    this._skyNightColor = new THREE.Color(0x1a2540);
    this._groundDayColor = new THREE.Color(0xb8a890);
    this._groundNightColor = new THREE.Color(0x2a2630);
    this._time = 0;
    this._ready = false;
  }

  init() {
    if (!this.scene || !this.camera) {
      throw new Error('metaverse-sky: CloudLayer requires `scene` and `camera`');
    }
    if (this._ready) return this;
    this._regenMask();
    this._applyPieceSize();

    for (let li = 0; li < CLOUD_LAYERS.length; li += 1) {
      const layer = CLOUD_LAYERS[li];
      const mat = createPuffMaterial(
        this.params.cloudColor,
        this.params.opacity * layer.opacityMul,
        this.params.roundness,
        this.params.softness,
        this.params.quality,
      );

      const geo = new THREE.BoxGeometry(1, 1, 1);
      geo.setAttribute(
        'instanceSeed',
        new THREE.InstancedBufferAttribute(new Float32Array(this._maxInstances), 1),
      );
      geo.setAttribute(
        'instanceFade',
        new THREE.InstancedBufferAttribute(new Float32Array(this._maxInstances), 1),
      );
      geo.setAttribute(
        'instanceDensity',
        new THREE.InstancedBufferAttribute(new Float32Array(this._maxInstances), 1),
      );
      geo.setAttribute(
        'instanceRand',
        new THREE.InstancedBufferAttribute(new Float32Array(this._maxInstances * 4), 4),
      );

      const mesh = new THREE.InstancedMesh(geo, mat, this._maxInstances);
      mesh.count = 0;
      mesh.frustumCulled = false;
      mesh.renderOrder = 2 + li;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

      const group = new THREE.Group();
      group.add(mesh);
      this.scene.add(group);

      this._layers.push({ layer, group, mesh, material: mat });
    }

    this._ready = true;
    this._syncVisibility();
    this._syncFog();
    this._syncSunLighting();
    this._updateGroupPositions(this.camera.position);
    this._rebuildAll(true);
    return this;
  }

  _applyPieceSize() {
    this._piece = 12 * (6 / Math.max(3, this.params.tile));
    this._radius = Math.min(75, Math.ceil((this.camera?.far ?? 900) / this._piece) + 2);
    this._maxInstances = (this._radius * 2 + 1) ** 2;
  }

  _syncVisibility() {
    const on = this.params.enabled && this._ready;
    for (const { group } of this._layers) group.visible = on;
  }

  _applyColors() {
    for (const { material } of this._layers) material.uniforms.uColor.value.copy(this.params.cloudColor);
  }

  _applyOpacity() {
    const p = this.params;
    for (const { layer, material } of this._layers) material.uniforms.uOpacity.value = p.opacity * layer.opacityMul;
  }

  _applyShaderStyle() {
    const p = this.params;
    const desiredSide = p.quality ? THREE.DoubleSide : THREE.FrontSide;
    for (const { material } of this._layers) {
      material.uniforms.uRoundness.value = p.roundness;
      material.uniforms.uSoftness.value = p.softness;
      material.uniforms.uQuality.value = p.quality ? 1 : 0;
      material.uniforms.uDarkness.value = p.darkness;
      if (material.side !== desiredSide) {
        material.side = desiredSide;
        material.needsUpdate = true;
      }
    }
  }

  setSunDirection(direction) {
    if (!this.sky) return this;
    const d = vectorFrom(direction).normalize();
    this.sky.material.uniforms.sunPosition.value.copy(d);
    this._syncSunLighting();
    return this;
  }

  setWindDirection(direction) {
    const vec = Array.isArray(direction)
      ? direction
      : [Number(direction?.x) || 0, Number(direction?.y) || 0];
    const len = Math.hypot(vec[0], vec[1]) || 1;
    const nx = vec[0] / len;
    const ny = vec[1] / len;
    this.params.windDirection = (THREE.MathUtils.radToDeg(Math.atan2(ny, nx)) % 360 + 360) % 360;
    return this;
  }

  setWindSpeed(speed) {
    this.params.windSpeed = Math.max(0, speed);
    return this;
  }

  _regenMask() {
    const p = this.params;
    this._mask = createCloudMask({
      seed: p.noiseSeed,
      coverage: p.coverage,
      noiseScale: p.noiseScale,
      noiseOctaves: p.noiseOctaves,
      noiseJitter: p.noiseJitter,
    });
    this._maskDensity = createMaskDensity(this._mask);
  }

  _syncFog() {
    const fog = this.scene?.fog;
    const enabled = fog ? 1 : 0;
    for (const { material } of this._layers) {
      const u = material.uniforms;
      u.uFogEnabled.value = enabled;
      if (fog) {
        u.uFogColor.value.copy(fog.color);
        u.uFogNear.value = fog.near;
        u.uFogFar.value = fog.far;
      }
    }
  }

  _syncSunLighting() {
    if (!this.sky) return;
    const sun = this.sky.material.uniforms.sunPosition.value;
    const sunHeight = clamp(sun.y, -0.25, 1);
    const day = smoothstep(-0.08, 0.22, sunHeight);
    const warmth = 1 - smoothstep(0.12, 0.62, sunHeight);
    this._sunColor.setHex(0xffffff).lerp(this._sunWarmColor, warmth * 0.55);
    this._shadowColor.setHex(0x9fb7d5).lerp(this._shadowDayColor, day * 0.72);
    this._skyColor.copy(this._skyNightColor).lerp(this._skyDayColor, day);
    this._groundColor.copy(this._groundNightColor).lerp(this._groundDayColor, day * (1 - warmth * 0.3));

    for (const { material } of this._layers) {
      const u = material.uniforms;
      u.uSunDirection.value.copy(sun).normalize();
      u.uSunColor.value.copy(this._sunColor).multiplyScalar(0.35 + day * 0.65);
      u.uShadowColor.value.copy(this._shadowColor).multiplyScalar(0.75 + day * 0.25);
      u.uSkyColor.value.copy(this._skyColor).multiplyScalar(0.4 + day * 0.6);
      u.uGroundColor.value.copy(this._groundColor).multiplyScalar(0.3 + day * 0.7);
      u.uTime.value = this._time;
    }
  }

  _scrollState() {
    const piece = this._piece;
    return {
      cellX: Math.floor(this._scroll.x / piece),
      cellZ: Math.floor(this._scroll.y / piece),
      fracX: this._scroll.x - Math.floor(this._scroll.x / piece) * piece,
      fracZ: this._scroll.y - Math.floor(this._scroll.y / piece) * piece,
    };
  }

  _needsRebuild(cam) {
    const piece = this._piece;
    const cellX = Math.floor(cam.x / piece);
    const cellZ = Math.floor(cam.z / piece);
    const s = this._scrollState();
    const last = this._lastCell;
    if (cellX !== last.x || cellZ !== last.z || s.cellX !== last.sx || s.cellZ !== last.sz) {
      last.x = cellX;
      last.z = cellZ;
      last.sx = s.cellX;
      last.sz = s.cellZ;
      return true;
    }
    return false;
  }

  _groupOrigin(cam) {
    const piece = this._piece;
    const cellX = Math.floor(cam.x / piece);
    const cellZ = Math.floor(cam.z / piece);
    const s = this._scrollState();
    return {
      x: cellX * piece - s.fracX,
      z: cellZ * piece - s.fracZ,
      cellX,
      cellZ,
      scrollGX: s.cellX,
      scrollGZ: s.cellZ,
    };
  }

  _rebuildLayer(layerState, cam) {
    const { layer, mesh } = layerState;
    const piece = this._piece;
    const radius = this._radius;
    const mask = this._mask;
    const density = this._maskDensity;
    const g = this._groupOrigin(cam);
    const seeds = mesh.geometry.getAttribute('instanceSeed');
    const fades = mesh.geometry.getAttribute('instanceFade');
    const densAttr = mesh.geometry.getAttribute('instanceDensity');
    const rands = mesh.geometry.getAttribute('instanceRand');
    const fadeStart = Math.max(1, radius - Math.max(6, Math.min(14, radius * 0.16)));
    let count = 0;

    for (let dz = -radius; dz <= radius; dz += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        const radial = Math.hypot(dx + 0.5, dz + 0.5);
        if (radial > radius + 0.35) continue;

        const tu = floorMod(g.cellX + dx + g.scrollGX + layer.phaseU, MASK_SIZE);
        const tv = floorMod(g.cellZ + dz + g.scrollGZ + layer.phaseV, MASK_SIZE);
        const idx = tv * MASK_SIZE + tu;
        if (mask[idx] < 24) continue;

        const localDensity = density[idx] / 255;
        if (localDensity < 0.2 && cellHash(tu, tv, layer.phaseU + 11) > 0.35) continue;

        const h0 = cellHash(tu, tv, layer.phaseU);
        const h1 = cellHash(tu, tv, layer.phaseV + 3);
        const h2 = cellHash(tu, tv, layer.phaseU + layer.phaseV);
        const h3 = cellHash(tu, tv, layer.phaseU + layer.phaseV + 19);
        const edgeFade = smoothstep(radius + 0.35, fadeStart, radial);
        const densityFade = smoothstep(0.06, 0.72, localDensity);
        const instanceFade = edgeFade * (0.46 + densityFade * 0.54);
        if (instanceFade <= 0.015) continue;

        const wx = (g.cellX + dx) * piece + piece * (0.42 + h0 * 0.16);
        const wz = (g.cellZ + dz) * piece + piece * (0.42 + h1 * 0.16);
        const scale = this.params.puffScale;
        const layerLift = this.params.layerHeight;
        const mass = 0.82 + densityFade * 0.24;
        const sx = piece * (3.65 + h0 * 1.18 + h3 * 0.35) * scale * mass;
        const sy = layer.thickness * layerLift * (1.68 + h2 * 0.95 + densityFade * 0.24) * scale;
        const sz = piece * (3.3 + h1 * 1.18 + (1 - h3) * 0.28) * scale * mass;

        this._dummy.position.set(
          wx - g.x,
          this.params.altitude + layer.yOff * layerLift + sy * 0.5,
          wz - g.z,
        );
        this._dummy.rotation.set(0, 0, 0);
        this._dummy.scale.set(sx, sy, sz);
        this._dummy.updateMatrix();
        mesh.setMatrixAt(count, this._dummy.matrix);
        seeds.setX(count, h2);
        fades.setX(count, instanceFade);
        densAttr.setX(count, localDensity);
        rands.setXYZW(
          count,
          cellHash(tu, tv, layer.phaseU + 7),
          cellHash(tu, tv, layer.phaseV + 13),
          cellHash(tu, tv, layer.phaseU + layer.phaseV + 23),
          cellHash(tu, tv, layer.phaseU * 2 + 5),
        );
        count += 1;
        if (count >= this._maxInstances) break;
      }
      if (count >= this._maxInstances) break;
    }

    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    seeds.needsUpdate = true;
    fades.needsUpdate = true;
    densAttr.needsUpdate = true;
    rands.needsUpdate = true;
  }

  _rebuildAll(force = false) {
    if (!this._ready) return;
    const cam = this.camera.position;
    if (!force && !this._needsRebuild(cam)) return;
    for (const layerState of this._layers) this._rebuildLayer(layerState, cam);
  }

  _updateGroupPositions(cam) {
    const g = this._groupOrigin(cam);
    for (const { group } of this._layers) group.position.set(g.x, 0, g.z);
  }

  getAtmosphereSettings() {
    const p = this.params;
    return {
      cloudsEnabled: p.enabled,
      cloudOpacity: p.opacity,
      cloudAltitude: p.altitude,
      cloudWindSpeed: p.windSpeed,
      cloudWindDirection: p.windDirection,
      cloudTile: p.tile,
      cloudColor: p.cloudColor.getHex(),
      cloudAutoTint: p.autoTint,
      cloudPuffScale: p.puffScale,
      cloudLayerHeight: p.layerHeight,
      cloudCoverage: p.coverage,
      cloudNoiseSeed: p.noiseSeed,
      cloudNoiseScale: p.noiseScale,
      cloudNoiseOctaves: p.noiseOctaves,
      cloudNoiseJitter: p.noiseJitter,
      cloudRoundness: p.roundness,
      cloudSoftness: p.softness,
      cloudDarkness: p.darkness,
      cloudQuality: p.quality,
    };
  }

  applyAtmosphereSettings(data = {}) {
    const p = this.params;
    const prevNoise = {
      seed: p.noiseSeed,
      coverage: p.coverage,
      noiseScale: p.noiseScale,
      noiseOctaves: p.noiseOctaves,
      noiseJitter: p.noiseJitter,
    };

    if (data.cloudsEnabled != null) p.enabled = !!data.cloudsEnabled;
    if (data.cloudOpacity != null) p.opacity = data.cloudOpacity;
    if (data.cloudAltitude != null) p.altitude = data.cloudAltitude;
    if (data.cloudWindSpeed != null) p.windSpeed = data.cloudWindSpeed;
    if (data.cloudWindDirection != null) p.windDirection = data.cloudWindDirection;
    if (data.cloudTile != null) p.tile = data.cloudTile;
    if (data.cloudColor != null) p.cloudColor.set(data.cloudColor);
    if (data.cloudAutoTint != null) p.autoTint = !!data.cloudAutoTint;
    if (data.cloudPuffScale != null) p.puffScale = data.cloudPuffScale;
    if (data.cloudLayerHeight != null) p.layerHeight = data.cloudLayerHeight;
    if (data.cloudCoverage != null) p.coverage = data.cloudCoverage;
    if (data.cloudNoiseSeed != null) p.noiseSeed = data.cloudNoiseSeed;
    if (data.cloudNoiseScale != null) p.noiseScale = data.cloudNoiseScale;
    if (data.cloudNoiseOctaves != null) p.noiseOctaves = data.cloudNoiseOctaves;
    if (data.cloudNoiseJitter != null) p.noiseJitter = data.cloudNoiseJitter;
    if (data.cloudRoundness != null) p.roundness = data.cloudRoundness;
    if (data.cloudSoftness != null) p.softness = data.cloudSoftness;
    if (data.cloudDarkness != null) p.darkness = data.cloudDarkness;
    if (data.cloudQuality != null) p.quality = data.cloudQuality;

    const noiseChanged = prevNoise.seed !== p.noiseSeed
      || prevNoise.coverage !== p.coverage
      || prevNoise.noiseScale !== p.noiseScale
      || prevNoise.noiseOctaves !== p.noiseOctaves
      || prevNoise.noiseJitter !== p.noiseJitter;
    if (noiseChanged) this._regenMask();

    this._applyPieceSize();
    this._applyOpacity();
    this._applyColors();
    this._applyShaderStyle();
    this._syncVisibility();
    this._lastCell = { x: NaN, z: NaN, sx: NaN, sz: NaN };
    this._rebuildAll(true);
    return this;
  }

  update(dt) {
    if (!this._ready || !this._layers.length || !this.params.enabled) return;

    this._time += dt;
    const p = this.params;
    const cam = this.camera.position;
    const wind = p.windDirection * Math.PI / 180;

    this._scroll.x += Math.cos(wind) * p.windSpeed * dt;
    this._scroll.y += Math.sin(wind) * p.windSpeed * dt;

    this._updateGroupPositions(cam);
    this._rebuildAll(false);
    this._syncFog();
    this._syncSunLighting();

    if (this.sky && p.autoTint) {
      const sunY = this.sky.material.uniforms.sunPosition.value.y;
      const day = smoothstep(-0.06, 0.28, sunY);
      const goldenHour = 1 - smoothstep(0.08, 0.62, Math.max(0, sunY));
      const tint = this._autoTintColor;
      tint.setRGB(
        0.72 + day * 0.22 + goldenHour * 0.06,
        0.78 + day * 0.17 + goldenHour * 0.02,
        0.9 + day * 0.08 - goldenHour * 0.05,
      );
      for (const { material } of this._layers) material.uniforms.uColor.value.copy(tint);
    }
  }

  dispose() {
    for (const { group, mesh, material } of this._layers) {
      this.scene?.remove(group);
      mesh.geometry.dispose();
      material.dispose();
    }
    this._layers = [];
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
      ? new CloudLayer({ scene, camera, sky: this.sky, ...cloudOptions }).init()
      : null;
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
    this.clouds?.setWindDirection(direction);
    return this;
  }

  setWindSpeed(speed) {
    this.clouds?.setWindSpeed(speed);
    return this;
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
    return this;
  }

  dispose() {
    this.clouds?.dispose();
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
    this._slider('Altitude (m)', 55, 140, 1, p.altitude, (v) => c.applyAtmosphereSettings({ cloudAltitude: v }));
    this._slider('Tiling', 3, 10, 0.5, p.tile, (v) => c.applyAtmosphereSettings({ cloudTile: v }));

    this._section('Cloud shape');
    this._checkbox('High quality (volumetric)', p.quality > 0, (on) => c.applyAtmosphereSettings({ cloudQuality: on ? 1 : 0 }));
    this._slider('Puff scale', 0.5, 2, 0.05, p.puffScale, (v) => c.applyAtmosphereSettings({ cloudPuffScale: v }));
    this._slider('Layer height', 0.5, 2, 0.05, p.layerHeight, (v) => c.applyAtmosphereSettings({ cloudLayerHeight: v }));
    this._slider('Corner roundness', 0.05, 0.35, 0.01, p.roundness, (v) => c.applyAtmosphereSettings({ cloudRoundness: v }));
    this._slider('Edge softness', 0, 1, 0.01, p.softness, (v) => c.applyAtmosphereSettings({ cloudSoftness: v }));
    this._slider('Darkness', 0, 1, 0.01, p.darkness, (v) => c.applyAtmosphereSettings({ cloudDarkness: v }));

    this._section('Cloud noise');
    this._slider('Coverage', 0.2, 0.8, 0.01, p.coverage, (v) => c.applyAtmosphereSettings({ cloudCoverage: v }));
    this._slider('Pattern scale', 0.01, 0.08, 0.001, p.noiseScale, (v) => c.applyAtmosphereSettings({ cloudNoiseScale: v }));
    this._slider('Detail (octaves)', 3, 7, 1, p.noiseOctaves, (v) => c.applyAtmosphereSettings({ cloudNoiseOctaves: v }));
    this._slider('Jitter', 0, 0.2, 0.005, p.noiseJitter, (v) => c.applyAtmosphereSettings({ cloudNoiseJitter: v }));
    this._slider('Seed', 0, 999, 1, p.noiseSeed, (v) => c.applyAtmosphereSettings({ cloudNoiseSeed: v }));

    this._section('Cloud wind');
    this._slider('Speed', 0, 0.15, 0.001, p.windSpeed, (v) => c.applyAtmosphereSettings({ cloudWindSpeed: v }));
    this._slider('Direction', 0, 360, 1, p.windDirection, (v) => c.applyAtmosphereSettings({ cloudWindDirection: v }));

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
