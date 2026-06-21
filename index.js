/**
 * metaverse-sky - Three.js WebGPU/TSL sky, sun helpers, and procedural voxel clouds.
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

function hash13Node(p) {
  const q = fract(p.mul(0.1031));
  const d = dot(q, q.yzx.add(33.33));
  return fract(q.x.add(q.y).mul(q.z).add(d));
}

function vnoise3Node(p) {
  const i = floor(p);
  let f = fract(p);
  f = f.mul(f).mul(vec3(3.0).sub(f.mul(2.0)));
  const n000 = hash13Node(i);
  const n100 = hash13Node(i.add(vec3(1.0, 0.0, 0.0)));
  const n010 = hash13Node(i.add(vec3(0.0, 1.0, 0.0)));
  const n110 = hash13Node(i.add(vec3(1.0, 1.0, 0.0)));
  const n001 = hash13Node(i.add(vec3(0.0, 0.0, 1.0)));
  const n101 = hash13Node(i.add(vec3(1.0, 0.0, 1.0)));
  const n011 = hash13Node(i.add(vec3(0.0, 1.0, 1.0)));
  const n111 = hash13Node(i.add(vec3(1.0, 1.0, 1.0)));
  return mix(
    mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
    mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
    f.z,
  );
}

function createPuffMaterial(color, opacity, roundness, softness, quality) {
  const material = new THREE.MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthWrite = false;
  material.side = quality ? THREE.DoubleSide : THREE.FrontSide;
  material.toneMapped = false;

  const u = {
    uColor: assignUniform(material, 'uColor', uniform(color.clone())),
    uOpacity: assignUniform(material, 'uOpacity', uniform(opacity)),
    uRoundness: assignUniform(material, 'uRoundness', uniform(roundness)),
    uSoftness: assignUniform(material, 'uSoftness', uniform(softness)),
    uFogColor: assignUniform(material, 'uFogColor', uniform(new THREE.Color(0x9fb7d5))),
    uFogNear: assignUniform(material, 'uFogNear', uniform(300)),
    uFogFar: assignUniform(material, 'uFogFar', uniform(700)),
    uFogEnabled: assignUniform(material, 'uFogEnabled', uniform(1)),
    uSunDirection: assignUniform(material, 'uSunDirection', uniform(new THREE.Vector3(...DEFAULT_SUN_POSITION).normalize())),
    uSunColor: assignUniform(material, 'uSunColor', uniform(new THREE.Color(0xfff0d2))),
    uShadowColor: assignUniform(material, 'uShadowColor', uniform(new THREE.Color(0xc9d6e8))),
    uSkyColor: assignUniform(material, 'uSkyColor', uniform(new THREE.Color(0x9fc4ff))),
    uGroundColor: assignUniform(material, 'uGroundColor', uniform(new THREE.Color(0xb8a890))),
    uTime: assignUniform(material, 'uTime', uniform(0)),
    uPhaseG: assignUniform(material, 'uPhaseG', uniform(0.6)),
    uQuality: assignUniform(material, 'uQuality', uniform(quality ? 1 : 0)),
    uDarkness: assignUniform(material, 'uDarkness', uniform(0)),
  };

  const iSeed = attribute('instanceSeed', 'float');
  const iFade = attribute('instanceFade', 'float');
  const iDensity = attribute('instanceDensity', 'float');
  const iRand = attribute('instanceRand', 'vec4');

  material.fragmentNode = Fn(() => {
    const p = positionLocal;
    const q = vec3(p.x.mul(0.92 + 0.14).sub(iSeed.mul(0.14).mul(p.x)), p.y.mul(1.12), p.z.mul(0.92).add(iSeed.mul(0.14).mul(p.z)));
    const ellipsoid = length(q).sub(0.58).sub(iDensity.mul(0.06));
    const base = oneMinus(nodeSmoothstep(u.uSoftness.mul(0.18).add(0.03).negate(), u.uSoftness.mul(0.18).add(0.03), ellipsoid));
    const wob = vnoise3Node(p.mul(4.0).add(vec3(iSeed.mul(9.0), iSeed.mul(5.0), u.uTime.mul(0.1))));
    const dens = nodeClamp(base.mul(0.82).add(wob.mul(0.18)), 0.0, 1.0);
    If(dens.lessThan(0.002), () => { Discard(); });

    const heightNorm = nodeClamp(p.y.add(0.42).div(0.9), 0.0, 1.0);
    const n = normalize(vec3(p.x.mul(0.65), p.y.mul(1.1).add(0.12), p.z.mul(0.65)));
    const sunDir = normalize(u.uSunDirection);
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const sunFacing = nodeClamp(dot(n, sunDir).mul(0.5).add(0.5), 0.0, 1.0);
    const sunLit = nodeSmoothstep(0.2, 0.95, sunFacing);
    const phase = pow(max(dot(viewDir, sunDir), 0.0), 4.0).mul(nodeSmoothstep(0.15, 0.75, dens));
    const wisp = hash13Node(vec3(positionWorld.x.mul(0.04), positionWorld.z.mul(0.04), iSeed.mul(9.0).add(u.uTime.mul(0.05))));
    const topFade = mix(1.0, float(0.65).add(wisp.mul(0.25)), nodeSmoothstep(0.45, 0.98, heightNorm));
    const bellyShadow = mix(0.78, 1.0, nodeSmoothstep(-0.42, -0.20, p.y));
    const ambient = mix(u.uGroundColor, u.uSkyColor, heightNorm);
    const selfShadow = mix(0.76, 1.0, heightNorm).mul(mix(0.9, 1.08, sunLit));
    const shadowTint = u.uShadowColor.mul(mix(1.15, 0.85, iDensity)).mul(mix(1.0, 0.55, u.uDarkness));
    let col = u.uColor.mul(mix(0.82, 1.14, heightNorm)).mul(float(0.94).add(wisp.mul(0.06)));
    col = mix(col.mul(shadowTint), col, selfShadow);
    col = col.add(u.uSunColor.mul(phase).mul(float(0.1).add(heightNorm.mul(0.22))).mul(mix(1.0, 0.4, u.uDarkness)));
    col = col.add(ambient.mul(0.18).mul(mix(1.0, 0.5, u.uDarkness)));
    col = col.add(u.uSunColor.mul(pow(sunFacing, 16.0)).mul(heightNorm).mul(0.06));
    col = mix(col, col.mul(vec3(0.32, 0.36, 0.44)), u.uDarkness.mul(0.85));
    let alpha = dens.mul(float(0.88).add(wisp.mul(0.12))).mul(topFade).mul(bellyShadow).mul(u.uOpacity).mul(iFade).mul(float(0.6).add(iDensity.mul(0.4)));
    const fogDepth = length(positionWorld.sub(cameraPosition));
    const fogFactor = nodeSmoothstep(u.uFogNear, u.uFogFar, fogDepth).mul(u.uFogEnabled);
    col = mix(col, u.uFogColor, fogFactor);
    alpha = alpha.mul(oneMinus(fogFactor));
    return vec4(col, alpha);
  })();

  return material;
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
      ? new CloudLayer({ scene, camera, sky: this.sky, ...cloudOptions }).init()
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
    this.clouds?.setWindDirection(direction);
    this.precipitation?.setWindDirection(direction);
    return this;
  }

  setWindSpeed(speed) {
    this.clouds?.setWindSpeed(speed);
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
