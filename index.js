/**
 * metaverse-sky - Three.js WebGPU/TSL sky, sun helpers, and fast 2.5D sky-volume clouds.
 *
 * Peer dependency: the host app must resolve `three` and `three/addons/`.
 */

import * as THREE from 'three/webgpu';
import {
  Fn, uniform, texture, texture3D, instancedBufferAttribute,
  vec2, vec3, vec4, float, int,
  abs, max, min, mix, clamp as nodeClamp, smoothstep as nodeSmoothstep, dot, normalize, length, pow, exp, sin, sqrt, fract, oneMinus,
  If, Loop, Break, Discard, positionLocal, positionWorld, positionView, cameraPosition, uv, screenCoordinate,
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
export const DEFAULT_SUN_BALL_DISTANCE = DEFAULT_SKY_SCALE * 0.92;
export const DEFAULT_SUN_BALL_RADIUS = 12;
export const DEFAULT_SUN_BALL_COLOR = 0xfff7df;
export const DEFAULT_SUN_BALL_HORIZON_COLOR = 0xff6a24;
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
  coverage: 0.65,
  noiseScale: 0.012,
  detailStrength: 0,
  holes: 0.5,
  cloudType: 0.55,
  cloudBanks: 0.6,
  sharpness: 0.35,
  wispiness: 0.45,
  darkness: 0,
  cirrus: 0.35,
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
  cloudType: CLOUD_DEFAULTS.cloudType,
  cloudBanks: CLOUD_DEFAULTS.cloudBanks,
  sharpness: CLOUD_DEFAULTS.sharpness,
  wispiness: CLOUD_DEFAULTS.wispiness,
  darkness: CLOUD_DEFAULTS.darkness,
  cirrus: CLOUD_DEFAULTS.cirrus,
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

function smooth3(t) {
  return t * t * (3 - 2 * t);
}

// Per-octave lookup tables: gradients / Worley feature offsets only depend on
// the wrapped cell index, so precomputing them once per octave removes nearly
// all hash work from the bake's hot loops.
const _gradTables = new Map();
const _featTables = new Map();
const _wrapTables = new Map();

// wrap[v + 1] === tmod(v, cells) for v in [-1, cells + 1] — replaces modulo
// arithmetic in the bake's hot loops.
function getWrapTable(cells) {
  let tbl = _wrapTables.get(cells);
  if (tbl) return tbl;
  tbl = new Int32Array(cells + 3);
  for (let v = -1; v <= cells + 1; v++) tbl[v + 1] = tmod(v, cells);
  _wrapTables.set(cells, tbl);
  return tbl;
}

function getGradTable(cells) {
  let tbl = _gradTables.get(cells);
  if (tbl) return tbl;
  tbl = new Uint8Array(cells * cells * cells);
  let i = 0;
  for (let z = 0; z < cells; z++) {
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        tbl[i++] = Math.floor(hash31(x, y, z) * _GRAD.length) % _GRAD.length;
      }
    }
  }
  _gradTables.set(cells, tbl);
  return tbl;
}

function getFeatTable(cells) {
  let tbl = _featTables.get(cells);
  if (tbl) return tbl;
  tbl = new Float32Array(cells * cells * cells * 3);
  let i = 0;
  for (let z = 0; z < cells; z++) {
    for (let y = 0; y < cells; y++) {
      for (let x = 0; x < cells; x++) {
        tbl[i++] = hash31(x, y, z);
        tbl[i++] = hash31(y, z, x);
        tbl[i++] = hash31(z, x, y);
      }
    }
  }
  _featTables.set(cells, tbl);
  return tbl;
}

// Proper gradient (Perlin) 3D noise — smooth and free of value-noise blockiness.
// `cells` is the lattice period; lattice points wrap modulo it so the result is
// seamlessly tileable.
function perlin3(x, y, z, cells, grad, wrap) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = smooth3(xf), v = smooth3(yf), w = smooth3(zf);
  const wx0 = wrap[xi + 1], wy0 = wrap[yi + 1], wz0 = wrap[zi + 1];
  const wx1 = wrap[xi + 2], wy1 = wrap[yi + 2], wz1 = wrap[zi + 2];
  const row0 = wy0 * cells, row1 = wy1 * cells;
  const slab0 = wz0 * cells * cells, slab1 = wz1 * cells * cells;
  const g000 = _GRAD[grad[wx0 + row0 + slab0]], g100 = _GRAD[grad[wx1 + row0 + slab0]];
  const g010 = _GRAD[grad[wx0 + row1 + slab0]], g110 = _GRAD[grad[wx1 + row1 + slab0]];
  const g001 = _GRAD[grad[wx0 + row0 + slab1]], g101 = _GRAD[grad[wx1 + row0 + slab1]];
  const g011 = _GRAD[grad[wx0 + row1 + slab1]], g111 = _GRAD[grad[wx1 + row1 + slab1]];
  const d000 = g000[0] * xf + g000[1] * yf + g000[2] * zf;
  const d100 = g100[0] * (xf - 1) + g100[1] * yf + g100[2] * zf;
  const d010 = g010[0] * xf + g010[1] * (yf - 1) + g010[2] * zf;
  const d110 = g110[0] * (xf - 1) + g110[1] * (yf - 1) + g110[2] * zf;
  const d001 = g001[0] * xf + g001[1] * yf + g001[2] * (zf - 1);
  const d101 = g101[0] * (xf - 1) + g101[1] * yf + g101[2] * (zf - 1);
  const d011 = g011[0] * xf + g011[1] * (yf - 1) + g011[2] * (zf - 1);
  const d111 = g111[0] * (xf - 1) + g111[1] * (yf - 1) + g111[2] * (zf - 1);
  const x00 = d000 + (d100 - d000) * u, x10 = d010 + (d110 - d010) * u;
  const x01 = d001 + (d101 - d001) * u, x11 = d011 + (d111 - d011) * u;
  const y0 = x00 + (x10 - x00) * v, y1 = x01 + (x11 - x01) * v;
  // Perlin output is in ~[-1,1]; remap to [0,1].
  return y0 + (y1 - y0) * w + 0.5;
}

function worley3(x, y, z, cells, feat, wrap) {
  // Inverted Worley: 1 near cell feature points, 0 between. Tileable: feature
  // points live inside wrapped cells so the field matches at boundaries.
  let minD = 1e9;
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const cx = wrap[xi + dx + 1], cy = wrap[yi + dy + 1], cz = wrap[zi + dz + 1];
        const fi = (cx + cy * cells + cz * cells * cells) * 3;
        // Use unwrapped coordinates for the distance check so crossing the edge
        // still measures true distance to the (wrapped) feature point.
        const ddx = (xi + dx) + feat[fi] - x;
        const ddy = (yi + dy) + feat[fi + 1] - y;
        const ddz = (zi + dz) + feat[fi + 2] - z;
        const d = ddx * ddx + ddy * ddy + ddz * ddz;
        if (d < minD) minD = d;
      }
    }
  }
  return 1 - clamp(Math.sqrt(minD), 0, 1);
}

// FBM over the tileable noise. Inputs are normalized [0,1) coordinates within
// one texture period; `baseCells` is the number of lattice cells across that
// period for the first octave. Each octave doubles the cell count (staying an
// integer, so every octave tiles seamlessly).
function fbmPerlin(x, y, z, octaves, baseCells) {
  let total = 0, amp = 0.5, cells = baseCells, max = 0;
  for (let i = 0; i < octaves; i++) {
    total += perlin3(x * cells, y * cells, z * cells, cells, getGradTable(cells), getWrapTable(cells)) * amp;
    max += amp;
    cells *= 2;
    amp *= 0.5;
  }
  return total / max;
}

function fbmWorley(x, y, z, octaves, baseCells) {
  let total = 0, amp = 0.5, cells = baseCells, max = 0;
  for (let i = 0; i < octaves; i++) {
    total += worley3(x * cells, y * cells, z * cells, cells, getFeatTable(cells), getWrapTable(cells)) * amp;
    max += amp;
    cells *= 2;
    amp *= 0.5;
  }
  return total / max;
}

const CLOUD_NOISE_SIZE = 64;

let _cloudNoiseTexture = null;

function getCloudNoiseTexture() {
  if (_cloudNoiseTexture) return _cloudNoiseTexture;
  const size = CLOUD_NOISE_SIZE;
  // Two channels: R = Perlin-Worley shape composite, G = Worley detail for
  // edge erosion. Baking these once means the ray march does cheap trilinear
  // texture fetches instead of evaluating procedural noise per step.
  const voxels = size * size * size;
  const shapeF = new Float32Array(voxels);
  const worleyF = new Float32Array(voxels);
  let shapeMin = Infinity;
  let shapeMax = -Infinity;
  let i = 0;
  for (let z = 0; z < size; z++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        // Normalized coordinates within the (periodic) texture volume. Perlin
        // uses 4 base cells per period for large billowy masses; Worley uses 6
        // for the smaller cellular puffs carved out of them.
        const nx = x / size;
        const ny = y / size;
        const nz = z / size;
        const perlin = fbmPerlin(nx, ny, nz, 4, 4);
        const worley = fbmWorley(nx, ny, nz, 2, 6);
        // Perlin-Worley: Perlin gives the billowy base, Worley carves cells.
        const shape = perlin * 0.45 + (perlin * worley + worley * 0.25) * 0.55;
        shapeF[i] = shape;
        worleyF[i] = worley;
        if (shape < shapeMin) shapeMin = shape;
        if (shape > shapeMax) shapeMax = shape;
        i += 1;
      }
    }
  }
  // FBM sums cluster in a narrow bell curve, which makes coverage thresholds
  // nearly inert. Histogram-equalize the shape channel to a uniform [0,1]
  // distribution so the coverage slider has a predictable, near-linear bite.
  const shapeRange = Math.max(1e-6, shapeMax - shapeMin);
  const hist = new Uint32Array(256);
  for (let v = 0; v < voxels; v++) {
    hist[Math.min(255, Math.floor(((shapeF[v] - shapeMin) / shapeRange) * 256))] += 1;
  }
  const cdf = new Float32Array(256);
  let acc = 0;
  for (let b = 0; b < 256; b++) {
    acc += hist[b];
    cdf[b] = acc / voxels;
  }
  const data = new Uint8Array(voxels * 2);
  for (let v = 0; v < voxels; v++) {
    const bin = Math.min(255, Math.floor(((shapeF[v] - shapeMin) / shapeRange) * 256));
    data[v * 2] = Math.round(cdf[bin] * 255);
    data[v * 2 + 1] = Math.round(clamp(worleyF[v], 0, 1) * 255);
  }
  const tex = new THREE.Data3DTexture(data, size, size, size);
  tex.format = THREE.RGFormat;
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
// Coverage-thresholded base density (no billow/detail): the shared core of the
// view march and the cheaper sun-light taps. Two texture fetches.
function cloudBaseDensity(u, p, noise, scale) {
  // Nubis-style density model:
  //   dimensional_profile = vertical_profile * cloud_coverage
  //   density = saturate(perlin_worley_composite - (1.0 - dimensional_profile))
  const heightFraction = nodeClamp(p.y.sub(u.uAltitude).div(u.uThickness), 0.0, 1.0);
  const cloudType = nodeClamp(u.uCloudType, 0.0, 1.0);

  // Vertical profile from the Nubis slides, lightly blended by cloud type.
  const bottomGradient = pow(heightFraction, mix(2.0, 1.25, cloudType));
  const topGradient = pow(oneMinus(heightFraction), mix(1.5, 0.85, cloudType));
  const verticalProfile = nodeClamp(bottomGradient.mul(topGradient).mul(mix(9.5, 6.5, cloudType)), 0.0, 1.0);

  // Flattened sampling prevents obvious repeated vertical stacks. The wind
  // offset scrolls the noise field so the cloudscape drifts (Nubis: deform
  // the sample coordinates by wind * time).
  const cloudP = p.sub(u.uWindOffset).mul(vec3(1.0, mix(0.12, 0.28, cloudType), 1.0));

  // Optional broad coverage/influence field, equivalent to a tiny procedural
  // weather map. High enough frequency that several coverage clumps fit inside
  // the visible dome — at very low frequencies one weather blob spans the
  // whole sky and the layer reads as uniform cotton.
  const weather = noise(cloudP.mul(scale.mul(0.15))).x;
  const coverageMap = mix(1.0, nodeSmoothstep(0.32, 0.66, weather), u.uCloudBanks);
  const dimensionalProfile = verticalProfile.mul(u.uCoverage).mul(coverageMap);

  // Nubis early-out: if the cheap dimensional profile says empty space, skip
  // the shape fetch entirely (most of a partly-cloudy sky).
  const thresholded = float(0.0).toVar();
  If(dimensionalProfile.greaterThan(0.02), () => {
    // Perlin-Worley composite, baked into the R channel.
    const shapeP = cloudP.mul(scale.mul(mix(0.20, 0.30, cloudType)));
    const composite = nodeSmoothstep(0.18, 0.82, noise(shapeP).x);
    thresholded.assign(nodeClamp(composite.sub(oneMinus(dimensionalProfile)), 0.0, 1.0));
  });
  return { thresholded, cloudP, cloudType, coverageMap, dimensionalProfile };
}

// Light-tap density: 1 texture fetch. Reuses the view sample's coverage map
// (the weather field is effectively constant over a light-tap distance) but
// re-evaluates the vertical profile and shape at the tap position — those are
// what make tops lit and bottoms/far-sides shadowed.
function cloudLightDensity(u, p, noise, scale, coverageMap) {
  const heightFraction = nodeClamp(p.y.sub(u.uAltitude).div(u.uThickness), 0.0, 1.0);
  const cloudType = nodeClamp(u.uCloudType, 0.0, 1.0);
  const bottomGradient = pow(heightFraction, mix(2.0, 1.25, cloudType));
  const topGradient = pow(oneMinus(heightFraction), mix(1.5, 0.85, cloudType));
  const verticalProfile = nodeClamp(bottomGradient.mul(topGradient).mul(mix(9.5, 6.5, cloudType)), 0.0, 1.0);
  const cloudP = p.sub(u.uWindOffset).mul(vec3(1.0, mix(0.12, 0.28, cloudType), 1.0));
  const dimensionalProfile = verticalProfile.mul(u.uCoverage).mul(coverageMap);
  const density = float(0.0).toVar();
  If(dimensionalProfile.greaterThan(0.02), () => {
    const shapeP = cloudP.mul(scale.mul(mix(0.20, 0.30, cloudType)));
    const composite = nodeSmoothstep(0.18, 0.82, noise(shapeP).x);
    density.assign(nodeClamp(composite.sub(oneMinus(dimensionalProfile)), 0.0, 1.0));
  });
  return density;
}

// Returns vec2(fineDensity, coarseDensity). `p` is world-space inside the fixed
// cloud slab. `noise` samples the baked Perlin-Worley 3D texture (R = shape
// composite, G = Worley detail) — trilinear fetches instead of hundreds of
// procedural hash evaluations.
function sampleCloudDensity(u, p, noise, scale) {
  const { thresholded, cloudP, cloudType, coverageMap, dimensionalProfile } = cloudBaseDensity(u, p, noise, scale);

  const fine = float(0.0).toVar();
  const coarse = float(0.0).toVar();
  // Nubis early-out: billow/detail fetches only run where the base density
  // actually produced cloud.
  If(thresholded.greaterThan(0.0), () => {
    // Mid-frequency billow: modulates density *inside* the mass so interiors
    // get lumpy structure (and, through the density-driven shading, lumpy
    // lighting) instead of saturating into a flat fill.
    const billow = noise(cloudP.mul(scale.mul(mix(0.85, 1.25, cloudType)))).x;
    const coarseDensity = nodeClamp(thresholded.mul(mix(0.55, 1.45, billow)), 0.0, 1.0);

    // Fine erosion: strongest at the edges, but reaching partway into the
    // interior so puff cores keep some cauliflower texture.
    const detailP = cloudP.mul(scale.mul(mix(4.2, 6.0, cloudType)));
    const detail = noise(detailP).y;
    const edgeMask = oneMinus(nodeSmoothstep(0.22, 0.75, coarseDensity).mul(0.6));
    const horizonFade = nodeSmoothstep(0.06, 0.24, normalize(p.sub(cameraPosition)).y);
    const erosion = oneMinus(detail).mul(u.uHoles).mul(edgeMask).mul(horizonFade).mul(0.72);
    coarse.assign(coarseDensity);
    fine.assign(nodeClamp(coarseDensity.sub(erosion), 0.0, 1.0));
  });

  return { fine, coarse, coverageMap, dimensionalProfile };
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
const CLOUD_STEPS = 28;
const CLOUD_ABSORPTION = 7.0;

function createSkyVolumeCloudMaterial(color, opacity, coverage, darkness, detailStrength, holes, cloudType, cloudBanks, sharpness, wispiness) {
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
    uCloudType: assignUniform(material, 'uCloudType', uniform(cloudType)),
    uCloudBanks: assignUniform(material, 'uCloudBanks', uniform(cloudBanks)),
    uSharpness: assignUniform(material, 'uSharpness', uniform(sharpness)),
    uWispiness: assignUniform(material, 'uWispiness', uniform(wispiness)),
    uScale: assignUniform(material, 'uScale', uniform(0.04)),
    uSunDirection: assignUniform(material, 'uSunDirection', uniform(new THREE.Vector3(...DEFAULT_SUN_POSITION).normalize())),
    uSunColor: assignUniform(material, 'uSunColor', uniform(new THREE.Color(0xfff0d2))),
    uShadowColor: assignUniform(material, 'uShadowColor', uniform(new THREE.Color(0xc9d6e8))),
    uFogColor: assignUniform(material, 'uFogColor', uniform(new THREE.Color(0x9fb7d5))),
    uDarkness: assignUniform(material, 'uDarkness', uniform(darkness)),
    uWindOffset: assignUniform(material, 'uWindOffset', uniform(new THREE.Vector3())),
    uCirrus: assignUniform(material, 'uCirrus', uniform(CLOUD_DEFAULTS.cirrus)),
    uRadius: assignUniform(material, 'uRadius', uniform(420)),
    uThickness: assignUniform(material, 'uThickness', uniform(60)),
    uAltitude: assignUniform(material, 'uAltitude', uniform(80)),
  };

  const noiseTex = getCloudNoiseTexture();
  // Baked Perlin-Worley sampler: R = shape composite, G = Worley detail.
  const noise = (coord) => texture3D(noiseTex, coord);

  material.fragmentNode = Fn(() => {
    const px = positionWorld;
    const cam = cameraPosition;
    const viewDir = normalize(px.sub(cam));
    const dist = length(px.sub(cam));

    const scale = u.uScale;
    const sunDir = normalize(u.uSunDirection);

    // Horizon fade: clouds fill the sky above the horizon, fade out below it.
    // Broad lower-horizon fade: clouds reach the horizon, but do not wrap down
    // around/under the scene like a full enclosing sphere.
    const horizonFade = nodeSmoothstep(float(0.0), float(0.16), viewDir.y);

    // Dual-lobe Henyey-Greenstein (Nubis): a strong forward lobe for direct
    // scattering plus a wide soft lobe applied to the multiple-scattering
    // term, so dense sunlit regions glow instead of flattening to grey.
    const mu = dot(viewDir, sunDir);
    const hg = (g) => {
      const g2 = g * g;
      return float(1.0 - g2).div(pow(float(1.0 + g2).sub(mu.mul(2.0 * g)), 1.5));
    };
    const phasePrimary = hg(0.55).mul(0.28).add(0.12);
    const phaseSecondary = hg(0.15).mul(0.45).add(0.20);

    // Fixed world-altitude cloud slab. This keeps the noise locked in the sky
    // instead of making the camera fly through arbitrary sphere noise.
    const up = max(viewDir.y, float(0.08));
    const slabStart = max(u.uAltitude.sub(cam.y).div(up), float(0.0));
    // Nubis distance-based stepping: start fine at the slab entry and grow the
    // step geometrically along the ray; far/horizon entries start coarser.
    // Combined with the slab-top break below, near-zenith rays finish in ~23
    // effective steps and horizon rays in ~12.
    const distanceFactor = nodeClamp(slabStart.div(u.uRadius), 0.0, 1.0);
    const initialVerticalStep = u.uThickness.div(float(CLOUD_STEPS)).mul(mix(0.8, 1.45, distanceFactor));
    const verticalStep = initialVerticalStep.toVar();
    const stepLen = initialVerticalStep.div(up).toVar();
    const transmittance = float(1.0).toVar();
    const lightEnergy = vec3(0.0).toVar();
    // Per-pixel start jitter hides ray-step banding. It must come from screen
    // pixel coordinates: the sphere uv varies smoothly across the screen, so
    // hashing it gives neighboring pixels nearly the same phase and the step
    // layers alias into coherent wavy bands instead of imperceptible noise.
    const jitter = fract(sin(dot(screenCoordinate.xy, vec2(12.9898, 78.233))).mul(43758.5453));
    const depth = slabStart.add(stepLen.mul(mix(0.05, 0.95, jitter))).toVar();

    // The entire march is skipped below the horizon (half the sky sphere) and
    // aborts once the view ray is effectively opaque.
    If(horizonFade.greaterThan(0.001), () => {
      Loop({ start: int(0), end: int(CLOUD_STEPS), type: 'int', name: 'i' }, () => {
        If(transmittance.lessThan(0.015), () => { Break(); });
        const p = cam.add(viewDir.mul(depth));
        // Ray exited the top of the cloud slab — nothing left to accumulate.
        If(p.y.greaterThan(u.uAltitude.add(u.uThickness)), () => { Break(); });
        const densitySample = sampleCloudDensity(u, p, noise, scale);
        const d = densitySample.fine.mul(horizonFade);
        const coarseD = densitySample.coarse.mul(horizonFade);

        // Only accumulate where there's actually cloud.
        If(d.greaterThan(float(0.005)), () => {
          // Nubis-style directional lighting: march two density taps toward
          // the sun so cloud sides facing the light brighten and far sides
          // fall into shadow — lighting tracks sun elevation *and* azimuth.
          const heightFraction = nodeClamp(p.y.sub(u.uAltitude).div(u.uThickness), 0.0, 1.0);
          const topLight = pow(heightFraction, 0.55);
          const baseShadow = oneMinus(topLight).mul(0.42);
          const selfShadow = nodeClamp(oneMinus(baseShadow), 0.4, 1.0);
          const diffuse = nodeClamp(sunDir.y.mul(0.5).add(0.5), 0.35, 1.0).mul(selfShadow);

          const lightStep = u.uThickness.mul(0.45);
          const toSun1 = cloudLightDensity(u, p.add(sunDir.mul(lightStep)), noise, scale, densitySample.coverageMap);
          const toSun2 = cloudLightDensity(u, p.add(sunDir.mul(lightStep.mul(2.3))), noise, scale, densitySample.coverageMap);
          const toSun3 = cloudLightDensity(u, p.add(sunDir.mul(lightStep.mul(4.6))), noise, scale, densitySample.coverageMap);
          const opticalToSun = toSun1.add(toSun2.mul(0.65)).add(toSun3.mul(0.4)).mul(2.0);
          const directT = exp(opticalToSun.negate());
          // Beer-powder: thin backlit edges dip dark before brightening — the
          // signature crisp rims of the Nubis model.
          const powder = oneMinus(exp(coarseD.negate().mul(3.5))).mul(0.65).add(0.35);
          const lightTransmittance = directT.mul(powder);
          // Nubis multiple-scattering approximation: light that bounces deeper
          // into dense, high, sun-attenuated regions re-emerges as a soft glow
          // (ms_volume from the Nubis Evolved slides).
          const msVolume = nodeClamp(densitySample.dimensionalProfile.mul(verticalStep).sub(0.1).div(0.9), 0.0, 1.0)
            .mul(pow(nodeClamp(u.uCoverage.mul(max(u.uCloudType, 0.05)), 0.0, 1.0), 0.25))
            .mul(pow(directT, 0.6))
            .mul(pow(heightFraction, 0.7));
          const luminance = float(0.06).add(d.mul(phasePrimary.add(phaseSecondary.mul(msVolume))));
          const directLight = mix(u.uShadowColor, u.uSunColor, nodeClamp(lightTransmittance.mul(diffuse).add(msVolume.mul(0.7)), 0.0, 1.0));
          // Nubis ambient scattering: sky light reaches the cloud in proportion
          // to sqrt(1 - dimensional_profile) — bright translucent edges/tops,
          // naturally darker dense bases — weighted toward the layer top.
          const ambientScattering = pow(oneMinus(nodeClamp(densitySample.dimensionalProfile, 0.0, 1.0)), 0.5)
            .mul(mix(0.35, 1.0, topLight));
          const ambientLight = u.uFogColor.mul(ambientScattering).mul(0.55);
          const stepColor = u.uColor.mul(directLight.mul(luminance).add(ambientLight.mul(0.28)));

          // Front-to-back compositing via Beer's law for the view ray.
          // Normalize optical depth by step count instead of raw world distance;
          // raw stepLen made the volume read like opaque paint.
          const tau = d.mul(verticalStep.div(u.uThickness)).mul(CLOUD_ABSORPTION).mul(u.uOpacity);
          lightEnergy.addAssign(transmittance.mul(stepColor).mul(tau).mul(3.5));
          transmittance.mulAssign(exp(tau.negate()));
        });

        depth.addAssign(stepLen);
        stepLen.mulAssign(1.028);
        verticalStep.mulAssign(1.028);
      });
    });

    // Cirrus sub-layer (Nubis 2.5-D model): a thin streaky ice-cloud sheet
    // far above the volumetric layer. One flat anisotropic lookup, no march;
    // composited behind the cumulus via the remaining transmittance.
    const cirrusFade = nodeSmoothstep(0.03, 0.14, viewDir.y).mul(u.uCirrus);
    If(cirrusFade.greaterThan(0.002), () => {
      const cirrusAlt = u.uAltitude.mul(2.5).add(160.0);
      const tCir = cirrusAlt.sub(cam.y).div(up);
      const pc = cam.add(viewDir.mul(tCir)).sub(u.uWindOffset.mul(1.6));
      const cuv = pc.xz.mul(scale.mul(0.35));
      // Strongly anisotropic sampling of the baked noise reads as wind-combed
      // ice streaks rather than puffs.
      const s1 = noise(vec3(cuv.x.mul(0.22), 0.37, cuv.y)).x;
      const s2 = noise(vec3(cuv.x.mul(0.6).add(13.7), 0.71, cuv.y.mul(2.1))).y;
      const streaks = nodeSmoothstep(0.45, 0.9, s1.mul(0.72).add(s2.mul(0.28)));
      const cd = streaks.mul(cirrusFade);
      // Thin ice is translucent and mostly sun-lit, with forward scatter.
      const cirrusColor = mix(u.uFogColor, u.uSunColor, float(0.45).add(pow(max(mu, 0.0), 3.0).mul(0.4)));
      lightEnergy.addAssign(transmittance.mul(cirrusColor).mul(cd).mul(0.85));
      transmittance.mulAssign(oneMinus(cd.mul(0.5)));
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
    // Gentle default drift so the sky is alive without any wind configured;
    // setWindDirection/setWindSpeed (same units as Precipitation) override.
    this._windX = 0.98;
    this._windZ = 0.2;
    this._windSpeed = 0.02;
    this._windOffset = new THREE.Vector3();
    this._ready = false;
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
      this.params.cloudType,
      this.params.cloudBanks,
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
      cloudType: p.cloudType,
      cloudBanks: p.cloudBanks,
      cloudSharpness: p.sharpness,
      cloudWispiness: p.wispiness,
      cloudDarkness: p.darkness,
      cloudCirrus: p.cirrus,
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
    if (data.cloudType != null) p.cloudType = data.cloudType;
    if (data.cloudBanks != null) p.cloudBanks = data.cloudBanks;
    if (data.cloudSharpness != null) p.sharpness = data.cloudSharpness;
    if (data.cloudWispiness != null) p.wispiness = data.cloudWispiness;
    if (data.cloudDarkness != null) p.darkness = data.cloudDarkness;
    if (data.cloudCirrus != null) p.cirrus = clamp(data.cloudCirrus, 0, 1);
    if (this._material) {
      const u = this._material.uniforms;
      u.uOpacity.value = p.opacity;
      u.uCoverage.value = p.coverage;
      u.uScale.value = Math.max(0.0005, p.noiseScale * (p.tile / 6) * 0.45);
      u.uDetailStrength.value = p.detailStrength;
      u.uHoles.value = p.holes;
      u.uCloudType.value = p.cloudType;
      u.uCloudBanks.value = p.cloudBanks;
      u.uSharpness.value = p.sharpness;
      u.uWispiness.value = p.wispiness;
      u.uRadius.value = p.drawDistance;
      u.uAltitude.value = p.altitude;
      u.uThickness.value = Math.max(20, p.drawDistance * 0.18);
      u.uDarkness.value = p.darkness;
      u.uCirrus.value = p.cirrus;
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
    // Clouds ride the wind at half the precipitation drift rate.
    const windMag = this._windSpeed * 30 * 0.5;
    this._windOffset.x += this._windX * windMag * (dt || 0);
    this._windOffset.z += this._windZ * windMag * (dt || 0);
    u.uWindOffset.value.copy(this._windOffset);
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
    // Ozone absorption (Chappuis band, peaks in green): deepens the zenith blue
    // and pushes twilight toward saturated orange/red. Pure absorption term.
    const betaOzone = vec3(0.650, 1.881, 0.085).mul(0.0013);
    const scaleRayleigh = float(8.0);
    const scaleMie = float(1.2);
    // Tent-shaped ozone layer centered at 25km, ~15km half-width.
    const ozoneDensity = (h) => max(oneMinus(abs(h.sub(25.0)).div(15.0)), 0.0);

    const opticalDepthR = float(0.0).toVar();
    const opticalDepthM = float(0.0).toVar();
    const opticalDepthO = float(0.0).toVar();
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
      opticalDepthO.addAssign(ozoneDensity(height).mul(segmentLength));

      const bLight = dot(p, sunDir);
      const cLight = dot(p, p).sub(atmosphereRadius.mul(atmosphereRadius));
      const hLight = max(bLight.mul(bLight).sub(cLight), 0.0);
      const lightLength = bLight.negate().add(sqrt(hLight)).div(lightSamples);
      const lightDepthR = float(0.0).toVar();
      const lightDepthM = float(0.0).toVar();
      const lightDepthO = float(0.0).toVar();
      const lt = float(0.0).toVar();
      Loop({ start: int(0), end: int(4), type: 'int', name: 'j' }, () => {
        const lp = p.add(sunDir.mul(lt.add(lightLength.mul(0.5))));
        const lh = max(length(lp).sub(planetRadius), 0.0);
        lightDepthR.addAssign(exp(lh.div(scaleRayleigh).negate()).mul(lightLength));
        lightDepthM.addAssign(exp(lh.div(scaleMie).negate()).mul(lightLength));
        lightDepthO.addAssign(ozoneDensity(lh).mul(lightLength));
        lt.addAssign(lightLength);
      });

      const tau = betaRayleigh.mul(opticalDepthR.add(lightDepthR))
        .add(betaMie.mul(opticalDepthM.add(lightDepthM)))
        .add(betaOzone.mul(opticalDepthO.add(lightDepthO)));
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
    const sunDisc = pow(sunDot, 14000.0).mul(day).mul(6.0);
    const sunGlow = pow(sunDot, mix(80.0, 24.0, nodeClamp(u.turbidity.div(12.0), 0.0, 1.0))).mul(nodeSmoothstep(-0.08, 0.35, sunY)).mul(0.08);
    const warm = mix(vec3(1.0, 0.38, 0.16), vec3(1.0, 0.92, 0.72), day);
    scatter = scatter.add(warm.mul(sunDisc.add(sunGlow)));

    const twilightTint = mix(vec3(0.90, 0.28, 0.18), vec3(0.22, 0.16, 0.42), horizon);
    scatter = mix(scatter, scatter.add(twilightTint.mul(0.22)), twilight);

    const night = mix(vec3(0.008, 0.014, 0.035), vec3(0.025, 0.040, 0.090), pow(horizon, 0.7));
    let col = mix(night, scatter, nodeSmoothstep(-0.18, 0.02, sunY));
    col = oneMinus(exp(col.mul(-1.15)));

    // Tiny stable dither breaks up gradient banding in the smooth sky.
    const dither = fract(sin(dot(viewDir.xy.add(viewDir.z), vec2(12.9898, 78.233))).mul(43758.5453));
    col = col.add(vec3(dither.sub(0.5).mul(1.0 / 160.0)));

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

// The sun quad extends this many disc radii past the disc so the corona has
// room to fall off without a visible clipping square.
const SUN_GLOW_EXTENT = 5;

export function createSunBall({
  distance = DEFAULT_SUN_BALL_DISTANCE,
  radius = DEFAULT_SUN_BALL_RADIUS,
  color = DEFAULT_SUN_BALL_COLOR,
  horizonColor = DEFAULT_SUN_BALL_HORIZON_COLOR,
  segments = 32, // kept for API compatibility; the disc is now shader-drawn
} = {}) {
  // Camera-facing quad with a shaded solar disc: limb-darkened HDR core, soft
  // rim, and a two-lobe additive corona (tight aureole + broad halo). Reads as
  // a glowing star instead of a flat matte ball.
  const size = radius * 2 * SUN_GLOW_EXTENT;
  const geometry = new THREE.PlaneGeometry(size, size);
  const material = new THREE.MeshBasicNodeMaterial();
  material.transparent = true;
  material.blending = THREE.AdditiveBlending;
  material.toneMapped = false;
  material.fog = false;
  material.depthWrite = false;
  // Keep depth testing on so terrain/objects can occlude the sun ball.
  material.depthTest = true;

  const u = {
    uColor: assignUniform(material, 'uColor', uniform(new THREE.Color(color))),
    uGlowColor: assignUniform(material, 'uGlowColor', uniform(new THREE.Color(color))),
    uOpacity: assignUniform(material, 'uOpacity', uniform(1)),
    uGlow: assignUniform(material, 'uGlow', uniform(1)),
  };

  material.fragmentNode = Fn(() => {
    const q = uv().sub(0.5).mul(2.0); // [-1, 1] across the quad
    const r = length(q);
    // Radial distance in disc radii: 1.0 is the limb, SUN_GLOW_EXTENT the quad edge.
    const rr = r.mul(SUN_GLOW_EXTENT);

    // Solar limb darkening: hot white center falling to the tinted rim.
    const rd = nodeClamp(rr, 0.0, 1.0);
    const limb = sqrt(max(oneMinus(rd.mul(rd)), 0.0));
    const discShape = oneMinus(nodeSmoothstep(0.94, 1.03, rr));
    const discColor = mix(u.uColor, vec3(1.0), limb.mul(0.55));
    const disc = discColor.mul(float(0.6).add(limb.mul(0.4))).mul(discShape).mul(2.4);

    // Corona: tight aureole hugging the limb plus a broad soft halo, faded out
    // well before the quad edge so the billboard never shows a hard border.
    const aureole = exp(max(rr.sub(1.0), 0.0).mul(-1.7));
    const halo = pow(max(oneMinus(r), 0.0), 2.6).mul(0.55);
    const edgeFade = oneMinus(nodeSmoothstep(0.78, 1.0, r));
    const glow = u.uGlowColor.mul(aureole.mul(0.6).add(halo)).mul(u.uGlow).mul(edgeFade);

    const col = disc.add(glow.mul(oneMinus(discShape.mul(0.85))));
    return vec4(col.mul(u.uOpacity), 1.0);
  })();

  const sunBall = new THREE.Mesh(geometry, material);
  sunBall.name = 'metaverse-sky-sun-ball';
  // Draw after the sky dome; otherwise the sky shader overpaints the mesh.
  sunBall.renderOrder = 998;
  sunBall.userData.distance = distance;
  sunBall.userData.radius = radius;
  sunBall.userData.highColor = new THREE.Color(color);
  sunBall.userData.horizonColor = new THREE.Color(horizonColor);
  return sunBall;
}

export function syncSunBall(sunBall, camera, direction, distance = sunBall?.userData?.distance ?? DEFAULT_SUN_BALL_DISTANCE) {
  if (!sunBall || !camera) return null;
  const d = vectorFrom(direction).normalize();
  const sunY = d.y;
  sunBall.position.copy(camera.position).add(d.multiplyScalar(distance));
  sunBall.visible = sunY > -0.08;
  // Billboard the glow quad at the camera.
  sunBall.quaternion.copy(camera.quaternion);

  const material = sunBall.material;
  const day = smoothstep(-0.08, 0.12, sunY);
  const horizonWarmth = 1 - smoothstep(0.02, 0.55, Math.max(0, sunY));
  const highColor = sunBall.userData.highColor ?? new THREE.Color(DEFAULT_SUN_BALL_COLOR);
  const horizonColor = sunBall.userData.horizonColor ?? new THREE.Color(DEFAULT_SUN_BALL_HORIZON_COLOR);
  const opacity = clamp(day * (0.45 + 0.55 * smoothstep(-0.02, 0.22, sunY)), 0, 1);
  if (material?.uniforms?.uColor) {
    const un = material.uniforms;
    un.uColor.value.copy(highColor).lerp(horizonColor, horizonWarmth * 0.95);
    un.uGlowColor.value.copy(highColor).lerp(horizonColor, Math.min(1, horizonWarmth * 1.2));
    un.uOpacity.value = opacity;
    // The corona swells and warms as the sun approaches the horizon.
    un.uGlow.value = 0.85 + horizonWarmth * 0.75;
  } else if (material) {
    // Custom/legacy meshes with a plain color material.
    material.color.copy(highColor).lerp(horizonColor, horizonWarmth * 0.95);
    material.opacity = opacity;
    material.needsUpdate = true;
  }

  const radius = sunBall.userData.radius ?? DEFAULT_SUN_BALL_RADIUS;
  const sizeScale = sunBall.userData.sizeScale ?? 1;
  const horizonScale = 1 + (1 - smoothstep(0.0, 0.35, Math.max(0, sunY))) * 0.18;
  // Atmospheric refraction visibly flattens the disc right at the horizon.
  const squash = 1 - (1 - smoothstep(0.0, 0.14, Math.max(0, sunY))) * 0.22;
  const s = horizonScale * (radius / DEFAULT_SUN_BALL_RADIUS) * sizeScale;
  sunBall.scale.set(s, s * squash, s);
  return sunBall;
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
    color: 0xc4d6e4,
    size: 0.2,
    speed: 60,
    windDrift: 0.6,
    softness: 0.5,
    slant: 1,
    swirl: 0,
    streak: 1,
    // Rain is mostly transparent water: faint in flat light, strongly visible
    // when backlit by the sun (forward scatter).
    alpha: 0.55,
    backlight: 1.1,
    flutter: 0,
  },
  snow: {
    color: 0xffffff,
    size: 0.28,
    speed: 8,
    windDrift: 1.4,
    softness: 0.18,
    slant: 0.2,
    swirl: 1,
    streak: 0,
    alpha: 0.9,
    backlight: 0.45,
    flutter: 1,
  },
  hail: {
    color: 0xdce8f2,
    size: 0.16,
    speed: 40,
    windDrift: 0.3,
    softness: 0.08,
    slant: 0.1,
    swirl: 0,
    streak: 0.35,
    alpha: 0.8,
    backlight: 0.6,
    flutter: 0.15,
  },
};

const PRECIP_MAX = 6000;
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
    this._camRight = new THREE.Vector3();
    this._sunViewDir = new THREE.Vector3(0, 1, 0);
    this._textures = {};
    this._textureLoader = null;
    this._ready = false;

    if (textures && Object.keys(textures).length > 0) {
      this.setTextures(textures);
    }
  }

  init() {
    if (this._ready) return this;
    // WebGPU renders THREE.Points as 1px primitives, so sized particles must
    // be an instanced THREE.Sprite driven by per-instance attributes.
    this._posAttr = new THREE.InstancedBufferAttribute(this._pos, 3).setUsage(THREE.DynamicDrawUsage);
    this._seedAttr = new THREE.InstancedBufferAttribute(this._seed, 1).setUsage(THREE.DynamicDrawUsage);
    this._sizeAttr = new THREE.InstancedBufferAttribute(this._size, 1).setUsage(THREE.DynamicDrawUsage);

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
      uSunDirView: assignUniform(this._pointMat, 'uSunDirView', uniform(new THREE.Vector3(0, 1, 0))),
      uDayFactor: assignUniform(this._pointMat, 'uDayFactor', uniform(1)),
      uBacklight: assignUniform(this._pointMat, 'uBacklight', uniform(0)),
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
      uStreak: assignUniform(this._pointMat, 'uStreak', uniform(0)),
      uTilt: assignUniform(this._pointMat, 'uTilt', uniform(0)),
    };
    const aSeed = instancedBufferAttribute(this._seedAttr);
    const aSize = instancedBufferAttribute(this._sizeAttr);
    // Per-instance world position; the sprite object itself stays at origin.
    this._pointMat.positionNode = instancedBufferAttribute(this._posAttr);
    // Sprite center in view space — constant per instance, usable in fragment.
    const dist = length(positionView);
    // Fade far particles out, and fade very-near ones too — a drop crossing
    // right in front of the lens would otherwise fill the screen as a giant
    // readable billboard.
    const fade = oneMinus(nodeSmoothstep(u.uFadeNear, u.uFadeFar, dist)).mul(nodeSmoothstep(1.5, 5.0, dist));
    // World-space particle size; streaking particles get a larger sprite to
    // hold the elongated shape. rotationNode slants the whole quad toward the
    // screen-space fall direction (gravity plus projected wind drift).
    this._pointMat.sizeNode = aSize.mul(u.uSizeScale).mul(u.uStreak.mul(0.9).add(1.0));
    this._pointMat.rotationNode = u.uTilt;
    this._pointMat.fragmentNode = Fn(() => {
      const c = uv().sub(0.5);
      const streak = nodeClamp(u.uStreak, 0.0, 1.5);
      const streak01 = nodeClamp(streak, 0.0, 1.0);

      // Procedural shape: a capsule that reads as a round drop at streak=0 and
      // a thin motion-blurred rain streak at streak=1.
      const halfLen = streak.mul(0.34);
      const dCap = length(vec2(c.x, max(abs(c.y).sub(halfLen), 0.0)));
      const width = mix(float(0.5), float(0.10), streak01);
      const inner = max(width.sub(u.uSoftness.mul(mix(1.0, 0.3, streak01))), 0.005);
      const tipFade = mix(
        float(1.0),
        oneMinus(nodeSmoothstep(0.05, halfLen.add(width), abs(c.y))).mul(0.8).add(0.2),
        streak01,
      );
      const proceduralAlpha = nodeSmoothstep(width, inner, dCap).mul(tipFade);

      // Textured sprites get a horizontal squeeze so rain art elongates with
      // the streak (the quad rotation comes from rotationNode). Mask outside
      // the squeezed bounds so clamped edge texels never smear.
      const stretch = streak.mul(1.2).add(1.0);
      const tuv = vec2(c.x.mul(stretch), c.y).add(0.5);
      const inBounds = oneMinus(nodeSmoothstep(0.475, 0.5, max(abs(tuv.x.sub(0.5)), abs(tuv.y.sub(0.5)))));
      const tex = texture(u.uTexture, tuv);
      const shapeAlpha = u.uTextureEnabled.greaterThan(0.5).select(tex.a.mul(inBounds), proceduralAlpha);
      const texColor = u.uTextureEnabled.greaterThan(0.5).select(tex.rgb, vec3(1.0));
      If(shapeAlpha.lessThan(0.01), () => { Discard(); });
      const twinkle = oneMinus(u.uTwinkle).add(u.uTwinkle.mul(float(0.55).add(sin(u.uTime.mul(3.0).add(aSeed.mul(50.0))).mul(0.45))));

      // Multiplicative lighting model: particles are lit by a sky-tinted
      // ambient plus sunlight, so they dim at night and inherit scene mood
      // instead of glowing a fixed tint.
      const backlit = pow(max(dot(normalize(positionView), normalize(u.uSunDirView)), 0.0), 6.0).mul(u.uBacklight);
      const ambient = mix(vec3(0.9), u.uFogColor, 0.45);
      const sunLight = u.uSunColor.mul(u.uSunFactor.mul(0.5).add(backlit.mul(0.9)));
      let col = u.uColor.mul(texColor).mul(ambient.add(sunLight)).mul(u.uDayFactor);
      const ff = nodeSmoothstep(u.uFogNear, u.uFogFar, dist).mul(u.uFogEnabled);
      col = mix(col, u.uFogColor, ff.mul(0.7));
      // Per-particle brightness variation plus backlit visibility boost:
      // precipitation reads much stronger looking toward the sun.
      const sparkle = float(0.8).add(aSeed.mul(0.4));
      const alpha = shapeAlpha.mul(u.uOpacity).mul(fade).mul(twinkle).mul(sparkle)
        .mul(float(1.0).add(backlit.mul(0.5)))
        .mul(oneMinus(ff.mul(0.5)));
      return vec4(col, alpha);
    })();
    this._points = new THREE.Sprite(this._pointMat);
    this._points.count = 0;
    this._points.frustumCulled = false;
    // Draw after the volumetric clouds (renderOrder 2) so near rain composites
    // over the far cloud layer instead of under it.
    this._points.renderOrder = 3;
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
      // Keep the last profile so particles continue falling and thinning out
      // during the fade; nulling it froze them mid-air until the timer hid
      // the whole group at once.
      this._startFade(0, 1.5);
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
    let day = 1;
    if (this.sky) {
      const sun = this.sky.material.uniforms.sunPosition.value;
      const sunY = clamp(sun.y, -0.1, 1);
      sunFactor = Math.max(0, sunY) * 0.6;
      day = smoothstep(-0.06, 0.25, sunY);
      const warmth = 1 - smoothstep(0.12, 0.62, sunY);
      const c = new THREE.Color(0xffffff).lerp(new THREE.Color(0xffb36f), warmth * 0.55);
      sunColor = c.getHex();
      this._sunViewDir.copy(sun).normalize().transformDirection(this.camera.matrixWorldInverse);
    }
    const pu = this._pointMat.uniforms;
    pu.uSunColor.value.setHex(sunColor);
    pu.uSunFactor.value = sunFactor;
    pu.uSunDirView.value.copy(this._sunViewDir);
    pu.uDayFactor.value = 0.12 + 0.88 * day;
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

    // Intensity can exceed 1 (editor slider goes to 3); never spawn past the
    // allocated instance buffers.
    const wantActive = Math.min(PRECIP_MAX, Math.floor(PRECIP_MAX * env));
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
    const t = this._time;
    // Gusting: the wind breathes on two slow frequencies instead of pushing
    // with a constant force. Never reverses (factor stays in ~[0.4, 1.6]).
    const gust = 1 + Math.sin(t * 0.5) * 0.35 + Math.sin(t * 0.13 + 1.7) * 0.25;
    const wx = this._windX * windMag * gust;
    const wz = this._windZ * windMag * gust;
    const swirl = prof.swirl;
    const flutter = prof.flutter ?? 0;

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
        // Two-frequency meander: a slow drift with a faster wobble on top
        // reads as fluttering flakes instead of synchronized pendulums.
        vx += (Math.sin(t * 1.3 + seed * 37) + 0.5 * Math.sin(t * 2.7 + seed * 61)) * swirl * 3.2;
        vz += (Math.cos(t * 1.1 + seed * 23) + 0.5 * Math.cos(t * 2.3 + seed * 47)) * swirl * 3.2;
      }
      let vy = fallSpeed * (0.8 + seed * 0.4);
      // Flutter: flakes momentarily hang and then drop, instead of descending
      // at a perfectly constant rate.
      if (flutter > 0) vy *= 1 + Math.sin(t * 1.9 + seed * 43) * 0.35 * flutter;
      pos[ix] += vx * dt;
      pos[iy] -= vy * dt;
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
    const opacity = env * (prof.alpha ?? 1);
    this._pointMat.uniforms.uBacklight.value = prof.backlight ?? 0;
    // Screen-space slant: project the horizontal wind drift onto the camera's
    // right axis and tilt the sprite toward the resulting fall direction. The
    // drift is exaggerated (x4) so slanting rain reads at gentle wind speeds.
    this._camRight.set(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const screenDrift = (wx * this._camRight.x + wz * this._camRight.z) * 4;
    this._pointMat.uniforms.uTilt.value = Math.atan2(screenDrift, Math.max(fallSpeed, 0.001)) * prof.slant;
    // Faster fall stretches the streaks, like a longer motion-blur exposure.
    this._pointMat.uniforms.uStreak.value = prof.streak * clamp(0.5 + this.params.speed * 0.5, 0.5, 1.5);
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
    // A zero instance count skips the draw during the fade-in ramp.
    this._points.count = this._active;
    this._points.visible = this._active > 0;
    this._posAttr.needsUpdate = true;
    this._seedAttr.needsUpdate = true;
    this._sizeAttr.needsUpdate = true;
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
    // Note: the sprite's quad geometry is shared by all THREE.Sprite instances
    // and must not be disposed here.
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
    sunBall = true,
    sunBallOptions = {},
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
    this.sunBall = sunBall
      ? (sunBall?.isObject3D ? sunBall : createSunBall({ distance: skyScale * 0.92, ...sunBallOptions }))
      : null;
    if (this.sunBall) this.scene.add(this.sunBall);

    if (renderer && atmosphere.exposure != null) renderer.toneMappingExposure = atmosphere.exposure;

    const angles = getSunAnglesFromDirection(this.sky.material.uniforms.sunPosition.value);
    this.elevation = atmosphere.elevation ?? angles.elevation;
    this.azimuth = atmosphere.azimuth ?? angles.azimuth;
    setSkySun(this.sky, { elevation: this.elevation, azimuth: this.azimuth, light: this.light, lightDistance: this.lightDistance });
    this._syncSunBall();

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

  _syncSunBall() {
    return syncSunBall(this.sunBall, this.camera, this.sky.material.uniforms.sunPosition.value);
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
    this._syncSunBall();
    this.syncEnvironmentLighting();
    this.onSunChange?.(direction);
    return this;
  }

  setSunDirection(direction) {
    const d = vectorFrom(direction).normalize();
    const angles = getSunAnglesFromDirection(d);
    return this.setSun(angles.elevation, angles.azimuth);
  }

  /** Visual sun disc size multiplier (1 = default). Corona scales with it. */
  setSunSize(scale) {
    if (this.sunBall) {
      this.sunBall.userData.sizeScale = Math.max(0.05, Number(scale) || 1);
      this._syncSunBall();
    }
    return this;
  }

  getSunSize() {
    return this.sunBall?.userData.sizeScale ?? 1;
  }

  setWindDirection(direction) {
    this.precipitation?.setWindDirection(direction);
    this.clouds?.setWindDirection(direction);
    return this;
  }

  setWindSpeed(speed) {
    this.precipitation?.setWindSpeed(speed);
    this.clouds?.setWindSpeed(speed);
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
      sunSize: this.getSunSize(),
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
    if (data.sunSize != null && this.sunBall) this.sunBall.userData.sizeScale = Math.max(0.05, Number(data.sunSize) || 1);
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
    this._syncSunBall();
    this.clouds?.applyAtmosphereSettings(data);
    this.syncEnvironmentLighting();
    return this;
  }

  update(deltaTime) {
    this.sky.position.copy(this.camera.position);
    this._syncSunBall();
    this.clouds?.update(deltaTime);
    this.precipitation?.update(deltaTime);
    return this;
  }

  dispose() {
    this.clouds?.dispose();
    this.precipitation?.dispose();
    if (this.sunBall) {
      this.scene.remove(this.sunBall);
      this.sunBall.geometry?.dispose?.();
      this.sunBall.material?.dispose?.();
    }
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
    sunBall = null,
    onSunChange = null,
    envIntensityMin = DEFAULT_ENV_INTENSITY_MIN,
    envIntensityMax = DEFAULT_ENV_INTENSITY_MAX,
  }) {
    this.sky = sky;
    this.u = sky.material.uniforms;
    this.light = light;
    this.renderer = renderer;
    this.clouds = clouds;
    this.sunBall = sunBall;
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
    if (this.sunBall) {
      this._slider('Sun size', 0.25, 3, 0.05, this.sunBall.userData.sizeScale ?? 1, (v) => {
        this.sunBall.userData.sizeScale = v;
        this.onSunChange?.();
      });
    }

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

    this._section('Cloud shape');
    this._slider('Coverage', 0.2, 1.2, 0.01, p.coverage, (v) => c.applyAtmosphereSettings({ cloudCoverage: v }));
    this._slider('Cirrus', 0, 1, 0.01, p.cirrus, (v) => c.applyAtmosphereSettings({ cloudCirrus: v }));
    this._slider('Cloud Type', 0, 1, 0.01, p.cloudType, (v) => c.applyAtmosphereSettings({ cloudType: v }));
    this._slider('Holes', 0, 1, 0.01, p.holes, (v) => c.applyAtmosphereSettings({ cloudHoles: v }));

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
