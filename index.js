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
  windSpeed: 0.045,
  windDirection: 255,
  tile: 6,
  drawDistance: 420,
  cloudColor: new THREE.Color(0xf2f6fc),
  autoTint: true,
  coverage: 0.5,
  noiseScale: 0.028,
  detailStrength: 0.45,
  sharpness: 0.35,
  wispiness: 0.45,
  darkness: 0,
};

export const DEFAULT_CLOUD_SETTINGS = Object.freeze({
  enabled: CLOUD_DEFAULTS.enabled,
  renderMode: CLOUD_DEFAULTS.renderMode,
  altitude: CLOUD_DEFAULTS.altitude,
  opacity: CLOUD_DEFAULTS.opacity,
  windSpeed: CLOUD_DEFAULTS.windSpeed,
  windDirection: CLOUD_DEFAULTS.windDirection,
  tile: CLOUD_DEFAULTS.tile,
  drawDistance: CLOUD_DEFAULTS.drawDistance,
  cloudColor: CLOUD_DEFAULTS.cloudColor.getHex(),
  autoTint: CLOUD_DEFAULTS.autoTint,
  coverage: CLOUD_DEFAULTS.coverage,
  noiseScale: CLOUD_DEFAULTS.noiseScale,
  detailStrength: CLOUD_DEFAULTS.detailStrength,
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

function createSkyVolumeCloudMaterial(color, opacity, coverage, darkness, detailStrength, sharpness, wispiness) {
  const material = new THREE.MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthWrite = false;
  material.side = THREE.DoubleSide;
  material.toneMapped = false;

  const u = {
    uColor: assignUniform(material, 'uColor', uniform(color.clone())),
    uOpacity: assignUniform(material, 'uOpacity', uniform(opacity)),
    uCoverage: assignUniform(material, 'uCoverage', uniform(coverage)),
    uDetailStrength: assignUniform(material, 'uDetailStrength', uniform(detailStrength)),
    uSharpness: assignUniform(material, 'uSharpness', uniform(sharpness)),
    uWispiness: assignUniform(material, 'uWispiness', uniform(wispiness)),
    uScale: assignUniform(material, 'uScale', uniform(0.006)),
    uScroll: assignUniform(material, 'uScroll', uniform(new THREE.Vector2())),
    uTime: assignUniform(material, 'uTime', uniform(0)),
    uSunDirection: assignUniform(material, 'uSunDirection', uniform(new THREE.Vector3(...DEFAULT_SUN_POSITION).normalize())),
    uSunColor: assignUniform(material, 'uSunColor', uniform(new THREE.Color(0xfff0d2))),
    uShadowColor: assignUniform(material, 'uShadowColor', uniform(new THREE.Color(0xc9d6e8))),
    uFogColor: assignUniform(material, 'uFogColor', uniform(new THREE.Color(0x9fb7d5))),
    uDarkness: assignUniform(material, 'uDarkness', uniform(darkness)),
    uRadius: assignUniform(material, 'uRadius', uniform(420)),
  };

  material.fragmentNode = Fn(() => {
    const rel = positionWorld.xz.sub(cameraPosition.xz);
    const dist = length(rel);
    const horizonFade = oneMinus(nodeSmoothstep(u.uRadius.mul(0.58), u.uRadius.mul(0.98), dist));
    const p = positionWorld.xz.add(u.uScroll).mul(u.uScale);
    const t = u.uTime.mul(0.015);
    const n0 = vnoise3Node(vec3(p.x, p.y, t));
    const wispCoord = vec2(p.x.mul(1.4).add(p.y.mul(0.28)), p.y.mul(5.6).sub(p.x.mul(0.16)));
    const n1 = vnoise3Node(vec3(wispCoord.x.add(17.0), wispCoord.y.sub(9.0), t.add(4.0)));
    const n2 = vnoise3Node(vec3(p.x.mul(6.8).sub(31.0), p.y.mul(6.8).add(12.0), t.mul(1.7)));
    const threshold = mix(0.78, 0.38, nodeClamp(u.uCoverage, 0.0, 1.0));
    const coarse = n0.mul(0.68).add(n1.mul(mix(0.16, 0.30, u.uWispiness))).add(n2.mul(0.08));
    const coarseDens = nodeSmoothstep(threshold.sub(0.20), threshold.add(0.16), coarse);
    const edgeMask = oneMinus(nodeSmoothstep(0.42, 0.96, coarseDens));
    const eroded = coarse.sub(n2.mul(edgeMask).mul(u.uDetailStrength).mul(0.22));
    const width = mix(0.22, 0.065, nodeClamp(u.uSharpness, 0.0, 1.0));
    const dens = nodeSmoothstep(threshold.sub(width), threshold.add(width), eroded);
    const wisps = nodeSmoothstep(0.24, 0.86, n1.add(n2.mul(0.5)));
    let alpha = dens.mul(mix(float(0.58).sub(u.uWispiness.mul(0.22)), 1.0, wisps)).mul(horizonFade).mul(u.uOpacity);
    If(alpha.lessThan(0.002), () => { Discard(); });

    const sunDir = normalize(u.uSunDirection);
    const viewDir = normalize(cameraPosition.sub(positionWorld));
    const forward = pow(max(dot(viewDir, sunDir), 0.0), 4.0);
    const topLight = nodeClamp(sunDir.y.mul(0.5).add(0.5), 0.15, 1.0);
    let col = u.uColor.mul(mix(0.72, 1.12, n0)).mul(mix(0.75, 1.12, topLight));
    col = mix(col.mul(u.uShadowColor), col, mix(0.45, 0.95, wisps));
    col = col.add(u.uSunColor.mul(forward).mul(0.22).mul(oneMinus(u.uDarkness)));
    col = mix(col, col.mul(vec3(0.34, 0.38, 0.48)), u.uDarkness.mul(0.85));
    col = mix(col, u.uFogColor, nodeSmoothstep(u.uRadius.mul(0.48), u.uRadius, dist).mul(0.45));
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
    this._scroll = new THREE.Vector2();
    this._mesh = null;
    this._material = null;
    this._autoTintColor = new THREE.Color();
    this._sunColor = new THREE.Color(0xfff0d2);
    this._shadowColor = new THREE.Color(0xc9d6e8);
    this._time = 0;
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
      this.params.sharpness,
      this.params.wispiness,
    );
    const size = Math.max(100, this.params.drawDistance * 2);
    const geo = new THREE.PlaneGeometry(size, size, 1, 1);
    this._mesh = new THREE.Mesh(geo, this._material);
    this._mesh.rotation.x = -Math.PI / 2;
    this._mesh.position.set(this.camera.position.x, this.params.altitude, this.camera.position.z);
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

  setWindDirection(direction) {
    const vec = Array.isArray(direction) ? direction : [Number(direction?.x) || 0, Number(direction?.y) || 0];
    const len = Math.hypot(vec[0], vec[1]) || 1;
    this.params.windDirection = (THREE.MathUtils.radToDeg(Math.atan2(vec[1] / len, vec[0] / len)) % 360 + 360) % 360;
    return this;
  }

  setWindSpeed(speed) {
    this.params.windSpeed = Math.max(0, speed);
    return this;
  }

  getAtmosphereSettings() {
    const p = this.params;
    return {
      cloudsEnabled: p.enabled,
      cloudRenderMode: 'volume',
      cloudOpacity: p.opacity,
      cloudAltitude: p.altitude,
      cloudWindSpeed: p.windSpeed,
      cloudWindDirection: p.windDirection,
      cloudTile: p.tile,
      cloudDrawDistance: p.drawDistance,
      cloudColor: p.cloudColor.getHex(),
      cloudAutoTint: p.autoTint,
      cloudCoverage: p.coverage,
      cloudNoiseScale: p.noiseScale,
      cloudDetailStrength: p.detailStrength,
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
    if (data.cloudWindSpeed != null) p.windSpeed = data.cloudWindSpeed;
    if (data.cloudWindDirection != null) p.windDirection = data.cloudWindDirection;
    if (data.cloudTile != null) p.tile = data.cloudTile;
    if (data.cloudDrawDistance != null) p.drawDistance = data.cloudDrawDistance;
    if (data.cloudColor != null) p.cloudColor.set(data.cloudColor);
    if (data.cloudAutoTint != null) p.autoTint = !!data.cloudAutoTint;
    if (data.cloudCoverage != null) p.coverage = data.cloudCoverage;
    if (data.cloudNoiseScale != null) p.noiseScale = data.cloudNoiseScale;
    if (data.cloudDetailStrength != null) p.detailStrength = data.cloudDetailStrength;
    if (data.cloudSharpness != null) p.sharpness = data.cloudSharpness;
    if (data.cloudWispiness != null) p.wispiness = data.cloudWispiness;
    if (data.cloudDarkness != null) p.darkness = data.cloudDarkness;
    if (this._material) {
      const u = this._material.uniforms;
      u.uOpacity.value = p.opacity;
      u.uCoverage.value = p.coverage;
      u.uScale.value = Math.max(0.0005, p.noiseScale * 0.22 * (p.tile / 6));
      u.uDetailStrength.value = p.detailStrength;
      u.uSharpness.value = p.sharpness;
      u.uWispiness.value = p.wispiness;
      u.uRadius.value = p.drawDistance;
      u.uDarkness.value = p.darkness;
      u.uColor.value.copy(p.cloudColor);
    }
    if (this._mesh) {
      const size = Math.max(100, p.drawDistance * 2);
      this._mesh.geometry.dispose();
      this._mesh.geometry = new THREE.PlaneGeometry(size, size, 1, 1);
      this._mesh.position.set(this.camera.position.x, p.altitude, this.camera.position.z);
    }
    this._syncVisibility();
    this._syncSunLighting();
    return this;
  }

  update(dt) {
    if (!this._ready || !this.params.enabled) return;
    this._time += dt;
    const p = this.params;
    const wind = p.windDirection * Math.PI / 180;
    this._scroll.x += Math.cos(wind) * p.windSpeed * dt * 60;
    this._scroll.y += Math.sin(wind) * p.windSpeed * dt * 60;
    this._mesh.position.set(this.camera.position.x, p.altitude, this.camera.position.z);
    const u = this._material.uniforms;
    u.uScroll.value.copy(this._scroll);
    u.uTime.value = this._time;
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
    this._slider('Altitude (m)', 55, 220, 1, p.altitude, (v) => c.applyAtmosphereSettings({ cloudAltitude: v }));
    this._slider('Draw distance (m)', 120, 900, 10, p.drawDistance, (v) => c.applyAtmosphereSettings({ cloudDrawDistance: v }));
    this._slider('Tiling', 3, 10, 0.5, p.tile, (v) => c.applyAtmosphereSettings({ cloudTile: v }));
    this._slider('Darkness', 0, 1, 0.01, p.darkness, (v) => c.applyAtmosphereSettings({ cloudDarkness: v }));

    this._section('Cloud noise');
    this._slider('Coverage', 0.2, 0.9, 0.01, p.coverage, (v) => c.applyAtmosphereSettings({ cloudCoverage: v }));
    this._slider('Pattern scale', 0.01, 0.08, 0.001, p.noiseScale, (v) => c.applyAtmosphereSettings({ cloudNoiseScale: v }));

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
