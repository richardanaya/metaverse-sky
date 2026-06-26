import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MetaverseSky, DEFAULT_CLOUD_SETTINGS } from 'metaverse-sky';

const canvas = document.querySelector('#scene');
const fpsValue = document.querySelector('#fps-value');

const sunAzimuthInput = document.querySelector('#sun-azimuth');
const sunElevationInput = document.querySelector('#sun-elevation');
const sunAzimuthValue = document.querySelector('#sun-azimuth-value');
const sunElevationValue = document.querySelector('#sun-elevation-value');
const iblMinInput = document.querySelector('#ibl-min');
const iblMaxInput = document.querySelector('#ibl-max');
const iblMinValue = document.querySelector('#ibl-min-value');
const iblMaxValue = document.querySelector('#ibl-max-value');
const exposureInput = document.querySelector('#exposure');
const exposureValue = document.querySelector('#exposure-value');
const turbidityInput = document.querySelector('#turbidity');
const turbidityValue = document.querySelector('#turbidity-value');
const rayleighInput = document.querySelector('#rayleigh');
const rayleighValue = document.querySelector('#rayleigh-value');
const mieInput = document.querySelector('#mie');
const mieValue = document.querySelector('#mie-value');
const mieGInput = document.querySelector('#mie-g');
const mieGValue = document.querySelector('#mie-g-value');
const cloudsEnabledInput = document.querySelector('#clouds-enabled');
const cloudOpacityInput = document.querySelector('#cloud-opacity');
const cloudOpacityValue = document.querySelector('#cloud-opacity-value');
const cloudAltitudeInput = document.querySelector('#cloud-altitude');
const cloudAltitudeValue = document.querySelector('#cloud-altitude-value');
const cloudDrawDistanceInput = document.querySelector('#cloud-draw-distance');
const cloudDrawDistanceValue = document.querySelector('#cloud-draw-distance-value');
const cloudTileInput = document.querySelector('#cloud-tile');
const cloudTileValue = document.querySelector('#cloud-tile-value');
const darknessInput = document.querySelector('#darkness');
const darknessValue = document.querySelector('#darkness-value');
const coverageInput = document.querySelector('#coverage');
const coverageValue = document.querySelector('#coverage-value');
const noiseScaleInput = document.querySelector('#noise-scale');
const noiseScaleValue = document.querySelector('#noise-scale-value');
const detailStrengthInput = document.querySelector('#detail-strength');
const detailStrengthValue = document.querySelector('#detail-strength-value');
const cloudHolesInput = document.querySelector('#cloud-holes');
const cloudHolesValue = document.querySelector('#cloud-holes-value');
const sharpnessInput = document.querySelector('#sharpness');
const sharpnessValue = document.querySelector('#sharpness-value');
const wispinessInput = document.querySelector('#wispiness');
const wispinessValue = document.querySelector('#wispiness-value');
const windDirectionInput = document.querySelector('#wind-direction');
const windSpeedInput = document.querySelector('#wind-speed');
const windDirectionValue = document.querySelector('#wind-direction-value');
const windSpeedValue = document.querySelector('#wind-speed-value');
const autoTintInput = document.querySelector('#auto-tint');
const cloudColorInput = document.querySelector('#cloud-color');
const cloudControls = document.querySelector('.cloud-controls');
const precipButtons = [...document.querySelectorAll('[data-precip]')];
const precipIntensityInput = document.querySelector('#precip-intensity');
const precipIntensityValue = document.querySelector('#precip-intensity-value');
const precipSpeedInput = document.querySelector('#precip-speed');
const precipSpeedValue = document.querySelector('#precip-speed-value');
const precipSizeInput = document.querySelector('#precip-size');
const precipSizeValue = document.querySelector('#precip-size-value');
const precipWindInput = document.querySelector('#precip-wind');
const precipWindValue = document.querySelector('#precip-wind-value');

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x9fb7d5, 300, 700);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 900);
camera.position.set(52, 42, 96);

const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
await renderer.init();
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.6;

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 12, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.maxDistance = 360;
controls.update();

const sunLight = new THREE.DirectionalLight(0xffffff, 2.2);
scene.add(sunLight);
scene.add(new THREE.AmbientLight(0x8090a8, 0.6));

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(120, 96),
  new THREE.MeshStandardMaterial({ color: 0x42566a, roughness: 0.95 }),
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// A few reference objects so the sky lighting has something to read against.
const sky = new MetaverseSky({
  scene,
  camera,
  renderer,
  light: sunLight,
  clouds: true,
  precipitation: true,
  precipitationOptions: {
    textures: {
      rain: '../../textures/raindrop.png',
      snow: '../../textures/snowflake.png',
      hail: '../../textures/hailstone.png',
    },
  },
  atmosphere: {
    elevation: Number(sunElevationInput.value),
    azimuth: Number(sunAzimuthInput.value),
    exposure: Number(exposureInput.value),
  },
  envIntensityMin: Number(iblMinInput.value),
  envIntensityMax: Number(iblMaxInput.value),
  onSunChange: () => {
    const u = sky.sky.material.uniforms;
    turbidityInput.value = String(u.turbidity.value);
    turbidityValue.textContent = u.turbidity.value.toFixed(1);
    rayleighInput.value = String(u.rayleigh.value);
    rayleighValue.textContent = u.rayleigh.value.toFixed(2);
  },
});

// Reference objects so the sky lighting has something to read against.
const pillarMaterial = new THREE.MeshStandardMaterial({ color: 0xd8e0ea, roughness: 0.6 });
for (let i = 0; i < 6; i += 1) {
  const angle = (i / 6) * Math.PI * 2;
  const pillar = new THREE.Mesh(new THREE.BoxGeometry(4, 22, 4), pillarMaterial);
  pillar.position.set(Math.cos(angle) * 36, 11, Math.sin(angle) * 36);
  scene.add(pillar);
}
sky.addEnvironmentMaterial(pillarMaterial);


bindPanel();
syncCloudControlsEnabled();

const clock = new THREE.Clock();
let fpsFrames = 0;
let fpsElapsed = 0;
renderer.setAnimationLoop(() => {
  const rawDt = clock.getDelta();
  const dt = Math.min(rawDt, 0.05);
  fpsFrames += 1;
  fpsElapsed += rawDt;
  if (fpsElapsed >= 0.5) {
    fpsValue.textContent = String(Math.round(fpsFrames / fpsElapsed));
    fpsFrames = 0;
    fpsElapsed = 0;
  }
  controls.update();
  sky.update(dt);
  renderer.render(scene, camera);
});

function bindPanel() {
  window.addEventListener('resize', resize);

  bindRange(sunAzimuthInput, sunAzimuthValue, (value) => {
    updateSunPosition(value, Number(sunElevationInput.value));
    return `${value}°`;
  });

  bindRange(sunElevationInput, sunElevationValue, (value) => {
    updateSunPosition(Number(sunAzimuthInput.value), value);
    return `${value}°`;
  });

  updateSunPosition(Number(sunAzimuthInput.value), Number(sunElevationInput.value));

  bindRange(iblMinInput, iblMinValue, (value) => {
    sky.envIntensityMin = value;
    if (sky.envIntensityMin > sky.envIntensityMax) {
      sky.envIntensityMax = sky.envIntensityMin;
      iblMaxInput.value = String(value);
      iblMaxValue.textContent = value.toFixed(2);
    }
    sky.syncEnvironmentLighting();
    return value.toFixed(2);
  });

  bindRange(iblMaxInput, iblMaxValue, (value) => {
    sky.envIntensityMax = value;
    if (sky.envIntensityMax < sky.envIntensityMin) {
      sky.envIntensityMin = sky.envIntensityMax;
      iblMinInput.value = String(value);
      iblMinValue.textContent = value.toFixed(2);
    }
    sky.syncEnvironmentLighting();
    return value.toFixed(2);
  });

  bindRange(exposureInput, exposureValue, (value) => {
    renderer.toneMappingExposure = value;
    return value.toFixed(2);
  });

  bindRange(turbidityInput, turbidityValue, (value) => {
    sky.sky.material.uniforms.turbidity.value = value;
    return value.toFixed(1);
  });

  bindRange(rayleighInput, rayleighValue, (value) => {
    sky.sky.material.uniforms.rayleigh.value = value;
    return value.toFixed(2);
  });

  bindRange(mieInput, mieValue, (value) => {
    sky.sky.material.uniforms.mieCoefficient.value = value;
    return value.toFixed(3);
  });

  bindRange(mieGInput, mieGValue, (value) => {
    sky.sky.material.uniforms.mieDirectionalG.value = value;
    return value.toFixed(2);
  });

  cloudsEnabledInput.addEventListener('change', () => {
    sky.applyAtmosphereSettings({ cloudsEnabled: cloudsEnabledInput.checked });
    syncCloudControlsEnabled();
  });

  bindRange(cloudOpacityInput, cloudOpacityValue, (value) => {
    sky.applyAtmosphereSettings({ cloudOpacity: value });
    return value.toFixed(2);
  });

  bindRange(cloudAltitudeInput, cloudAltitudeValue, (value) => {
    sky.applyAtmosphereSettings({ cloudAltitude: value });
    return `${value}m`;
  });

  bindRange(cloudDrawDistanceInput, cloudDrawDistanceValue, (value) => {
    sky.applyAtmosphereSettings({ cloudDrawDistance: value });
    return `${value}m`;
  });

  bindRange(cloudTileInput, cloudTileValue, (value) => {
    sky.applyAtmosphereSettings({ cloudTile: value });
    return `${value}x`;
  });

  bindRange(darknessInput, darknessValue, (value) => {
    sky.applyAtmosphereSettings({ cloudDarkness: value });
    if (value < 0.2) return 'Bright';
    if (value < 0.45) return 'Cloudy';
    if (value < 0.7) return 'Overcast';
    return 'Storm';
  });

  bindRange(coverageInput, coverageValue, (value) => {
    sky.applyAtmosphereSettings({ cloudCoverage: value });
    return value.toFixed(2);
  });

  bindRange(noiseScaleInput, noiseScaleValue, (value) => {
    sky.applyAtmosphereSettings({ cloudNoiseScale: value });
    return value.toFixed(3);
  });

  bindRange(detailStrengthInput, detailStrengthValue, (value) => {
    sky.applyAtmosphereSettings({ cloudDetailStrength: value });
    return value.toFixed(2);
  });

  bindRange(cloudHolesInput, cloudHolesValue, (value) => {
    sky.applyAtmosphereSettings({ cloudHoles: value });
    return value.toFixed(2);
  });

  bindRange(sharpnessInput, sharpnessValue, (value) => {
    sky.applyAtmosphereSettings({ cloudSharpness: value });
    return value.toFixed(2);
  });

  bindRange(wispinessInput, wispinessValue, (value) => {
    sky.applyAtmosphereSettings({ cloudWispiness: value });
    return value.toFixed(2);
  });

  bindRange(windDirectionInput, windDirectionValue, (value) => {
    const rad = THREE.MathUtils.degToRad(value);
    sky.setWindDirection([Math.cos(rad), Math.sin(rad)]);
    return `${value}°`;
  });

  bindRange(windSpeedInput, windSpeedValue, (value) => {
    sky.setWindSpeed(value);
    return value.toFixed(3);
  });

  precipButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const type = button.dataset.precip;
      precipButtons.forEach((b) => b.classList.toggle('active', b === button));
      sky.setPrecipitation({ type });
      syncPrecipControlsEnabled(type);
    });
  });

  bindRange(precipIntensityInput, precipIntensityValue, (value) => {
    sky.setPrecipitation({ intensity: value });
    return value.toFixed(2);
  });

  bindRange(precipSpeedInput, precipSpeedValue, (value) => {
    sky.setPrecipitation({ speed: value });
    return `${value.toFixed(1)}x`;
  });

  bindRange(precipSizeInput, precipSizeValue, (value) => {
    sky.setPrecipitation({ size: value });
    return `${value.toFixed(1)}x`;
  });

  bindRange(precipWindInput, precipWindValue, (value) => {
    sky.setPrecipitation({ windDrift: value });
    return `${value.toFixed(1)}x`;
  });

  autoTintInput.addEventListener('change', () => {
    sky.applyAtmosphereSettings({ cloudAutoTint: autoTintInput.checked });
  });

  cloudColorInput.addEventListener('input', () => {
    const hex = parseInt(cloudColorInput.value.slice(1), 16);
    sky.applyAtmosphereSettings({ cloudColor: hex, cloudAutoTint: false });
    autoTintInput.checked = false;
  });

  document.querySelector('#reset-defaults').addEventListener('click', () => {
    resetDefaults();
  });
}

function updateSunPosition(azimuthDeg, elevationDeg) {
  const az = THREE.MathUtils.degToRad(azimuthDeg);
  const el = THREE.MathUtils.degToRad(elevationDeg);
  const cosEl = Math.cos(el);
  const direction = new THREE.Vector3(
    cosEl * Math.cos(az),
    Math.sin(el),
    cosEl * Math.sin(az),
  );
  sky.setSunDirection(direction);
}

function syncCloudControlsEnabled() {
  const on = cloudsEnabledInput.checked;
  cloudControls.classList.toggle('is-disabled', !on);
  autoTintInput.disabled = !on;
  cloudColorInput.disabled = !on;
}

function syncPrecipControlsEnabled(type) {
  const on = type !== 'none';
  [precipIntensityInput, precipSpeedInput, precipSizeInput, precipWindInput].forEach((input) => {
    input.disabled = !on;
  });
}

function resetDefaults() {
  const d = DEFAULT_CLOUD_SETTINGS;
  const setRange = (input, value, label, fmt) => {
    input.value = String(value);
    label.textContent = fmt(value);
  };
  setRange(cloudOpacityInput, d.opacity, cloudOpacityValue, (v) => v.toFixed(2));
  setRange(cloudAltitudeInput, d.altitude, cloudAltitudeValue, (v) => `${v}m`);
  setRange(cloudDrawDistanceInput, d.drawDistance, cloudDrawDistanceValue, (v) => `${v}m`);
  setRange(cloudTileInput, d.tile, cloudTileValue, (v) => `${v}x`);
  setRange(darknessInput, d.darkness, darknessValue, (v) => (v < 0.2 ? 'Bright' : v < 0.45 ? 'Cloudy' : v < 0.7 ? 'Overcast' : 'Storm'));
  setRange(coverageInput, d.coverage, coverageValue, (v) => v.toFixed(2));
  setRange(noiseScaleInput, d.noiseScale, noiseScaleValue, (v) => v.toFixed(3));
  setRange(detailStrengthInput, d.detailStrength, detailStrengthValue, (v) => v.toFixed(2));
  setRange(cloudHolesInput, d.holes, cloudHolesValue, (v) => v.toFixed(2));
  setRange(sharpnessInput, d.sharpness, sharpnessValue, (v) => v.toFixed(2));
  setRange(wispinessInput, d.wispiness, wispinessValue, (v) => v.toFixed(2));
  setRange(windDirectionInput, 255, windDirectionValue, (v) => `${v}°`);
  setRange(windSpeedInput, 0.045, windSpeedValue, (v) => v.toFixed(3));

  cloudColorInput.value = `#${d.cloudColor.toString(16).padStart(6, '0')}`;
  autoTintInput.checked = d.autoTint;

  sky.applyAtmosphereSettings({
    cloudsEnabled: cloudsEnabledInput.checked,
    cloudOpacity: d.opacity,
    cloudAltitude: d.altitude,
    cloudDrawDistance: d.drawDistance,
    cloudTile: d.tile,
    cloudCoverage: d.coverage,
    cloudNoiseScale: d.noiseScale,
    cloudDetailStrength: d.detailStrength,
    cloudHoles: d.holes,
    cloudSharpness: d.sharpness,
    cloudWispiness: d.wispiness,
    cloudDarkness: d.darkness,
    cloudColor: d.cloudColor,
    cloudAutoTint: d.autoTint,
  });
  const rad = THREE.MathUtils.degToRad(255);
  sky.setWindDirection([Math.cos(rad), Math.sin(rad)]);
  sky.setWindSpeed(0.045);
}

function bindRange(input, label, onInput) {
  input.addEventListener('input', () => {
    label.textContent = onInput(Number(input.value));
  });
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

resize();
