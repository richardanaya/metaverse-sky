import type * as THREE from 'three';
import type { Sky } from 'three/addons/objects/Sky.js';

export const DEFAULT_SKY_SCALE: number;
export const DEFAULT_TURBIDITY: number;
export const DEFAULT_RAYLEIGH: number;
export const DEFAULT_MIE_COEFFICIENT: number;
export const DEFAULT_MIE_DIRECTIONAL_G: number;
export const DEFAULT_EXPOSURE: number;
export const DEFAULT_SUN_POSITION: [number, number, number];
export const DEFAULT_ENV_INTENSITY_MIN: number;
export const DEFAULT_ENV_INTENSITY_MAX: number;
export const ELEVATION_MIN: number;
export const ELEVATION_MAX: number;

export interface CloudSettings {
  enabled: boolean;
  altitude: number;
  opacity: number;
  windSpeed: number;
  windDirection: number;
  tile: number;
  cloudColor: THREE.ColorRepresentation;
  autoTint: boolean;
  puffScale: number;
  layerHeight: number;
  coverage: number;
  noiseSeed: number;
  noiseScale: number;
  noiseOctaves: number;
  noiseJitter: number;
  roundness: number;
  softness: number;
  darkness: number;
  quality: number;
}

export const DEFAULT_CLOUD_SETTINGS: Readonly<CloudSettings>;

export interface AtmosphereSettings {
  elevation?: number;
  azimuth?: number;
  turbidity?: number;
  rayleigh?: number;
  mieCoefficient?: number;
  mieDirectionalG?: number;
  exposure?: number;
  envIntensityMin?: number;
  envIntensityMax?: number;
  sunPosition?: [number, number, number] | THREE.Vector3;
  cloudsEnabled?: boolean;
  cloudOpacity?: number;
  cloudAltitude?: number;
  cloudWindSpeed?: number;
  cloudWindDirection?: number;
  cloudTile?: number;
  cloudColor?: THREE.ColorRepresentation;
  cloudAutoTint?: boolean;
  cloudPuffScale?: number;
  cloudLayerHeight?: number;
  cloudCoverage?: number;
  cloudNoiseSeed?: number;
  cloudNoiseScale?: number;
  cloudNoiseOctaves?: number;
  cloudNoiseJitter?: number;
  cloudRoundness?: number;
  cloudSoftness?: number;
  cloudDarkness?: number;
  cloudQuality?: number;
}

export interface CreateAtmosphereSkyOptions {
  scale?: number;
  turbidity?: number;
  rayleigh?: number;
  mieCoefficient?: number;
  mieDirectionalG?: number;
  sunPosition?: [number, number, number] | THREE.Vector3;
}

export interface SunAngles {
  elevation: number;
  azimuth: number;
}

export function sunDirectionFromAngles(elevation: number, azimuth: number): THREE.Vector3;
export function getSunAnglesFromDirection(direction: THREE.Vector3 | [number, number, number]): SunAngles;
export function createAtmosphereSky(options?: CreateAtmosphereSkyOptions): Sky;
export function setSkySun(
  sky: Sky,
  options?: { elevation?: number; azimuth?: number; light?: THREE.Light | null; lightDistance?: number },
): THREE.Vector3;
export function syncEnvironmentIntensity(options?: {
  scene?: THREE.Scene | null;
  materials?: THREE.Material[];
  elevation: number;
  envIntensityMin?: number;
  envIntensityMax?: number;
}): number;

export function showPanel(panel: HTMLElement, options?: { display?: string }): void;
export function hidePanel(panel: HTMLElement): void;
export function isPanelOpen(panel: HTMLElement): boolean;

export interface CloudLayerOptions extends Partial<CloudSettings> {
  scene: THREE.Scene;
  camera: THREE.Camera;
  sky?: Sky | null;
}

export class CloudLayer {
  scene: THREE.Scene;
  camera: THREE.Camera;
  sky: Sky | null;
  params: CloudSettings & { cloudColor: THREE.Color };
  constructor(options: CloudLayerOptions);
  init(): this;
  getAtmosphereSettings(): AtmosphereSettings;
  applyAtmosphereSettings(data?: AtmosphereSettings): this;
  setSunDirection(direction: THREE.Vector3 | [number, number, number]): this;
  setWindDirection(direction: [number, number] | THREE.Vector2): this;
  setWindSpeed(speed: number): this;
  update(deltaTime: number): void;
  dispose(): void;
}

export interface MetaverseSkyOptions {
  scene: THREE.Scene;
  camera: THREE.Camera;
  renderer?: THREE.WebGLRenderer | null;
  light?: THREE.Light | null;
  sky?: Sky | null;
  clouds?: boolean;
  cloudOptions?: Partial<CloudSettings>;
  skyScale?: number;
  atmosphere?: CreateAtmosphereSkyOptions & AtmosphereSettings;
  envIntensityMin?: number;
  envIntensityMax?: number;
  environmentMaterials?: THREE.Material[];
  onSunChange?: ((direction: THREE.Vector3) => void) | null;
}

export class MetaverseSky {
  scene: THREE.Scene;
  camera: THREE.Camera;
  renderer: THREE.WebGLRenderer | null;
  light: THREE.Light | null;
  sky: Sky;
  clouds: CloudLayer | null;
  elevation: number;
  azimuth: number;
  envIntensityMin: number;
  envIntensityMax: number;
  environmentMaterials: THREE.Material[];
  constructor(options: MetaverseSkyOptions);
  setSun(elevation?: number, azimuth?: number): this;
  setSunDirection(direction: THREE.Vector3 | [number, number, number]): this;
  setWindDirection(direction: [number, number] | THREE.Vector2): this;
  setWindSpeed(speed: number): this;
  syncEnvironmentLighting(materials?: THREE.Material[]): number;
  addEnvironmentMaterial(material: THREE.Material): this;
  getAtmosphereSettings(): AtmosphereSettings;
  applyAtmosphereSettings(data?: AtmosphereSettings): this;
  update(deltaTime: number): this;
  dispose(): void;
}

export interface SkyEditorOptions {
  sky: Sky;
  light?: THREE.Light | null;
  renderer?: THREE.WebGLRenderer;
  clouds?: CloudLayer | null;
  onSunChange?: (() => void) | null;
  envIntensityMin?: number;
  envIntensityMax?: number;
}

export class SkyEditor {
  sky: Sky;
  light: THREE.Light | null;
  renderer?: THREE.WebGLRenderer;
  clouds: CloudLayer | null;
  elevation: number;
  azimuth: number;
  envIntensityMin: number;
  envIntensityMax: number;
  active: boolean;
  panel: HTMLDivElement;
  constructor(options: SkyEditorOptions);
  open(): void;
  close(): void;
}
