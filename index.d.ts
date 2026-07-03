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
export const DEFAULT_SUN_BALL_DISTANCE: number;
export const DEFAULT_SUN_BALL_RADIUS: number;
export const DEFAULT_SUN_BALL_COLOR: number;
export const DEFAULT_SUN_BALL_HORIZON_COLOR: number;
export const ELEVATION_MIN: number;
export const ELEVATION_MAX: number;

export type CloudRenderMode = 'volume';

export interface CloudSettings {
  enabled: boolean;
  renderMode: CloudRenderMode;
  altitude: number;
  opacity: number;
  tile: number;
  drawDistance: number;
  cloudColor: THREE.ColorRepresentation;
  autoTint: boolean;
  coverage: number;
  noiseScale: number;
  detailStrength: number;
  holes: number;
  cloudType: number;
  cloudBanks: number;
  sharpness: number;
  wispiness: number;
  darkness: number;
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
  sunSize?: number;
  sunPosition?: [number, number, number] | THREE.Vector3;
  cloudsEnabled?: boolean;
  cloudRenderMode?: CloudRenderMode;
  cloudOpacity?: number;
  cloudAltitude?: number;
  cloudTile?: number;
  cloudDrawDistance?: number;
  cloudColor?: THREE.ColorRepresentation;
  cloudAutoTint?: boolean;
  cloudCoverage?: number;
  cloudNoiseScale?: number;
  cloudDetailStrength?: number;
  cloudHoles?: number;
  cloudType?: number;
  cloudBanks?: number;
  cloudSharpness?: number;
  cloudWispiness?: number;
  cloudDarkness?: number;
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
export function createSunBall(options?: {
  distance?: number;
  radius?: number;
  color?: THREE.ColorRepresentation;
  horizonColor?: THREE.ColorRepresentation;
  segments?: number;
}): THREE.Mesh;
export function syncSunBall(
  sunBall: THREE.Object3D | null | undefined,
  camera: THREE.Camera,
  direction: THREE.Vector3 | [number, number, number],
  distance?: number,
): THREE.Object3D | null;
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

export type PrecipType = 'none' | 'rain' | 'snow' | 'hail';

export interface PrecipSettings {
  type: PrecipType;
  intensity: number;
  speed: number;
  size: number;
  windDrift: number;
}

export interface PrecipitationOptions extends Partial<PrecipSettings> {
  scene: THREE.Scene;
  camera: THREE.Camera;
  sky?: Sky | null;
  textures?: Partial<Record<PrecipType, string>>;
}

export class Precipitation {
  scene: THREE.Scene;
  camera: THREE.Camera;
  sky: Sky | null;
  params: PrecipSettings;
  constructor(options: PrecipitationOptions);
  init(): this;
  setPrecipitation(data?: Partial<PrecipSettings>): this;
  setTextures(textures?: Partial<Record<PrecipType, string>>): this;
  setWindDirection(direction: [number, number] | THREE.Vector2): this;
  setWindSpeed(speed: number): this;
  update(deltaTime: number): void;
  dispose(): void;
}

export interface CloudSkyLayerOptions extends Partial<CloudSettings> {
  scene: THREE.Scene;
  camera: THREE.Camera;
  sky?: Sky | null;
}

export class CloudSkyLayer {
  scene: THREE.Scene;
  camera: THREE.Camera;
  sky: Sky | null;
  params: CloudSettings & { cloudColor: THREE.Color };
  constructor(options: CloudSkyLayerOptions);
  init(): this;
  getAtmosphereSettings(): AtmosphereSettings;
  applyAtmosphereSettings(data?: AtmosphereSettings): this;
  setSunDirection(direction: THREE.Vector3 | [number, number, number]): this;
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
  precipitation?: boolean;
  precipitationOptions?: Partial<PrecipSettings> & { type?: PrecipType; textures?: Partial<Record<PrecipType, string>> };
  sunBall?: boolean | THREE.Object3D;
  sunBallOptions?: {
    distance?: number;
    radius?: number;
    color?: THREE.ColorRepresentation;
    horizonColor?: THREE.ColorRepresentation;
    segments?: number;
  };
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
  sunBall: THREE.Object3D | null;
  clouds: CloudSkyLayer | null;
  precipitation: Precipitation | null;
  elevation: number;
  azimuth: number;
  envIntensityMin: number;
  envIntensityMax: number;
  environmentMaterials: THREE.Material[];
  constructor(options: MetaverseSkyOptions);

  setSun(elevation?: number, azimuth?: number): this;
  setSunDirection(direction: THREE.Vector3 | [number, number, number]): this;
  /** Visual sun disc size multiplier (1 = default). */
  setSunSize(scale: number): this;
  getSunSize(): number;

  setWind(directionOrAngle: number | [number, number] | THREE.Vector2, speed?: number): this;
  setWindDirection(direction: [number, number] | THREE.Vector2): this;
  setWindSpeed(speed: number): this;

  setPrecipitation(data?: Partial<PrecipSettings> & { type?: PrecipType }): this;
  setPrecipitationTextures(textures?: Partial<Record<PrecipType, string>>): this;
  getPrecipSettings(): PrecipSettings | null;

  setExposure(value: number): this;
  getExposure(): number | null;

  setAtmosphere(data?: AtmosphereSettings): this;
  getAtmosphere(): AtmosphereSettings;
  setClouds(data?: Partial<CloudSettings>): this;
  getClouds(): AtmosphereSettings | null;

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
  clouds?: CloudSkyLayer | null;
  sunBall?: THREE.Object3D | null;
  onSunChange?: (() => void) | null;
  envIntensityMin?: number;
  envIntensityMax?: number;
}

export class SkyEditor {
  sky: Sky;
  light: THREE.Light | null;
  renderer?: THREE.WebGLRenderer;
  clouds: CloudSkyLayer | null;
  sunBall: THREE.Object3D | null;
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
