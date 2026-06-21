# metaverse-sky

Three.js WebGPU/TSL sky and atmosphere library with an analytic atmospheric dome, sun helpers, IBL intensity syncing, fast 2.5D sky-volume clouds, precipitation (rain/snow/hail), and an optional sky editor panel.

## Install

```bash
npm install metaverse-sky three
```

Requires `three` >= 0.160 as a peer dependency. Rendering uses Three.js Shading Language (TSL), so apps should use `WebGPURenderer` from `three/webgpu`.

## CDN Import Map

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three/build/three.module.js",
    "three/webgpu": "https://cdn.jsdelivr.net/npm/three/build/three.webgpu.js",
    "three/tsl": "https://cdn.jsdelivr.net/npm/three/build/three.tsl.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three/examples/jsm/",
    "metaverse-sky": "https://cdn.jsdelivr.net/npm/metaverse-sky/index.js"
  }
}
</script>
```

## Minimal Usage

```js
import * as THREE from 'three/webgpu';
import { MetaverseSky } from 'metaverse-sky';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 900);
const renderer = new THREE.WebGPURenderer({ antialias: true });
await renderer.init();
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.6;

const sun = new THREE.DirectionalLight(0xffffff, 2);
scene.add(sun);

const sky = new MetaverseSky({ scene, camera, renderer, light: sun, clouds: true });

function animate(dt) {
  sky.update(dt);
  renderer.render(scene, camera);
}
```

## API

### `MetaverseSky`

The main facade. Owns the sky, clouds, precipitation, sun state, and IBL syncing.

```js
const sky = new MetaverseSky({
  scene,          // required
  camera,         // required
  renderer,       // recommended (for exposure control)
  light,          // DirectionalLight that follows the sun
  clouds: true,   // enable cloud layer (default: true)
  precipitation: true,  // enable precipitation system (default: false)
  precipitationOptions: {
    textures: {   // sprite textures for each precip type
      snow: 'metaverse-sky/textures/snowflake.png',
      hail: 'metaverse-sky/textures/hailstone.png',
      rain: 'metaverse-sky/textures/raindrop.png',
    },
  },
});
```

#### Sun

```js
sky.setSun(45, 135);                    // by elevation/azimuth degrees
sky.setSunDirection(new THREE.Vector3(0.3, 0.8, 0.5));  // by direction vector
```

#### Wind (drives both clouds and precipitation)

```js
sky.setWind(255, 0.05);    // direction in degrees + speed
sky.setWind([0.8, 0.6], 0.1);  // or direction as [x, z] + speed
```

#### Clouds

```js
sky.setClouds({
  opacity: 0.9,
  coverage: 0.6,
  drawDistance: 420,
  detailStrength: 0.5,
  sharpness: 0.4,
  wispiness: 0.5,
  darkness: 0.3,
});
sky.getClouds();  // -> current cloud settings
```

#### Atmosphere (sky shader params)

```js
sky.setAtmosphere({ turbidity: 10, rayleigh: 2.5, exposure: 0.5 });
sky.setExposure(0.6);
sky.getExposure();  // -> 0.6
```

#### Precipitation

```js
sky.setPrecipitation({ type: 'snow', intensity: 0.8, speed: 1.5, size: 1.2 });
sky.setPrecipitationTextures({ snow: './my-snow.png' });
sky.getPrecipSettings();  // -> current precip settings
```

Types: `'none'`, `'rain'`, `'snow'`, `'hail'`. Each eases in/out over 2.4s when toggled.

#### Per-frame

```js
sky.update(deltaTime);  // call every frame
sky.dispose();          // cleanup
```

### Exports

| Export | Purpose |
|--------|---------|
| `MetaverseSky` | Facade: sky + clouds + precipitation + sun + IBL |
| `CloudSkyLayer` | Fast procedural 2.5D sky-volume cloud layer |
| `Precipitation` | Particle-based rain/snow/hail with wind response |
| `SkyEditor` | DOM panel for live-tweaking all settings |
| `createAtmosphereSky` | Create and initialize a Three.js `Sky` object |
| `setSkySun` | Apply elevation and azimuth to sky uniforms and optional light |
| `syncEnvironmentIntensity` | Update `scene.environmentIntensity` and material `envMapIntensity` from sun elevation |
| `sunDirectionFromAngles` | Convert elevation and azimuth to a normalized sun direction |
| `getSunAnglesFromDirection` | Convert a sun direction to elevation and azimuth |

## Examples

```bash
cd metaverse-sky
python3 -m http.server 8080
```

- [Minimal](http://localhost:8080/example/simple/) — smallest integration
- [Editor](http://localhost:8080/example/editor/) — full control panel with all sliders

## License

MIT - Richard Anaya. See [LICENSE](LICENSE).
