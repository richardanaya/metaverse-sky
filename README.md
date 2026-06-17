# metaverse-sky

Three.js sky and atmosphere library with Preetham atmospheric scattering, sun helpers, IBL intensity syncing, a procedural voxel cloud deck, and an optional sky editor panel.

## Install

```bash
npm install metaverse-sky three
```

Requires `three` >= 0.160 as a peer dependency.

## CDN Import Map

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three/examples/jsm/",
    "metaverse-sky": "https://cdn.jsdelivr.net/npm/metaverse-sky/index.js"
  }
}
</script>
```

## Minimal Usage

```js
import * as THREE from 'three';
import { MetaverseSky } from 'metaverse-sky';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 900);
const renderer = new THREE.WebGLRenderer({ antialias: true });
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

## Exports

| Export | Purpose |
|--------|---------|
| `MetaverseSky` | Owns a Three.js `Sky`, optional `CloudLayer`, sun state, and environment intensity syncing |
| `CloudLayer` | Procedural scrolling voxel cloud deck from `metaverse-world` |
| `SkyEditor` | DOM panel for sun, atmosphere, exposure, IBL, and cloud settings |
| `createAtmosphereSky` | Create and initialize a Three.js `Sky` object |
| `setSkySun` | Apply elevation and azimuth to sky uniforms and optional light |
| `syncEnvironmentIntensity` | Update `scene.environmentIntensity` and material `envMapIntensity` from sun elevation |
| `sunDirectionFromAngles` | Convert elevation and azimuth to a normalized sun direction |
| `getSunAnglesFromDirection` | Convert a sun direction to elevation and azimuth |

## Example

```bash
cd metaverse-sky
python3 -m http.server 8080
```

Open [http://localhost:8080/example/simple/](http://localhost:8080/example/simple/).

## License

MIT - Richard Anaya. See [LICENSE](LICENSE).
