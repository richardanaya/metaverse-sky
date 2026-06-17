import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MetaverseSky } from 'metaverse-sky';

const canvas = document.querySelector('canvas');

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x9fb7d5, 300, 700);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 900);
camera.position.set(52, 42, 96);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.6;

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 12, 0);
controls.update();

const sun = new THREE.DirectionalLight(0xffffff, 2.2);
scene.add(sun);
scene.add(new THREE.AmbientLight(0x8090a8, 0.6));

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(90, 96),
  new THREE.MeshStandardMaterial({ color: 0x42566a, roughness: 0.95 }),
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const atmosphere = new MetaverseSky({ scene, camera, renderer, light: sun, clouds: true });

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

const clock = new THREE.Clock();

function animate() {
  const dt = clock.getDelta();
  controls.update();
  atmosphere.update(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

window.addEventListener('resize', resize);
resize();
animate();
