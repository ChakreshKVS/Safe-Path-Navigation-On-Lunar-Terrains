import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class LunarViewer {
    constructor(container) {
        this.container = container;
        this.init();
    }

    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);

        this.loadTextures();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    loadTextures() {
        const textureLoader = new THREE.TextureLoader();
        const heightMapLoader = new THREE.TextureLoader();

        Promise.all([
            new Promise(resolve => textureLoader.load('/assets/lunar-texture.jpg', resolve)),
            new Promise(resolve => heightMapLoader.load('/assets/height-map.jpg', resolve))
        ]).then(([texture, heightMap]) => {
            this.createTerrain(texture, heightMap);
            this.animate();
        });
    }

    createTerrain(texture, heightMap) {
        const geometry = new THREE.PlaneGeometry(10, 10, 256, 256);
        
        const material = new THREE.MeshStandardMaterial({
            map: texture,
            displacementMap: heightMap,
            displacementScale: 2.0,
            roughness: 0.8,
            metalness: 0.2
        });

        this.terrain = new THREE.Mesh(geometry, material);
        this.terrain.rotation.x = -Math.PI / 2;
        this.scene.add(this.terrain);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Initialize the viewer
const container = document.getElementById('container');
const viewer = new LunarViewer(container); 