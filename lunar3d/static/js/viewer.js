import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class LunarViewer {
    constructor() {
        this.container = document.getElementById('viewer-container');
        this.init();
    }

    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, 3, 3);

        // Renderer setup with better shadows
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            shadowMap: true,
            precision: 'highp'
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Enhanced controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 2;
        this.controls.maxDistance = 8;
        this.controls.maxPolarAngle = Math.PI / 2;

        // Advanced lighting setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        this.scene.add(ambientLight);

        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(5, 5, 5);
        mainLight.castShadow = true;
        mainLight.shadow.mapSize.width = 2048;
        mainLight.shadow.mapSize.height = 2048;
        this.scene.add(mainLight);

        // Add rim light for crater edges
        const rimLight = new THREE.DirectionalLight(0x404040, 0.5);
        rimLight.position.set(-5, 3, -5);
        this.scene.add(rimLight);

        this.loadTextures();
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    loadTextures() {
        const textureLoader = new THREE.TextureLoader();
        const loadingManager = new THREE.LoadingManager();

        Promise.all([
            new Promise(resolve => textureLoader.load('/static/assets/lunar-texture.jpg', resolve)),
            new Promise(resolve => textureLoader.load('/static/assets/height-map.png', resolve))
        ]).then(([colorTexture, heightTexture]) => {
            // Enhance texture properties
            colorTexture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
            colorTexture.encoding = THREE.sRGBEncoding;
            
            // Process height map
            heightTexture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
            
            this.createTerrain(colorTexture, heightTexture);
            this.animate();
        }).catch(error => {
            console.error('Error loading textures:', error);
        });
    }

    createTerrain(colorTexture, heightTexture) {
        // Create high-resolution geometry
        const geometry = new THREE.PlaneGeometry(5, 5, 512, 512);

        // Generate normal map from height map
        const normalMap = this.generateNormalMap(heightTexture);

        // Create enhanced material
        const material = new THREE.MeshStandardMaterial({
            map: colorTexture,
            displacementMap: heightTexture,
            displacementScale: 0.3,
            displacementBias: -0.15,
            normalMap: normalMap,
            normalScale: new THREE.Vector2(1.5, 1.5),
            roughness: 0.85,
            metalness: 0.15,
            side: THREE.DoubleSide,
            // Enable better detail preservation
            flatShading: false,
            wireframe: false
        });

        // Create and position mesh
        this.terrain = new THREE.Mesh(geometry, material);
        this.terrain.rotation.x = -Math.PI / 2;
        this.terrain.castShadow = true;
        this.terrain.receiveShadow = true;
        this.scene.add(this.terrain);
    }

    generateNormalMap(heightTexture) {
        // Create normal map from height map
        const normalMap = heightTexture.clone();
        normalMap.wrapS = THREE.RepeatWrapping;
        normalMap.wrapT = THREE.RepeatWrapping;
        return normalMap;
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
new LunarViewer(); 