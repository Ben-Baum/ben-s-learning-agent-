import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

export class Char3DViewer {
  constructor(container, options = {}) {
    console.log('🚀 Char3DViewer constructor called');
    this.container = container;
    this.options = {
      width: options.width || container.clientWidth || 240,
      height: options.height || container.clientHeight || 280,
      modelPath: options.modelPath || '/models/character.glb',
      defaultState: {
        body: '#FAD4A0',
        outfit: '#1A2472',
        accent: '#F0C030',
        animation: 'idle',
        speed: 1.0,
        ...options.defaultState,
      },
      onProgress: options.onProgress || (() => {}),
      onLoad: options.onLoad || (() => {}),
      onError: options.onError || (() => {}),
    };

    this.state = { ...this.options.defaultState };
    this.state.body = new THREE.Color(this.state.body);
    this.state.outfit = new THREE.Color(this.state.outfit);
    this.state.accent = new THREE.Color(this.state.accent);

    this.meshGroups = { body: [], outfit: [], accent: [], all: [] };
    this.rafId = null;
    this.clock = new THREE.Clock();
    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.userYaw = 0;
    this.userPitch = 0;
    this.velocityYaw = 0;

    this.resizeObserver = null;
    this.intersectionObserver = null;
    this.isVisible = true;
    this.isPaused = false;

    this.boundPointerDown = this._onPointerDown.bind(this);
    this.boundPointerMove = this._onPointerMove.bind(this);
    this.boundPointerUp = this._onPointerUp.bind(this);

    this._initCanvas();
    this._initRenderer();
    this._initScene();
    this._initInteraction();
    this._initResizeObserver();
    this._initVisibility();
    this._animate();
    this._loadModel(this.options.modelPath);
  }

  _initCanvas() {
    let canvas = this.container.querySelector('canvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.className = 'char3d-canvas';
      this.container.appendChild(canvas);
    }
    this.canvas = canvas;
  }

  _initRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.options.width, this.options.height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
  }

  _initScene() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      35,
      this.options.width / this.options.height,
      0.1,
      100
    );
    this.camera.position.set(0, 1.4, 4.0);
    this.camera.lookAt(0, 1.0, 0);

    const hemi = new THREE.HemisphereLight(0xC8E0FF, 0x1A2472, 0.85);
    this.scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xFFE6B0, 1.4);
    keyLight.position.set(2.5, 4, 3);
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x00C2AE, 0.8);
    rimLight.position.set(-3, 2, -2);
    this.scene.add(rimLight);

    const fillLight = new THREE.DirectionalLight(0x7B68EE, 0.4);
    fillLight.position.set(0, -1, 2);
    this.scene.add(fillLight);

    this.root = new THREE.Group();
    this.scene.add(this.root);
  }

  _initInteraction() {
    this.canvas.addEventListener('mousedown', this.boundPointerDown);
    window.addEventListener('mousemove', this.boundPointerMove);
    window.addEventListener('mouseup', this.boundPointerUp);
    this.canvas.addEventListener('touchstart', this.boundPointerDown, {
      passive: false,
    });
    this.canvas.addEventListener('touchmove', this.boundPointerMove, {
      passive: false,
    });
    this.canvas.addEventListener('touchend', this.boundPointerUp);
  }

  _initResizeObserver() {
    this.resizeObserver = new ResizeObserver(() => {
      const w = this.container.clientWidth || this.options.width;
      const h = this.container.clientHeight || this.options.height;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    });
    this.resizeObserver.observe(this.container);
  }

  _initVisibility() {
    this.intersectionObserver = new IntersectionObserver(([entry]) => {
      this.isVisible = entry.isIntersecting;
    });
    this.intersectionObserver.observe(this.canvas);

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.isPaused = true;
      else this.isPaused = false;
    });
  }

  _onPointerDown(e) {
    this.dragging = true;
    const p = e.touches ? e.touches[0] : e;
    this.lastX = p.clientX;
    this.lastY = p.clientY;
    this.velocityYaw = 0;
  }

  _onPointerMove(e) {
    if (!this.dragging) return;
    e.preventDefault();
    const p = e.touches ? e.touches[0] : e;
    const dx = p.clientX - this.lastX;
    const dy = p.clientY - this.lastY;
    this.userYaw += dx * 0.01;
    this.userPitch = Math.max(-0.5, Math.min(0.5, this.userPitch + dy * 0.005));
    this.velocityYaw = dx * 0.01;
    this.lastX = p.clientX;
    this.lastY = p.clientY;
  }

  _onPointerUp() {
    this.dragging = false;
  }

  _classifyMesh(mesh) {
    if (!mesh.material) return 'outfit';

    // Try to classify by mesh name first
    const name = mesh.name.toLowerCase();
    if (name.includes('face') || name.includes('head') || name.includes('skin') || name.includes('body'))
      return 'body';
    if (name.includes('eye') || name.includes('iris') || name.includes('pupil'))
      return 'accent';
    if (name.includes('glass') || name.includes('shine') || name.includes('hair') || name.includes('highlight'))
      return 'accent';

    // Fallback to color-based classification
    const mat = Array.isArray(mesh.material)
      ? mesh.material[0]
      : mesh.material;
    if (!mat || !mat.color) return 'outfit';

    const c = mat.color;
    const hsl = { h: 0, s: 0, l: 0 };
    c.getHSL(hsl);

    // Skin tones: warm colors with good saturation and lightness
    if ((hsl.h < 0.08 || hsl.h > 0.95) && hsl.s > 0.1 && hsl.l > 0.3 && hsl.l < 0.8)
      return 'body';
    // Very bright parts (eyes, highlights, accessories)
    if (hsl.l > 0.8) return 'accent';
    // Dark parts (eyes pupils, shadows)
    if (hsl.l < 0.3) return 'accent';

    return 'outfit';
  }

  _applyColor(target) {
    const targetColor = this.state[target];
    const meshCount = this.meshGroups[target].length;
    console.log(`🎨 Applying ${target} color`, targetColor, `to ${meshCount} meshes`);
    this.meshGroups[target].forEach(({ mesh }) => {
      const mats = Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material];
      mats.forEach((mat) => {
        if (mat && mat.color) {
          mat.color.copy(targetColor);
          mat.needsUpdate = true;
        }
      });
    });
  }

  _loadModel(path) {
    console.log('📥 Loading model from:', path);
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);

    loader.load(
      path,
      (gltf) => {
        const model = gltf.scene;

        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;
        model.scale.setScalar(scale);
        model.position.sub(center.multiplyScalar(scale));
        model.position.y += (size.y * scale) / 2 - 0.05;

        model.traverse((child) => {
          if (child.isMesh && child.material) {
            const category = this._classifyMesh(child);
            if (Array.isArray(child.material)) {
              child.material = child.material.map((m) => m.clone());
            } else {
              child.material = child.material.clone();
            }
            const finalMat = Array.isArray(child.material)
              ? child.material[0]
              : child.material;
            const origColor = finalMat.color
              ? finalMat.color.clone()
              : new THREE.Color(0xffffff);
            this.meshGroups[category].push({ mesh: child, originalColor: origColor });
            this.meshGroups.all.push({ mesh: child, originalColor: origColor });
            console.log(`📦 Mesh: ${child.name} → ${category}`, child);
          }
        });
        console.log('📊 Mesh groups:', {
          body: this.meshGroups.body.length,
          outfit: this.meshGroups.outfit.length,
          accent: this.meshGroups.accent.length,
          total: this.meshGroups.all.length
        });

        this.root.add(model);

        this._applyColor('body');
        this._applyColor('outfit');
        this._applyColor('accent');

        this.options.onLoad();
      },
      (xhr) => {
        if (xhr.lengthComputable && xhr.total > 0) {
          const pct = (xhr.loaded / xhr.total) * 100;
          const mbLoaded = (xhr.loaded / 1024 / 1024).toFixed(1);
          const mbTotal = (xhr.total / 1024 / 1024).toFixed(1);
          this.options.onProgress(pct, mbLoaded, mbTotal);
        } else {
          const mbLoaded = (xhr.loaded / 1024 / 1024).toFixed(1);
          this.options.onProgress(null, mbLoaded, null);
        }
      },
      (err) => {
        console.error('GLB load error', err);
        this.options.onError(err);
      }
    );
  }

  _animate() {
    this.rafId = requestAnimationFrame(() => this._animate());

    if (!this.isVisible || this.isPaused) return;

    const t = this.clock.getElapsedTime() * this.state.speed;

    if (!this.dragging && Math.abs(this.velocityYaw) > 0.0001) {
      this.userYaw += this.velocityYaw;
      this.velocityYaw *= 0.93;
    }

    let baseY = 0,
      baseX = 0,
      baseRotY = this.userYaw,
      baseRotX = this.userPitch,
      scaleMul = 1;

    switch (this.state.animation) {
      case 'float':
        baseY = Math.sin(t * 1.5) * 0.1;
        baseRotY += Math.sin(t * 0.6) * 0.15;
        break;
      case 'bounce':
        baseY = Math.abs(Math.sin(t * 3.5)) * 0.18;
        scaleMul = 1 + Math.sin(t * 7) * 0.02;
        break;
      case 'sway':
        baseX = Math.sin(t * 2) * 0.1;
        baseRotX += Math.sin(t * 2) * 0.08;
        break;
      case 'spin':
        baseRotY += (t * 1.2) % (Math.PI * 2);
        baseY = Math.sin(t * 1.0) * 0.05;
        break;
      case 'wave':
        baseRotY += Math.sin(t * 2.5) * 0.4;
        baseRotX += Math.cos(t * 2.5) * 0.1;
        baseY = Math.sin(t * 5) * 0.04;
        break;
      case 'idle':
      default:
        scaleMul = 1 + Math.sin(t * 1.6) * 0.012;
        baseY = Math.sin(t * 0.9) * 0.03;
        break;
    }

    this.root.position.set(baseX, baseY, 0);
    this.root.rotation.y = baseRotY;
    this.root.rotation.x = baseRotX;
    this.root.scale.setScalar(scaleMul);

    this.renderer.render(this.scene, this.camera);
  }

  setColor(target, hex) {
    if (!this.state[target]) return;
    this.state[target].set(hex);
    this._applyColor(target);
  }

  setAnimation(name) {
    if (['idle', 'float', 'bounce', 'sway', 'spin', 'wave'].includes(name)) {
      this.state.animation = name;
    }
  }

  setSpeed(value) {
    this.state.speed = Math.max(0.1, Math.min(2.0, parseFloat(value)));
  }

  reset() {
    this.state.body.set(this.options.defaultState.body);
    this.state.outfit.set(this.options.defaultState.outfit);
    this.state.accent.set(this.options.defaultState.accent);
    this.state.animation = this.options.defaultState.animation;
    this.state.speed = this.options.defaultState.speed;
    this._applyColor('body');
    this._applyColor('outfit');
    this._applyColor('accent');
  }

  destroy() {
    if (this.rafId) cancelAnimationFrame(this.rafId);

    this.canvas.removeEventListener('mousedown', this.boundPointerDown);
    window.removeEventListener('mousemove', this.boundPointerMove);
    window.removeEventListener('mouseup', this.boundPointerUp);
    this.canvas.removeEventListener('touchstart', this.boundPointerDown);
    this.canvas.removeEventListener('touchmove', this.boundPointerMove);
    this.canvas.removeEventListener('touchend', this.boundPointerUp);

    if (this.resizeObserver) this.resizeObserver.disconnect();
    if (this.intersectionObserver) this.intersectionObserver.disconnect();

    this.scene.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach((m) => m.dispose());
        } else {
          object.material.dispose();
        }
      }
    });

    this.renderer.dispose();
    this.canvas.remove();
  }
}
