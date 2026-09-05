import { useEffect, useRef } from "react";
import * as THREE from "three";

function Globe3D() {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;

    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      35,
      width / height,
      0.1,
      100
    );
    camera.position.set(0, 0.4, 3.4);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Earth sphere, textured with the real NASA image
    const textureLoader = new THREE.TextureLoader();
    const earthTexture = textureLoader.load(
      "/earth-blue-marble.jpg"
    );

    const earthGeometry = new THREE.SphereGeometry(1, 48, 48);
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughness: 0.85,
      metalness: 0.05,
    });
    const earth = new THREE.Mesh(earthGeometry, earthMaterial);
    scene.add(earth);

    // Soft atmospheric rim glow — a slightly larger
    // transparent sphere rendered from the inside
    const atmosphereGeometry = new THREE.SphereGeometry(
      1.04,
      48,
      48
    );
    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: 0x4dd8f0,
      transparent: true,
      opacity: 0.12,
      side: THREE.BackSide,
    });
    const atmosphere = new THREE.Mesh(
      atmosphereGeometry,
      atmosphereMaterial
    );
    scene.add(atmosphere);

    // Lighting — one directional "sun" plus soft ambient fill
    const sunLight = new THREE.DirectionalLight(0xffffff, 1.4);
    sunLight.position.set(3, 1.5, 2);
    scene.add(sunLight);

    const ambientLight = new THREE.AmbientLight(0x1a2a33, 1.1);
    scene.add(ambientLight);

    // Satellite — small body + two solar panel wings,
    // parented to a pivot so it genuinely orbits in 3D
    const satellitePivot = new THREE.Object3D();
    satellitePivot.rotation.z = 0.35;
    scene.add(satellitePivot);

    const satelliteGroup = new THREE.Group();
    satelliteGroup.position.set(1.7, 0.3, 0);

    const bodyGeometry = new THREE.BoxGeometry(0.09, 0.09, 0.09);
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0xcfd8dc,
      metalness: 0.6,
      roughness: 0.4,
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    satelliteGroup.add(body);

    const panelGeometry = new THREE.BoxGeometry(0.16, 0.06, 0.005);
    const panelMaterial = new THREE.MeshStandardMaterial({
      color: 0x2b6ea8,
      metalness: 0.3,
      roughness: 0.5,
    });

    const panelLeft = new THREE.Mesh(panelGeometry, panelMaterial);
    panelLeft.position.set(-0.14, 0, 0);
    satelliteGroup.add(panelLeft);

    const panelRight = new THREE.Mesh(panelGeometry, panelMaterial);
    panelRight.position.set(0.14, 0, 0);
    satelliteGroup.add(panelRight);

    satellitePivot.add(satelliteGroup);

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let animationFrameId = 0;
    const clock = new THREE.Clock();

    function renderFrame() {
      if (!prefersReducedMotion) {
        const delta = clock.getDelta();

        earth.rotation.y += delta * 0.12;
        satellitePivot.rotation.y += delta * 0.5;

        animationFrameId = requestAnimationFrame(renderFrame);
      }

      renderer.render(scene, camera);
    }

    renderFrame();

    function handleResize() {
      if (!container) return;

      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;

      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    }

    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", handleResize);

      earthGeometry.dispose();
      earthMaterial.dispose();
      atmosphereGeometry.dispose();
      atmosphereMaterial.dispose();
      bodyGeometry.dispose();
      bodyMaterial.dispose();
      panelGeometry.dispose();
      panelMaterial.dispose();
      earthTexture.dispose();
      renderer.dispose();

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div ref={containerRef} className="globe-3d-container" />;
}

export default Globe3D;