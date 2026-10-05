import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// - State -
let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let arrowGroup = null;
let animId = null;
let legendHost = null;
// The dataset on screen and the largest speed seen in it, so stepping through time keeps the
// camera where the user put it and colours that compare across frames.
let currentDatasetId = null;
let scaleMax = 0;

// - Depth exaggeration (matches hydro viewer: 10x base * slider) -
const DEPTH_EXAG = 8.0;  // equivalent to hydro viewer slider default
const DEPTH_BASE_MULT = 10.0;

// - Color map: depth-aware (cyan surface → teal mid → magenta deep) -
// Ported from hydro viewer oceanICColor
function depthColor(mag, maxMag, depthFrac) {
    const d = Math.min(1, depthFrac || 0);
    const t = Math.min(1, mag / Math.max(maxMag, 0.001));
    let r, g, b;

    if (d < 0.33) {
        // Electric cyan → vivid teal (surface)
        const s = d / 0.33;
        r = 0.0 + 0.1 * s;
        g = 1.0 - 0.2 * s;
        b = 1.0;
    } else if (d < 0.66) {
        // Vivid teal → hot blue (mid-depth)
        const s = (d - 0.33) / 0.33;
        r = 0.1 + 0.25 * s;
        g = 0.8 - 0.5 * s;
        b = 1.0;
    } else {
        // Hot blue → neon magenta (deep)
        const s = (d - 0.66) / 0.34;
        r = 0.35 + 0.65 * s;
        g = 0.3 - 0.15 * s;
        b = 1.0;
    }

    // Modulate brightness by velocity magnitude
    const bright = 0.4 + 0.6 * t;
    return new THREE.Color(r * bright, g * bright, b * bright);
}

// - Merge simple geometries -
// - Color map for single-level fields (atmosphere, or a surface-only ocean field): speed -
// Plasma stops (matplotlib), perceptually uniform and legible on the dark background.
const PLASMA = [
    [0.00, [0.050, 0.031, 0.529]],
    [0.25, [0.494, 0.012, 0.659]],
    [0.50, [0.800, 0.278, 0.471]],
    [0.75, [0.973, 0.584, 0.251]],
    [1.00, [0.941, 0.976, 0.129]],
];

function speedColor(mag, maxMag) {
    const t = Math.min(1, mag / Math.max(maxMag, 0.001));
    for (let i = 1; i < PLASMA.length; i++) {
        const [t1, c1] = PLASMA[i];
        if (t <= t1) {
            const [t0, c0] = PLASMA[i - 1];
            const s = (t - t0) / (t1 - t0);
            return new THREE.Color(
                c0[0] + (c1[0] - c0[0]) * s,
                c0[1] + (c1[1] - c0[1]) * s,
                c0[2] + (c1[2] - c0[2]) * s,
            );
        }
    }
    return new THREE.Color(...PLASMA[PLASMA.length - 1][1]);
}

// A speed legend over the 3D canvas: the plasma ramp from 0 to the frame's maximum speed.
function showSpeedLegend(maxMag) {
    hideSpeedLegend();
    if (!legendHost) return;
    if (getComputedStyle(legendHost).position === 'static') legendHost.style.position = 'relative';
    const stops = PLASMA.map(([s, c]) =>
        `rgb(${c.map(x => Math.round(x * 255)).join(',')}) ${s * 100}%`).join(', ');
    const el = document.createElement('div');
    el.id = 'preview3dLegend';
    el.style.cssText = 'position:absolute;right:16px;bottom:16px;padding:8px 10px;border-radius:6px;' +
        'background:rgba(13,17,23,0.75);color:#c9d1d9;font:12px system-ui,sans-serif;pointer-events:none;';
    el.innerHTML = `<div style="margin-bottom:4px">Speed (m/s)</div>` +
        `<div style="width:160px;height:10px;border-radius:2px;background:linear-gradient(to right, ${stops})"></div>` +
        `<div style="display:flex;justify-content:space-between;margin-top:2px"><span>0</span><span>${maxMag.toFixed(1)}</span></div>`;
    legendHost.appendChild(el);
}

function hideSpeedLegend() {
    const el = document.getElementById('preview3dLegend');
    if (el) el.remove();
}

function mergeGeos(geos) {
    let totalV = 0, allIdx = [];
    for (const g of geos) { totalV += g.attributes.position.count; }
    const pos = new Float32Array(totalV * 3);
    const nor = new Float32Array(totalV * 3);
    let vOff = 0;
    for (const g of geos) {
        const p = g.attributes.position, n = g.attributes.normal;
        for (let i = 0; i < p.count; i++) {
            pos[(vOff + i) * 3] = p.getX(i); pos[(vOff + i) * 3 + 1] = p.getY(i); pos[(vOff + i) * 3 + 2] = p.getZ(i);
            if (n) { nor[(vOff + i) * 3] = n.getX(i); nor[(vOff + i) * 3 + 1] = n.getY(i); nor[(vOff + i) * 3 + 2] = n.getZ(i); }
        }
        if (g.index) { for (let i = 0; i < g.index.count; i++) allIdx.push(g.index.array[i] + vOff); }
        vOff += p.count;
    }
    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    merged.setIndex(allIdx);
    return merged;
}

function initThree(container) {
    legendHost = container;
    const canvas = document.createElement('canvas');
    canvas.id = 'preview3dCanvas';
    container.appendChild(canvas);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0d1117);
    renderer.setSize(container.clientWidth, container.clientHeight);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 5000);
    camera.position.set(0, 150, 250);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Lighting: ambient + directional + uplight for subsurface visibility
    scene.add(new THREE.AmbientLight(0x404060, 0.8));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(100, 200, 150);
    scene.add(dir);
    // Uplight from below to illuminate subsurface arrows
    const uplight = new THREE.DirectionalLight(0x334466, 0.4);
    uplight.position.set(0, -100, 0);
    scene.add(uplight);

    // Grid helper at surface (y=0)
    const grid = new THREE.GridHelper(200, 20, 0x1a2030, 0x1a2030);
    scene.add(grid);

    function animate() {
        animId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    const ro = new ResizeObserver(() => {
        const w = container.clientWidth, h = container.clientHeight;
        if (w > 0 && h > 0) {
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        }
    });
    ro.observe(container);
}

function clearArrows() {
    hideSpeedLegend();
    if (arrowGroup) {
        scene.remove(arrowGroup);
        arrowGroup.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
        arrowGroup = null;
    }
}

function buildVectorField(data, fitCamera) {
    clearArrows();
    if (!data.vectors || data.vectors.length === 0) return;

    arrowGroup = new THREE.Group();

    const [lonMin, latMin, lonMax, latMax] = data.bounds;
    const lonCenter = (lonMin + lonMax) / 2;
    const latCenter = (latMin + latMax) / 2;
    let cc = latCenter; if (cc > 90 || cc < -90) cc = 0; const cosLat = Math.cos(cc * Math.PI / 180);
    const degToM = 111320;

    // Scale scene to ~200 units across
    const lonSpan = Math.max(lonMax - lonMin, 0.001) * cosLat * degToM;
    const latSpan = Math.max(latMax - latMin, 0.001) * degToM;
    const maxSpan = Math.max(lonSpan, latSpan);
    const sceneScale = 200 / maxSpan;

    // Compute max magnitude
    let maxMag = 0;
    for (const v of data.vectors) {
        const mag = Math.sqrt(v.u * v.u + v.v * v.v);
        if (mag > maxMag) maxMag = mag;
    }
    scaleMax = Math.max(scaleMax, maxMag);
    maxMag = scaleMax;

    // Build arrow geometry (shaft + cone)
    const shaft = new THREE.CylinderGeometry(0.12, 0.12, 1, 6);
    shaft.translate(0, 0.5, 0);
    shaft.rotateX(Math.PI / 2);
    const cone = new THREE.ConeGeometry(0.3, 0.5, 6);
    cone.rotateX(Math.PI / 2);
    cone.translate(0, 0, 1.0);
    const arrowGeo = mergeGeos([shaft, cone]);

    const count = data.vectors.length;
    // One level (an atmosphere, or a surface-only ocean field): colour by speed, not depth.
    const singleLevel = !data.depth_levels || data.depth_levels.length <= 1;
    // depthTest: false so subsurface arrows render through the grid plane
    // Unlit for a single level, so the speed colour map shows true; shaded for depth layers.
    const mat = singleLevel
        ? new THREE.MeshBasicMaterial({ depthTest: false })
        : new THREE.MeshPhongMaterial({ flatShading: true, depthTest: false });
    const mesh = new THREE.InstancedMesh(arrowGeo, mat, count);
    mesh.renderOrder = 10;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < count; i++) {
        const v = data.vectors[i];
        const mag = Math.sqrt(v.u * v.u + v.v * v.v);

        const sx = (v.lon - lonCenter) * cosLat * degToM * sceneScale;
        const sz = -(v.lat - latCenter) * degToM * sceneScale;
        // Vertical exaggeration: depth (negative meters) * exag * base_mult * sceneScale
        const sy = v.depth * DEPTH_EXAG * DEPTH_BASE_MULT * sceneScale;

        const rotY = Math.atan2(v.u, -v.v);
        const arrowLen = Math.max(1, Math.min(6, (mag / Math.max(maxMag, 0.001)) * 5));

        dummy.position.set(sx, sy, sz);
        dummy.rotation.set(0, rotY, 0);
        dummy.scale.set(arrowLen, arrowLen, arrowLen);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        mesh.setColorAt(i, singleLevel ? speedColor(mag, maxMag) : depthColor(mag, maxMag, v.depth_frac || 0));
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    arrowGroup.add(mesh);

    // Add depth level planes as subtle reference
    for (const d of data.depth_levels) {
        if (d === 0) continue;
        const planeY = d * DEPTH_EXAG * DEPTH_BASE_MULT * sceneScale;
        const planeGeo = new THREE.PlaneGeometry(200, 200);
        const planeMat = new THREE.MeshBasicMaterial({
            color: 0x1a2a40, transparent: true, opacity: 0.12,
            side: THREE.DoubleSide, depthTest: false
        });
        const plane = new THREE.Mesh(planeGeo, planeMat);
        plane.rotation.x = -Math.PI / 2;
        plane.position.y = planeY;
        plane.renderOrder = 5;
        arrowGroup.add(plane);
    }

    scene.add(arrowGroup);
    if (singleLevel) showSpeedLegend(maxMag);

    // Fit camera to vector field bounds, only for a newly selected dataset
    if (!fitCamera) return;
    const box = new THREE.Box3().setFromObject(arrowGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const dist = Math.max(size.x, size.y, size.z) * 1.2;
    controls.target.copy(center);
    camera.position.set(center.x + dist * 0.3, center.y + dist * 0.5, center.z + dist * 0.8);
    controls.update();
}

// - Public API (called from app.js) -
window.preview3d = {
    active: false,

    show(container) {
        // Re-init if canvas was destroyed (e.g. by 2D preview's innerHTML wipe)
        const existing = document.getElementById('preview3dCanvas');
        if (!renderer || !existing) {
            if (renderer) { renderer.dispose(); renderer = null; scene = null; camera = null; controls = null; }
            initThree(container);
        }
        const canvas = document.getElementById('preview3dCanvas');
        if (canvas) canvas.style.display = 'block';
        this.active = true;
    },

    hide() {
        const canvas = document.getElementById('preview3dCanvas');
        if (canvas) canvas.style.display = 'none';
        this.active = false;
        currentDatasetId = null;
        clearArrows();
    },

    async load(datasetId, timeIdx) {
        const url = `/api/v1/cache/preview3d?dataset_id=${encodeURIComponent(datasetId)}&time_idx=${timeIdx || 0}`;
        try {
            const res = await fetch(url);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            const newDataset = datasetId !== currentDatasetId;
            if (newDataset) { currentDatasetId = datasetId; scaleMax = 0; }
            buildVectorField(data, newDataset);
            return { count: data.count, u_var: data.u_var, v_var: data.v_var, depths: data.depth_levels.length, time: data.time };
        } catch (e) {
            console.error('3D preview failed:', e);
            throw e;
        }
    },

    dispose() {
        clearArrows();
        if (animId) cancelAnimationFrame(animId);
        if (renderer) renderer.dispose();
        renderer = null; scene = null; camera = null; controls = null;
        this.active = false;
    }
};
