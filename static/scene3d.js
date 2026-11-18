// The 3D preview: a store's own cells at the resolution of the data and no finer.
//
//   * Surface: one flat cell per element at the sea surface, see-through, in the field's color
//     table (one byte per element, looked up in a shader), with faint cell edges that fade out
//     where they would be closer than a few pixels on screen.
//   * Bed: lit and tinted from shallow to deep. A mesh with the depth at its nodes is drawn as
//     its own faceted triangles; otherwise every cell is flat at its own bed, with a wall where a
//     neighbor is shallower (a staircase), so nothing is smoothed.
//   * Tracers (on by default where the store has a current or wind): in a thinned set of columns
//     and up to six levels down each, one short segment per cell moving along that cell's
//     velocity and wrapping inside the cell (its chord through the centroid along the flow), so
//     it shows direction and speed without travelling into structure the data does not hold.
//     Colored by depth in the column.
//   * Level maps (off by default): up to four layers drawn flat per cell at their own heights.
//   * The vertical section, when one is drawn on the map, hangs as a curtain.
//
// Records are held, never blended; tracers move continuously over the record on screen.
// Navigation is vantagekit (orbit and fly, keys while the canvas has focus).

import * as THREE from "three";
import CameraControls from "camera-controls";
import { createVantage } from "vantagekit/three";
import * as api from "./view_api.js";

const SPAN_UNITS = 200;          // the domain is scaled to about this many scene units across
const DEG = 111320;              // meters per degree of latitude
const MAX_LEVEL_MAPS = 4;
export const SURFACE_OPACITY = 0.6;
const LEVEL_OPACITY = 0.5;
const SURFACE_LIFT_M = 0.5;      // the surface cells sit this far above still water
const EDGE_OPACITY = 0.45;
// The bed's tint, shallow to deep (reached at BED_DEEP_M), for the dark page: a near-colorless
// slate kept in its own dark lightness band (about L* 15 and below), so the pale tracers and the
// surface colors never sink into it.
const BED_SHALLOW = 0x2b2f34, BED_DEEP = 0x121417, BED_DEEP_M = 60;
// Tracers: model seconds shown per display second, trail length (model s), at most this many
// crossings of a cell a second, and a tail at most this share of the cell.
const TRACER_RATE = 1800, TRACER_TRAIL = 3600, CROSSINGS_MAX = 1.5, TAIL_MAX = 0.9;
// Tracer depth as a share of the water column: pale at the surface to deep blue at the bed.
export const DEPTH_RAMP = [[191, 224, 242], [141, 184, 222], [107, 149, 207], [85, 119, 189], [74, 98, 168]];

function depthRgb(t) {
    const s = Math.min(3.999, Math.max(0, t) * 4), i = Math.floor(s), u = s - i, r = DEPTH_RAMP;
    return [0, 1, 2].map((c) => (r[i][c] + u * (r[i + 1][c] - r[i][c])) / 255);
}
const toLinear = (c) => c.map((v) => Math.pow(v, 2.2));

// The field's 256-entry color table as an sRGB texture, so the shader samples linear colors.
function lutTexture(rgba) {
    const tex = new THREE.DataTexture(rgba, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = tex.minFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    return tex;
}

// Triangles colored by a per-face byte code (attribute `code`); code 0 (no value) is not drawn.
function codedMaterial(lut, opacity) {
    return new THREE.ShaderMaterial({
        uniforms: { lut: { value: lut }, opacity: { value: opacity } },
        vertexShader: `attribute float code; varying float vCode;
            void main() { vCode = code; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: `uniform sampler2D lut; uniform float opacity; varying float vCode;
            void main() {
                if (vCode < 0.5) discard;
                gl_FragColor = vec4(texture2D(lut, vec2((vCode + 0.5) / 256.0, 0.5)).rgb, opacity);
                #include <colorspace_fragment>
            }`,
        side: THREE.DoubleSide,
        transparent: opacity < 1,
        depthWrite: opacity >= 1,
    });
}

// Cell edges (attribute `len`, the edge's length in scene units), fading out where an edge would
// span fewer than about 4 px, so a dense mesh seen from afar does not alias.
function edgeMaterial(opacity) {
    return new THREE.ShaderMaterial({
        uniforms: { opacity: { value: opacity }, pxPerRad: { value: 1000 } },
        transparent: true,
        depthWrite: false,
        vertexShader: `attribute float len; uniform float pxPerRad; varying float vPx;
            void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0);
                vPx = len * pxPerRad / max(-mv.z, 1e-3); gl_Position = projectionMatrix * mv; }`,
        fragmentShader: `uniform float opacity; varying float vPx;
            void main() { float a = opacity * clamp((vPx - 3.0) / 6.0, 0.0, 1.0);
                if (a < 0.004) discard; gl_FragColor = vec4(0.0, 0.0, 0.0, a); }`,
    });
}

// Tracers as one LineSegments: head and tail of each segment share its attributes. Positions are
// meters (east, up, north) about the scene center; the shader scales them.
const TRACER_VERT = `
attribute vec3 aCentre; attribute vec2 aDir; attribute vec4 aMove; attribute float aEnd; attribute vec3 aColour;
uniform float time; uniform float exag; uniform float invMpu;
varying vec3 vColour; varying float vAlpha;
void main() {
    float f = fract(time * aMove.y + aMove.z);
    float head = (f - 0.5) * aMove.x;
    float along = aEnd < 0.5 ? head : max(head - aMove.w, -0.5 * aMove.x);
    vec3 p = vec3((aCentre.x + aDir.x * along) * invMpu, aCentre.y * exag * invMpu, -(aCentre.z + aDir.y * along) * invMpu);
    vColour = aColour;
    vAlpha = (aEnd < 0.5 ? 1.0 : 0.15) * smoothstep(0.0, 0.08, f) * smoothstep(1.0, 0.92, f);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;
const TRACER_FRAG = `
varying vec3 vColour; varying float vAlpha;
void main() { gl_FragColor = vec4(vColour, vAlpha);
    #include <colorspace_fragment>
}`;

class TracerLines {
    constructor() {
        const geo = new THREE.BufferGeometry();
        this.object = new THREE.LineSegments(geo, new THREE.ShaderMaterial({
            vertexShader: TRACER_VERT, fragmentShader: TRACER_FRAG, transparent: true, depthWrite: false,
            uniforms: { time: { value: 0 }, exag: { value: 1 }, invMpu: { value: 1 } },
        }));
        this.object.frustumCulled = false;
        this.object.renderOrder = 6;
    }

    set(items, exag, invMpu) {
        const n = items.length, g = new THREE.BufferGeometry();
        const cen = new Float32Array(n * 6), dir = new Float32Array(n * 4), mov = new Float32Array(n * 8);
        const end = new Float32Array(n * 2), col = new Float32Array(n * 6);
        items.forEach((it, q) => {
            const c = toLinear(depthRgb(it.depthFrac));
            const phase = (q * 0.61803398875) % 1;
            for (let e = 0; e < 2; e++) {
                const p = q * 2 + e;
                cen.set([it.x, it.y, it.n], p * 3);
                dir.set([it.ex, it.en], p * 2);
                mov.set([it.across, it.rate, phase, it.tail], p * 4);
                end[p] = e;
                col.set(c, p * 3);
            }
        });
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 6), 3));
        g.setAttribute("aCentre", new THREE.BufferAttribute(cen, 3));
        g.setAttribute("aDir", new THREE.BufferAttribute(dir, 2));
        g.setAttribute("aMove", new THREE.BufferAttribute(mov, 4));
        g.setAttribute("aEnd", new THREE.BufferAttribute(end, 1));
        g.setAttribute("aColour", new THREE.BufferAttribute(col, 3));
        this.object.geometry.dispose();
        this.object.geometry = g;
        this.setScale(exag, invMpu);
        this.object.visible = n > 0;
    }

    setScale(exag, invMpu) {
        this.object.material.uniforms.exag.value = exag;
        this.object.material.uniforms.invMpu.value = invMpu;
    }

    tick(seconds) { this.object.material.uniforms.time.value = seconds % 3600; }
    dispose() { this.object.geometry.dispose(); this.object.material.dispose(); }
}

// The stretch [a, b] (meters along the line) of the line through (x, y) along unit (dx, dy) that
// lies inside the triangle with corners P, or null.
function chord(P, x, y, dx, dy) {
    const area = (P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[1][1] - P[0][1]) * (P[2][0] - P[0][0]);
    let a = -Infinity, b = Infinity;
    for (let k = 0; k < 3; k++) {
        const e0 = P[k], e1 = P[(k + 1) % 3];
        const nx = (e1[1] - e0[1]) * Math.sign(area), ny = -(e1[0] - e0[0]) * Math.sign(area);  // outward
        const num = nx * (e0[0] - x) + ny * (e0[1] - y), den = nx * dx + ny * dy;
        if (Math.abs(den) < 1e-12) { if (num < 0) return null; continue; }
        const t = num / den;
        if (den > 0) b = Math.min(b, t); else a = Math.max(a, t);
    }
    return a < b ? [a, b] : null;
}

export class Scene3D {
    constructor(container) {
        this.container = container;
        this.exag = 1;
        this.surfaceOpacity = SURFACE_OPACITY;
        this.levels = [];          // level maps shown (layer indices), none by default
        this.showSurface = true;
        this.showTracers = true;
        this.showBed = true;
        this.section = null;       // {data, field} from the controller
        this.objects = { surface: null, edges: null, bed: null, levels: new Map(), curtain: null };
        this.tracers = new TracerLines();
        this.token = 0;
        this._init();
    }

    _init() {
        const c = this.container;
        this.canvas = document.createElement("canvas");
        this.canvas.className = "scene3d-canvas";
        this.canvas.tabIndex = 0;
        c.appendChild(this.canvas);
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x0d1117);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 20000);
        this.camera.position.set(0, 150, 250);
        this.nav = createVantage({
            THREE, CameraControls, camera: this.camera, dom: this.canvas,
            metersPerUnit: 1, exaggeration: () => this.exag,
            keys: "focus", minDistance: 0.5, smoothTime: 0.2,
            // Double-click the surface, a level map, the bed or the curtain to turn about it.
            pivotPick: { objects: () => [this.objects.surface, this.objects.bed, this.objects.curtain,
                ...this.objects.levels.values()].filter(Boolean) },
        });
        this.scene.add(new THREE.HemisphereLight(0xdfe8f2, 0x202428, 1.6));
        const sun = new THREE.DirectionalLight(0xffffff, 1.4);
        sun.position.set(-40, 90, -30);
        this.scene.add(sun);
        this.scene.add(this.tracers.object);
        const clock = new THREE.Clock();
        let elapsed = 0;
        const loop = () => {
            this.raf = requestAnimationFrame(loop);
            const dt = clock.getDelta();
            elapsed += dt;
            this.nav.update(dt);
            this.tracers.tick(elapsed);
            if (this.objects.edges) {
                const h = this.renderer.domElement.clientHeight || 1;
                this.objects.edges.material.uniforms.pxPerRad.value = h / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)));
            }
            this.renderer.render(this.scene, this.camera);
        };
        loop();
        this.ro = new ResizeObserver(() => this.resize());
        this.ro.observe(c);
        this.resize();
    }

    resize() {
        const w = this.container.clientWidth, h = this.container.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    // A new store: its 3D manifest and cells.
    async load(id, m3, geom) {
        this.clear();
        this.id = id;
        this.m = m3;
        this.geom = geom;
        const [w, s, e, n] = m3.bbox;
        this.lon0 = (w + e) / 2;
        this.lat0 = (s + n) / 2;
        this.kx = Math.cos(this.lat0 * Math.PI / 180) * DEG;
        const span = Math.max((e - w) * this.kx, (n - s) * DEG, 1);
        this.mpu = span / SPAN_UNITS;
        this.nav.metersPerUnit = this.mpu;
        // Node positions in meters east and north of the center.
        const nn = geom.lon.length;
        this.mx = new Float64Array(nn);
        this.my = new Float64Array(nn);
        for (let i = 0; i < nn; i++) {
            this.mx[i] = (geom.lon[i] - this.lon0) * this.kx;
            this.my[i] = (geom.lat[i] - this.lat0) * DEG;
        }
        this.bed = m3.has_bed ? await api.bed(id, "3d") : null;
        this.bedNodes = m3.bed_at_nodes ? await api.bed(id, "3d", "node") : null;
        this.levels = [];
        // Vertical range for the default exaggeration: the deepest layer or the bed.
        const zs = m3.layers.map((l) => l.z_mean).filter(Number.isFinite);
        let depth = zs.length ? Math.max(...zs.map(Math.abs)) : 0;
        if (this.bed) for (const b of this.bed) if (Number.isFinite(b)) depth = Math.max(depth, Math.abs(b));
        this.nominal = !zs.length && m3.vertical !== "surface";   // sigma layers with no depths
        this.exag = this.nominal ? 1 : niceExag(depth > 0 ? 0.2 * span / depth : 1);
        this._buildSurface();
        this._buildEdges();
        this._buildBed();
        this._fitPending = true;
        return { exag: this.exag, nominal: this.nominal };
    }

    // Scene position of a point in meters (east, up, north).
    _p(x, z, y) { return [x / this.mpu, z * this.exag / this.mpu, -y / this.mpu]; }

    _nominalZ(k) {
        // Layers without depths: stacked evenly down to 15% of the span, labeled as such.
        return -(k / Math.max(1, this.m.layers.length - 1)) * 0.15 * SPAN_UNITS * this.mpu;
    }

    _flatCells(z) {
        // Flat triangles, one per element, at height z(e) (meters); a cell without a height
        // collapses onto its first node.
        const tri = this.geom.tri, n = tri.length / 3, pos = new Float32Array(n * 9);
        for (let e = 0; e < n; e++) {
            const ze = z(e), ok = Number.isFinite(ze);
            for (let j = 0; j < 3; j++) {
                const v = tri[3 * e + (ok ? j : 0)];
                pos.set(this._p(this.mx[v], ok ? ze : 0, this.my[v]), 9 * e + 3 * j);
            }
        }
        return pos;
    }

    _buildSurface() {
        const n = this.geom.tri.length / 3, g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(this._flatCells(() => SURFACE_LIFT_M), 3));
        g.setAttribute("code", new THREE.BufferAttribute(new Float32Array(n * 3), 1));
        const mesh = new THREE.Mesh(g, codedMaterial(null, this.surfaceOpacity));
        mesh.renderOrder = 5;
        this.objects.surface = mesh;
        this.scene.add(mesh);
    }

    // The cells' own edges, once each; a grid's diagonal (inside one cell) is left out.
    _buildEdges() {
        const tri = this.geom.tri, n = tri.length / 3, seen = new Map(), pos = [], len = [];
        const grid = this.m.store === "grid";
        for (let e = 0; e < n; e++) {
            for (let j = 0; j < 3; j++) {
                const a = tri[3 * e + j], b = tri[3 * e + (j + 1) % 3];
                const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
                const other = seen.get(key);
                if (other === undefined) { seen.set(key, e); continue; }
                seen.set(key, -1);
                if (grid && (other >> 1) === (e >> 1)) seen.set(key, -2);   // the diagonal
            }
        }
        for (const [key, owner] of seen) {
            if (owner === -2) continue;
            const a = Math.floor(key / 4294967296), b = key % 4294967296;
            pos.push(...this._p(this.mx[a], SURFACE_LIFT_M + 0.1, this.my[a]), ...this._p(this.mx[b], SURFACE_LIFT_M + 0.1, this.my[b]));
            const l = Math.hypot(this.mx[a] - this.mx[b], this.my[a] - this.my[b]) / this.mpu;
            len.push(l, l);
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
        g.setAttribute("len", new THREE.BufferAttribute(new Float32Array(len), 1));
        const lines = new THREE.LineSegments(g, edgeMaterial(EDGE_OPACITY));
        lines.renderOrder = 7;
        this.objects.edges = lines;
        this.scene.add(lines);
    }

    _buildBed() {
        if (this.objects.bed) { this.scene.remove(this.objects.bed); dispose(this.objects.bed); this.objects.bed = null; }
        if (!this.bed || this.nominal) return;
        const tri = this.geom.tri, n = tri.length / 3, verts = [], heights = [];
        const push = (x, z, y) => { verts.push(...this._p(x, z, y)); heights.push(z); };
        if (this.bedNodes) {
            // The mesh's own bed: each triangle linear between its nodes' depths.
            for (let e = 0; e < n; e++) {
                for (let j = 0; j < 3; j++) {
                    const v = tri[3 * e + j];
                    push(this.mx[v], this.bedNodes[v], this.my[v]);
                }
            }
        } else {
            // A staircase: every cell flat at its bed, a wall where a neighbor's bed is higher (up
            // to it) or where there is no neighbor (land, up to the surface).
            const b = this.bed;
            const owner = new Map();
            for (let e = 0; e < n; e++) {
                if (!Number.isFinite(b[e])) continue;
                const P = [0, 1, 2].map((j) => tri[3 * e + j]);
                for (const v of P) push(this.mx[v], b[e], this.my[v]);
                for (let j = 0; j < 3; j++) {
                    const a = P[j], c = P[(j + 1) % 3];
                    const key = a < c ? a * 4294967296 + c : c * 4294967296 + a;
                    if (owner.has(key)) owner.get(key).push(e); else owner.set(key, [e]);
                }
            }
            const wall = (a, c, lo, hi) => {
                push(this.mx[a], lo, this.my[a]); push(this.mx[c], lo, this.my[c]); push(this.mx[c], hi, this.my[c]);
                push(this.mx[a], lo, this.my[a]); push(this.mx[c], hi, this.my[c]); push(this.mx[a], hi, this.my[a]);
            };
            for (const [key, es] of owner) {
                const a = Math.floor(key / 4294967296), c = key % 4294967296;
                if (es.length === 1) wall(a, c, b[es[0]], 0);
                else if (b[es[0]] !== b[es[1]]) wall(a, c, Math.min(b[es[0]], b[es[1]]), Math.max(b[es[0]], b[es[1]]));
            }
        }
        const shallow = new THREE.Color(BED_SHALLOW), deep = new THREE.Color(BED_DEEP), tint = new THREE.Color();
        const col = new Float32Array(verts.length);
        heights.forEach((z, i) => {
            tint.lerpColors(shallow, deep, Math.min(1, Math.max(0, -z / BED_DEEP_M)));
            col.set([tint.r, tint.g, tint.b], i * 3);
        });
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
        g.setAttribute("color", new THREE.BufferAttribute(col, 3));
        g.computeVertexNormals();
        const mesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide }));
        mesh.renderOrder = 1;
        mesh.visible = this.showBed;
        this.objects.bed = mesh;
        this.scene.add(mesh);
    }

    // Draw a record: the surface, the level maps, the tracers and the curtain.
    async draw(field, t) {
        if (!this.m) return;
        const token = ++this.token;
        this.field = field;
        this.t = t;
        const f = this.m.fields.find((x) => x.name === field);
        if (!this.luts) this.luts = new Map();
        if (!this.luts.has(field)) this.luts.set(field, lutTexture(api.ramp(f)));
        const lut = this.luts.get(field);
        const levels = f.layered ? this.levels : [];
        const [codes, levelData] = await Promise.all([
            api.frame(this.id, field, 0, t, "3d"),
            Promise.all(levels.map(async (k) => ({
                k,
                codes: await api.frame(this.id, field, k, t, "3d"),
                z: this.nominal ? null : await api.heights(this.id, k, t, "3d"),
            }))),
        ]);
        if (token !== this.token) return;
        this._setCodes(this.objects.surface, codes, lut);
        this.objects.surface.visible = this.showSurface;
        this.objects.edges.visible = this.showSurface;
        const keep = new Set(levels);
        for (const [k, obj] of this.objects.levels) {
            if (!keep.has(k)) { this.scene.remove(obj); dispose(obj); this.objects.levels.delete(k); }
        }
        for (const { k, codes: c, z } of levelData) this._levelMap(k, c, z, lut);
        await this._tracers(t, token);
        this._curtain();
        if (this._fitPending) {
            this._fitPending = false;
            const box = new THREE.Box3().setFromObject(this.objects.surface);
            if (this.objects.bed) box.expandByObject(this.objects.bed);
            if (!box.isEmpty()) {
                this.nav.fitBox(box, { bearingDeg: 0, elevationDeg: 40, factor: 1.15 });
                this.nav.saveHome();
            }
        }
    }

    _setCodes(mesh, codes, lut) {
        const a = mesh.geometry.attributes.code, arr = a.array;
        for (let e = 0; e < codes.length; e++) arr[3 * e] = arr[3 * e + 1] = arr[3 * e + 2] = codes[e];
        a.needsUpdate = true;
        mesh.material.uniforms.lut.value = lut;
    }

    _levelMap(k, codes, z, lut) {
        let obj = this.objects.levels.get(k);
        const n = this.geom.tri.length / 3;
        const pos = this._flatCells((e) => (z ? z[e] : this._nominalZ(k)));
        if (!obj) {
            const g = new THREE.BufferGeometry();
            g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
            g.setAttribute("code", new THREE.BufferAttribute(new Float32Array(n * 3), 1));
            obj = new THREE.Mesh(g, codedMaterial(lut, LEVEL_OPACITY));
            obj.renderOrder = 4;
            this.objects.levels.set(k, obj);
            this.scene.add(obj);
        } else {
            obj.geometry.attributes.position.array.set(pos);
            obj.geometry.attributes.position.needsUpdate = true;
        }
        obj.geometry.computeBoundingSphere();
        this._setCodes(obj, codes, lut);
    }

    async _tracers(t, token) {
        const vec = this.m.vectors[0];
        if (!this.showTracers || !vec) { this.tracers.object.visible = false; return; }
        const d = await api.tracers(this.id, vec.name, t, "3d");
        if (token !== this.token) return;
        const tri = this.geom.tri, items = [], nl = Math.max(1, this.m.layers.length);
        d.element.forEach((e, c) => {
            const P = [0, 1, 2].map((j) => { const v = tri[3 * e + j]; return [this.mx[v], this.my[v]]; });
            const cx = (P[0][0] + P[1][0] + P[2][0]) / 3, cy = (P[0][1] + P[1][1] + P[2][1]) / 3;
            const bed = this.bed ? this.bed[e] : NaN;
            for (const L of d.levels) {
                const u = L.u[c], v = L.v[c];
                if (u == null || v == null) continue;
                const sp = Math.hypot(u, v);
                if (!(sp > 1e-3)) continue;
                const ex = u / sp, en = v / sp, ch = chord(P, cx, cy, ex, en);
                if (!ch) continue;
                const across = ch[1] - ch[0], mid = 0.5 * (ch[0] + ch[1]);
                const k = L.layer[c];
                const z = L.z ? L.z[c] : this.nominal ? this._nominalZ(k) : SURFACE_LIFT_M;
                if (z == null) continue;
                const depthFrac = Number.isFinite(bed) && bed < 0 ? Math.min(1, Math.max(0, z / bed)) : k / nl;
                items.push({ x: cx + ex * mid, y: z, n: cy + en * mid, ex, en, across,
                    rate: Math.min(CROSSINGS_MAX, sp * TRACER_RATE / across),
                    tail: Math.min(sp * TRACER_TRAIL, TAIL_MAX * across), depthFrac });
            }
        });
        this.tracers.set(items, this.exag, 1 / this.mpu);
        this.tracerCount = items.length;
    }

    setSection(section) {
        this.section = section;
        this._curtain();
    }

    _curtain() {
        if (this.objects.curtain) { this.scene.remove(this.objects.curtain); dispose(this.objects.curtain); this.objects.curtain = null; }
        const s = this.section;
        if (!s || !this.m || !s.data.z) return;
        const f = this.m.fields.find((x) => x.name === s.field);
        if (!f) return;
        if (!this.luts) this.luts = new Map();
        if (!this.luts.has(s.field)) this.luts.set(s.field, lutTexture(api.ramp(f)));
        const pos = [], code = [];
        for (const c of sectionCells(s.data)) {
            const k = api.encode(f, c.value);
            if (!k) continue;
            const x0 = (c.lon0 - this.lon0) * this.kx, y0 = (c.lat0 - this.lat0) * DEG;
            const x1 = (c.lon1 - this.lon0) * this.kx, y1 = (c.lat1 - this.lat0) * DEG;
            const a0 = this._p(x0, c.top, y0), a1 = this._p(x1, c.top, y1), b0 = this._p(x0, c.bottom, y0), b1 = this._p(x1, c.bottom, y1);
            pos.push(...a0, ...a1, ...b1, ...a0, ...b1, ...b0);
            for (let j = 0; j < 6; j++) code.push(k);
        }
        if (!pos.length) return;
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
        g.setAttribute("code", new THREE.BufferAttribute(new Float32Array(code), 1));
        const mesh = new THREE.Mesh(g, codedMaterial(this.luts.get(s.field), 1));
        mesh.renderOrder = 3;
        this.objects.curtain = mesh;
        this.scene.add(mesh);
    }

    setExaggeration(x) {
        const prev = this.exag;
        this.exag = x;
        this.nav.rescaleVertical(prev, x, this.nav.getTarget().y);
        if (!this.m) return;
        // Heights scale with the exaggeration: rebuild what carries them.
        this.scene.remove(this.objects.surface); dispose(this.objects.surface);
        this.scene.remove(this.objects.edges); dispose(this.objects.edges);
        this._buildSurface();
        this._buildEdges();
        this._buildBed();
        if (this.field != null) this.draw(this.field, this.t);
    }

    setLevels(list) { this.levels = list.slice(0, MAX_LEVEL_MAPS); }
    setOpacity(a) {
        this.surfaceOpacity = a;
        const m = this.objects.surface && this.objects.surface.material;
        if (m) { m.uniforms.opacity.value = a; m.transparent = a < 1; m.depthWrite = a >= 1; m.needsUpdate = true; }
    }
    setBed(on) { this.showBed = on; if (this.objects.bed) this.objects.bed.visible = on; }
    setSurface(on) {
        this.showSurface = on;
        if (this.objects.surface) this.objects.surface.visible = on;
        if (this.objects.edges) this.objects.edges.visible = on;
    }
    setTracers(on) { this.showTracers = on; if (this.field != null) this.draw(this.field, this.t); }

    clear() {
        for (const k of ["surface", "edges", "bed", "curtain"]) {
            if (this.objects[k]) { this.scene.remove(this.objects[k]); dispose(this.objects[k]); this.objects[k] = null; }
        }
        for (const o of this.objects.levels.values()) { this.scene.remove(o); dispose(o); }
        this.objects.levels.clear();
        this.tracers.object.visible = false;
        if (this.luts) { for (const t of this.luts.values()) t.dispose(); this.luts.clear(); }
        this.m = null;
    }

    show(on) { this.canvas.style.display = on ? "block" : "none"; if (on) this.resize(); }
}

function dispose(o) {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
}

function niceExag(x) {
    if (!(x > 1)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(x)));
    const m = x / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

// The cells of a section, as drawn: each sample spans halfway to its neighbors along the line,
// each layer from halfway to the layer above (the surface for the top layer) to halfway to the
// layer below (the bed for the bottom layer, where the store has one).
export function sectionCells(d) {
    const n = d.distance.length, out = [];
    const mid = (a, b) => (a + b) / 2;
    for (let i = 0; i < n; i++) {
        if (d.element[i] < 0) continue;
        const i0 = Math.max(0, i - 1), i1 = Math.min(n - 1, i + 1);
        const a = i === 0 ? 0 : 0.5, b = i === n - 1 ? 0 : 0.5;
        const lon0 = d.lon[i] + (d.lon[i0] - d.lon[i]) * a, lat0 = d.lat[i] + (d.lat[i0] - d.lat[i]) * a;
        const lon1 = d.lon[i] + (d.lon[i1] - d.lon[i]) * b, lat1 = d.lat[i] + (d.lat[i1] - d.lat[i]) * b;
        const x0 = d.distance[i] + (d.distance[i0] - d.distance[i]) * a;
        const x1 = d.distance[i] + (d.distance[i1] - d.distance[i]) * b;
        const L = d.values.length;
        for (let k = 0; k < L; k++) {
            const v = d.values[k][i];
            const zk = d.z ? d.z[k][i] : -k;
            if (v == null || zk == null) continue;
            const up = k > 0 && d.z && d.z[k - 1][i] != null ? d.z[k - 1][i] : null;
            const dn = k < L - 1 && d.z && d.z[k + 1][i] != null ? d.z[k + 1][i] : null;
            const top = up != null ? mid(up, zk) : dn != null ? zk + (zk - dn) / 2 : zk + 0.5;
            let bottom = dn != null ? mid(zk, dn) : up != null ? zk - (up - zk) / 2 : zk - 0.5;
            if (dn == null && d.bed && d.bed[i] != null && d.bed[i] < zk) bottom = d.bed[i];
            out.push({ i, k, value: v, lon0, lat0, lon1, lat1, x0, x1, top, bottom });
        }
    }
    return out;
}
