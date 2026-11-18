// The preview map: a store's own cells drawn flat, one color per cell, over the same basemap as
// the coverage map. A MapLibre custom layer (WebGL2) holds the triangles once; each record is one
// byte per cell, uploaded as a texture and looked up in the field's 256-entry color ramp, so
// stepping through records sends no geometry. Nothing is interpolated or blended.

const CODES_WIDTH = 4096;

const VERT = `#version 300 es
precision highp float;
precision highp int;
in vec2 a_pos;
uniform mat4 u_matrix;
uniform sampler2D u_codes;
uniform sampler2D u_ramp;
flat out vec4 v_color;
void main() {
    int e = gl_VertexID / 3;
    float c = texelFetch(u_codes, ivec2(e % ${CODES_WIDTH}, e / ${CODES_WIDTH}), 0).r * 255.0;
    v_color = texelFetch(u_ramp, ivec2(int(c + 0.5), 0), 0);
    gl_Position = u_matrix * vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision mediump float;
flat in vec4 v_color;
uniform float u_opacity;
out vec4 color;
void main() {
    if (v_color.a == 0.0) discard;
    color = vec4(v_color.rgb, 1.0) * (v_color.a * u_opacity);
}`;

function compile(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
}

// A MapLibre custom layer drawing triangles with one byte-coded color each.
class CellsLayer {
    constructor(id) {
        this.id = id;
        this.type = "custom";
        this.renderingMode = "2d";
        this.opacity = 0.9;
        this.count = 0;
        this.pending = null;
    }

    onAdd(map, gl) {
        this.map = map;
        if (typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) {
            this.unsupported = true;
            return;
        }
        this.gl = gl;
        const prog = gl.createProgram();
        gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
        gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
        this.prog = prog;
        this.loc = {
            pos: gl.getAttribLocation(prog, "a_pos"),
            matrix: gl.getUniformLocation(prog, "u_matrix"),
            codes: gl.getUniformLocation(prog, "u_codes"),
            ramp: gl.getUniformLocation(prog, "u_ramp"),
            opacity: gl.getUniformLocation(prog, "u_opacity"),
        };
        this.buf = gl.createBuffer();
        this.codesTex = this._texture();
        this.rampTex = this._texture();
        if (this.pending) {
            const p = this.pending;
            this.pending = null;
            if (p.geometry) this.setGeometry(...p.geometry);
            if (p.ramp) this.setRamp(p.ramp);
            if (p.codes) this.setCodes(p.codes);
        }
    }

    _texture() {
        const gl = this.gl;
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
    }

    _later(key, value) {
        this.pending = this.pending || {};
        this.pending[key] = value;
    }

    // Cells: node lon/lat (Float64Array) and triangles (Uint32Array). Positions are Mercator
    // offsets from the mesh's own origin, so float32 keeps them to well under a meter.
    setGeometry(lon, lat, tri) {
        if (!this.gl) { this._later("geometry", [lon, lat, tri]); return; }
        const n = tri.length / 3;
        let lo0 = 0, la0 = 0;
        for (let i = 0; i < lon.length; i++) { lo0 += lon[i]; la0 += lat[i]; }
        const origin = maplibregl.MercatorCoordinate.fromLngLat([lo0 / lon.length, la0 / lat.length]);
        this.origin = [origin.x, origin.y];
        const mx = new Float64Array(lon.length), my = new Float64Array(lon.length);
        for (let i = 0; i < lon.length; i++) {
            const m = maplibregl.MercatorCoordinate.fromLngLat([lon[i], lat[i]]);
            mx[i] = m.x - origin.x;
            my[i] = m.y - origin.y;
        }
        const pos = new Float32Array(n * 6);
        for (let k = 0; k < n * 3; k++) {
            const v = tri[k];
            pos[2 * k] = mx[v];
            pos[2 * k + 1] = my[v];
        }
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
        gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
        this.count = n * 3;
        this.nElem = n;
        this.map.triggerRepaint();
    }

    setRamp(rgba) {
        if (!this.gl) { this._later("ramp", rgba); return; }
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
        this.map.triggerRepaint();
    }

    setCodes(codes) {
        if (!this.gl) { this._later("codes", codes); return; }
        const gl = this.gl;
        const rows = Math.max(1, Math.ceil(codes.length / CODES_WIDTH));
        let data = codes;
        if (codes.length !== rows * CODES_WIDTH) {
            data = new Uint8Array(rows * CODES_WIDTH);
            data.set(codes);
        }
        gl.bindTexture(gl.TEXTURE_2D, this.codesTex);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, CODES_WIDTH, rows, 0, gl.RED, gl.UNSIGNED_BYTE, data);
        this.map.triggerRepaint();
    }

    setOpacity(a) {
        this.opacity = a;
        if (this.map) this.map.triggerRepaint();
    }

    render(gl, matrix) {
        if (!this.prog || !this.count) return;
        // The matrix maps Mercator (0 to 1) to clip space; fold in the mesh origin.
        const m = Array.from(matrix);
        const [ox, oy] = this.origin;
        for (let r = 0; r < 4; r++) m[12 + r] = matrix[r] * ox + matrix[4 + r] * oy + matrix[12 + r];
        gl.useProgram(this.prog);
        gl.uniformMatrix4fv(this.loc.matrix, false, new Float32Array(m));
        gl.uniform1f(this.loc.opacity, this.opacity);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.codesTex);
        gl.uniform1i(this.loc.codes, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
        gl.uniform1i(this.loc.ramp, 1);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
        gl.enableVertexAttribArray(this.loc.pos);
        gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArrays(gl.TRIANGLES, 0, this.count);
        gl.disableVertexAttribArray(this.loc.pos);
    }
}

// The preview map: cells layer, site circles (river mouths and other point stores), the section
// line being drawn, and pointer events for the readout and the section tool.
export class CellsMap {
    constructor(container, { onHover, onSectionDone } = {}) {
        this.container = container;
        this.onHover = onHover || (() => {});
        this.onSectionDone = onSectionDone || (() => {});
        this.sectionPts = null;
        this.ready = this._init();
    }

    async _init() {
        let style = window.fkCoverage ? await window.fkCoverage.basemapStyle()
            : "https://tiles.openfreemap.org/styles/dark";
        if (typeof style === "object") style = structuredClone(style);   // shared with the coverage map
        this.map = new maplibregl.Map({
            container: this.container,
            style,
            center: [-71, 40.5],
            zoom: 6,
            attributionControl: { compact: true },
        });
        this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
        this.map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
        await new Promise((r) => this.map.on("load", r));
        // The basemap credit behind its (i) button, as on the coverage map.
        if (window.fkCoverage) window.fkCoverage.collapseAttribution(this.map);
        this.cells = new CellsLayer("cells");
        this.map.addLayer(this.cells);
        const empty = { type: "FeatureCollection", features: [] };
        this.map.addSource("sites", { type: "geojson", data: empty });
        this.map.addLayer({
            id: "sites", type: "circle", source: "sites",
            paint: {
                "circle-radius": 7, "circle-color": ["get", "color"],
                "circle-stroke-color": "#0d1117", "circle-stroke-width": 1.5,
            },
        });
        this.map.addSource("section", { type: "geojson", data: empty });
        this.map.addLayer({
            id: "section-line", type: "line", source: "section",
            paint: { "line-color": "#ffffff", "line-width": 2, "line-dasharray": [2, 1] },
        });
        this.map.addLayer({
            id: "section-pts", type: "circle", source: "section", filter: ["==", "$type", "Point"],
            paint: { "circle-radius": 4, "circle-color": "#ffffff" },
        });
        this._wire();
        // The pane is hidden and shown with the tabs; follow the container's size.
        new ResizeObserver(() => this.map.resize()).observe(this.container);
        return this;
    }

    _wire() {
        const map = this.map;
        let last = 0, timer = null;
        map.on("mousemove", (e) => {
            clearTimeout(timer);
            const go = () => { last = performance.now(); this.onHover(e.lngLat, e.point); };
            if (performance.now() - last > 120) go(); else timer = setTimeout(go, 120);
        });
        map.on("mouseout", () => { clearTimeout(timer); this.onHover(null); });
        map.on("click", (e) => {
            if (!this.sectionPts) return;
            this.sectionPts.push([e.lngLat.lng, e.lngLat.lat]);
            this._drawSection();
        });
        map.on("dblclick", (e) => {
            if (!this.sectionPts) return;
            e.preventDefault();
            this.finishSection();
        });
    }

    startSection() {
        this.sectionPts = [];
        this.map.doubleClickZoom.disable();
        this.map.getCanvas().style.cursor = "crosshair";
        this._drawSection();
    }

    finishSection() {
        const pts = this.sectionPts || [];
        // A double-click also lands two clicks on the same spot; keep distinct points.
        const clean = pts.filter((p, i) => i === 0 || p[0] !== pts[i - 1][0] || p[1] !== pts[i - 1][1]);
        this.sectionPts = null;
        this.map.getCanvas().style.cursor = "";
        setTimeout(() => this.map.doubleClickZoom.enable(), 0);
        if (clean.length >= 2) {
            this.showSectionLine(clean);
            this.onSectionDone(clean);
        } else {
            this.showSectionLine(null);
        }
    }

    cancelSection() {
        if (!this.sectionPts) return;
        this.sectionPts = null;
        this.map.getCanvas().style.cursor = "";
        this.map.doubleClickZoom.enable();
        this.showSectionLine(null);
    }

    _drawSection() {
        this.showSectionLine(this.sectionPts);
    }

    showSectionLine(pts) {
        const src = this.map && this.map.getSource("section");
        if (!src) return;
        const features = [];
        if (pts && pts.length) {
            pts.forEach((p) => features.push({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
            if (pts.length > 1) features.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: pts } });
        }
        src.setData({ type: "FeatureCollection", features });
    }

    setCells(geom, rgba, fit) {
        this.cells.setGeometry(geom.lon, geom.lat, geom.tri);
        this.cells.setRamp(rgba);
        this.map.getSource("sites").setData({ type: "FeatureCollection", features: [] });
        if (fit) this.fit(fit);
    }

    setCodes(codes) { this.cells.setCodes(codes); }
    setRamp(rgba) { this.cells.setRamp(rgba); }
    setOpacity(a) { this.cells.setOpacity(a); }
    get unsupported() { return !!(this.cells && this.cells.unsupported); }

    clearCells() {
        if (this.cells) { this.cells.count = 0; this.map.triggerRepaint(); }
    }

    // Point stores: one circle per site, colored from the field's ramp by its byte code.
    setSites(sites, codes, rgba, fit) {
        this.clearCells();
        const features = sites.map((s, i) => {
            const c = codes ? codes[i] : 0;
            const color = c ? `rgb(${rgba[4 * c]},${rgba[4 * c + 1]},${rgba[4 * c + 2]})` : "#6e7681";
            return { type: "Feature", properties: { color, name: s.name, i },
                geometry: { type: "Point", coordinates: [s.lon, s.lat] } };
        });
        this.map.getSource("sites").setData({ type: "FeatureCollection", features });
        if (fit) this.fit(fit);
    }

    fit([w, s, e, n]) {
        this.map.fitBounds([[w, s], [e, n]], { padding: 40, animate: false, maxZoom: 14 });
    }

    resize() { if (this.map) this.map.resize(); }
}
