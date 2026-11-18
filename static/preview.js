// The Preview pane: the selected store on its own cells, as a map or in 3D, one record at a time.
// Field, layer and record choices, the legend, the readout under the pointer, the vertical
// section tool and the shareable link (#<store>&f=&k=&t=&v=) live here; drawing is in
// cells_map.js and scene3d.js, the server calls in view_api.js.

import * as api from "./view_api.js";
import { CellsMap } from "./cells_map.js";
import { Scene3D, sectionCells, DEPTH_RAMP } from "./scene3d.js";

const $ = (id) => document.getElementById(id);
const FRAME_MS = 500;   // the fastest playback step; slower records take as long as they load

const S = {
    item: null, m: null, geom: null, m3: null, geom3: null,
    field: null, layer: 0, t: 0, mode: "2d",
    playing: false, section: null, sectionPts: null, token: 0,
};
let map = null;
let scene = null;

const fieldOf = (m, name) => (m ? m.fields.find((f) => f.name === name) : null);
const nRecords = () => (S.m && S.m.times.length ? S.m.times.length : 1);
const isPoints = () => S.m && S.m.store === "points";

function setStatus(text, error) {
    const el = $("previewStatus");
    el.textContent = text || "";
    el.classList.toggle("error", !!error);
    el.hidden = !text;
}

// ---- opening a store ------------------------------------------------------------------------

async function open(item, want = {}) {
    const token = ++S.token;
    setPlaying(false);
    S.item = item;
    S.m = S.m3 = S.geom = S.geom3 = null;
    S.section = S.sectionPts = null;
    S._fitted = null;
    if (map && map.map) map.showSectionLine(null);
    if (scene) scene.setSection(null);
    $("sectionPanel").hidden = true;
    $("previewControls").classList.remove("hidden");
    $("previewEmpty").hidden = true;
    setStatus("Loading the store's cells");
    try {
        const m = await api.manifest(item.id, "2d");
        if (token !== S.token) return;
        S.m = m;
        S.field = fieldOf(m, want.f) ? want.f : m.fields[0].name;
        S.layer = Math.min(Math.max(0, +want.k || 0), Math.max(0, m.layers.length - 1));
        S.t = Math.min(Math.max(0, +want.t || 0), nRecords() - 1);
        fillControls();
        if (!map) {
            setStatus("Loading the basemap");
            map = new CellsMap($("cellsMap"), { onHover: hover, onSectionDone: sectionDone });
        }
        await map.ready;
        if (map.unsupported) {
            setStatus("This browser has no WebGL2, which the cell map needs.", true);
            return;
        }
        if (isPoints()) {
            S.geom = null;
            setStatus("");
        } else {
            S.geom = await api.mesh(item.id, "2d", m);
            if (token !== S.token) return;
            // Show the map before fitting it: a hidden map has no size to fit to.
            if (want.v !== "3d") await setMode("2d", true);
            map.setCells(S.geom, api.ramp(fieldOf(m, S.field)), m.bbox);
            setStatus("");
        }
        await setMode(want.v === "3d" && !isPoints() ? "3d" : "2d", true);
        await draw();
    } catch (e) {
        if (token !== S.token) return;
        console.error("preview:", e);
        setStatus(`This store cannot be previewed: ${e.message}`, true);
        $("previewControls").classList.add("hidden");
        if (map) { map.clearCells(); map.setSites([], null, null); }
    }
}

function close() {
    S.token++;
    setPlaying(false);
    S.item = S.m = null;
    $("previewControls").classList.add("hidden");
    $("previewEmpty").hidden = false;
    $("sectionPanel").hidden = true;
    setStatus("");
    if (map) map.clearCells();
}

// ---- controls -------------------------------------------------------------------------------

function fillControls() {
    const m = S.m;
    const sel = $("fieldSelect");
    sel.replaceChildren(...m.fields.map((f) => {
        const o = document.createElement("option");
        o.value = f.name;
        o.textContent = f.units ? `${f.label} (${f.units})` : f.label;
        return o;
    }));
    sel.value = S.field;
    const lr = $("layerRange");
    lr.max = Math.max(0, m.layers.length - 1);
    lr.value = S.layer;
    const tr = $("timeRange");
    tr.max = nRecords() - 1;
    tr.value = S.t;
    $("timeGroup").hidden = m.times.length < 2;
    $("viewSeg").hidden = isPoints();
    $("sectionBtn").hidden = isPoints() || m.layers.length < 2;
    updateLayerControls();
    updateTimeLabel();
    fill3dControls();
}

function updateLayerControls() {
    const f = fieldOf(S.m, S.field);
    const show = S.m.layers.length > 1 && f && f.layered && S.mode === "2d";
    $("layerGroup").hidden = !show;
    const ly = S.m.layers[S.layer];
    $("layerLabel").textContent = !f || !f.layered ? "" : ly ? ly.label : "";
}

function updateTimeLabel() {
    const m = S.m;
    if (!m) return;
    if (m.times.length) {
        const extra = m.times.length > 1 ? `, record ${S.t + 1} of ${m.times.length}` +
            (m.cadence ? `, ${m.cadence}` : "") : "";
        $("timeLabel").textContent = `${api.fmtTime(m.times[S.t])}${extra}`;
    } else {
        $("timeLabel").textContent = m.snapshot ? `Snapshot ${api.fmtTime(m.snapshot)}` : "No time in this store";
    }
    $("timeLabel").hidden = false;
}

function fill3dControls() {
    const m = S.m;
    const chips = $("layerChips");
    chips.replaceChildren();
    if (!m || isPoints()) return;
    $("tracerGroup").hidden = !m.vectors.length;
}

function render3dChips() {
    const chips = $("layerChips");
    chips.replaceChildren();
    if (!scene || !S.m) return;
    const f = fieldOf(S.m, S.field);
    $("layerChipsGroup").hidden = !(f && f.layered && S.m.layers.length > 1);
    // Four pickers, each a layer drawn whole at its heights, or none (the default).
    const current = scene.levels.slice();
    for (let slot = 0; slot < 4; slot++) {
        const sel = document.createElement("select");
        sel.className = "form-select form-select-sm";
        sel.title = `Level map ${slot + 1}`;
        const none = document.createElement("option");
        none.value = "";
        none.textContent = "none";
        sel.appendChild(none);
        S.m.layers.forEach((ly) => {
            const o = document.createElement("option");
            o.value = ly.index;
            o.textContent = ly.label;
            sel.appendChild(o);
        });
        sel.value = current[slot] != null ? current[slot] : "";
        sel.addEventListener("change", () => {
            const picked = [...chips.querySelectorAll("select")].map((s) => s.value).filter((v) => v !== "").map(Number);
            scene.setLevels([...new Set(picked)].sort((a, c) => a - c));
            draw();
        });
        chips.appendChild(sel);
    }
}

async function setMode(mode, quiet) {
    S.mode = mode;
    document.querySelectorAll("#viewSeg [data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    $("controls3d").hidden = mode !== "3d";
    $("cellsMap").hidden = mode !== "2d";
    if (mode === "3d") {
        if (!scene) scene = new Scene3D($("scene3d"));
        $("scene3d").hidden = false;
        scene.show(true);
        if (!S.m3) {
            setStatus("Loading cells for 3D");
            S.m3 = await api.manifest(S.item.id, "3d");
            S.geom3 = await api.mesh(S.item.id, "3d", S.m3);
            const info = await scene.load(S.item.id, S.m3, S.geom3);
            $("exagRange").value = Math.log10(info.exag);
            $("exagLabel").textContent = `${info.exag}x`;
            // Layers without depths are stacked evenly; there is no vertical scale to exaggerate.
            $("exagGroup").hidden = info.nominal;
            // The toggles as the user left them, for the next store too.
            scene.setSurface($("surfaceToggle").checked);
            scene.setBed($("bedToggle").checked);
            scene.showTracers = $("tracerToggle").checked;
            scene.setOpacity(+$("opacityRange").value);
            setStatus("");
        }
        if (S.section) scene.setSection(S.section);
        render3dChips();
    } else {
        $("scene3d").hidden = true;
        if (scene) scene.show(false);
        if (map) map.resize();
    }
    updateLayerControls();
    updateLegend();
    if (!quiet) { await draw(); }
    updateHash();
}

// ---- drawing --------------------------------------------------------------------------------

async function draw() {
    if (!S.m) return;
    const token = S.token;
    const f = fieldOf(S.m, S.field);
    try {
        if (S.mode === "3d" && scene) {
            await scene.draw(S.field, S.t);
        } else if (isPoints()) {
            const codes = await api.frame(S.item.id, S.field, 0, S.t, "2d");
            if (token !== S.token) return;
            const sites = S.m.sites;
            map.setSites(sites, codes, api.ramp(f), S._fitted === S.item.id ? null : sitesBox(sites));
            S._fitted = S.item.id;
        } else {
            const codes = await api.frame(S.item.id, S.field, f.layered ? S.layer : 0, S.t, "2d");
            if (token !== S.token) return;
            map.setCodes(codes);
            // Prefetch the next record while this one is on screen.
            if (nRecords() > 1) api.frame(S.item.id, S.field, f.layered ? S.layer : 0, (S.t + 1) % nRecords(), "2d").catch(() => {});
        }
    } catch (e) {
        console.error("draw:", e);
        setStatus(`Could not draw this record: ${e.message}`, true);
    }
    updateLegend();
    updateHash();
}

function sitesBox(sites) {
    const lon = sites.map((s) => s.lon), lat = sites.map((s) => s.lat);
    const pad = 0.05;
    return [Math.min(...lon) - pad, Math.min(...lat) - pad, Math.max(...lon) + pad, Math.max(...lat) + pad];
}

// ---- legend and readout ---------------------------------------------------------------------

function updateLegend() {
    const el = $("legend");
    const f = fieldOf(S.m, S.field);
    if (!f) { el.hidden = true; return; }
    el.hidden = false;
    $("legendTitle").textContent = f.units ? `${f.label} (${f.units})` : f.label;
    const cv = $("legendBar");
    const ctx = cv.getContext("2d");
    const rgba = api.ramp(f);
    const img = ctx.createImageData(254, 1);
    for (let i = 0; i < 254; i++) img.data.set(rgba.subarray(4 * (i + 1), 4 * (i + 2)), 4 * i);
    cv.width = 254;
    cv.height = 1;
    ctx.putImageData(img, 0, 0);
    const ticks = $("legendTicks");
    let tv = (f.ticks || []).filter((v) => v >= Math.min(f.lo, f.hi) && v <= Math.max(f.lo, f.hi));
    while (tv.length > 5) tv = tv.filter((_, i) => i % 2 === 0);
    ticks.replaceChildren(...tv.map((v) => {
        const s = document.createElement("span");
        s.style.left = `${100 * api.barPos(f, v)}%`;
        s.textContent = api.fmt(v);
        return s;
    }));
    const notes = [];
    if (S.m.stride > 1) notes.push(`every ${ordinal(S.m.stride)} cell shown`);
    if (S.mode === "3d" && scene && scene.nominal) notes.push("no depths in store: layers evenly stacked");
    if (S.mode === "3d" && scene && !scene.nominal) notes.push(`vertical ${scene.exag}x`);
    $("legendNote").textContent = notes.join(", ");
    // Tracers get their own small bar: depth in the water column.
    const tr = $("legendTracers");
    tr.hidden = !(S.mode === "3d" && scene && scene.showTracers && S.m.vectors.length);
    if (!tr.hidden) {
        tr.querySelector(".fk-legend-mini").style.background =
            `linear-gradient(to right, ${DEPTH_RAMP.map((c) => `rgb(${c.join(",")})`).join(", ")})`;
    }
}

// A layer center's height: depth below the datum, or height above it (the top ocean layer at
// high water, or an atmosphere level).
function zLabel(z) {
    if (z <= 0) return `${(-z).toFixed(1)} m deep`;
    return z < 10 ? `${z.toFixed(1)} m above datum` : `${z.toFixed(0)} m above sea level`;
}

function ordinal(n) {
    return n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`;
}

let hoverToken = 0;
async function hover(lngLat) {
    const el = $("readout");
    if (!lngLat || !S.m || S.mode !== "2d") { el.hidden = true; return; }
    const my = ++hoverToken;
    const f = fieldOf(S.m, S.field);
    try {
        const p = await api.point(S.item.id, lngLat.lng, lngLat.lat, S.field, f.layered ? S.layer : 0, S.t, "2d");
        if (my !== hoverToken) return;
        if ((p.element != null && p.element < 0) || (p.site == null && p.element == null)) { el.hidden = true; return; }
        const parts = [p.name ? `${p.name}: ${api.fmt(p.value, f.units)}` : api.fmt(p.value, f.units)];
        if (p.z != null) parts.push(zLabel(p.z));
        if (p.bed != null) parts.push(p.bed <= 0 ? `bed ${(-p.bed).toFixed(1)} m` : `ground ${p.bed.toFixed(0)} m`);
        if (p.cell != null) parts.push(Array.isArray(p.cell) ? `cell ${p.cell[0]}, ${p.cell[1]}` : `cell ${p.cell}`);
        el.textContent = parts.join(" | ");
        el.hidden = false;
    } catch (_) {
        el.hidden = true;
    }
}

// ---- vertical sections ----------------------------------------------------------------------

function startSection() {
    if (S.mode !== "2d") setMode("2d");
    map.startSection();
    setStatus("Click points along the section; double-click (or Enter) to finish, Esc to cancel");
}

async function sectionDone(pts) {
    setStatus("");
    S.sectionPts = pts;
    await loadSection();
}

async function loadSection() {
    if (!S.sectionPts || !S.m) return;
    try {
        const data = await api.section(S.item.id, S.sectionPts, S.field, S.t, "2d");
        S.section = { data, field: S.field };
        drawSection();
        if (scene && S.m3) scene.setSection(S.section);
    } catch (e) {
        setStatus(`Section failed: ${e.message}`, true);
    }
}

function drawSection() {
    const panel = $("sectionPanel");
    const sec = S.section;
    if (!sec) { panel.hidden = true; return; }
    panel.hidden = false;
    const f = fieldOf(S.m, sec.field);
    const d = sec.data;
    $("sectionTitle").textContent = `${f.label}${f.units ? ` (${f.units})` : ""}, ${api.fmtTime(d.time)}` +
        (d.z ? "" : ", by layer (no depths in this store)");
    const cv = $("sectionCanvas");
    const w = cv.clientWidth, h = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr;
    cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const cells = sectionCells(d);
    S.sectionCells = cells;
    if (!cells.length) { ctx.fillStyle = "#8b949e"; ctx.fillText("The line does not cross the store's cells.", 12, 20); return; }
    const L = 46, R = 8, T = 6, B = 22;
    const xmax = d.distance[d.distance.length - 1] || 1;
    let ytop = -Infinity, ybot = Infinity;
    for (const c of cells) { ytop = Math.max(ytop, c.top); ybot = Math.min(ybot, c.bottom); }
    if (d.bed) for (const b of d.bed) if (b != null) ybot = Math.min(ybot, b);
    const X = (x) => L + (x / xmax) * (w - L - R);
    const Y = (y) => T + ((ytop - y) / (ytop - ybot || 1)) * (h - T - B);
    S.sectionScale = { X, Y, L, R, T, B, xmax, ytop, ybot, w, h };
    const rgba = api.ramp(f);
    for (const c of cells) {
        const code = api.encode(f, c.value);
        ctx.fillStyle = `rgb(${rgba[4 * code]},${rgba[4 * code + 1]},${rgba[4 * code + 2]})`;
        const x0 = X(c.x0), x1 = X(c.x1), y0 = Y(c.top), y1 = Y(c.bottom);
        ctx.fillRect(x0, y0, Math.max(1, x1 - x0 + 0.5), Math.max(1, y1 - y0 + 0.5));
    }
    if (d.bed) {
        ctx.strokeStyle = "#c9d1d9";
        ctx.lineWidth = 1;
        ctx.beginPath();
        let pen = false;
        d.bed.forEach((b, i) => {
            if (b == null) { pen = false; return; }
            const x = X(d.distance[i]), y = Y(b);
            if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
            pen = true;
        });
        ctx.stroke();
    }
    ctx.fillStyle = "#8b949e";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    const unit = d.z ? " m" : "";
    for (const y of [ytop, (ytop + ybot) / 2, ybot]) {
        ctx.fillText(d.z ? `${y.toFixed(0)}${unit}` : `layer ${Math.round(-y)}`, L - 4, Y(y) + 4);
    }
    ctx.textAlign = "center";
    ctx.fillText("0 km", X(0) + 10, h - 6);
    ctx.fillText(`${(xmax / 1000).toFixed(xmax < 10000 ? 1 : 0)} km`, X(xmax) - 18, h - 6);
}

function sectionHover(ev) {
    const s = S.sectionScale, cells = S.sectionCells, el = $("sectionReadout");
    if (!s || !cells) return;
    const r = ev.target.getBoundingClientRect();
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    const f = fieldOf(S.m, S.section.field);
    const hit = cells.find((c) => px >= s.X(c.x0) && px <= s.X(c.x1) + 0.5 && py >= s.Y(c.top) && py <= s.Y(c.bottom) + 0.5);
    if (!hit) { el.textContent = ""; return; }
    const z = S.section.data.z ? S.section.data.z[hit.k][hit.i] : null;
    el.textContent = `${api.fmt(hit.value, f.units)} | ${(S.section.data.distance[hit.i] / 1000).toFixed(2)} km` +
        (z != null ? ` | ${zLabel(z)}` : ` | layer ${hit.k}`);
}

// ---- time -----------------------------------------------------------------------------------

function setPlaying(on) {
    S.playing = on;
    $("playBtn").textContent = on ? "Pause" : "Play";
}

async function playLoop() {
    while (S.playing && S.m && nRecords() > 1) {
        const started = performance.now();
        S.t = (S.t + 1) % nRecords();
        $("timeRange").value = S.t;
        updateTimeLabel();
        await draw();
        if (S.section) loadSection();
        await new Promise((r) => setTimeout(r, Math.max(0, FRAME_MS - (performance.now() - started))));
    }
    setPlaying(false);
}

// ---- link -----------------------------------------------------------------------------------

function updateHash() {
    if (!S.item) return;
    const p = [`f=${encodeURIComponent(S.field)}`];
    if (S.layer) p.push(`k=${S.layer}`);
    if (S.t) p.push(`t=${S.t}`);
    if (S.mode === "3d") p.push("v=3d");
    history.replaceState(null, "", `#${encodeURIComponent(S.item.id)}&${p.join("&")}`);
}

export function parseHash(hash) {
    const [id, ...rest] = hash.replace(/^#/, "").split("&");
    const want = {};
    rest.forEach((kv) => { const [k, v] = kv.split("="); want[k] = decodeURIComponent(v || ""); });
    return { id: decodeURIComponent(id || ""), want };
}

// ---- wiring ---------------------------------------------------------------------------------

function wire() {
    $("fieldSelect").addEventListener("change", (e) => {
        S.field = e.target.value;
        const f = fieldOf(S.m, S.field);
        if (map && !isPoints()) map.setRamp(api.ramp(f));
        updateLayerControls();
        if (S.mode === "3d") render3dChips();
        draw();
        if (S.sectionPts) loadSection();
    });
    $("layerRange").addEventListener("input", (e) => {
        S.layer = +e.target.value;
        updateLayerControls();
        draw();
    });
    $("timeRange").addEventListener("input", (e) => {
        S.t = +e.target.value;
        updateTimeLabel();
        draw();
        if (S.sectionPts) loadSection();
    });
    $("playBtn").addEventListener("click", () => {
        if (S.playing) { setPlaying(false); return; }
        setPlaying(true);
        playLoop();
    });
    document.querySelectorAll("#viewSeg [data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    $("sectionBtn").addEventListener("click", startSection);
    $("sectionClose").addEventListener("click", () => {
        S.section = S.sectionPts = null;
        $("sectionPanel").hidden = true;
        if (map) map.showSectionLine(null);
        if (scene) scene.setSection(null);
    });
    $("sectionCanvas").addEventListener("mousemove", sectionHover);
    $("sectionCanvas").addEventListener("mouseleave", () => { $("sectionReadout").textContent = ""; });
    document.addEventListener("keydown", (e) => {
        if (!map || !map.sectionPts) return;
        if (e.key === "Enter") map.finishSection();
        if (e.key === "Escape") { map.cancelSection(); setStatus(""); }
    });
    $("exagRange").addEventListener("input", (e) => {
        if (!scene) return;
        const x = Math.round(Math.pow(10, +e.target.value));
        $("exagLabel").textContent = `${x}x`;
        scene.setExaggeration(x);
        updateLegend();
    });
    $("opacityRange").addEventListener("input", (e) => scene && scene.setOpacity(+e.target.value));
    $("bedToggle").addEventListener("change", (e) => scene && scene.setBed(e.target.checked));
    $("surfaceToggle").addEventListener("change", (e) => scene && scene.setSurface(e.target.checked));
    $("tracerToggle").addEventListener("change", (e) => {
        if (!scene) return;
        scene.setTracers(e.target.checked);
        updateLegend();
    });
    $("infoBtn").addEventListener("click", () => $("infoDialog").showModal());
    $("infoClose").addEventListener("click", () => $("infoDialog").close());
    new ResizeObserver(() => { if (S.section) drawSection(); }).observe($("sectionPanel"));
}

wire();
window.fkPreview = {
    open, close, parseHash,
    // For debugging in the console: the cell map and the 3D scene, once made.
    get map() { return map; },
    get scene() { return scene; },
};
