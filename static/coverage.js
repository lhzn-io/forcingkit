// Coverage view for the cache inventory: a map of store footprints and a strip of hours covered
// per source family per day, both following the filters above the list. Reads the inventory's
// structured fields (bbox, time_start, time_end, family, kind); stores without them are listed
// but not drawn.
(function () {
    "use strict";

    const PALETTE = ["#58a6ff", "#3fb950", "#d29922", "#f778ba", "#a371f7", "#39c5cf",
        "#ff7b72", "#e3b341", "#7ee787", "#79c0ff", "#ffa657", "#d2a8ff"];
    const UNKNOWN = "Unclassified";
    const state = {
        items: [],
        families: [],
        hiddenFamilies: new Set(),
        kind: "",
        from: null,   // Date (UTC midnight) or null
        to: null,     // Date (UTC end of day) or null
        inView: false,
        view: null,   // [west, south, east, north] of the map
        selectedId: null,
        fitted: false,
    };
    const lastChip = { family: null, at: 0 };
    let onChange = () => {};
    let onSelect = () => {};

    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" })[c]);
    const fam = (it) => it.family || UNKNOWN;
    const colorOf = (f) => (f === UNKNOWN ? "#8b949e" : PALETTE[state.families.indexOf(f) % PALETTE.length]);
    const t0 = (it) => (it.time_start ? Date.parse(it.time_start) : NaN);
    const t1 = (it) => (it.time_end ? Date.parse(it.time_end) : t0(it));

    function overlapsTime(it) {
        if (!state.from && !state.to) return true;
        const a = t0(it), b = t1(it);
        if (Number.isNaN(a)) return false;
        if (state.from && b < state.from.getTime()) return false;
        if (state.to && a > state.to.getTime()) return false;
        return true;
    }

    function overlapsView(it) {
        if (!state.view) return true;
        if (!it.bbox) return false;
        const [w, s, e, n] = it.bbox, [vw, vs, ve, vn] = state.view;
        return w <= ve && e >= vw && s <= vn && n >= vs;
    }

    // Family and kind filters only (the strip spans all dates; the map ignores the view).
    const passesType = (it) => !state.hiddenFamilies.has(fam(it)) && (!state.kind || it.kind === state.kind);

    function filtered(items) {
        return items.filter((it) => passesType(it) && overlapsTime(it) && (!state.inView || overlapsView(it)));
    }

    // ---- filter controls ----
    function renderFilters() {
        const chips = $("familyChips");
        chips.innerHTML = "";
        const counts = {};
        state.items.forEach((it) => { counts[fam(it)] = (counts[fam(it)] || 0) + 1; });
        state.families.forEach((f) => {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "chip" + (state.hiddenFamilies.has(f) ? " chip-off" : "");
            b.style.setProperty("--chip", colorOf(f));
            b.textContent = `${f} ${counts[f] || 0}`;
            b.title = "Click to show or hide this source; double-click to show only it";
            // The chips are rebuilt on every click, so a double-click is told by timing, not dblclick.
            b.addEventListener("click", () => {
                const now = Date.now();
                if (lastChip.family === f && now - lastChip.at < 350) {
                    state.hiddenFamilies = new Set(state.families.filter((g) => g !== f));
                    lastChip.family = null;
                } else {
                    if (state.hiddenFamilies.has(f)) state.hiddenFamilies.delete(f); else state.hiddenFamilies.add(f);
                    lastChip.family = f;
                    lastChip.at = now;
                }
                changed();
            });
            chips.appendChild(b);
        });
        const kinds = [...new Set(state.items.map((it) => it.kind).filter(Boolean))].sort();
        const sel = $("kindSelect");
        const keep = state.kind;
        sel.innerHTML = "<option value=\"\">All kinds</option>" +
            kinds.map((k) => `<option value="${esc(k)}">${esc(k)}</option>`).join("");
        sel.value = kinds.includes(keep) ? keep : "";
        state.kind = sel.value;
    }

    function wireControls() {
        $("kindSelect").addEventListener("change", (e) => { state.kind = e.target.value; changed(); });
        $("dateFrom").addEventListener("change", (e) => {
            state.from = e.target.value ? new Date(e.target.value + "T00:00:00Z") : null; changed();
        });
        $("dateTo").addEventListener("change", (e) => {
            state.to = e.target.value ? new Date(e.target.value + "T23:59:59Z") : null; changed();
        });
        $("inViewToggle").addEventListener("change", (e) => { state.inView = e.target.checked; changed(); });
        $("clearFilters").addEventListener("click", () => {
            state.hiddenFamilies.clear();
            state.kind = "";
            state.from = state.to = null;
            $("dateFrom").value = $("dateTo").value = "";
            changed();
        });
        document.querySelectorAll("[data-pane]").forEach((b) =>
            b.addEventListener("click", () => showPane(b.dataset.pane)));
    }

    function showPane(name) {
        const panel = document.querySelector(".preview-panel");
        panel.classList.toggle("show-coverage", name === "coverage");
        document.querySelectorAll("[data-pane]").forEach((b) => b.classList.toggle("active", b.dataset.pane === name));
        if (name === "coverage") {
            drawMap();
            drawStrip();
            if (window.Plotly && $("coverageStrip").data) Plotly.Plots.resize($("coverageStrip"));
        }
    }

    const coverageShown = () => !!document.querySelector(".preview-panel.show-coverage");

    // ---- map (MapLibre) ----
    // Basemap: the service's own relief archives when FORCINGKIT_BASEMAP_DIR is set (PMTiles read
    // by byte range, dimmed for the dark page), else the OpenFreeMap vector style it names.
    let map = null;
    let mapReady = false;
    let popup = null;

    async function basemapStyle() {
        let cfg = { basemaps: [], attribution: "", fallback_style: "https://tiles.openfreemap.org/styles/dark" };
        try {
            cfg = await (await fetch("/api/v1/cache/basemap", { cache: "no-store" })).json();
        } catch (_) { /* fall back */ }
        if (!cfg.basemaps || !cfg.basemaps.length || !window.pmtiles) return cfg.fallback_style;
        const protocol = new pmtiles.Protocol();
        maplibregl.addProtocol("pmtiles", protocol.tile);
        const style = {
            version: 8,
            sources: {},
            layers: [{ id: "background", type: "background", paint: { "background-color": "#0d1117" } }],
        };
        const relief = [];
        for (const b of cfg.basemaps) {
            const url = new URL(`/api/v1/cache/basemap/files/${encodeURIComponent(b.file)}`, location.href).href;
            try {
                const archive = new pmtiles.PMTiles(url);
                protocol.add(archive);
                const h = await archive.getHeader();
                relief.push({ ...b, url, maxzoom: h.maxZoom });
            } catch (e) {
                console.warn("basemap archive unreadable", b.file, e);
            }
        }
        // Coarse first: the regional background under the detailed regions.
        relief.sort((a, b) => a.maxzoom - b.maxzoom || a.name.localeCompare(b.name));
        relief.forEach((b, i) => {
            style.sources[`relief-${b.name}`] = { type: "raster", url: `pmtiles://${b.url}`, tileSize: 256 };
            if (i === 0 && cfg.attribution) style.sources[`relief-${b.name}`].attribution = esc(cfg.attribution);
            style.layers.push({
                id: `relief-${b.name}`, type: "raster", source: `relief-${b.name}`,
                paint: { "raster-brightness-max": 0.6, "raster-saturation": -0.25 },
            });
            if (b.shoreline) {
                style.sources[`shore-${b.name}`] = {
                    type: "geojson",
                    data: new URL(`/api/v1/cache/basemap/files/${encodeURIComponent(b.shoreline)}`, location.href).href,
                };
                style.layers.push({
                    id: `shore-${b.name}`, type: "line", source: `shore-${b.name}`,
                    paint: { "line-color": "#3b5868", "line-width": 1 },
                });
            }
        });
        return style;
    }

    function storesGeoJSON() {
        const items = state.items.filter((it) => it.bbox && passesType(it) && overlapsTime(it));
        return {
            type: "FeatureCollection",
            features: items.map((it) => {
                const [w, s, e, n] = it.bbox;
                const f = fam(it);
                return {
                    type: "Feature",
                    properties: {
                        id: it.id, color: colorOf(f),
                        label: `<b>${esc(f)}</b>, ${esc(it.kind || "unknown kind")}<br>${esc(it.id)}<br>` +
                            `${esc(it.time_start || "?")} to ${esc(it.time_end || "?")}<br>${it.size_mb} MB`,
                    },
                    geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] },
                };
            }),
        };
    }

    function fitToStores(animate) {
        const boxes = state.items.filter((it) => it.bbox && passesType(it)).map((it) => it.bbox);
        if (!map || !boxes.length) return;
        const w = Math.min(...boxes.map((b) => b[0])), s = Math.min(...boxes.map((b) => b[1]));
        const e = Math.max(...boxes.map((b) => b[2])), n = Math.max(...boxes.map((b) => b[3]));
        map.fitBounds([[w, s], [e, n]], { padding: 30, maxZoom: 10, animate });
    }

    async function ensureMap() {
        if (map || !window.maplibregl || !$("coverageMap")) return;
        map = new maplibregl.Map({
            container: "coverageMap",
            style: await basemapStyle(),
            center: [-71, 40.5],
            zoom: 4,
            attributionControl: { compact: true },
        });
        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
        map.on("error", (e) => console.warn("coverage map:", e && e.error ? e.error.message : e));
        popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: "fk-popup" });
        map.on("load", () => {
            map.addSource("stores", { type: "geojson", data: storesGeoJSON() });
            map.addLayer({
                id: "stores", type: "line", source: "stores",
                paint: { "line-color": ["get", "color"], "line-width": 1.5 },
            });
            map.addSource("selected", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
            map.addLayer({
                id: "selected", type: "line", source: "selected",
                paint: { "line-color": "#ffffff", "line-width": 3.5 },
            });
            mapReady = true;
            fitToStores(false);
            state.fitted = true;
            drawMap();
        });
        // Thin outlines are hard to hit: query a few pixels around the pointer.
        const near = (e) => map.queryRenderedFeatures(
            [[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]], { layers: ["stores"] });
        map.on("mousemove", (e) => {
            if (!mapReady) return;
            const f = near(e);
            map.getCanvas().style.cursor = f.length ? "pointer" : "";
            if (f.length) {
                const more = f.length > 1 ? `<br><i>${f.length - 1} more here</i>` : "";
                popup.setLngLat(e.lngLat).setHTML(f[0].properties.label + more).addTo(map);
            } else {
                popup.remove();
            }
        });
        map.on("click", (e) => {
            if (!mapReady) return;
            const f = near(e);
            if (f.length) onSelect(f[0].properties.id);
        });
        map.on("moveend", () => {
            const b = map.getBounds();
            state.view = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
            drawStrip();
            if (state.inView) onChange();
        });
    }

    function drawMap() {
        if (!coverageShown()) return;
        if (!map) { ensureMap(); return; }
        map.resize();
        if (!mapReady) return;
        map.getSource("stores").setData(storesGeoJSON());
        const sel = state.items.find((it) => it.id === state.selectedId && it.bbox);
        const [w, s, e, n] = sel ? sel.bbox : [0, 0, 0, 0];
        map.getSource("selected").setData(sel ? {
            type: "Feature", properties: {},
            geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] },
        } : { type: "FeatureCollection", features: [] });
        if (!state.fitted) { fitToStores(false); state.fitted = true; }
    }

    // ---- coverage strip: hours covered per family per UTC day, for stores in the map view ----
    const DAY = 86400000, HOUR = 3600000;

    function popcount(x) { let c = 0; while (x) { x &= x - 1; c++; } return c; }

    function drawStrip() {
        const el = $("coverageStrip");
        if (!window.Plotly || !el || !coverageShown()) return;
        const items = state.items.filter((it) => passesType(it) && !Number.isNaN(t0(it)) && overlapsView(it));
        $("stripNote").textContent = (state.view ? "In the map view: " : "") +
            `${items.length} store${items.length === 1 ? "" : "s"} with a time span. Click a day to filter to it.`;
        if (!items.length) { Plotly.purge(el); return; }
        const fams = state.families.filter((f) => items.some((it) => fam(it) === f));
        const dayMin = Math.floor(Math.min(...items.map(t0)) / DAY);
        const dayMax = Math.floor(Math.max(...items.map(t1)) / DAY);
        const nDays = dayMax - dayMin + 1;
        if (nDays > 20000) return;
        const bits = fams.map(() => new Uint32Array(nDays));
        const count = fams.map(() => new Uint16Array(nDays));
        items.forEach((it) => {
            const fi = fams.indexOf(fam(it));
            const a = Math.floor(t0(it) / HOUR), b = Math.floor(t1(it) / HOUR);
            for (let d = Math.floor(a / 24); d <= Math.floor(b / 24); d++) {
                const lo = Math.max(a, d * 24) - d * 24, hi = Math.min(b, d * 24 + 23) - d * 24;
                const mask = (((1 << (hi + 1)) >>> 0) - 1) & ~((1 << lo) - 1);
                bits[fi][d - dayMin] |= hi === 23 ? (0xffffff & ~((1 << lo) - 1)) : mask;
                count[fi][d - dayMin]++;
            }
        });
        const x = Array.from({ length: nDays }, (_, k) => new Date((dayMin + k) * DAY).toISOString().slice(0, 10));
        const z = bits.map((row) => Array.from(row, (v) => (v ? popcount(v) : null)));
        const text = count.map((row, fi) => Array.from(row, (c, k) =>
            c ? `${esc(fams[fi])}, ${x[k]}<br>${popcount(bits[fi][k])} of 24 h covered by ${c} store${c > 1 ? "s" : ""}` : ""));
        Plotly.react(el, [{
            type: "heatmap", x, y: fams, z, text, hoverinfo: "text", zmin: 0, zmax: 24, xgap: 1, ygap: 2,
            colorscale: [[0, "#1f2a37"], [0.5, "#1f6feb"], [1, "#79c0ff"]],
            colorbar: {
                title: { text: "h/day", side: "right", font: { color: "#8b949e", size: 10 } },
                thickness: 8, len: 0.9, tickfont: { color: "#8b949e", size: 10 },
            },
        }], {
            margin: { l: 110, r: 10, t: 6, b: 34 },
            paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
            xaxis: { type: "date", color: "#8b949e", gridcolor: "rgba(255,255,255,0.06)" },
            yaxis: { color: "#c9d1d9", automargin: true },
            uirevision: "strip",
        }, { displaylogo: false, responsive: true });
        if (!el._fkWired) {
            el._fkWired = true;
            el.on("plotly_click", (ev) => {
                const p = ev.points && ev.points[0];
                if (!p || p.z == null) return;
                const day = String(p.x).slice(0, 10);
                state.from = new Date(day + "T00:00:00Z");
                state.to = new Date(day + "T23:59:59Z");
                $("dateFrom").value = $("dateTo").value = day;
                changed();
            });
        }
    }

    function changed() {
        renderFilters();
        drawMap();
        drawStrip();
        onChange();
    }

    window.fkCoverage = {
        init({ rerender, select }) {
            onChange = rerender;
            onSelect = select;
            wireControls();
            showPane("coverage");
        },
        setInventory(items) {
            state.items = items;
            state.families = [...new Set(items.map(fam))].sort((a, b) =>
                (a === UNKNOWN) - (b === UNKNOWN) || a.localeCompare(b));
            renderFilters();
            drawMap();
            drawStrip();
        },
        filter: filtered,
        setSelected(id) {
            state.selectedId = id;
            showPane("preview");
        },
        showPane,
        // For debugging in the console: the MapLibre map, once made.
        get map() { return map; },
    };
})();
