// Client for the cache viewer's drawing API (/api/v1/cache/view): manifests, cells, one byte per
// cell per record, points, sections and tracer velocities. Frames are kept in a small cache so stepping
// back and forth through records does not refetch them.

const BASE = "/api/v1/cache/view";
const FRAME_CACHE = 96;
const frames = new Map();

async function get(url, kind) {
    const res = await fetch(url);
    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try { detail = (await res.json()).detail || detail; } catch (_) { /* binary or empty */ }
        const err = new Error(detail);
        err.status = res.status;
        throw err;
    }
    return kind === "json" ? res.json() : res.arrayBuffer();
}

const q = (params) => new URLSearchParams(params).toString();
const path = (id, what, params) => `${BASE}/${encodeURIComponent(id)}/${what}?${q(params)}`;

export function manifest(id, detail = "2d") {
    return get(path(id, "manifest", { detail }), "json");
}

// Node longitudes and latitudes (Float64) then triangles (Uint32), sized by the manifest.
export async function mesh(id, detail, m) {
    const buf = await get(path(id, "mesh", { detail }), "bin");
    const n = m.n_node, e = m.n_elem;
    return {
        lon: new Float64Array(buf, 0, n),
        lat: new Float64Array(buf, n * 8, n),
        tri: new Uint32Array(buf, n * 16, e * 3),
    };
}

export async function frame(id, field, layer, t, detail = "2d") {
    const key = `${id}|${detail}|${field}|${layer}|${t}`;
    if (frames.has(key)) {
        const hit = frames.get(key);
        frames.delete(key);
        frames.set(key, hit);
        return hit;
    }
    const p = get(path(id, "frame", { field, layer, t, detail }), "bin").then((b) => new Uint8Array(b));
    frames.set(key, p);
    p.catch(() => frames.delete(key));
    while (frames.size > FRAME_CACHE) frames.delete(frames.keys().next().value);
    return p;
}

export async function heights(id, layer, t, detail = "3d") {
    try {
        return new Float32Array(await get(path(id, "heights", { layer, t, detail }), "bin"));
    } catch (e) {
        if (e.status === 404) return null;
        throw e;
    }
}

export async function bed(id, detail = "3d", at = "element") {
    try {
        return new Float32Array(await get(path(id, "bed", { detail, at }), "bin"));
    } catch (e) {
        if (e.status === 404) return null;
        throw e;
    }
}

export function point(id, lon, lat, field, layer, t, detail = "2d") {
    return get(path(id, "point", { lon, lat, field, layer, t, detail }), "json");
}

export function section(id, pts, field, t, detail = "2d", n = 240) {
    const s = pts.map(([lo, la]) => `${lo.toFixed(6)},${la.toFixed(6)}`).join(";");
    return get(path(id, "section", { pts: s, field, t, n, detail }), "json");
}

export function tracers(id, pair, t, detail = "3d", levels = 6, limit = 3000) {
    return get(path(id, "tracers", { pair, t, levels, limit, detail }), "json");
}

// Field scales: the byte code of a value, as the server encodes it (FieldMeta.encode).
export function encode(f, v) {
    if (v == null || !Number.isFinite(v)) return 0;
    let u;
    if (f.log) {
        const c = Math.min(Math.max(v, f.lo), f.hi);
        u = (Math.log(c) - Math.log(f.lo)) / (Math.log(f.hi) - Math.log(f.lo));
    } else {
        u = Math.min(Math.max((v - f.lo) / (f.hi - f.lo), 0), 1);
    }
    return 1 + Math.round(u * 254);
}

// Position (0 to 1) of a value along the color bar.
export function barPos(f, v) {
    return (encode(f, v) - 1) / 254;
}

// 256 RGBA entries per field, one per byte code (code 0: no value, transparent).
export function ramp(f) {
    if (!f._rgba) {
        const bin = atob(f.ramp_rgba);
        const a = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
        f._rgba = a;
    }
    return f._rgba;
}

export function fmt(v, units) {
    if (v == null || !Number.isFinite(v)) return "no value";
    const a = Math.abs(v);
    const s = a !== 0 && (a < 0.01 || a >= 1e5) ? v.toExponential(2) : v.toFixed(a < 1 ? 3 : a < 100 ? 2 : 0);
    return units ? `${s} ${units}` : s;
}

export function fmtTime(iso) {
    return iso ? iso.replace("T", " ").replace(/:00Z$/, " UTC").replace(/Z$/, " UTC") : "";
}
