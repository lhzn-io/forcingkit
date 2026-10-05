document.addEventListener("DOMContentLoaded", () => {
    console.log("forcingkit UI Initialized");
    const datasetList = document.getElementById("datasetList");
    const datasetCount = document.getElementById("datasetCount");
    const purgeBtn = document.getElementById("purgeBtn");

    // Preview Canvas Elements
    const previewCanvas = document.getElementById("previewCanvas");
    const previewControls = document.getElementById("previewControls");
    const varSelect = document.getElementById("varSelect");
    const timeRange = document.getElementById("timeRange");
    const timeIdxDisplay = document.getElementById("timeIdxDisplay");

    let currentInventory = [];
    let selectedDataset = null;
    let previewDebounce = null;
    let mode3d = false;
    const vectorPairs = [["u", "v"], ["water_u", "water_v"], ["u10", "v10"]];

    // - Custom Modal System -
    function createModal() {
        const overlay = document.createElement("div");
        overlay.className = "modal-overlay";
        overlay.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-icon"></div>
                <h3 class="modal-title"></h3>
                <p class="modal-message"></p>
                <div class="modal-actions"></div>
            </div>`;
        document.body.appendChild(overlay);
        return overlay;
    }

    function showModal({ title, message, icon = "warn", actions = [], danger = false }) {
        const overlay = createModal();
        const dialog = overlay.querySelector(".modal-dialog");
        if (danger) dialog.classList.add("modal-danger");

        const iconEl = overlay.querySelector(".modal-icon");
        if (icon === "warn") {
            iconEl.innerHTML = `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
        } else if (icon === "trash") {
            iconEl.innerHTML = `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`;
        } else if (icon === "success") {
            iconEl.innerHTML = `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
        } else if (icon === "error") {
            iconEl.innerHTML = `<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;
        }

        overlay.querySelector(".modal-title").textContent = title;
        overlay.querySelector(".modal-message").innerHTML = message;

        const actionsEl = overlay.querySelector(".modal-actions");
        return new Promise(resolve => {
            actions.forEach(({ label, cls, value }) => {
                const btn = document.createElement("button");
                btn.className = `btn ${cls || "btn-secondary"}`;
                btn.textContent = label;
                btn.addEventListener("click", () => {
                    overlay.classList.add("modal-closing");
                    setTimeout(() => { overlay.remove(); resolve(value); }, 150);
                });
                actionsEl.appendChild(btn);
            });
            // Close on backdrop click
            overlay.addEventListener("click", (e) => {
                if (e.target === overlay) {
                    overlay.classList.add("modal-closing");
                    setTimeout(() => { overlay.remove(); resolve(false); }, 150);
                }
            });
            requestAnimationFrame(() => overlay.classList.add("modal-visible"));
        });
    }

    function showAlert({ title, message, icon = "success" }) {
        return showModal({
            title, message, icon,
            actions: [{ label: "OK", cls: "btn-primary", value: true }],
        });
    }

    // - Initialize View & Auto-Poll -
    let inventoryFingerprint = "";
    let pollPaused = false;

    fetchInventory().then(() => {
        const hash = window.location.hash.replace('#', '');
        if (hash && currentInventory.length > 0) {
            const match = currentInventory.find(item => item.id === hash);
            if (match) {
                console.log('Deep-link: auto-selecting', hash);
                const cards = document.querySelectorAll('.dataset-card');
                const idx = currentInventory.indexOf(match);
                if (cards[idx]) {
                    selectDataset(match, cards[idx]);
                    cards[idx].scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }
        // Start auto-polling after initial load
        setInterval(pollInventory, 2000);
    });

    async function pollInventory() {
        if (pollPaused) return;
        try {
            const resp = await fetch("/api/v1/cache/inventory", { cache: "no-store" });
            if (!resp.ok) return;
            const items = await resp.json();
            // Only re-render if the inventory actually changed
            const fp = items.map(i => `${i.id}:${i.modified_time}:${i.size_mb}`).join("|");
            if (fp === inventoryFingerprint) return;
            inventoryFingerprint = fp;
            console.log("Inventory changed, re-rendering");
            currentInventory = items;
            renderInventory(items);
        } catch (_) {
            // Silently ignore poll failures
        }
    }

    if (purgeBtn) {
        purgeBtn.addEventListener("click", async () => {
            const confirmed = await showModal({
                title: "Purge Entire Cache",
                message: "This will <strong>permanently delete all</strong> cached boundary conditions, initial conditions, and atmospheric forcing data.<br><br>This action <strong>cannot be undone</strong>.",
                icon: "warn",
                danger: true,
                actions: [
                    { label: "Cancel", cls: "btn-secondary", value: false },
                    { label: "Purge Everything", cls: "btn-danger-fill", value: true },
                ],
            });

            if (!confirmed) return;

            pollPaused = true;
            purgeBtn.disabled = true;
            purgeBtn.textContent = "Purging...";

            try {
                const resp = await fetch("/api/v1/cache/purge", { method: "POST" });
                if (!resp.ok) throw new Error("Failed to purge cache (HTTP " + resp.status + ")");
                const data = await resp.json();
                console.log("Purge successful:", data.message);

                selectedDataset = null;
                previewControls.classList.add("hidden");
                previewCanvas.innerHTML = '<div class="empty-state"><p>Cache purged. Select a new dataset after generation.</p></div>';
                await fetchInventory();
                await showAlert({ title: "Cache Purged", message: data.message, icon: "success" });
            } catch (err) {
                console.error("Purge failed:", err);
                await showAlert({ title: "Purge Failed", message: err.message, icon: "error" });
            } finally {
                purgeBtn.disabled = false;
                purgeBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                    Purge Cache`;
                pollPaused = false;
            }
        });
    }

    async function deleteDataset(item, e) {
        e.stopPropagation();

        const confirmed = await showModal({
            title: "Delete Dataset",
            message: `Delete <strong>${item.label || item.id}</strong> (${item.size_mb} MB)?<br><span style="color:var(--text-secondary);font-size:0.85rem;">${item.id}</span>`,
            icon: "trash",
            actions: [
                { label: "Cancel", cls: "btn-secondary", value: false },
                { label: "Delete", cls: "btn-danger-fill", value: true },
            ],
        });

        if (!confirmed) return;

        try {
            const resp = await fetch(`/api/v1/cache/dataset/${encodeURIComponent(item.id)}`, { method: "DELETE" });
            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}));
                throw new Error(errJson.detail || `HTTP ${resp.status}`);
            }
            console.log(`Deleted dataset: ${item.id}`);
            if (selectedDataset && selectedDataset.id === item.id) {
                selectedDataset = null;
                previewControls.classList.add("hidden");
                previewCanvas.innerHTML = '<div class="empty-state"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg><p>Select a dataset to render map.</p></div>';
                window.location.hash = '';
            }
            await fetchInventory();
        } catch (err) {
            console.error("Delete failed:", err);
            await showAlert({ title: "Delete Failed", message: err.message, icon: "error" });
        }
    }

    async function fetchInventory() {
        try {
            const resp = await fetch("/api/v1/cache/inventory", { cache: "no-store" });
            if (!resp.ok) throw new Error(`HTTP Error: ${resp.status}`);
            currentInventory = await resp.json();
            inventoryFingerprint = currentInventory.map(i => `${i.id}:${i.modified_time}:${i.size_mb}`).join("|");
            renderInventory(currentInventory);
        } catch (err) {
            console.error("Failed to fetch inventory:", err);
            datasetList.innerHTML = `
                <div class="empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    <p>Failed to load cache inventory.<br><small>${err.message}</small></p>
                </div>`;
        }
    }

    function renderInventory(items) {
        datasetCount.textContent = items.length;
        datasetList.innerHTML = "";

        if (items.length === 0) {
            datasetList.innerHTML = `
                <div class="empty-state">
                    <p>No processed datasets found in the local cache.</p>
                </div>`;
            return;
        }

        items.forEach(item => {
            const date = new Date(item.modified_time * 1000).toLocaleString(undefined, {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
            });

            const card = document.createElement("div");
            card.className = "dataset-card";
            if (selectedDataset && selectedDataset.id === item.id) {
                card.classList.add("selected");
            }

            const isIC = item.id.startsWith("ic_");
            const isBC = item.id.startsWith("bc_");
            const isOBC = item.id.startsWith("obc_");
            const isAtmos = isBC && !isOBC;

            let catClass = isIC ? "type-ic" : (isOBC ? "type-zarr" : (isAtmos ? "type-bc" : "type-bc"));
            let catLabel = isIC ? "INITIAL CONDITIONS" : (isOBC ? "OPEN BOUNDARY CONDITIONS" : "ATMOSPHERIC FORCING");

            // If it's a raw grib file and not part of the standard processed zarrs
            if (item.id.startsWith("hrrr") || item.id.startsWith("era")) {
              catClass = "type-grib";
              catLabel = item.id.startsWith("era") ? "RAW GRB/NC DONOR (ATMOSPHERIC/WIND)" : "RAW GRB/NC DONOR (ATMOSPHERIC)";
            }
            if (item.id.startsWith("necofs") && !item.id.startsWith("bc_") && !item.id.startsWith("ic_") && !item.id.startsWith("obc_")) {
              catClass = "type-grib";
              catLabel = "RAW GRB/NC DONOR (OCEAN/CURRENT)";
            }

            const typeLabel = catLabel;
            const typeClass = catClass;

            const donorLabel = item.donor_model ? `<span class="donor-label">${item.donor_model}</span>` : `<span class="donor-label">${item.source || ''}</span>`;
            const resLabel = item.resolution ? `<span class="res-label">${item.resolution}</span>` : '';
            // Short hash from dataset ID (e.g. "bc_5439f3a2bf41" → "5439f3a2bf41")
            const hashId = item.id.replace(/^(bc_|ic_|obc_)/, '');

            // Build display label
            let displayName = item.label || item.id;
            if (isIC) displayName = "Ocean Initial Conditions";
            else if (isOBC) displayName = "Open Boundary Conditions";
            else if (isAtmos) displayName = "Atmospheric Boundary Conditions";

            const gridInfo = item.grid_shape ? `${item.grid_shape} grid` : '';
            const timeInfo = item.time_steps ? `${item.time_steps} steps` : '';
            const statsLine = [gridInfo, timeInfo].filter(Boolean).join(' \u00b7 ');
            const varsLine = item.variables ? item.variables.join(', ') : '';

            card.innerHTML = `
                <div class="card-header">
                    <div class="card-title">${displayName}</div>
                    <button class="card-delete-btn" title="Delete dataset">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
                <div class="card-meta">
                    <span class="type-tag ${typeClass}">${typeLabel}${donorLabel}${resLabel}<span class="hash-label">${hashId}</span></span>
                    <span>${item.size_mb} MB</span>
                    <span>${date}</span>
                </div>
                ${statsLine ? `<div class="card-stats">${statsLine}</div>` : ''}
                ${varsLine ? `<div class="card-vars">${varsLine}</div>` : ''}
                ${item.spatial_extent ? `<div class="card-stats">\ud83c\udf10 ${item.spatial_extent}</div>` : ''}
                ${item.time_range ? `<div class="card-time">${item.time_steps && item.time_steps <= 1 ? '⏱️' : '📅'} ${item.time_range}</div>` : ''}
            `;

            // Wire up delete button
            card.querySelector(".card-delete-btn").addEventListener("click", (e) => deleteDataset(item, e));

            card.addEventListener("click", () => selectDataset(item, card));
            datasetList.appendChild(card);
        });
    }

    function selectDataset(item, cardElement) {
        console.log("Dataset selected:", item.id);
        window.location.hash = item.id;  // Update URL for deep-linking
        document.querySelectorAll(".dataset-card").forEach(c => c.classList.remove("selected"));
        cardElement.classList.add("selected");

        setPlaying(false);
        selectedDataset = item;
        previewControls.classList.remove("hidden");
        frameTime.textContent = "";
        timeRange.value = 0;
        timeIdxDisplay.textContent = "0";

        // Set time slider max from metadata; hide for single-snapshot datasets (e.g. ICs)
        const timeSliderGroup = document.getElementById("timeSliderGroup");
        const snapshotLabel = document.getElementById("snapshotLabel");
        const snapshotLabelText = document.getElementById("snapshotLabelText");
        if (item.time_steps && item.time_steps > 1) {
            timeRange.max = item.time_steps - 1;
            timeIdxDisplay.textContent = `0 (${item.time_steps})`;
            if (timeSliderGroup) timeSliderGroup.style.display = "";
            if (snapshotLabel) snapshotLabel.style.display = "none";
        } else {
            timeRange.max = 0;
            timeRange.value = 0;
            if (timeSliderGroup) timeSliderGroup.style.display = "none";
            // Show snapshot timestamp from target_date if available
            if (snapshotLabel && snapshotLabelText && item.target_date) {
                try {
                    const d = new Date(item.target_date);
                    snapshotLabelText.textContent = `Snapshot: ${d.toUTCString().replace('GMT', 'UTC')}`;
                } catch (_) {
                    snapshotLabelText.textContent = `Snapshot: ${item.target_date}`;
                }
                snapshotLabel.style.display = "";
            } else if (snapshotLabel) {
                snapshotLabel.style.display = "none";
            }
        }

        // Populate variable dropdown from metadata if available
        // Human-readable labels for common variable names
        const varLabels = {
            temp: "Temperature (temp)", salt: "Salinity (salt)",
            zeta: "Surface Elevation (zeta)", ssh: "Sea Surface Height (ssh)",
            t2m: "Air Temp 2m (t2m)", sp: "Surface Pressure (sp)",
            hs: "Sig. Wave Height (hs)", water_level: "Water Level",
        };
        // Known u/v vector pairs - second component is hidden, first triggers 3D
        const knownPairs = [["u", "v"], ["water_u", "water_v"], ["u10", "v10"]];
        const pairLabels = {
            u: "Current Velocity (u, v)", water_u: "Current Velocity (water_u, water_v)",
            u10: "Wind 10m (u10, v10)",
        };

        if (item.variables && item.variables.length > 0) {
            varSelect.innerHTML = '';
            // Collect which variables are the "hidden" half of a vector pair
            const hiddenVars = new Set();
            for (const [uName, vName] of knownPairs) {
                if (item.variables.includes(uName) && item.variables.includes(vName)) {
                    hiddenVars.add(vName);
                }
            }
            item.variables.forEach(v => {
                if (hiddenVars.has(v)) return;  // Skip v-component, it's merged into u
                const opt = document.createElement('option');
                opt.value = v;
                opt.textContent = pairLabels[v] || varLabels[v] || v;
                varSelect.appendChild(opt);
            });
            // Default to a vector variable for IC datasets so 3D auto-triggers
            const isIC = item.id.startsWith('ic_');
            const preferred = isIC || item.id.startsWith('obc_')
                ? ['u', 'water_u', 'u10', 'zeta', 'temp']
                : ['u10', 'zeta', 'temp'];
            const defaultVar = preferred.find(p => item.variables.includes(p) && !hiddenVars.has(p)) || item.variables[0];
            varSelect.value = defaultVar;
        } else {
            // Fallback: show/hide hardcoded options
            Array.from(varSelect.options).forEach(opt => {
                opt.style.display = "none";
                if (item.id.startsWith("bc_") || item.id.startsWith("hrrr") || item.id.startsWith("era5")) {
                    if (["u10", "v10", "t2m", "water_level"].includes(opt.value)) opt.style.display = "";
                }
                if (item.id.startsWith("ic_") || item.id.includes("hycom") || item.id.includes("nyhops") || item.id.includes("maracoos")) {
                    if (["u", "v", "temp", "salt", "zeta", "water_u", "water_v", "hs"].includes(opt.value)) opt.style.display = "";
                }
            });
            for (let opt of varSelect.options) {
                if (opt.style.display !== "none") { varSelect.value = opt.value; break; }
            }
        }

        if (window.preview3d) window.preview3d.hide();

        updatePreview();
    }

    varSelect.addEventListener("change", updatePreview);

    let isRendering = false;
    timeRange.addEventListener("input", (e) => {
        const total = selectedDataset?.time_steps || '?';
        const t = e.target.value;
        timeIdxDisplay.textContent = `${t} (${total})`;

        // Instant visual update if we hold the datacube!
        if (window.currentDataCube) {
            let plotDiv = document.getElementById("plotlyMap");
            if (plotDiv && window.currentDataCube[t]) {
                if (!isRendering) {
                    isRendering = true;
                    requestAnimationFrame(() => {
                        Plotly.restyle(plotDiv, 'z', [window.currentDataCube[t]]);
                        isRendering = false;
                    });
                }
                return;
            }
        }

        clearTimeout(previewDebounce);
        previewDebounce = setTimeout(updatePreview, 300);
    });

    // Auto-play: step the time slider, waiting for each frame to render, and loop.
    const playBtn = document.getElementById("playBtn");
    const frameTime = document.getElementById("frameTime");
    const FRAME_MS = 400;  // the fastest step; slower frames take as long as they take to load
    let playing = false;

    function setPlaying(on) {
        playing = on;
        if (playBtn) playBtn.textContent = on ? "Pause" : "Play";
    }

    async function playLoop() {
        while (playing && selectedDataset) {
            const last = parseInt(timeRange.max) || 0;
            if (last < 1) break;
            const next = (parseInt(timeRange.value) + 1) % (last + 1);
            const started = performance.now();
            timeRange.value = next;
            if (window.currentDataCube && window.currentDataCube[next]) {
                timeRange.dispatchEvent(new Event("input"));
            } else {
                timeIdxDisplay.textContent = `${next} (${selectedDataset.time_steps || '?'})`;
                await updatePreview();
            }
            await new Promise(r => setTimeout(r, Math.max(0, FRAME_MS - (performance.now() - started))));
        }
        setPlaying(false);
    }

    if (playBtn) {
        playBtn.addEventListener("click", () => {
            if (playing) { setPlaying(false); return; }
            setPlaying(true);
            playLoop();
        });
    }

    const lodSelect = document.getElementById("lodSelect");
    if (lodSelect) {
        lodSelect.addEventListener("change", () => {
            window.currentDataCube = null; // force refetch
            updatePreview();
        });
    }

    async function updatePreview() {
        if (!selectedDataset) return;

        const t = timeRange.value;
        const id = selectedDataset.id;

        // Determine mode from selected variable: 3D if it's part of a u/v pair
        const selectedVar = varSelect.value;
        mode3d = selectedDataset.type === 'zarr' &&
            vectorPairs.some(([u, v]) => (selectedVar === u || selectedVar === v) &&
                selectedDataset.variables?.includes(u) && selectedDataset.variables?.includes(v));

        // - 3D Vector Mode -
        if (mode3d && window.preview3d) {
            // Hide 2D content, show 3D canvas
            const img = document.getElementById("renderedMap");
            if (img) img.style.display = 'none';
            // Clear empty-state divs
            const emptyState = previewCanvas.querySelector('.empty-state');
            if (emptyState) emptyState.style.display = 'none';

            window.preview3d.show(previewCanvas);

            try {
                const info = await window.preview3d.load(id, parseInt(t));
                frameTime.textContent = info.time ? info.time.replace("T", " ").replace("Z", " UTC") : "";
                console.log(`3D preview: ${info.count} vectors (${info.u_var}/${info.v_var}), ${info.depths} depth levels`);
            } catch (err) {
                console.error("3D preview failed:", err);
                // Fall through - keep the canvas visible with whatever was there
            }
            return;
        }

        // - 2D Matplotlib Mode -
        if (window.preview3d) window.preview3d.hide();

        let loader = document.querySelector(".preview-loader");
        if (!loader) {
            loader = document.createElement("div");
            loader.className = "preview-loader";
            loader.innerHTML = '<div class="spinner"></div>';
            previewCanvas.appendChild(loader);
        }

        const v = varSelect.value;
        const ext = selectedDataset.type;

        const lod = document.getElementById("lodSelect")?.value || 0;

        // Ship the entire datacube inside the browser if we don't have it for this specific var
        if (!window.currentDataCube || window.currentDataCube.var_name !== v || window.currentDataCube.dataset_id !== id) {
            window.currentDataCube = null; // Destroy old
            const fullUrl = `/api/v1/cache/data?dataset_id=${encodeURIComponent(id)}&ext=${ext}&var_name=${v}&time_idx=all&lod=${lod}`;
            console.log("Fetching full Plotly DataCube to browser memory:", fullUrl);

            try {
                const resp = await fetch(fullUrl);
                if (resp.ok) {
                    const fullData = await resp.json();
                    if (fullData.cube) {
                        window.currentDataCube = fullData.z;
                        window.currentDataCube.var_name = v;
                        window.currentDataCube.dataset_id = id;
                        window.currentDataCube.x_vals = fullData.x;
                        window.currentDataCube.y_vals = fullData.y;
                    }
                }
            } catch(e) {
                console.warn("Datacube pre-fetch failed:", e);
            }
        }

        const url = `/api/v1/cache/data?dataset_id=${encodeURIComponent(id)}&ext=${ext}&var_name=${v}&time_idx=${t}&lod=${lod}`;
        console.log("Fetching Plotly data:", url);

        try {
            let data;
            if (window.currentDataCube && window.currentDataCube[t]) {
                data = {
                    z: window.currentDataCube[t],
                    x: window.currentDataCube.x_vals,
                    y: window.currentDataCube.y_vals
                };
            } else {
                const resp = await fetch(url);
                if (!resp.ok) {
                    const errJson = await resp.json().catch(() => ({}));
                    throw new Error(errJson.detail || `HTTP ${resp.status}`);
                }
                data = await resp.json();
            }

            // Destroy previous image/tooltips if existing
            const oldImg = document.getElementById("renderedMap");
            if (oldImg) oldImg.remove();

            let plotDiv = document.getElementById("plotlyMap");
            if (!plotDiv) {
                previewCanvas.innerHTML = "";
                plotDiv = document.createElement("div");
                plotDiv.id = "plotlyMap";
                plotDiv.style.width = "100%";
                plotDiv.style.height = "100%";
                previewCanvas.appendChild(plotDiv);
            }

            const isTemp = v.includes('temp') || v === 't2m';
            const vUnit = isTemp ? '°' : '';

            const trace = {
                z: data.z,
                x: data.x,
                y: data.y,
                type: 'heatmap',
                colorscale: 'Viridis',
                zsmooth: 'best',
                hoverongaps: false,
                hovertemplate: `<b>${v}</b>: %{z:.3f}${vUnit}<br>Index/Coords: [%{y}, %{x}]<extra></extra>`
            };

            const layout = {
                margin: { t: 30, r: 10, b: 10, l: 10 },
                xaxis: { visible: false },
                yaxis: { visible: false },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#c9d1d9' }
            };

            const config = { responsive: true, displayModeBar: false };

            Plotly.react(plotDiv, [trace], layout, config);

            if (loader && loader.parentNode) loader.parentNode.removeChild(loader);

        } catch (err) {
            console.error("Preview render failed:", err);
            previewCanvas.innerHTML = `
                <div class="empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    <p>Render Failed<br><small style="word-break: break-all;">${err.message}</small></p>
                </div>`;
            if (loader && loader.parentNode) loader.parentNode.removeChild(loader);
        }
    }
});
