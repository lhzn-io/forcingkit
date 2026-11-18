// The cache inventory page: the list of stores (polled), deleting and purging, and selection,
// which opens the store in the Preview pane (preview.js) and outlines it on the coverage map
// (coverage.js). A link #<store id>[&f=&k=&t=&v=] opens a store with a field, layer, record and
// view chosen.
document.addEventListener("DOMContentLoaded", () => {
    const datasetList = document.getElementById("datasetList");
    const datasetCount = document.getElementById("datasetCount");
    const purgeBtn = document.getElementById("purgeBtn");

    let currentInventory = [];
    let selectedDataset = null;

    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);

    // Coverage map and timeline (coverage.js): its filters narrow the list below.
    function selectById(id, want) {
        const item = currentInventory.find(i => i.id === id);
        if (!item) return;
        const card = document.querySelector(`.dataset-card[data-id="${CSS.escape(id)}"]`);
        selectDataset(item, card || document.createElement("div"), want);
        if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    if (window.fkCoverage) {
        window.fkCoverage.init({ rerender: () => renderInventory(currentInventory), select: (id) => selectById(id) });
    }

    // - Modal dialogs (messages are built from escaped strings) -
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

    // - Initial load, deep link and polling -
    let inventoryFingerprint = "";
    let pollPaused = false;
    const fingerprint = (items) => items.map(i => `${i.id}:${i.modified_time}:${i.size_mb}`).join("|");

    fetchInventory().then(() => {
        if (window.location.hash && window.fkPreview) {
            const { id, want } = window.fkPreview.parseHash(window.location.hash);
            if (id && currentInventory.some(item => item.id === id)) selectById(id, want);
        }
        setInterval(pollInventory, 2000);
    });

    async function pollInventory() {
        if (pollPaused) return;
        try {
            const resp = await fetch("/api/v1/cache/inventory", { cache: "no-store" });
            if (!resp.ok) return;
            const items = await resp.json();
            const fp = fingerprint(items);
            if (fp === inventoryFingerprint) return;
            inventoryFingerprint = fp;
            currentInventory = items;
            renderInventory(items);
        } catch (_) {
            // A missed poll is retried two seconds later.
        }
    }

    if (purgeBtn) {
        const purgeLabel = purgeBtn.innerHTML;
        purgeBtn.addEventListener("click", async () => {
            const confirmed = await showModal({
                title: "Purge Entire Cache",
                message: "This will <strong>permanently delete every</strong> store in the cache: built stores and raw donor files.<br><br>This action <strong>cannot be undone</strong>.",
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
                selectedDataset = null;
                if (window.fkPreview) window.fkPreview.close();
                await fetchInventory();
                await showAlert({ title: "Cache Purged", message: esc(data.message), icon: "success" });
            } catch (err) {
                console.error("Purge failed:", err);
                await showAlert({ title: "Purge Failed", message: esc(err.message), icon: "error" });
            } finally {
                purgeBtn.disabled = false;
                purgeBtn.innerHTML = purgeLabel;
                pollPaused = false;
            }
        });
    }

    async function deleteDataset(item, e) {
        e.stopPropagation();
        const confirmed = await showModal({
            title: "Delete Dataset",
            message: `Delete <strong>${esc(item.label || item.id)}</strong> (${esc(item.size_mb)} MB)?<br><span style="color:var(--text-secondary);font-size:0.85rem;">${esc(item.id)}</span>`,
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
            if (selectedDataset && selectedDataset.id === item.id) {
                selectedDataset = null;
                if (window.fkPreview) window.fkPreview.close();
                history.replaceState(null, "", window.location.pathname);
            }
            await fetchInventory();
        } catch (err) {
            console.error("Delete failed:", err);
            await showAlert({ title: "Delete Failed", message: esc(err.message), icon: "error" });
        }
    }

    async function fetchInventory() {
        try {
            const resp = await fetch("/api/v1/cache/inventory", { cache: "no-store" });
            if (!resp.ok) throw new Error(`HTTP Error: ${resp.status}`);
            currentInventory = await resp.json();
            inventoryFingerprint = fingerprint(currentInventory);
            renderInventory(currentInventory);
        } catch (err) {
            console.error("Failed to fetch inventory:", err);
            datasetList.innerHTML = `
                <div class="empty-state">
                    <p>Failed to load cache inventory.<br><small>${esc(err.message)}</small></p>
                </div>`;
        }
    }

    // The store's kind from the inventory; the id prefix only for entries without one.
    function kindOf(item) {
        if (item.kind) return item.kind;
        if (item.id.startsWith("ic_")) return "initial condition";
        if (item.id.startsWith("obc_")) return "ocean boundary";
        if (item.type !== "zarr") return "raw donor file";
        return "store";
    }

    function kindClass(kind) {
        if (kind === "raw donor file") return "type-grib";
        if (kind.startsWith("ocean") || kind === "initial condition") return "type-ic";
        return "type-bc";
    }

    function renderInventory(allItems) {
        if (window.fkCoverage) window.fkCoverage.setInventory(allItems);
        const items = window.fkCoverage ? window.fkCoverage.filter(allItems) : allItems;
        datasetCount.textContent = items.length === allItems.length
            ? items.length : `${items.length} of ${allItems.length}`;
        datasetList.innerHTML = "";

        if (items.length === 0) {
            datasetList.innerHTML = `<div class="empty-state"><p>${allItems.length
                ? "No stores match the filters." : "No stores in the cache."}</p></div>`;
            return;
        }

        items.forEach(item => {
            const date = new Date(item.modified_time * 1000).toLocaleString(undefined, {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
            });
            const card = document.createElement("div");
            card.className = "dataset-card";
            card.dataset.id = item.id;
            if (selectedDataset && selectedDataset.id === item.id) card.classList.add("selected");

            const kind = kindOf(item);
            const family = item.family || item.donor_model || item.source || "";
            const hashId = item.id.replace(/^(bc_|ic_|obc_)/, "");
            const title = item.label || item.id;
            const stats = [item.grid_shape ? `${item.grid_shape} grid` : "",
                item.time_steps ? `${item.time_steps} records` : ""].filter(Boolean).join(" · ");
            const vars = item.variables ? [...item.variables].sort().join(", ") : "";

            card.innerHTML = `
                <div class="card-header">
                    <div class="card-title">${esc(title)}</div>
                    <button class="card-delete-btn" title="Delete dataset">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
                <div class="card-meta">
                    <span class="type-tag ${kindClass(kind)}">${esc(kind.toUpperCase())}${family ? `<span class="donor-label">${esc(family)}</span>` : ""}${item.resolution ? `<span class="res-label">${esc(item.resolution)}</span>` : ""}<span class="hash-label">${esc(hashId)}</span></span>
                    <span>${esc(item.size_mb)} MB</span>
                    <span>${esc(date)}</span>
                </div>
                ${stats ? `<div class="card-stats">${esc(stats)}</div>` : ""}
                ${vars ? `<div class="card-vars">${esc(vars)}</div>` : ""}
                ${item.spatial_extent ? `<div class="card-stats">${esc(item.spatial_extent)}</div>` : ""}
                ${item.time_range ? `<div class="card-time">${esc(item.time_range)}</div>` : ""}
            `;
            card.querySelector(".card-delete-btn").addEventListener("click", (e) => deleteDataset(item, e));
            card.addEventListener("click", () => selectDataset(item, card));
            datasetList.appendChild(card);
        });
    }

    function selectDataset(item, cardElement, want) {
        if (window.fkCoverage) window.fkCoverage.setSelected(item.id);
        document.querySelectorAll(".dataset-card").forEach(c => c.classList.remove("selected"));
        cardElement.classList.add("selected");
        selectedDataset = item;
        if (window.fkPreview) window.fkPreview.open(item, want || {});
    }
});
