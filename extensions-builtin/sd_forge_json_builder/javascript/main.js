onUiLoaded(() => {
    const tab = document.getElementById("tab_sd-forge-json-builder");

    const t2i_prompt = document.getElementById("txt2img_prompt").querySelector("textarea");
    const t2i_width = document.getElementById("txt2img_width").querySelector("input");
    const t2i_height = document.getElementById("txt2img_height").querySelector("input");

    const i2i_prompt = document.getElementById("img2img_prompt").querySelector("textarea");
    const i2i_width = document.getElementById("img2img_width").querySelector("input");
    const i2i_height = document.getElementById("img2img_height").querySelector("input");

    const LABELS = ["object", "character", "text"];
    const CANVAS_TYPES = ["single", "multi_panel", "collage"];
    const ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
    const MP_PIXELS = 1024 * 1024;

    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const round16 = (v) => Math.max(16, Math.round(v / 16) * 16);

    let idCounter = 0;

    const state = {
        width: 1024,
        height: 1024,
        summary: { caption: "", meta: "", extra: "", canvas: "single" },
        items: [],
    };

    const listRefs = new Map();
    const canvasRefs = new Map();

    const MIN_ZOOM = 0.25;
    const MAX_ZOOM = 8;
    const view = { scale: 1, panX: 0, panY: 0 };

    function h(tag, attrs = {}, children = []) {
        const el = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
            if (k === "class") el.className = v;
            else if (k === "style") Object.assign(el.style, v);
            else if (k === "dataset") Object.assign(el.dataset, v);
            else if (k === "value" || k === "checked") el[k] = v;
            else if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
            else el.setAttribute(k, v);
        }
        for (const c of [].concat(children)) {
            if (c === null || c === undefined) continue;
            el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
        }
        return el;
    }

    const growables = new Set();

    function autoGrow(el) {
        if (el?.tagName !== "TEXTAREA") return;
        el.style.height = "auto";
        const cs = getComputedStyle(el);
        const borders = parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
        el.style.height = `${el.scrollHeight + borders}px`;
    }

    const growObserver = new ResizeObserver(() => growables.forEach(autoGrow));

    function addGrowable(el) {
        if (growables.has(el)) return;
        growables.add(el);
        growObserver.observe(el);
        requestAnimationFrame(() => autoGrow(el));
    }

    function pruneGrowables() {
        for (const el of growables) {
            if (el.isConnected) continue;
            growObserver.unobserve(el);
            growables.delete(el);
        }
    }

    function defaultBbox() {
        const m = Math.round(Math.min(state.width, state.height) * 0.1);
        const s = Math.round(Math.min(state.width, state.height) * 0.3);
        return [m, m, m + s, m + s];
    }

    function newItem() {
        return { id: ++idCounter, label: "object", caption: "", text: "", bbox: defaultBbox() };
    }

    state.items.push(newItem());

    // ----- canvas -----

    const canvasSection = h("div", { class: "jb-canvas" });
    const stage = h("div", { class: "jb-stage" });
    canvasSection.appendChild(stage);

    // ----- side panel -----

    const widthInput = h("input", { type: "number", class: "jb-width", min: "16", step: "16", value: state.width });
    const heightInput = h("input", { type: "number", class: "jb-height", min: "16", step: "16", value: state.height });
    const mpInput = h("input", { type: "number", class: "jb-mp", min: "0.1", step: "0.1", value: "1" });
    const arSelect = h(
        "select",
        { class: "jb-ar" },
        ASPECT_RATIOS.map((r) => h("option", { value: r }, [r])),
    );
    const setResBtn = h("button", { type: "button", class: "jb-set-res" }, ["Set Resolution"]);

    const resolutionSection = h("div", { class: "jb-section" }, [
        h("div", { class: "jb-section-title" }, ["Resolution"]),
        h("div", { class: "jb-row" }, [h("label", {}, ["Width"]), widthInput, h("label", {}, ["Height"]), heightInput]),
        h("div", { class: "jb-row" }, [h("label", {}, ["MP"]), mpInput, h("label", {}, ["AR"]), arSelect, setResBtn]),
    ]);

    const sumCaption = h("textarea", { class: "jb-sum-caption", placeholder: "global caption", rows: "1" });
    const sumMeta = h("input", { type: "text", class: "jb-sum-meta", placeholder: "meta tags (comma separated)" });
    const sumExtra = h("input", { type: "text", class: "jb-sum-extra", placeholder: "e.g. LoRA" });
    const canvasRadioInputs = [];
    const canvasRadios = CANVAS_TYPES.map((t) => {
        const id = `jb-canvas-${t}`;
        const radio = h("input", { type: "radio", name: "jb-canvas", value: t, id, checked: state.summary.canvas === t });
        radio.addEventListener("change", () => {
            if (radio.checked) state.summary.canvas = t;
        });
        canvasRadioInputs.push(radio);
        return h("label", { class: "jb-radio", for: id }, [radio, t]);
    });

    const summarySection = h("div", { class: "jb-section" }, [
        h("div", { class: "jb-section-title" }, ["Summary"]),
        h("div", { class: "jb-row" }, canvasRadios),
        h("label", { class: "jb-fld-label", for: "jb-sum-caption" }, ["Caption"]),
        sumCaption,
        h("div", { class: "jb-row" }, [
            h("div", { class: "jb-half" }, [h("label", { class: "jb-fld-label", for: "jb-sum-meta" }, ["Meta"]), sumMeta]),
            h("div", { class: "jb-half" }, [h("label", { class: "jb-fld-label", for: "jb-sum-extra" }, ["Extra"]), sumExtra]),
        ]),
    ]);
    sumCaption.id = "jb-sum-caption";
    sumMeta.id = "jb-sum-meta";
    sumExtra.id = "jb-sum-extra";

    const addBtn = h("button", { type: "button", class: "jb-add" }, ["Add Item"]);
    const clearBtn = h("button", { type: "button", class: "jb-clear" }, ["Clear All"]);
    const listEl = h("div", { class: "jb-list" });
    const listSection = h("div", { class: "jb-section jb-list-section" }, [h("div", { class: "jb-list-toolbar" }, [addBtn, clearBtn]), listEl]);

    const sendT2I = h("button", { type: "button", class: "jb-send-t2i" }, ["Send to txt2img"]);
    const sendI2I = h("button", { type: "button", class: "jb-send-i2i" }, ["Send to img2img"]);
    const submitSection = h("div", { class: "jb-section jb-submit" }, [sendT2I, sendI2I]);

    // ----- save / load -----

    const loadPromptBtn = h("button", { type: "button", class: "jb-load-prompt" }, ["Load Prompt"]);
    const saveJsonBtn = h("button", { type: "button", class: "jb-save-json" }, ["Save JSON"]);
    const loadJsonBtn = h("button", { type: "button", class: "jb-load-json" }, ["Load JSON"]);
    const saveLoadSection = h("div", { class: "jb-section jb-save" }, [
        h("div", { class: "jb-section-title" }, ["Save/Load"]),
        h("div", { class: "jb-row" }, [loadPromptBtn]),
        h("div", { class: "jb-row" }, [saveJsonBtn, loadJsonBtn]),
    ]);

    const dialogInput = h("textarea", { class: "jb-dialog-input", placeholder: "Compiled Prompt or Layout Draft", spellcheck: "false" });
    const dialogWidth = h("input", { type: "number", min: "16", step: "16", class: "jb-dialog-width" });
    const dialogHeight = h("input", { type: "number", min: "16", step: "16", class: "jb-dialog-height" });
    const dialogClose = h("button", { type: "button", class: "jb-dialog-close", title: "Close" }, ["\u00D7"]);
    const dialogLoad = h("button", { type: "button", class: "jb-dialog-load" }, ["Load"]);
    const modal = h("div", { class: "jb-modal jb-hidden" }, [
        h("div", { class: "jb-dialog" }, [
            dialogClose,
            h("div", { class: "jb-dialog-body" }, [dialogInput, h("div", { class: "jb-dialog-dims" }, [h("label", {}, ["Width"]), dialogWidth, h("label", {}, ["Height"]), dialogHeight])]),
            dialogLoad,
        ]),
    ]);

    const fileInput = h("input", { type: "file", accept: ".json,application/json", class: "jb-file" });
    fileInput.style.display = "none";

    const topRow = h("div", { class: "jb-top" }, [resolutionSection, saveLoadSection]);
    const side = h("div", { class: "jb-side" }, [topRow, summarySection, listSection, submitSection]);
    const root = h("div", { class: "jb-root" }, [canvasSection, side, modal, fileInput]);

    tab.textContent = "";
    tab.appendChild(root);
    addGrowable(sumCaption);

    function loadDraft(draft) {
        state.width = draft.width;
        state.height = draft.height;
        state.summary.caption = draft.summary.caption;
        state.summary.meta = draft.summary.meta.join(", ");
        state.summary.canvas = draft.summary.canvas;
        state.summary.extra = "";
        state.items = draft.items.map((item) => ({
            id: ++idCounter,
            label: item.label,
            caption: item.caption,
            text: item.text || "",
            bbox: item.bbox.slice(),
        }));

        widthInput.value = state.width;
        heightInput.value = state.height;
        sumCaption.value = state.summary.caption;
        sumMeta.value = state.summary.meta;
        sumExtra.value = state.summary.extra;
        autoGrow(sumCaption);
        canvasRadioInputs.forEach((radio) => {
            radio.checked = radio.value === state.summary.canvas;
        });

        fitStage();
        renderAll();
    }

    function openDialog() {
        loadPromptBtn.classList.remove("jb-invalid");
        dialogInput.value = "";
        dialogWidth.value = state.width;
        dialogHeight.value = state.height;
        modal.classList.remove("jb-hidden");
        dialogInput.focus();
    }

    function closeDialog() {
        modal.classList.add("jb-hidden");
    }

    function handleDialogLoad() {
        const text = dialogInput.value.trim();
        if (text === "") {
            closeDialog();
            return;
        }
        try {
            const draft = decompilePrompt(text, parseFloat(dialogWidth.value), parseFloat(dialogHeight.value));
            loadDraft(draft);
        } catch {
            loadPromptBtn.classList.add("jb-invalid");
        }
        closeDialog();
    }

    loadPromptBtn.addEventListener("click", openDialog);
    dialogClose.addEventListener("click", closeDialog);
    dialogLoad.addEventListener("click", handleDialogLoad);

    saveJsonBtn.addEventListener("click", () => {
        const name = prompt("Save JSON as", "layout.json");
        if (name === null || name.trim() === "") return;
        const blob = new Blob([JSON.stringify(buildJson(), null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = h("a", { href: url, download: name.trim() });
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    });

    loadJsonBtn.addEventListener("click", () => {
        loadJsonBtn.classList.remove("jb-invalid");
        fileInput.value = "";
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.addEventListener("load", () => {
            try {
                const draft = decompilePrompt(reader.result, NaN, NaN);
                loadDraft(draft);
            } catch {
                loadJsonBtn.classList.add("jb-invalid");
            }
        });
        reader.readAsText(file);
    });

    // ----- canvas sizing -----

    function fitStage() {
        const inset = 32;
        const cw = Math.max(1, canvasSection.clientWidth - inset);
        const ch = Math.max(1, canvasSection.clientHeight - inset);
        const ratio = state.width / state.height;
        let w = cw;
        let hgt = cw / ratio;
        if (hgt > ch) {
            hgt = ch;
            w = ch * ratio;
        }
        stage.style.width = w + "px";
        stage.style.height = hgt + "px";
        clampPan();
        resetView();
    }

    function applyView() {
        stage.style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.scale})`;
        stage.style.setProperty("--jb-text-scale", String(1 / Math.max(1, view.scale)));
    }

    function clampAxis(pan, canvasSize, stageSize) {
        const diff = (stageSize - canvasSize) / 2;
        return diff >= 0 ? clamp(pan, -diff, diff) : clamp(pan, diff, -diff);
    }

    function clampPan() {
        const c = canvasSection.getBoundingClientRect();
        const r = stage.getBoundingClientRect();
        view.panX = clampAxis(view.panX, c.width, r.width);
        view.panY = clampAxis(view.panY, c.height, r.height);
    }

    function zoomAt(clientX, clientY, factor) {
        const next = clamp(view.scale * factor, MIN_ZOOM, MAX_ZOOM);
        const ratio = next / view.scale;
        if (ratio === 1) return;
        const r = stage.getBoundingClientRect();
        view.panX += (clientX - (r.left + r.width / 2)) * (1 - ratio);
        view.panY += (clientY - (r.top + r.height / 2)) * (1 - ratio);
        view.scale = next;
        applyView();
        clampPan();
        applyView();
    }

    function resetView() {
        view.scale = 1;
        view.panX = 0;
        view.panY = 0;
        applyView();
    }

    function startPan(e) {
        const sx = e.clientX;
        const sy = e.clientY;
        const px = view.panX;
        const py = view.panY;
        stage.classList.add("jb-panning");
        function move(ev) {
            view.panX = px + (ev.clientX - sx);
            view.panY = py + (ev.clientY - sy);
            clampPan();
            applyView();
        }
        function up() {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            stage.classList.remove("jb-panning");
        }
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    stage.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();
            zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.2 : 1 / 1.2);
        },
        { passive: false },
    );

    stage.addEventListener("pointerdown", (e) => {
        if (e.button !== 1) return;
        e.preventDefault();
        startPan(e);
    });

    stage.addEventListener("dblclick", (e) => {
        if (e.target.closest(".jb-box")) return;
        resetView();
    });

    function updateBoxPosition(item) {
        const ref = canvasRefs.get(item.id);
        if (!ref) return;
        const [x1, y1, x2, y2] = item.bbox;
        ref.el.style.left = (x1 / state.width) * 100 + "%";
        ref.el.style.top = (y1 / state.height) * 100 + "%";
        ref.el.style.width = ((x2 - x1) / state.width) * 100 + "%";
        ref.el.style.height = ((y2 - y1) / state.height) * 100 + "%";
    }

    function syncListNums(item) {
        const ref = listRefs.get(item.id);
        if (!ref) return;
        ref.numInputs.forEach((inp, i) => {
            if (document.activeElement !== inp) inp.value = item.bbox[i];
        });
    }

    function makeEditable(item, field) {
        const ta = h("textarea", { class: "jb-field", placeholder: field, readonly: "", spellcheck: "false" });
        ta.value = item[field] || "";
        ta.addEventListener("dblclick", () => {
            ta.readOnly = false;
            ta.focus();
        });
        ta.addEventListener("blur", () => {
            ta.readOnly = true;
        });
        ta.addEventListener("input", () => {
            item[field] = ta.value;
            const ref = listRefs.get(item.id);
            if (!ref) return;
            const inp = field === "caption" ? ref.captionInput : ref.textInput;
            if (inp && document.activeElement !== inp) {
                inp.value = ta.value;
                autoGrow(inp);
            }
        });
        return ta;
    }

    function startDrag(item, e) {
        const ref = canvasRefs.get(item.id);
        const rect = stage.getBoundingClientRect();
        const scaleX = state.width / rect.width;
        const scaleY = state.height / rect.height;
        const sx = e.clientX;
        const sy = e.clientY;
        const [x1, y1, x2, y2] = item.bbox;
        const w = x2 - x1;
        const hgt = y2 - y1;
        let moved = false;

        function move(ev) {
            const dx = (ev.clientX - sx) * scaleX;
            const dy = (ev.clientY - sy) * scaleY;
            if (!moved && Math.hypot(dx, dy) < 3) return;
            moved = true;
            ref.el.classList.add("jb-active");
            const nx1 = clamp(Math.round(x1 + dx), 0, state.width - w);
            const ny1 = clamp(Math.round(y1 + dy), 0, state.height - hgt);
            item.bbox = [nx1, ny1, nx1 + w, ny1 + hgt];
            updateBoxPosition(item);
            syncListNums(item);
        }

        function up() {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            ref.el.classList.remove("jb-active");
        }

        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    function startResize(item, e) {
        e.stopPropagation();
        const ref = canvasRefs.get(item.id);
        const rect = stage.getBoundingClientRect();
        const scaleX = state.width / rect.width;
        const scaleY = state.height / rect.height;
        const sx = e.clientX;
        const sy = e.clientY;
        const [x1, y1, x2, y2] = item.bbox;
        ref.el.classList.add("jb-active");

        function move(ev) {
            const nx2 = clamp(Math.round(x2 + (ev.clientX - sx) * scaleX), x1 + 16, state.width);
            const ny2 = clamp(Math.round(y2 + (ev.clientY - sy) * scaleY), y1 + 16, state.height);
            item.bbox = [x1, y1, nx2, ny2];
            updateBoxPosition(item);
            syncListNums(item);
        }

        function up() {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            ref.el.classList.remove("jb-active");
        }

        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    function createBox(item) {
        const el = h("div", { class: "jb-box", dataset: { id: String(item.id) } });
        el.appendChild(h("div", { class: "jb-box-tag" }, [item.label]));

        const caption = makeEditable(item, "caption");
        el.appendChild(caption);

        let text = null;
        if (item.label === "text") {
            text = makeEditable(item, "text");
            text.classList.add("jb-box-text");
            el.appendChild(text);
        }

        const handle = h("div", { class: "jb-box-handle" });
        handle.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            startResize(item, ev);
        });
        el.appendChild(handle);

        el.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            if (ev.target === handle) return;
            if (ev.target.tagName === "TEXTAREA" && !ev.target.readOnly) return;
            ev.preventDefault();
            startDrag(item, ev);
        });

        canvasRefs.set(item.id, { el, caption, text });
        updateBoxPosition(item);
        return el;
    }

    function renderCanvas() {
        stage.textContent = "";
        canvasRefs.clear();
        for (const item of state.items) stage.appendChild(createBox(item));
    }

    // ----- list -----

    function createRow(item) {
        const row = h("div", { class: "jb-item", dataset: { id: String(item.id) } });

        const top = h("div", { class: "jb-item-top" });
        const radios = LABELS.map((label) => {
            const id = `jb-${item.id}-${label}`;
            const radio = h("input", {
                type: "radio",
                name: `jb-label-${item.id}`,
                value: label,
                id,
                checked: item.label === label,
            });
            radio.addEventListener("change", () => {
                if (radio.checked) {
                    item.label = label;
                    renderAll();
                }
            });
            return h("label", { class: "jb-radio", for: id }, [radio, label]);
        });
        top.append(...radios);

        const remove = h("button", { type: "button", class: "jb-remove", title: "Remove" }, ["\u00D7"]);
        remove.addEventListener("click", () => removeItem(item.id));
        top.appendChild(remove);
        row.appendChild(top);

        const caption = h("textarea", { class: "jb-item-caption", placeholder: "caption", rows: "1", value: item.caption });
        caption.addEventListener("input", () => {
            item.caption = caption.value;
            autoGrow(caption);
            const ref = canvasRefs.get(item.id);
            if (ref && document.activeElement !== ref.caption) ref.caption.value = caption.value;
        });
        row.appendChild(caption);
        addGrowable(caption);

        let textInput = null;
        if (item.label === "text") {
            textInput = h("input", { type: "text", class: "jb-item-text", placeholder: "text", value: item.text });
            textInput.addEventListener("input", () => {
                item.text = textInput.value;
                const ref = canvasRefs.get(item.id);
                if (ref?.text && document.activeElement !== ref.text) ref.text.value = textInput.value;
            });
            row.appendChild(textInput);
        }

        const placeholders = ["x1", "y1", "x2", "y2"];
        const nums = h("div", { class: "jb-item-nums" });
        const numInputs = item.bbox.map((value, i) => {
            const inp = h("input", { type: "number", class: "jb-num", placeholder: placeholders[i], value });
            inp.addEventListener("input", () => {
                const v = parseFloat(inp.value);
                if (!Number.isNaN(v)) {
                    item.bbox[i] = v;
                    updateBoxPosition(item);
                }
            });
            nums.appendChild(inp);
            return inp;
        });
        row.appendChild(nums);

        listRefs.set(item.id, { row, captionInput: caption, textInput, numInputs });
        return row;
    }

    function renderList() {
        listEl.textContent = "";
        listRefs.clear();
        pruneGrowables();
        for (const item of state.items) listEl.appendChild(createRow(item));
    }

    function renderAll() {
        renderList();
        renderCanvas();
    }

    function addItem() {
        state.items.push(newItem());
        renderAll();
    }

    function removeItem(id) {
        const index = state.items.findIndex((it) => it.id === id);
        if (index >= 0) state.items.splice(index, 1);
        if (state.items.length === 0) state.items.push(newItem());
        renderAll();
    }

    function clearItems() {
        state.items = [newItem()];
        renderAll();
    }

    // ----- inputs -----

    function applyResolution(w, h) {
        state.width = w;
        state.height = h;
        fitStage();
        renderCanvas();
    }

    function clampItemsToCanvas() {
        for (const item of state.items) {
            const bw = Math.min(item.bbox[2] - item.bbox[0], state.width);
            const bh = Math.min(item.bbox[3] - item.bbox[1], state.height);
            const x1 = clamp(item.bbox[0], 0, state.width - bw);
            const y1 = clamp(item.bbox[1], 0, state.height - bh);
            item.bbox = [x1, y1, x1 + bw, y1 + bh];
            syncListNums(item);
        }
        renderCanvas();
    }

    widthInput.addEventListener("input", () => {
        const v = parseInt(widthInput.value, 10);
        if (v > 0) applyResolution(v, state.height);
    });
    widthInput.addEventListener("change", clampItemsToCanvas);

    heightInput.addEventListener("input", () => {
        const v = parseInt(heightInput.value, 10);
        if (v > 0) applyResolution(state.width, v);
    });
    heightInput.addEventListener("change", clampItemsToCanvas);

    setResBtn.addEventListener("click", () => {
        const mp = parseFloat(mpInput.value);
        if (!(mp > 0)) return;
        const [aw, ah] = arSelect.value.split(":").map(Number);
        const area = mp * MP_PIXELS;
        const w = Math.sqrt((area * aw) / ah);
        const hgt = area / w;
        const width = round16(w);
        const height = round16(hgt);
        widthInput.value = width;
        heightInput.value = height;
        applyResolution(width, height);
        clampItemsToCanvas();
    });

    sumCaption.addEventListener("input", () => {
        state.summary.caption = sumCaption.value;
        autoGrow(sumCaption);
    });
    sumMeta.addEventListener("input", () => {
        state.summary.meta = sumMeta.value;
    });
    sumExtra.addEventListener("input", () => {
        state.summary.extra = sumExtra.value;
    });

    addBtn.addEventListener("click", addItem);
    clearBtn.addEventListener("click", clearItems);

    // ----- submit -----

    function buildJson() {
        const meta = state.summary.meta
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
        const summary = {};
        if (meta.length) summary.meta = meta;
        summary.canvas = state.summary.canvas;
        summary.caption = state.summary.caption;
        return {
            width: state.width,
            height: state.height,
            bbox_coord_space: "abs",
            bbox_format: "xyxy",
            summary,
            items: state.items.map((item) => {
                const out = { label: item.label, caption: item.caption, bbox: item.bbox.map((n) => Math.round(n)) };
                if (item.label === "text") out.text = item.text;
                return out;
            }),
        };
    }

    function validate() {
        root.querySelectorAll(".jb-invalid").forEach((el) => {
            el.classList.remove("jb-invalid");
        });
        const invalid = [];

        if (state.summary.caption.trim() === "") invalid.push(sumCaption);

        for (const item of state.items) {
            const list = listRefs.get(item.id);
            const canvas = canvasRefs.get(item.id);

            if (item.caption.trim() === "") {
                if (list) invalid.push(list.captionInput);
                if (canvas) invalid.push(canvas.caption);
            }
            if (item.label === "text" && item.text.trim() === "") {
                if (list?.textInput) invalid.push(list.textInput);
                if (canvas?.text) invalid.push(canvas.text);
            }
        }

        invalid.forEach((el) => {
            el.classList.add("jb-invalid");
        });
        return invalid.length === 0;
    }

    function submit(which) {
        if (!validate()) return;
        let prompt = null;

        try {
            prompt = compilePrompt(JSON.stringify(buildJson()));
        } catch (e) {
            alert(e);
            return;
        }

        if (state.summary.extra.trim() !== "") {
            prompt += "\n" + state.summary.extra;
        }

        const promptEl = which === "t2i" ? t2i_prompt : i2i_prompt;
        const widthEl = which === "t2i" ? t2i_width : i2i_width;
        const heightEl = which === "t2i" ? t2i_height : i2i_height;

        promptEl.value = prompt.replaceAll("\n", " ");
        widthEl.value = state.width;
        heightEl.value = state.height;
        updateInput(promptEl);
        updateInput(widthEl);
        updateInput(heightEl);

        if (which === "t2i") switch_to_txt2img();
        else switch_to_img2img();
    }

    sendT2I.addEventListener("click", () => submit("t2i"));
    sendI2I.addEventListener("click", () => submit("i2i"));

    // ----- init -----

    renderAll();
    fitStage();
    requestAnimationFrame(fitStage);
    if (window.ResizeObserver) new ResizeObserver(fitStage).observe(canvasSection);
});
