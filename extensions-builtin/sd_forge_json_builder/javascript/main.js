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
        summary: { caption: "", meta: "", canvas: "single" },
        items: [],
    };

    const listRefs = new Map();
    const canvasRefs = new Map();

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

    const sumCaption = h("input", { type: "text", class: "jb-sum-caption", placeholder: "global caption" });
    const sumMeta = h("input", { type: "text", class: "jb-sum-meta", placeholder: "meta tags (comma separated)" });
    const canvasRadios = CANVAS_TYPES.map((t) => {
        const id = `jb-canvas-${t}`;
        const radio = h("input", { type: "radio", name: "jb-canvas", value: t, id, checked: state.summary.canvas === t });
        radio.addEventListener("change", () => {
            if (radio.checked) state.summary.canvas = t;
        });
        return h("label", { class: "jb-radio", for: id }, [radio, t]);
    });

    const summarySection = h("div", { class: "jb-section" }, [
        h("div", { class: "jb-section-title" }, ["Summary"]),
        h("div", { class: "jb-row" }, canvasRadios),
        h("label", { class: "jb-fld-label", for: "jb-sum-caption" }, ["Caption"]),
        sumCaption,
        h("label", { class: "jb-fld-label", for: "jb-sum-meta" }, ["Meta"]),
        sumMeta,
    ]);
    sumCaption.id = "jb-sum-caption";
    sumMeta.id = "jb-sum-meta";

    const addBtn = h("button", { type: "button", class: "jb-add" }, ["Add Item"]);
    const clearBtn = h("button", { type: "button", class: "jb-clear" }, ["Clear All"]);
    const listEl = h("div", { class: "jb-list" });
    const listSection = h("div", { class: "jb-section jb-list-section" }, [h("div", { class: "jb-list-toolbar" }, [addBtn, clearBtn]), listEl]);

    const sendT2I = h("button", { type: "button", class: "jb-send-t2i" }, ["Send to txt2img"]);
    const sendI2I = h("button", { type: "button", class: "jb-send-i2i" }, ["Send to img2img"]);
    const submitSection = h("div", { class: "jb-section jb-submit" }, [sendT2I, sendI2I]);

    const side = h("div", { class: "jb-side" }, [resolutionSection, summarySection, listSection, submitSection]);
    const root = h("div", { class: "jb-root" }, [canvasSection, side]);

    tab.textContent = "";
    tab.appendChild(root);

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
    }

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
            if (inp && document.activeElement !== inp) inp.value = ta.value;
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
        handle.addEventListener("pointerdown", (ev) => startResize(item, ev));
        el.appendChild(handle);

        el.addEventListener("pointerdown", (ev) => {
            if (ev.target === handle) return;
            if (ev.target.tagName === "TEXTAREA" && !ev.target.readOnly) return;
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

        const remove = h("button", { type: "button", class: "jb-remove", title: "Remove" }, ["×"]);
        remove.addEventListener("click", () => removeItem(item.id));
        top.appendChild(remove);
        row.appendChild(top);

        const caption = h("input", { type: "text", class: "jb-item-caption", placeholder: "caption", value: item.caption });
        caption.addEventListener("input", () => {
            item.caption = caption.value;
            const ref = canvasRefs.get(item.id);
            if (ref && document.activeElement !== ref.caption) ref.caption.value = caption.value;
        });
        row.appendChild(caption);

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
    });
    sumMeta.addEventListener("input", () => {
        state.summary.meta = sumMeta.value;
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
        root.querySelectorAll(".jb-invalid").forEach((el) => el.classList.remove("jb-invalid"));
        const invalid = [];

        if (state.summary.caption.trim() === "") invalid.push(sumCaption);

        for (const item of state.items) {
            const list = listRefs.get(item.id);

            if (item.caption.trim() === "") {
                if (list) invalid.push(list.captionInput);
            }
            if (item.label === "text" && item.text.trim() === "") {
                if (list?.textInput) invalid.push(list.textInput);
            }
        }

        invalid.forEach((el) => el.classList.add("jb-invalid"));
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

        const promptEl = which === "t2i" ? t2i_prompt : i2i_prompt;
        const widthEl = which === "t2i" ? t2i_width : i2i_width;
        const heightEl = which === "t2i" ? t2i_height : i2i_height;

        promptEl.value = prompt;
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
