(function () {
    const TUTORIAL = `
Each item is one region: pick a label (object / character / text), write a caption, then drag the bounding box to position it, or drag the bottom-right corner to resize it. Double-clicking a bounding box or the text field in the list brings it to the front.
Fields in the side list are edited with a single click. Double-clicking any field syncs it to the main prompt: both fields become highlighted, and editing one updates the other, allowing the usage of Extensions like auto-complete. Double-click the same field again to unlink.
"Send to Prompt" validates the layout then compiles it into the main prompt, and sets the width and height; "Load from Prompt" reads a compiled prompt and the sizes back into the editor, mainly for reading infotext; "Save/Load JSON" stores the draft as a .json file on disk.
"Extra" is appended to the prompt after the JSON structure, mainly for LoRA syntax.
Canvas Controls: scroll to zoom, middle-drag to pan, double-click empty space or click magnifier button to reset view.
    `;

    let instance;
    let currentTab;
    let stopPromptLink = () => { };

    onUiTabChange(() => {
        stopPromptLink();
        currentTab = get_uiCurrentTab().textContent.trim();
        const div = document.getElementById(`${currentTab}_json_builder`);
        if (div == null) return;

        div.childNodes[0].style.display = "none";
        div.appendChild(instance);
    });

    onUiLoaded(() => {
        currentTab = get_uiCurrentTab().textContent.trim();
        const PASTEL_COLORS = ["#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff", "#d7baff", "#ffbaf2"];
        let activePromptLink = null;

        stopPromptLink = () => {
            if (!activePromptLink) return;
            activePromptLink.prompt.removeEventListener("input", activePromptLink.onPromptInput);
            for (const el of [activePromptLink.prompt, ...activePromptLink.fields]) {
                el.style.removeProperty("outline");
                el.style.removeProperty("outline-offset");
            }
            activePromptLink = null;
        };

        async function flushSelectedStyles(link) {
            const dropdown = document.getElementById(`${currentTab}_styles`);
            const names = Array.from(dropdown?.querySelectorAll(".token > span") ?? []).map((el) => el.textContent.trim());
            if (!dropdown || names.length === 0) return;

            let styles;
            try {
                const req = await (await fetch("/internal/json_builder_prompt_styles")).json();
                styles = Object.fromEntries(req.map((s) => [s.name, s]));
            } catch (e) {
                console.error("Failed to fetch Styles:", e.message);
                return;
            }

            const promptEl = link.prompt;
            let prompt = promptEl.value;

            for (let i = 0; i < names.length; i++) {
                if (!(names[i] in styles)) continue;
                const s = styles[names[i]].prompt.trim();
                if (s.includes("{prompt}")) prompt = s.replaceAll("{prompt}", prompt);
                else prompt = prompt + s;
            }

            link.setValue(prompt);
            dropdown.querySelector(".token-remove.remove-all")?.click();
        }

        function updatePromptLink(key, value) {
            const link = activePromptLink;
            if (!link || link.key !== key || link.syncing) return;
            link.syncing = true;
            link.prompt.value = value;
            updateInput(link.prompt);
            link.syncing = false;
        }

        function addPromptLink(field, key, color, getValue, setValue) {
            field.dataset.promptLinkKey = key;

            field.addEventListener("dblclick", async () => {
                if (activePromptLink) {
                    await flushSelectedStyles(activePromptLink);
                    const unlinkOnly = activePromptLink.key === key;
                    stopPromptLink();
                    if (unlinkOnly) return;
                }

                const prompt = document.getElementById(`${currentTab}_prompt`).querySelector("textarea");
                const fields = new Set(root.querySelectorAll(`[data-prompt-link-key="${key}"]`));
                const link = { key, fields, prompt, syncing: true, setValue };
                activePromptLink = link;

                link.onPromptInput = () => {
                    if (activePromptLink !== link || link.syncing) return;
                    link.syncing = true;
                    link.setValue(prompt.value);
                    link.syncing = false;
                };

                for (const el of [prompt, ...fields]) {
                    el.style.setProperty("outline", `2px solid ${color}`, "important");
                    el.style.setProperty("outline-offset", "1px");
                }

                prompt.addEventListener("input", link.onPromptInput);
                prompt.value = getValue();
                updateInput(prompt);
                link.syncing = false;
            });

            field.addEventListener("input", () => updatePromptLink(key, getValue()));

            if (activePromptLink?.key === key) {
                field.style.setProperty("outline", `2px solid ${color}`, "important");
                field.style.setProperty("outline-offset", "1px");
                activePromptLink.fields.add(field);
            }
        }

        const LABELS = ["object", "character", "text"];
        const itemColor = (index) => PASTEL_COLORS[index % PASTEL_COLORS.length];
        const itemFieldKey = (item, field) => `item-${item.id}-${field}`;
        const CANVAS_TYPES = ["single", "multi_panel", "collage"];
        const ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
        const MP_PIXELS = 1024 * 1024;

        const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
        const round16 = (v) => Math.max(16, Math.round(v / 16) * 16);

        let idCounter = 0;
        let spawnCount = 0;

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
            const lineHeight = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.2 || 16;
            const padding = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
            const oneLine = Math.ceil(lineHeight) + padding + borders;
            const next = `${Math.max(el.scrollHeight + borders, oneLine)}px`;
            if (el.style.height !== next) el.style.height = next;
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

        function defaultBbox(index) {
            const m = Math.round(Math.min(state.width, state.height) * 0.1);
            const s = Math.round(Math.min(state.width, state.height) * 0.3);
            const offset = Math.min(index * Math.max(16, Math.round(Math.min(state.width, state.height) * 0.05)), state.width - m - s, state.height - m - s);
            return [m + offset, m + offset, m + s + offset, m + s + offset];
        }

        function newItem() {
            const index = spawnCount++;
            return { id: ++idCounter, label: "object", caption: "", text: "", bbox: defaultBbox(index) };
        }

        state.items.push(newItem());

        // ----- canvas -----

        const canvasSection = h("div", { class: "jb-canvas" });
        canvasSection.style.position = "relative";
        const stage = h("div", { class: "jb-stage" });
        let selectedItemId = null;

        function selectItem(id) {
            selectedItemId = id;
            canvasRefs.forEach((ref, itemId) => {
                ref.el.style.zIndex = itemId === id ? "1" : "";
            });
            const box = canvasRefs.get(id)?.el;
            if (box) stage.appendChild(box);
        }

        canvasSection.appendChild(stage);

        const fitCanvasBtn = h("button", { type: "button", class: "jb-fit-canvas", title: "Fit to View", "aria-label": "Fit to View" }, ["\u{1F50D}"]);
        Object.assign(fitCanvasBtn.style, { position: "absolute", left: "8px", bottom: "8px", zIndex: "64" });
        const helpBtn = h("button", { type: "button", class: "jb-help", title: TUTORIAL, "aria-label": "How to Use" }, ["\u2754"]);
        Object.assign(helpBtn.style, { position: "absolute", right: "8px", bottom: "8px", zIndex: "64" });
        canvasSection.append(fitCanvasBtn, helpBtn);

        const bgGalleryBtn = h("button", { type: "button", class: "jb-bg-btn", title: "Use Selected Gallery Image" }, ["\u{1F5BC}"]);
        const bgFileBtn = h("button", { type: "button", class: "jb-bg-btn", title: "Load Background From File" }, ["\u{1F4C2}"]);
        const bgClearBtn = h("button", { type: "button", class: "jb-bg-btn", title: "Reset Background" }, ["\u{1F5D1}"]);

        const bgButtons = h("div", { style: { position: "absolute", left: "8px", top: "8px", zIndex: "64", display: "flex", gap: "4px" } }, [bgGalleryBtn, bgFileBtn, bgClearBtn]);
        canvasSection.appendChild(bgButtons);

        const bgFileInput = h("input", { type: "file", accept: "image/*" });
        bgFileInput.style.display = "none";
        canvasSection.appendChild(bgFileInput);

        function setStageBackground(src) {
            stage.style.backgroundImage = `url("${src}")`;
            stage.style.backgroundSize = "100% 100%";
            stage.style.backgroundRepeat = "no-repeat";
        }

        let bgFileURL = null;

        bgGalleryBtn.addEventListener("click", () => {
            const thumbs = Array.from(gradioApp().querySelectorAll(`#${currentTab}_gallery_container .thumbnail-item.thumbnail-small`));
            const src = (thumbs.find((elem) => elem.classList.contains("selected")) ?? thumbs[0])?.querySelector("img")?.src;
            if (src) setStageBackground(src);
        });
        bgFileBtn.addEventListener("click", () => {
            bgFileInput.value = "";
            bgFileInput.click();
        });
        bgClearBtn.addEventListener("click", () => {
            if (bgFileURL) URL.revokeObjectURL(bgFileURL);
            bgFileURL = null;
            for (const prop of ["background-image", "background-size", "background-repeat"]) {
                stage.style.removeProperty(prop);
            }
        });

        bgFileInput.addEventListener("change", () => {
            const file = bgFileInput.files?.[0];
            if (!file) return;
            if (bgFileURL) URL.revokeObjectURL(bgFileURL);
            bgFileURL = URL.createObjectURL(file);
            setStageBackground(bgFileURL);
        });

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
        const expandItemsBtn = h("button", { type: "button", class: "jb-expand-items", title: "Expand items", "aria-expanded": "false" }, ["\u{1F53C}"]);
        const listEl = h("div", { class: "jb-list" });
        const listSection = h("div", { class: "jb-section jb-list-section" }, [h("div", { class: "jb-list-toolbar" }, [addBtn, clearBtn, expandItemsBtn]), listEl]);

        const sendPromptBtn = h("button", { type: "button", class: "jb-send-prompt" }, ["Send to Prompt"]);
        const loadPromptBtn = h("button", { type: "button", class: "jb-load-prompt" }, ["Load from Prompt"]);
        const submitSection = h("div", { class: "jb-section jb-submit" }, [sendPromptBtn, loadPromptBtn]);

        // ----- save / load -----

        const saveJsonBtn = h("button", { type: "button", class: "jb-save-json" }, ["Save JSON"]);
        const loadJsonBtn = h("button", { type: "button", class: "jb-load-json" }, ["Load JSON"]);
        const saveLoadSection = h("div", { class: "jb-section jb-save" }, [h("div", { class: "jb-section-title" }, ["Save/Load"]), h("div", { class: "jb-row" }, [saveJsonBtn, loadJsonBtn])]);

        const fileInput = h("input", { type: "file", accept: ".json,application/json", class: "jb-file" });
        fileInput.style.display = "none";

        const topRow = h("div", { class: "jb-top" }, [resolutionSection, saveLoadSection]);
        const side = h("div", { class: "jb-side" }, [topRow, summarySection, listSection, submitSection]);
        const root = h("div", { class: "jb-root" }, [canvasSection, side, fileInput]);

        let itemsExpanded = false;
        let listSpacer;

        expandItemsBtn.addEventListener("click", () => {
            itemsExpanded = !itemsExpanded;
            if (itemsExpanded) {
                const sideRect = side.getBoundingClientRect();
                const listRect = listSection.getBoundingClientRect();
                listSpacer = h("div", { class: "jb-section jb-list-section" });
                Object.assign(listSpacer.style, { boxSizing: "border-box", flex: `0 0 ${listRect.height}px`, minHeight: "0", visibility: "hidden" });
                side.insertBefore(listSpacer, listSection);
                side.style.position = "relative";
                listSection.style.position = "absolute";
                listSection.style.inset = `0 0 ${sideRect.bottom - listRect.bottom}px`;
                listSection.style.zIndex = "5";
                submitSection.style.position = "relative";
                submitSection.style.zIndex = "6";
                expandItemsBtn.textContent = "\u{1F53D}";
                expandItemsBtn.title = "Collapse Items";
            } else {
                listSpacer.remove();
                listSpacer = null;
                listSection.style.position = "";
                listSection.style.inset = "";
                listSection.style.zIndex = "";
                submitSection.style.position = "";
                submitSection.style.zIndex = "";
                side.style.position = "";
                expandItemsBtn.textContent = "\u{1F53C}";
                expandItemsBtn.title = "Expand Items";
            }
            expandItemsBtn.setAttribute("aria-expanded", String(itemsExpanded));
        });

        fitCanvasBtn.addEventListener("click", fitStage);

        addGrowable(sumCaption);

        function loadDraft(draft) {
            state.width = draft.width;
            state.height = draft.height;
            committedWidth = state.width;
            committedHeight = state.height;
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
            spawnCount = state.items.length;

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
            root.querySelectorAll(".jb-invalid").forEach((el) => {
                el.classList.remove("jb-invalid");
            });
        }

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
            stopPromptLink();
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
                } catch (e) {
                    loadJsonBtn.classList.add("jb-invalid");
                    console.error("Failed to load JSON:", e.message);
                }
            });
            reader.addEventListener("error", () => {
                loadJsonBtn.classList.add("jb-invalid");
                console.error("Failed to load JSON:", reader.error);
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

        function setItemField(item, field, value, source) {
            item[field] = value;
            const ref = listRefs.get(item.id);
            const listField = field === "caption" ? ref?.captionInput : ref?.textInput;
            const canvasField = canvasRefs.get(item.id)?.[field === "caption" ? "caption" : "text"];
            for (const target of [listField, canvasField]) {
                if (!target || target === source) continue;
                target.value = value;
                autoGrow(target);
            }
        }

        function makeEditable(item, field, color) {
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
                setItemField(item, field, ta.value, ta);
            });
            const key = itemFieldKey(item, field);
            addPromptLink(
                ta,
                key,
                color,
                () => item[field] || "",
                (value) => setItemField(item, field, value),
            );
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

        function createBox(item, color) {
            const el = h("div", { class: "jb-box", dataset: { id: String(item.id) } });
            el.addEventListener("dblclick", () => selectItem(item.id));
            el.style.borderColor = color;
            el.style.background = `color-mix(in srgb, ${color} 20%, transparent)`;
            const tag = h("div", { class: "jb-box-tag" }, [item.label]);
            tag.style.background = color;
            tag.style.color = "var(--body-text-color)";
            el.appendChild(tag);

            const caption = makeEditable(item, "caption", color);
            el.appendChild(caption);

            let text = null;
            if (item.label === "text") {
                text = makeEditable(item, "text", color);
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
            state.items.forEach((item, index) => {
                stage.appendChild(createBox(item, itemColor(index)));
            });
            if (canvasRefs.has(selectedItemId)) selectItem(selectedItemId);
            else selectedItemId = null;
        }

        // ----- list -----

        function createRow(item, color) {
            const row = h("div", { class: "jb-item", dataset: { id: String(item.id) } });
            row.style.borderColor = color;
            row.style.background = `color-mix(in srgb, ${color} 16%, var(--background-fill-secondary))`;
            row.addEventListener("dblclick", (event) => {
                if (event.target.closest(".jb-remove, .jb-radio")) return;
                selectItem(item.id);
            });

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
                setItemField(item, "caption", caption.value, caption);
                autoGrow(caption);
            });
            addPromptLink(
                caption,
                itemFieldKey(item, "caption"),
                color,
                () => item.caption || "",
                (value) => setItemField(item, "caption", value),
            );
            row.appendChild(caption);
            addGrowable(caption);

            let textInput = null;
            if (item.label === "text") {
                textInput = h("input", { type: "text", class: "jb-item-text", placeholder: "text", value: item.text });
                textInput.addEventListener("input", () => {
                    setItemField(item, "text", textInput.value, textInput);
                });
                addPromptLink(
                    textInput,
                    itemFieldKey(item, "text"),
                    color,
                    () => item.text || "",
                    (value) => setItemField(item, "text", value),
                );
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
            state.items.forEach((item, index) => {
                listEl.appendChild(createRow(item, itemColor(index)));
            });
        }

        function renderAll() {
            renderList();
            renderCanvas();
            if (activePromptLink && !root.querySelector(`[data-prompt-link-key="${activePromptLink.key}"]`)) stopPromptLink();
        }

        function addItem() {
            state.items.push(newItem());
            renderAll();
        }

        function removeItem(id) {
            spawnCount--;
            const index = state.items.findIndex((it) => it.id === id);
            if (index >= 0) state.items.splice(index, 1);
            if (state.items.length === 0) state.items.push(newItem());
            renderAll();
        }

        function clearItems() {
            spawnCount = 0;
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

        let committedWidth = state.width;
        let committedHeight = state.height;

        function rescaleItemsToCanvas() {
            const rw = state.width / committedWidth;
            const rh = state.height / committedHeight;
            committedWidth = state.width;
            committedHeight = state.height;
            for (const item of state.items) {
                const [x1, y1, x2, y2] = item.bbox;
                const nx1 = clamp(Math.round(x1 * rw), 0, state.width - 16);
                const ny1 = clamp(Math.round(y1 * rh), 0, state.height - 16);
                const nx2 = clamp(Math.max(Math.round(x2 * rw), nx1 + 16), nx1 + 16, state.width);
                const ny2 = clamp(Math.max(Math.round(y2 * rh), ny1 + 16), ny1 + 16, state.height);
                item.bbox = [nx1, ny1, nx2, ny2];
                syncListNums(item);
            }
            renderCanvas();
        }

        widthInput.addEventListener("input", () => {
            const v = parseInt(widthInput.value, 10);
            if (v > 0) applyResolution(v, state.height);
        });
        widthInput.addEventListener("change", rescaleItemsToCanvas);

        heightInput.addEventListener("input", () => {
            const v = parseInt(heightInput.value, 10);
            if (v > 0) applyResolution(state.width, v);
        });
        heightInput.addEventListener("change", rescaleItemsToCanvas);

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
            rescaleItemsToCanvas();
        });

        sumCaption.addEventListener("input", () => {
            state.summary.caption = sumCaption.value;
            autoGrow(sumCaption);
        });
        addPromptLink(
            sumCaption,
            "summary-caption",
            "#808080",
            () => state.summary.caption,
            (value) => {
                state.summary.caption = value;
                sumCaption.value = value;
                autoGrow(sumCaption);
            },
        );
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

        async function submit() {
            if (activePromptLink) await flushSelectedStyles(activePromptLink);
            stopPromptLink();
            if (!validate()) return;
            let prompt = null;

            try {
                prompt = compilePrompt(JSON.stringify(buildJson()));
            } catch (e) {
                alert(e);
                return;
            }

            prompt = prompt.replaceAll("\\n", "\n").replaceAll("\n", " ");
            if (state.summary.extra.trim() !== "") {
                prompt += "\n" + state.summary.extra.replace(/\s+/g, " ").trim();
            }

            const promptEl = document.getElementById(`${currentTab}_prompt`).querySelector("textarea");
            const widthEl = document.getElementById(`${currentTab}_width`).querySelector("input");
            const heightEl = document.getElementById(`${currentTab}_height`).querySelector("input");

            promptEl.value = prompt;
            widthEl.value = state.width;
            heightEl.value = state.height;
            updateInput(promptEl);
            updateInput(widthEl);
            updateInput(heightEl);

            promptEl.scrollIntoView({ behavior: "smooth", block: "center" });
        }

        function loadPrompt() {
            stopPromptLink();
            loadPromptBtn.classList.remove("jb-invalid");
            const promptEl = document.getElementById(`${currentTab}_prompt`).querySelector("textarea");
            const widthEl = document.getElementById(`${currentTab}_width`).querySelector("input");
            const heightEl = document.getElementById(`${currentTab}_height`).querySelector("input");
            const text = promptEl.value.trim();
            if (text === "") return;
            try {
                loadDraft(decompilePrompt(text, parseFloat(widthEl.value), parseFloat(heightEl.value)));
            } catch (e) {
                loadPromptBtn.classList.add("jb-invalid");
                console.error("Failed to load Prompt:", e.message);
            }
        }

        sendPromptBtn.addEventListener("click", submit);
        loadPromptBtn.addEventListener("click", loadPrompt);

        // ----- init -----

        instance = root;
        renderAll();
        fitStage();
        requestAnimationFrame(fitStage);
        if (window.ResizeObserver) new ResizeObserver(fitStage).observe(canvasSection);
    });
})();
