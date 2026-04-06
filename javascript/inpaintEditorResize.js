const INPAINT_EDITOR_DEFAULT_HEIGHT = 512;
const INPAINT_EDITOR_MIN_HEIGHT = 320;
const INPAINT_EDITOR_MAX_HEIGHT = 1400;
const INPAINT_EDITOR_TARGETS = [
    "img2maskimg",
    "inpaint_sketch",
    "img_inpaint_mask",
];

function inpaintEditorStorageKey(rootId) {
    return `forge_inpaint_editor_height:${rootId}`;
}

function parseInpaintEditorHeight(value) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return null;
    return Math.max(INPAINT_EDITOR_MIN_HEIGHT, Math.min(INPAINT_EDITOR_MAX_HEIGHT, parsed));
}

function getSavedInpaintEditorHeight(rootId) {
    return parseInpaintEditorHeight(localGet(inpaintEditorStorageKey(rootId), ""));
}

function saveInpaintEditorHeight(rootId, height) {
    const normalized = parseInpaintEditorHeight(height);
    if (normalized == null) return;
    localSet(inpaintEditorStorageKey(rootId), String(normalized));
}

function applyInpaintEditorHeight(container, height) {
    const normalized = parseInpaintEditorHeight(height);
    if (!container || normalized == null) return;
    container.style.height = `${normalized}px`;
    if (container.parentElement) {
        container.parentElement.style.minHeight = `${normalized}px`;
    }
}

function ensureInpaintResizeHandle(rootId, container) {
    let handle = container.querySelector(".forge-inpaint-resize-handle");
    if (handle) return handle;

    handle = document.createElement("div");
    handle.className = "forge-inpaint-resize-handle";
    handle.title = "Drag to resize this inpaint editor. Double-click to reset.";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-label", "Resize inpaint editor");
    handle.innerHTML = '<span class="forge-inpaint-resize-grip" aria-hidden="true"></span>';

    handle.addEventListener("dblclick", (event) => {
        event.preventDefault();
        applyInpaintEditorHeight(container, INPAINT_EDITOR_DEFAULT_HEIGHT);
        saveInpaintEditorHeight(rootId, INPAINT_EDITOR_DEFAULT_HEIGHT);
    });

    handle.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        const startY = event.clientY;
        const startHeight = container.getBoundingClientRect().height || INPAINT_EDITOR_DEFAULT_HEIGHT;
        container.classList.add("is-resizing-inpaint");
        document.body.classList.add("forge-inpaint-resizing");

        const onMove = (moveEvent) => {
            const delta = moveEvent.clientY - startY;
            applyInpaintEditorHeight(container, startHeight + delta);
        };

        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            container.classList.remove("is-resizing-inpaint");
            document.body.classList.remove("forge-inpaint-resizing");
            saveInpaintEditorHeight(rootId, container.getBoundingClientRect().height);
        };

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    });

    container.appendChild(handle);
    return handle;
}

function setupInpaintEditorResize(rootId) {
    const root = gradioApp().getElementById(rootId);
    if (!root) return;
    const container = root.querySelector(".forge-container");
    if (!container) return;

    root.classList.add("forge-inpaint-root-resizable");
    container.classList.add("is-resizable-inpaint");

    const savedHeight = getSavedInpaintEditorHeight(rootId);
    if (savedHeight != null) {
        applyInpaintEditorHeight(container, savedHeight);
    } else {
        applyInpaintEditorHeight(container, container.getBoundingClientRect().height || INPAINT_EDITOR_DEFAULT_HEIGHT);
    }

    ensureInpaintResizeHandle(rootId, container);
}

onAfterUiUpdate(function () {
    INPAINT_EDITOR_TARGETS.forEach(setupInpaintEditorResize);
});
