(function () {
    const OVERLAY_ID = "inpaintPaddingPreview";
    const POLL_MS = 350;

    function app() {
        return gradioApp();
    }

    function isVisible(el) {
        return !!(el && el.offsetParent !== null);
    }

    function hideOverlay() {
        const overlay = app().querySelector(`#${OVERLAY_ID}`);
        if (overlay) {
            overlay.style.display = "none";
        }
    }

    function ensureOverlay() {
        let overlay = app().querySelector(`#${OVERLAY_ID}`);
        if (!overlay) {
            overlay = document.createElement("div");
            overlay.id = OVERLAY_ID;
            overlay.style.position = "absolute";
            overlay.style.pointerEvents = "none";
            overlay.style.zIndex = "1001";
            overlay.style.border = "2px dashed #ffb300";
            overlay.style.boxShadow = "0 0 0 1px rgba(0,0,0,0.6) inset";
            overlay.style.background = "rgba(255, 179, 0, 0.08)";
            overlay.style.display = "none";
            app().appendChild(overlay);
        }
        return overlay;
    }

    function getMaskCanvas() {
        const candidates = app().querySelectorAll(
            "#img2maskimg canvas.forge-drawing-canvas, #img2maskimg canvas[id^='drawingCanvas_'], #img2maskimg canvas",
        );
        if (!candidates.length) {
            return null;
        }

        for (const canvas of candidates) {
            if (canvas && canvas.width > 1 && canvas.height > 1) {
                return canvas;
            }
        }

        return candidates[candidates.length - 1];
    }

    function getInpaintImage() {
        return app().querySelector("#img2maskimg div.forge-image-container img");
    }

    function getPaddingValue() {
        const root = app().querySelector("#img2img_inpaint_full_res_padding");
        if (!root) {
            return 0;
        }

        const number = root.querySelector('input[type="number"]');
        const range = root.querySelector('input[type="range"]');
        const value = parseInt((number && number.value) || (range && range.value) || "0", 10);

        return Number.isFinite(value) ? value : 0;
    }

    function getDisplayedImageRect(img) {
        if (!img || !img.complete) {
            return null;
        }

        const naturalW = img.naturalWidth || img.width;
        const naturalH = img.naturalHeight || img.height;
        if (!naturalW || !naturalH || !img.clientWidth || !img.clientHeight) {
            return null;
        }

        const boxW = img.clientWidth;
        const boxH = img.clientHeight;
        const imageAspect = naturalW / naturalH;
        const boxAspect = boxW / boxH;

        let drawW;
        let drawH;

        if (imageAspect > boxAspect) {
            drawW = boxW;
            drawH = boxW / imageAspect;
        } else {
            drawH = boxH;
            drawW = boxH * imageAspect;
        }

        const viewport = img.getBoundingClientRect();

        return {
            left: viewport.left + window.scrollX + (boxW - drawW) / 2,
            top: viewport.top + window.scrollY + (boxH - drawH) / 2,
            width: drawW,
            height: drawH,
        };
    }

    function getMaskBounds(canvas) {
        if (!canvas || !canvas.width || !canvas.height) {
            return null;
        }

        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) {
            return null;
        }

        const w = canvas.width;
        const h = canvas.height;
        let pixels = null;
        try {
            pixels = ctx.getImageData(0, 0, w, h).data;
        } catch (err) {
            return null;
        }

        let minX = w;
        let minY = h;
        let maxX = -1;
        let maxY = -1;

        for (let y = 0; y < h; y++) {
            const rowStart = y * w * 4;
            for (let x = 0; x < w; x++) {
                const a = pixels[rowStart + x * 4 + 3];
                if (a > 10) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX < 0 || maxY < 0) {
            return null;
        }

        return { minX, minY, maxX, maxY, width: w, height: h };
    }

    function updateOverlay() {
        const tab = app().querySelector("#tab_img2img");
        if (!tab || tab.style.display !== "block") {
            hideOverlay();
            return;
        }

        const inpaintRoot = app().querySelector("#img2maskimg");
        if (!isVisible(inpaintRoot)) {
            hideOverlay();
            return;
        }

        const img = getInpaintImage();
        const rect = getDisplayedImageRect(img);
        if (!rect) {
            hideOverlay();
            return;
        }

        const canvas = getMaskCanvas();
        const bounds = getMaskBounds(canvas);
        if (!bounds) {
            hideOverlay();
            return;
        }

        const padding = getPaddingValue();
        const minX = Math.max(0, bounds.minX - padding);
        const minY = Math.max(0, bounds.minY - padding);
        const maxX = Math.min(bounds.width - 1, bounds.maxX + padding);
        const maxY = Math.min(bounds.height - 1, bounds.maxY + padding);

        const pxLeft = rect.left + (minX / bounds.width) * rect.width;
        const pxTop = rect.top + (minY / bounds.height) * rect.height;
        const pxWidth = ((maxX - minX + 1) / bounds.width) * rect.width;
        const pxHeight = ((maxY - minY + 1) / bounds.height) * rect.height;

        const overlay = ensureOverlay();
        overlay.style.left = `${pxLeft}px`;
        overlay.style.top = `${pxTop}px`;
        overlay.style.width = `${Math.max(pxWidth, 2)}px`;
        overlay.style.height = `${Math.max(pxHeight, 2)}px`;
        overlay.style.display = "block";
    }

    function scheduleUpdate() {
        window.requestAnimationFrame(updateOverlay);
    }

    onAfterUiUpdate(scheduleUpdate);
    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("scroll", scheduleUpdate, { passive: true });

    document.addEventListener("input", function (evt) {
        const target = evt.target;
        if (!target) {
            return;
        }

        if (target.closest("#img2img_inpaint_full_res") || target.closest("#img2img_inpaint_full_res_padding") || target.closest("#img2maskimg")) {
            scheduleUpdate();
        }
    });

    setInterval(updateOverlay, POLL_MS);
})();
