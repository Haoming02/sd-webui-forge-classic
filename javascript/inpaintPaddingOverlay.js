(function () {
    const OVERLAY_ID = "inpaintPaddingPreview";
    const HANDLE_ID = "inpaintPaddingPreviewHandle";
    const LABEL_ID = "inpaintPaddingPreviewLabel";
    const POLL_MS = 350;
    const MASK_BOUNDS_CACHE_MS = 120;

    let dragState = null;
    let pendingFrame = 0;
    let pendingSyncBiasControls = false;
    let maskBoundsCache = {
        canvas: null,
        width: 0,
        height: 0,
        bounds: null,
        expiresAt: 0,
        dirty: true,
    };

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

    function invalidateMaskBounds() {
        maskBoundsCache.dirty = true;
        maskBoundsCache.expiresAt = 0;
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

            const handle = document.createElement("button");
            handle.id = HANDLE_ID;
            handle.type = "button";
            handle.textContent = "Move";
            handle.title = "Drag to bias the masked inpaint area";
            handle.style.position = "absolute";
            handle.style.top = "-14px";
            handle.style.left = "-2px";
            handle.style.pointerEvents = "auto";
            handle.style.cursor = "grab";
            handle.style.border = "1px solid rgba(27, 38, 49, 0.5)";
            handle.style.borderRadius = "999px";
            handle.style.padding = "2px 8px";
            handle.style.background = "rgba(255, 179, 0, 0.95)";
            handle.style.color = "#1f2933";
            handle.style.fontSize = "11px";
            handle.style.fontWeight = "700";
            handle.style.boxShadow = "0 2px 8px rgba(0,0,0,0.18)";
            overlay.appendChild(handle);

            const label = document.createElement("div");
            label.id = LABEL_ID;
            label.style.position = "absolute";
            label.style.right = "6px";
            label.style.bottom = "6px";
            label.style.padding = "2px 6px";
            label.style.borderRadius = "999px";
            label.style.background = "rgba(27, 38, 49, 0.68)";
            label.style.color = "#fff6d5";
            label.style.fontSize = "11px";
            label.style.fontWeight = "600";
            label.style.pointerEvents = "none";
            overlay.appendChild(label);

            app().appendChild(overlay);

            handle.addEventListener("mousedown", beginDrag);
            handle.addEventListener("dblclick", (event) => {
                event.preventDefault();
                setBiasValue("img2img_inpaint_full_res_bias_x", 0);
                setBiasValue("img2img_inpaint_full_res_bias_y", 0);
                scheduleUpdate();
            });
        }
        return overlay;
    }

    function getSliderRoot(id) {
        return app().querySelector(`#${id}`);
    }

    function getSliderInputs(id) {
        const root = getSliderRoot(id);
        if (!root) {
            return { root: null, number: null, range: null };
        }

        return {
            root,
            number: root.querySelector('input[type="number"]'),
            range: root.querySelector('input[type="range"]'),
        };
    }

    function getSliderValue(id, fallback = 0) {
        const { number, range } = getSliderInputs(id);
        const value = parseInt((number && number.value) || (range && range.value) || `${fallback}`, 10);
        return Number.isFinite(value) ? value : fallback;
    }

    function setBiasValue(id, value) {
        const { root, number, range } = getSliderInputs(id);
        if (!root) {
            return;
        }

        const minimum = parseInt(root.dataset.min || root.getAttribute("data-min") || (range && range.min) || "-1024", 10);
        const maximum = parseInt(root.dataset.max || root.getAttribute("data-max") || (range && range.max) || "1024", 10);
        const clamped = Math.max(Number.isFinite(minimum) ? minimum : -1024, Math.min(Number.isFinite(maximum) ? maximum : 1024, Math.round(value)));

        if (number) number.value = String(clamped);
        if (range) range.value = String(clamped);
        if (number) {
            number.dispatchEvent(new Event("input", { bubbles: true }));
            number.dispatchEvent(new Event("change", { bubbles: true }));
        }
        if (range) {
            range.dispatchEvent(new Event("input", { bubbles: true }));
            range.dispatchEvent(new Event("change", { bubbles: true }));
        }
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
        return getSliderValue("img2img_inpaint_full_res_padding", 0);
    }

    function getBiasValues() {
        return {
            x: getSliderValue("img2img_inpaint_full_res_bias_x", 0),
            y: getSliderValue("img2img_inpaint_full_res_bias_y", 0),
        };
    }

    function getProcessingSize() {
        const width = getSliderValue("img2img_width", 1024);
        const height = getSliderValue("img2img_height", 1024);
        if (!width || !height) {
            return null;
        }

        return { width, height };
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

        const now = Date.now();
        if (
            maskBoundsCache.canvas === canvas
            && maskBoundsCache.width === canvas.width
            && maskBoundsCache.height === canvas.height
            && !maskBoundsCache.dirty
            && maskBoundsCache.expiresAt > now
        ) {
            return maskBoundsCache.bounds;
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
            maskBoundsCache = {
                canvas,
                width: canvas.width,
                height: canvas.height,
                bounds: null,
                expiresAt: now + MASK_BOUNDS_CACHE_MS,
                dirty: false,
            };
            return null;
        }

        const bounds = { minX, minY, maxX, maxY, width: w, height: h };
        maskBoundsCache = {
            canvas,
            width: canvas.width,
            height: canvas.height,
            bounds,
            expiresAt: now + MASK_BOUNDS_CACHE_MS,
            dirty: false,
        };
        return bounds;
    }

    function expandCropRegion(cropRegion, processingWidth, processingHeight, imageWidth, imageHeight) {
        let [x1, y1, x2, y2] = cropRegion;
        const ratioCropRegion = (x2 - x1) / (y2 - y1);
        const ratioProcessing = processingWidth / processingHeight;

        if (ratioCropRegion > ratioProcessing) {
            const desiredHeight = (x2 - x1) / ratioProcessing;
            const desiredHeightDiff = Math.trunc(desiredHeight - (y2 - y1));
            y1 -= Math.trunc(desiredHeightDiff / 2);
            y2 += desiredHeightDiff - Math.trunc(desiredHeightDiff / 2);
            if (y2 >= imageHeight) {
                const diff = y2 - imageHeight;
                y2 -= diff;
                y1 -= diff;
            }
            if (y1 < 0) {
                y2 -= y1;
                y1 -= y1;
            }
            if (y2 >= imageHeight) {
                y2 = imageHeight;
            }
        } else {
            const desiredWidth = (y2 - y1) * ratioProcessing;
            const desiredWidthDiff = Math.trunc(desiredWidth - (x2 - x1));
            x1 -= Math.trunc(desiredWidthDiff / 2);
            x2 += desiredWidthDiff - Math.trunc(desiredWidthDiff / 2);
            if (x2 >= imageWidth) {
                const diff = x2 - imageWidth;
                x2 -= diff;
                x1 -= diff;
            }
            if (x1 < 0) {
                x2 -= x1;
                x1 -= x1;
            }
            if (x2 >= imageWidth) {
                x2 = imageWidth;
            }
        }

        return [x1, y1, x2, y2];
    }

    function biasCropRegion(cropRegion, maskBox, biasX, biasY, imageWidth, imageHeight) {
        const [x1, y1, x2, y2] = cropRegion;
        const [mx1, my1, mx2, my2] = maskBox;
        let minDx = mx2 - x2;
        let maxDx = mx1 - x1;
        let minDy = my2 - y2;
        let maxDy = my1 - y1;

        minDx = Math.max(minDx, -x1);
        maxDx = Math.min(maxDx, imageWidth - x2);
        minDy = Math.max(minDy, -y1);
        maxDy = Math.min(maxDy, imageHeight - y2);

        const actualDx = Math.min(Math.max(Math.round(biasX || 0), minDx), maxDx);
        const actualDy = Math.min(Math.max(Math.round(biasY || 0), minDy), maxDy);

        return {
            crop: [x1 + actualDx, y1 + actualDy, x2 + actualDx, y2 + actualDy],
            actualBiasX: actualDx,
            actualBiasY: actualDy,
        };
    }

    function computeOverlayData() {
        const tab = app().querySelector("#tab_img2img");
        if (!tab || tab.style.display !== "block") {
            return null;
        }

        const inpaintRoot = app().querySelector("#img2maskimg");
        if (!isVisible(inpaintRoot)) {
            return null;
        }

        const img = getInpaintImage();
        const imageRect = getDisplayedImageRect(img);
        if (!imageRect) {
            return null;
        }

        const canvas = getMaskCanvas();
        const bounds = getMaskBounds(canvas);
        if (!bounds) {
            return null;
        }

        const processingSize = getProcessingSize();
        if (!processingSize) {
            return null;
        }

        const padding = getPaddingValue();
        const requestedBias = getBiasValues();

        const tightMaskBox = [
            bounds.minX,
            bounds.minY,
            bounds.maxX + 1,
            bounds.maxY + 1,
        ];

        const paddedCrop = [
            Math.max(tightMaskBox[0] - padding, 0),
            Math.max(tightMaskBox[1] - padding, 0),
            Math.min(tightMaskBox[2] + padding, bounds.width),
            Math.min(tightMaskBox[3] + padding, bounds.height),
        ];

        const expandedCrop = expandCropRegion(
            paddedCrop,
            processingSize.width,
            processingSize.height,
            bounds.width,
            bounds.height,
        );

        const shifted = biasCropRegion(
            expandedCrop,
            tightMaskBox,
            requestedBias.x,
            requestedBias.y,
            bounds.width,
            bounds.height,
        );

        return {
            imageRect,
            imageWidth: bounds.width,
            imageHeight: bounds.height,
            crop: shifted.crop,
            expandedCrop,
            tightMaskBox,
            actualBiasX: shifted.actualBiasX,
            actualBiasY: shifted.actualBiasY,
        };
    }

    function updateOverlay(syncBiasControls = false) {
        const data = computeOverlayData();
        if (!data) {
            hideOverlay();
            return;
        }

        const [x1, y1, x2, y2] = data.crop;
        const overlay = ensureOverlay();
        const label = overlay.querySelector(`#${LABEL_ID}`);

        const pxLeft = data.imageRect.left + (x1 / data.imageWidth) * data.imageRect.width;
        const pxTop = data.imageRect.top + (y1 / data.imageHeight) * data.imageRect.height;
        const pxWidth = ((x2 - x1) / data.imageWidth) * data.imageRect.width;
        const pxHeight = ((y2 - y1) / data.imageHeight) * data.imageRect.height;

        overlay.style.left = `${pxLeft}px`;
        overlay.style.top = `${pxTop}px`;
        overlay.style.width = `${Math.max(pxWidth, 2)}px`;
        overlay.style.height = `${Math.max(pxHeight, 2)}px`;
        overlay.style.display = "block";

        if (label) {
            label.textContent = `Bias X ${data.actualBiasX >= 0 ? "+" : ""}${data.actualBiasX}, Y ${data.actualBiasY >= 0 ? "+" : ""}${data.actualBiasY}`;
        }

        if (syncBiasControls) {
            if (data.actualBiasX !== getSliderValue("img2img_inpaint_full_res_bias_x", 0)) {
                setBiasValue("img2img_inpaint_full_res_bias_x", data.actualBiasX);
            }
            if (data.actualBiasY !== getSliderValue("img2img_inpaint_full_res_bias_y", 0)) {
                setBiasValue("img2img_inpaint_full_res_bias_y", data.actualBiasY);
            }
        }
    }

    function scheduleUpdate(syncBiasControls = false) {
        pendingSyncBiasControls = pendingSyncBiasControls || syncBiasControls;
        if (pendingFrame) {
            return;
        }

        pendingFrame = window.requestAnimationFrame(() => {
            pendingFrame = 0;
            const nextSyncBiasControls = pendingSyncBiasControls;
            pendingSyncBiasControls = false;
            updateOverlay(nextSyncBiasControls);
        });
    }

    function beginDrag(event) {
        const data = computeOverlayData();
        if (!data) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        dragState = {
            startClientX: event.clientX,
            startClientY: event.clientY,
            startBiasX: data.actualBiasX,
            startBiasY: data.actualBiasY,
            imageRect: data.imageRect,
            imageWidth: data.imageWidth,
            imageHeight: data.imageHeight,
            crop: data.expandedCrop,
            maskBox: data.tightMaskBox,
        };

        const handle = app().querySelector(`#${HANDLE_ID}`);
        if (handle) {
            handle.style.cursor = "grabbing";
        }
    }

    function dragMove(event) {
        if (!dragState) {
            return;
        }

        event.preventDefault();

        const deltaX = ((event.clientX - dragState.startClientX) / dragState.imageRect.width) * dragState.imageWidth;
        const deltaY = ((event.clientY - dragState.startClientY) / dragState.imageRect.height) * dragState.imageHeight;

        const desiredBiasX = dragState.startBiasX + deltaX;
        const desiredBiasY = dragState.startBiasY + deltaY;
        const shifted = biasCropRegion(
            dragState.crop,
            dragState.maskBox,
            desiredBiasX,
            desiredBiasY,
            dragState.imageWidth,
            dragState.imageHeight,
        );

        setBiasValue("img2img_inpaint_full_res_bias_x", shifted.actualBiasX);
        setBiasValue("img2img_inpaint_full_res_bias_y", shifted.actualBiasY);
        scheduleUpdate();
    }

    function endDrag() {
        if (!dragState) {
            return;
        }

        dragState = null;
        const handle = app().querySelector(`#${HANDLE_ID}`);
        if (handle) {
            handle.style.cursor = "grab";
        }
    }

    onAfterUiUpdate(() => scheduleUpdate(true));
    window.addEventListener("resize", () => scheduleUpdate());
    window.addEventListener("scroll", () => scheduleUpdate(), { passive: true });
    document.addEventListener("mousemove", dragMove);
    document.addEventListener("mouseup", endDrag);
    document.addEventListener(
        "pointerdown",
        function (evt) {
            if (evt.target?.closest?.("#img2maskimg")) {
                invalidateMaskBounds();
            }
        },
        true,
    );
    document.addEventListener(
        "pointerup",
        function (evt) {
            if (evt.target?.closest?.("#img2maskimg")) {
                invalidateMaskBounds();
                scheduleUpdate();
            }
        },
        true,
    );

    document.addEventListener("input", function (evt) {
        const target = evt.target;
        if (!target) {
            return;
        }

        if (
            target.closest("#img2img_inpaint_full_res")
            || target.closest("#img2img_inpaint_full_res_padding")
            || target.closest("#img2img_inpaint_full_res_bias_x")
            || target.closest("#img2img_inpaint_full_res_bias_y")
            || target.closest("#img2img_width")
            || target.closest("#img2img_height")
            || target.closest("#img2maskimg")
        ) {
            if (target.closest("#img2maskimg")) {
                invalidateMaskBounds();
            }
            scheduleUpdate();
        }
    });

    setInterval(() => updateOverlay(true), POLL_MS);
})();
