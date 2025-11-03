// ------------------------------------------------------------
// Utility Classes
// ------------------------------------------------------------
class GradioTextAreaBind {
    constructor(elementId, className) {
        this.target = document.querySelector(`#${elementId}.${className} textarea`);
        this.syncLock = false;
        this.previousValue = '';

        // If the target isn't found yet, don't bail out — try to attach later.
        // This happens when the Gradio DOM for this canvas/tab isn't rendered yet.
        // We'll poll briefly for the textarea and attach a MutationObserver once found.
        const attachObserverToTarget = (targetEl) => {
            this.target = targetEl;
            if (!this.target) return;
            this.observer = new MutationObserver(() => {
                if (this.target.value !== this.previousValue) {
                    this.previousValue = this.target.value;
                    if (!this.syncLock && this._callback) {
                        this.syncLock = true;
                        this._callback(this.target.value);
                        this.syncLock = false;
                    }
                }
            });

            // Observe changes
            this.observer.observe(this.target, {
                characterData: true,
                subtree: true,
                childList: true,
                attributes: true
            });
        };

        if (this.target) {
            attachObserverToTarget(this.target);
        } else {
            // Try a few times to find the textarea in case Gradio mounts it later
            console.warn(`GradioTextAreaBind: Target textarea not found for #${elementId}.${className} - will retry`);
            const selector = `#${elementId}.${className} textarea`;
            let attempts = 0;
            const maxAttempts = 25; // ~5 seconds if interval is 200ms
            const interval = 200;
            const tryFind = () => {
                const el = document.querySelector(selector);
                if (el) {
                    attachObserverToTarget(el);
                } else if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(tryFind, interval);
                } else {
                    console.warn(`GradioTextAreaBind: Failed to find textarea for ${selector} after ${attempts} attempts`);
                }
            };
            tryFind();
        }
    }

    setValue(newValue) {
        if (!this.target) return;
        if (this.syncLock) return;

        this.syncLock = true;
        this.target.value = newValue;
        this.previousValue = newValue;

        const inputEvent = new Event('input', { bubbles: true });
        Object.defineProperty(inputEvent, 'target', { value: this.target });
        this.target.dispatchEvent(inputEvent);

        this.syncLock = false;
    }

    listen(callback) {
        this._callback = callback;
    }
}

// Add this class before ForgeCanvas class definition
class UndoManager {
    constructor(maxStates = 20) {
        this.states = [];
        this.currentIndex = -1;
        this.maxStates = maxStates;
    }

    pushState(state) {
        // Remove future states if we're in the middle of history
        if (this.currentIndex < this.states.length - 1) {
            this.states = this.states.slice(0, this.currentIndex + 1);
        }

        // Remove oldest state if at capacity
        if (this.states.length >= this.maxStates) {
            this.states.shift();
            this.currentIndex--;
        }

        this.states.push(state);
        this.currentIndex++;

        // Clean up memory for very old states (keep last 5 uncompressed)
        this.cleanupMemory();
    }

    cleanupMemory() {
        // Keep detailed states for recent actions, compress older ones
        const keepDetailed = 5;
        for (let i = 0; i < this.states.length - keepDetailed; i++) {
            if (this.states[i] && (this.states[i].drawingDataURL || this.states[i].dataURL)) {
                // For older states, we can potentially remove background data since it's redundant
                if (this.states[i].backgroundImage && i > 0) {
                    // Only keep background in the most recent state to avoid duplication
                    delete this.states[i].backgroundImage;
                }
            }
        }

        // Also free ImageBitmap resources and blob URLs for older states
        try {
            for (let i = 0; i < this.states.length - keepDetailed; i++) {
                const s = this.states[i];
                if (s && s.drawingBitmap) {
                    try {
                        if (typeof s.drawingBitmap.close === 'function') s.drawingBitmap.close();
                    } catch (e) {
                        // ignore
                    }
                    s.drawingBitmap = null;
                }
                // Revoke any stored object URLs
                if (s && s.drawingObjectUrl) {
                    try { URL.revokeObjectURL(s.drawingObjectUrl); } catch (e) {}
                    s.drawingObjectUrl = null;
                }
            }
        } catch (e) {
            // ignore
        }
    }

    canUndo() {
        return this.currentIndex > 0;
    }

    canRedo() {
        return this.currentIndex < this.states.length - 1;
    }

    undo() {
        if (!this.canUndo()) return null;
        this.currentIndex--;
        return this.states[this.currentIndex];
    }

    redo() {
        if (!this.canRedo()) return null;
        this.currentIndex++;
        return this.states[this.currentIndex];
    }

    clear() {
        // Properly clean up state references
        this.states.forEach(state => {
            if (state) {
                const dataURL = state.drawingDataURL || state.dataURL;
                if (dataURL) {
                    // Allow garbage collection
                    if (state.drawingDataURL) state.drawingDataURL = null;
                    if (state.dataURL) state.dataURL = null;
                }
                if (state.backgroundImage) {
                    state.backgroundImage = null;
                }
                // Close ImageBitmap if present to free GPU memory
                if (state.drawingBitmap) {
                    try {
                        if (typeof state.drawingBitmap.close === 'function') state.drawingBitmap.close();
                    } catch (e) {
                        // ignore
                    }
                    state.drawingBitmap = null;
                }
                // Revoke any stored object URLs
                if (state.drawingObjectUrl) {
                    try { URL.revokeObjectURL(state.drawingObjectUrl); } catch (e) {}
                    state.drawingObjectUrl = null;
                }
            }
        });
        this.states = [];
        this.currentIndex = -1;
    }

    getCurrentState() {
        return this.states[this.currentIndex];
    }

    // Get memory usage estimate
    getMemoryUsage() {
        let totalBytes = 0;
        let drawingBytes = 0;
        let backgroundBytes = 0;

        this.states.forEach(state => {
            if (state) {
                const dataURL = state.drawingDataURL || state.dataURL;
                if (dataURL) {
                    // Rough estimate: base64 is ~33% larger than binary
                    const bytes = dataURL.length * 0.75;
                    drawingBytes += bytes;
                    totalBytes += bytes;
                }
                // Add background image size if present
                if (state.backgroundImage) {
                    const bgBytes = state.backgroundImage.length * 0.75;
                    backgroundBytes += bgBytes;
                    totalBytes += bgBytes;
                }
            }
        });

        return {
            total: totalBytes,
            drawing: drawingBytes,
            background: backgroundBytes,
            statesCount: this.states.length
        };
    }
}

// ------------------------------------------------------------
// Main Canvas Class: ForgeCanvas
// ------------------------------------------------------------

/**
 * ForgeCanvas - A powerful canvas component for Stable Diffusion WebUI Forge
 * Provides image loading, drawing tools, and mask editing capabilities
 */
class ForgeCanvas {
    /**
     * ForgeCanvas constructor
     * @param {string} uuid - Unique identifier for this canvas instance
     * @param {boolean} noUpload - Disable file upload functionality
     * @param {boolean} noScribbles - Disable drawing tools
     * @param {boolean} mask - Enable mask mode for inpainting
     * @param {number} initialHeight - Initial canvas height
     * @param {string} scribbleColor - Default brush color
     * @param {boolean} scribbleColorFixed - Lock brush color
     * @param {number} scribbleWidth - Default brush width
     * @param {boolean} scribbleWidthFixed - Lock brush width
     * @param {number} scribbleAlpha - Default brush opacity
     * @param {boolean} scribbleAlphaFixed - Lock brush opacity
     * @param {number} scribbleSoftness - Default brush softness
     * @param {boolean} scribbleSoftnessFixed - Lock brush softness
     */
    constructor(
        uuid,
        noUpload = false,
        noScribbles = false,
        mask = false,
        initialHeight = 512,
        scribbleColor = '#000000',
        scribbleColorFixed = false,
        scribbleWidth = 4,
        scribbleWidthFixed = false,
        scribbleAlpha = 100,
        scribbleAlphaFixed = false,
        scribbleSoftness = 0,
        scribbleSoftnessFixed = false
    ) {
        // ============================================================
        // CORE CONFIGURATION
        // ============================================================
        this.gradioConfig = typeof gradio_config !== 'undefined' ? gradio_config : null;
        this.uuid = uuid;
        this.noUpload = noUpload;
        this.noScribbles = noScribbles;
        this.mask = mask;
        this.initialHeight = initialHeight;

        // Load settings from global opts if available
        this.loadSettingsFromOpts();

        // Override with constructor parameters if provided
        this.scribbleColor = scribbleColor ?? this.scribbleColor;
        if (scribbleWidth !== 4 || !this.scribbleWidth) this.scribbleWidth = scribbleWidth;
        if (scribbleAlpha !== 100 || !this.scribbleAlpha) this.scribbleAlpha = scribbleAlpha;
        if (scribbleSoftness !== 0 || !this.scribbleSoftness) this.scribbleSoftness = scribbleSoftness;

        this.scribbleColorFixed = scribbleColorFixed;
        this.scribbleWidthFixed = scribbleWidthFixed;
        this.scribbleAlphaFixed = scribbleAlphaFixed;
        this.scribbleSoftnessFixed = scribbleSoftnessFixed;

        // ============================================================
        // TOOL & MODE PROPERTIES
        // ============================================================
        this.currentMode = 'normal';
        this.currentTool = 'brush';

        // ============================================================
        // CANVAS & DRAWING CONTEXT
        // ============================================================
        this.tempCanvas = document.createElement('canvas');
        this.drawingCtx = null;
        this.contrastPatternCanvas = null;
        this.contrastPattern = null;

        // ============================================================
        // ANIMATION LOOP PROPERTIES
        // ============================================================
        this.isDrawingLoopActive = false;
        this.drawPending = false;
        this.brushStrokes = []; // { x0, y0, x1, y1, type }

        // ============================================================
        // PERFORMANCE & UTILITY PROPERTIES
        // ============================================================
        this.uploadDebounceTimer = null;
        this.uploadDebounceDelay = 300; // ms
        this.tempDrawPoints = [];
        this.tempDrawBackground = null;
        this.lastMousePos = { x: 0, y: 0 };
        this.toolbarOffset = { x: 0, y: 0 };
        this.originalState = {};
        this.devicePixelRatio = window.devicePixelRatio || 1;
        this.lastIndicatorUpdateTime = 0;
        this.lastPointerMoveTime = 0;
        this.toolbarHideDistance = 100; // Distance (px) to hide toolbar when drawing
        this.toolbarHideTimeout = null;

        // ============================================================
        // TOAST NOTIFICATION PROPERTIES
        // ============================================================
        this.toastTimeoutId = null;
        this.toastVisibilityTimeoutId = null;
        this.lastToastTime = 0;
        this.toastDebounceDelay = 100; // ms
        this.pendingToastMessage = null;
        this.pendingToastDuration = 0;
        this.pendingToastTimeout = null;

        // ============================================================
        // ANIMATION CLEANUP PROPERTIES
        // ============================================================
        this.animationTimeouts = new Set();

        // ============================================================
        // HISTORY & BINDINGS
        // ============================================================
        this.history = [];
        this.historyIndex = -1;
        // Ensure maxUndoSteps is always set; support 'i' or <=0 for unlimited undo
        if (typeof this.maxUndoSteps === 'undefined') {
            this.maxUndoSteps = 20;
        }
        // Normalize string or numeric settings: allow 'i' (infinite) or numbers <= 0 to mean unlimited
        const normalizeMaxUndo = (v) => {
            if (typeof v === 'string') {
                const s = v.trim().toLowerCase();
                if (s === 'i' || s === 'infinite' || s === 'unlimited') return Infinity;
                const n = parseInt(v, 10);
                if (!isNaN(n)) return n > 0 ? n : Infinity;
                return 20;
            }
            if (typeof v === 'number') {
                if (!isFinite(v)) return Infinity;
                return v > 0 ? v : Infinity;
            }
            return 20;
        };

        this.maxUndoSteps = normalizeMaxUndo(this.maxUndoSteps);
        this.undoManager = new UndoManager(this.maxUndoSteps);
        this.lastSavedState = 0; // Timestamp of last save
        this.lastNonDrawingSave = 0; // Timestamp of last non-drawing save
        this.backgroundGradioBind = new GradioTextAreaBind(this.uuid, 'logical_image_background');
        this.foregroundGradioBind = new GradioTextAreaBind(this.uuid, 'logical_image_foreground');

    // Token for a server-stored background image (forge-canvas://id)
    this.imgToken = null;
    // Current object URL created from a fetched background blob (for revocation)
    this._backgroundObjectUrl = null;

        // Consecutive undo/redo tracking
        this.consecutiveUndos = 0;
        this.consecutiveRedos = 0;
        this.lastActionType = null;

        this.start(); // Initialize all logic
    }

    /**
     * Load settings from global opts object
     */
    loadSettingsFromOpts() {
        if (typeof opts !== 'undefined' && opts.data) {
            this.scribbleColor = opts.data.canvas_default_brush_color || '#000000';
            this.scribbleWidth = opts.data.canvas_default_brush_width || 4;
            this.scribbleAlpha = opts.data.canvas_default_brush_alpha || 100;
            this.scribbleSoftness = opts.data.canvas_default_brush_softness || 0;
            this.maxUndoSteps = opts.data.canvas_max_undo_steps || 20;
            // allow 'i' or <=0 to request unlimited undo
            if (this.maxUndoSteps === 'i' || this.maxUndoSteps === 'I' || String(this.maxUndoSteps).toLowerCase() === 'infinite' || String(this.maxUndoSteps).toLowerCase() === 'unlimited') {
                this.maxUndoSteps = Infinity;
            } else {
                const parsed = parseInt(this.maxUndoSteps, 10);
                if (!isNaN(parsed)) {
                    this.maxUndoSteps = parsed > 0 ? parsed : Infinity;
                }
            }
            this.autoSaveInterval = opts.data.canvas_auto_save_interval || 30;
            this.enableKeyboardShortcuts = opts.data.canvas_enable_keyboard_shortcuts !== false;
            this.zoomSensitivity = opts.data.canvas_zoom_sensitivity || 1.0;
            this.panSensitivity = opts.data.canvas_pan_sensitivity || 1.0;
        } else {
            // Default values if opts not available
            this.scribbleColor = '#000000';
            this.scribbleWidth = 4;
            this.scribbleAlpha = 100;
            this.scribbleSoftness = 0;
            this.maxUndoSteps = 20;
            this.autoSaveInterval = 30;
            this.enableKeyboardShortcuts = true;
            this.zoomSensitivity = 1.0;
            this.panSensitivity = 1.0;
        }
    }

    /**
     * High-level initialization function.
     */
    start() {
        // 1. Cache and store DOM references
        this.cacheDOMElements();

        // 2. Initialize UI states
        this.initUI();

        // 3. Bind event handlers
        this.bindToolbarEvents();
        this.bindCanvasEvents();
        this.bindDragDropEvents();
        this.bindGlobalEvents();

        // 4. Set up watchers
        this.observeContainerResize();

        // 5. Final touches
        this.updateUndoRedoButtons();
        this.backgroundGradioBind.listen(base64Data => this.uploadBase64(base64Data));
        this.foregroundGradioBind.listen(base64Data => this.uploadBase64DrawingCanvas(base64Data));

        // Prevent default scroll on the drawing canvas
        if (this.drawingCanvas) {
            this.drawingCanvas.addEventListener('wheel', e => e.preventDefault(), { passive: false });
            this.drawingCanvas.setAttribute('tabindex', '0');
        }

        // Kick off an animation loop for drawing changes
        this.startDrawingLoop();
    }

    // ============================================================
    // 1) DOM CACHING & INITIAL UI SETUP
    // ============================================================
    cacheDOMElements() {
        const ids = [
            'imageContainer', 'image', 'resizeLine', 'container', 'toolbar', 'uploadButton',
            'resetButton', 'centerButton', 'removeButton', 'undoButton', 'redoButton',
            'drawingCanvas', 'maxButton', 'minButton', 'scribbleIndicator', 'uploadHint',
            'uploadHintButton',
            'scribbleColor', 'scribbleColorBlock', 'scribbleWidth', 'widthLabel',
            'scribbleWidthBlock', 'scribbleAlpha', 'alphaLabel', 'scribbleAlphaBlock',
            'scribbleSoftness', 'softnessLabel', 'scribbleSoftnessBlock', 'eraserButton',
            'brushButton', 'brushDropdown', 'toolbarMessage'
        ];

        this.elems = {};
        ids.forEach(id => {
            const el = document.getElementById(`${id}_${this.uuid}`);
            this.elems[id] = el || null;
        });
    }

    initUI() {
        const {
            scribbleColor, scribbleWidth, scribbleAlpha, scribbleSoftness,
            scribbleIndicator, container, drawingCanvas, uploadButton, uploadHint
        } = this.elems;

        // Initialize scribble controls with loaded settings
        if (scribbleColor) scribbleColor.value = this.scribbleColor;
        if (scribbleWidth) scribbleWidth.value = this.scribbleWidth;
        if (scribbleAlpha) scribbleAlpha.value = this.scribbleAlpha;
        if (scribbleSoftness) scribbleSoftness.value = this.scribbleSoftness;

        // Indicator size
        if (scribbleIndicator) {
            const scribbleIndicatorSize = this.scribbleWidth * 20;
            scribbleIndicator.style.width = `${scribbleIndicatorSize}px`;
            scribbleIndicator.style.height = `${scribbleIndicatorSize}px`;
        }

        // Container height - use setting if available
        if (container) {
            const height = (typeof opts !== 'undefined' && opts.data && opts.data.canvas_default_height) 
                ? opts.data.canvas_default_height 
                : this.initialHeight;
            container.style.height = `${height}px`;
        }

        // Initialize drawing canvas with devicePixelRatio awareness
        if (drawingCanvas && this.elems.imageContainer) {
            const cssWidth = this.elems.imageContainer.clientWidth;
            const cssHeight = this.elems.imageContainer.clientHeight;
            const DPR = window.devicePixelRatio || 1;

            // Set internal pixel buffer to CSS size * DPR for crisp drawing
            drawingCanvas.width = Math.round(cssWidth * DPR);
            drawingCanvas.height = Math.round(cssHeight * DPR);
            drawingCanvas.style.width = `${cssWidth}px`;
            drawingCanvas.style.height = `${cssHeight}px`;

            this.drawingCanvas = drawingCanvas;
            // Cache context and store DPR scaling to avoid repeated context reacquisition
            this.drawingCtx = drawingCanvas.getContext('2d', { willReadFrequently: false });
            this.devicePixelRatio = DPR;
            try {
                this.drawingCtx.setTransform(DPR, 0, 0, DPR, 0, 0);
            } catch (e) {
                // some older browsers may not support setTransform with these args
                this.drawingCtx.scale(DPR, DPR);
            }
        }

        // Hide scribble-related elements if noScribbles is true
        if (this.noScribbles) {
            [
                'resetButton', 'undoButton', 'redoButton', 'scribbleColor', 'scribbleColorBlock',
                'scribbleWidthBlock', 'scribbleAlphaBlock', 'scribbleSoftnessBlock',
                'scribbleIndicator', 'drawingCanvas', 'brushButton', 'brushDropdown'
            ].forEach(id => {
                if (this.elems[id]) this.elems[id].style.display = 'none';
            });
        }

        // Hide upload button & hint if noUpload is true
        if (this.noUpload && uploadButton) {
            uploadButton.style.display = 'none';
            if (uploadHint) uploadHint.style.display = 'none';
        }

        // Mask mode
        if (this.mask) {
            this.configureMaskMode();
        }

        // Hide/fix scribble controls if flagged as fixed
        if (this.scribbleColorFixed && this.elems.scribbleColorBlock) {
            this.elems.scribbleColorBlock.style.display = 'none';
        }
        if (this.scribbleWidthFixed && this.elems.scribbleWidthBlock) {
            this.elems.scribbleWidthBlock.style.display = 'none';
        }
        if (this.scribbleAlphaFixed && this.elems.scribbleAlphaBlock) {
            this.elems.scribbleAlphaBlock.style.display = 'none';
        }
        if (this.scribbleSoftnessFixed && this.elems.scribbleSoftnessBlock) {
            this.elems.scribbleSoftnessBlock.style.display = 'none';
        }

        // Initialize brush dropdown active state
        const { brushDropdown, brushButton } = this.elems;
        if (brushDropdown) {
            const items = brushDropdown.querySelectorAll('.forge-dropdown-item');
            items.forEach(item => {
                item.classList.toggle('active', item.dataset.tool === this.currentTool);
            });
        }
        if (brushButton) {
            brushButton.classList.toggle('active', this.currentTool === 'brush' || this.currentTool === 'eraser');
            if (this.currentTool === 'brush') {
                brushButton.textContent = '🖌️';
                brushButton.title = 'Brush Tool (B)';
            } else if (this.currentTool === 'eraser') {
                brushButton.textContent = '🧽';
                brushButton.title = 'Eraser Tool (E)';
            } else {
                // No tool selected - show default brush icon
                brushButton.textContent = '🖌️';
                brushButton.title = 'Brush Tool (B) - Hover for more options';
            }
        }
    }

    configureMaskMode() {
        const { scribbleColorBlock, scribbleAlphaBlock, scribbleSoftnessBlock, drawingCanvas } = this.elems;

        // Hide color/alpha/softness controls
        [scribbleColorBlock, scribbleAlphaBlock, scribbleSoftnessBlock].forEach(el => {
            if (el) el.style.display = 'none';
        });

        // Create the contrast pattern
        if (drawingCanvas) {
            const patternCanvas = document.createElement('canvas');
            patternCanvas.width = 20;
            patternCanvas.height = 20;
            const patternContext = patternCanvas.getContext('2d');
            patternContext.fillStyle = '#ffffff';
            patternContext.fillRect(0, 0, 10, 10);
            patternContext.fillRect(10, 10, 10, 10);
            patternContext.fillStyle = '#000000';
            patternContext.fillRect(10, 0, 10, 10);
            patternContext.fillRect(0, 10, 10, 10);

            this.contrastPatternCanvas = patternCanvas;
            this.contrastPattern = this.drawingCtx.createPattern(patternCanvas, 'repeat');
            drawingCanvas.style.opacity = '0.5';
            this.currentMode = 'inpainting';
        }
    }

    initTooltips() {
        const tooltips = {
            uploadButton: 'Upload Image (or drag & drop)',
            resetButton: 'Reset Canvas (R)',
            undoButton: 'Undo (Ctrl+Z)',
            redoButton: 'Redo (Ctrl+Y)',
            scribbleWidth: 'Brush Size ([ and ]) - Zoom (Z)',
        };

        Object.entries(tooltips).forEach(([id, tooltip]) => {
            const elem = this.elems[id];
            if (elem) elem.title = tooltip;
        });
    }

    // ============================================================
    // 2) EVENT BINDING
    // ============================================================
    bindToolbarEvents() {
        const {
            uploadButton, resetButton, centerButton, removeButton,
            undoButton, redoButton, maxButton, minButton, uploadHintButton
        } = this.elems;

        // File upload input
        const imageInput = document.getElementById(`imageInput_${this.uuid}`);
        if (imageInput) {
            imageInput.addEventListener('change', e => this.handleFileUpload(e.target.files[0]));
        }

        if (uploadButton && imageInput && !this.noUpload) {
            uploadButton.addEventListener('click', () => imageInput.click());
        }
        if (uploadHintButton && imageInput && !this.noUpload) {
            uploadHintButton.addEventListener('click', e => {
                e.preventDefault();
                imageInput.click();
            });
        }
        if (resetButton) {
            resetButton.addEventListener('click', () => this.resetImage());
        }
        if (centerButton) {
            centerButton.addEventListener('click', () => {
                this.adjustInitialPositionAndScale();
                this.drawImage();
            });
        }
        if (removeButton) {
            removeButton.addEventListener('click', () => this.removeImage());
        }
        if (undoButton) {
            undoButton.addEventListener('click', () => {
                this.undo();
            });
        }
        if (redoButton) {
            redoButton.addEventListener('click', () => {
                this.redo();
            });
        }
        if (maxButton) {
            maxButton.addEventListener('click', () => this.maximize());
        }
        if (minButton) {
            minButton.addEventListener('click', () => this.minimize());
        }

        // Brush dropdown functionality
        const { brushButton, brushDropdown } = this.elems;
        if (brushButton && brushDropdown) {
            // Show dropdown on hover
            const container = brushButton.parentElement;
            if (container) {
                container.addEventListener('mouseenter', () => {
                    brushDropdown.classList.add('show');
                });
                container.addEventListener('mouseleave', () => {
                    brushDropdown.classList.remove('show');
                });
            }

            // Handle dropdown item clicks
            brushDropdown.addEventListener('click', (e) => {
                e.stopPropagation();
                const item = e.target.closest('.forge-dropdown-item');
                if (item) {
                    const tool = item.dataset.tool;
                    if (tool) {
                        this.setTool(tool);
                        brushDropdown.classList.remove('show');
                    }
                }
            });
        }

        // Close dropdowns when clicking outside
        document.addEventListener('click', () => {
            this.hideAllDropdowns();
        });

        // Draggable toolbar
        const toolbarHandle = document.getElementById(`toolbarHandle_${this.uuid}`);
        if (toolbarHandle && this.elems.toolbar) {
            toolbarHandle.addEventListener('mousedown', e => {
                this.toolbarDragging = true;
                const toolbarRect = this.elems.toolbar.getBoundingClientRect();
                this.toolbarOffset = {
                    x: e.clientX - toolbarRect.left,
                    y: e.clientY - toolbarRect.top
                };
                e.preventDefault();
            });

            document.addEventListener('mousemove', e => {
                if (!this.toolbarDragging || !this.elems.toolbar || !this.elems.imageContainer) return;
                const containerRect = this.elems.imageContainer.getBoundingClientRect();
                const toolbarRect = this.elems.toolbar.getBoundingClientRect();

                let newX = e.clientX - containerRect.left - this.toolbarOffset.x;
                let newY = e.clientY - containerRect.top - this.toolbarOffset.y;

                // Keep toolbar within container bounds
                newX = Math.max(0, Math.min(newX, containerRect.width - toolbarRect.width));
                newY = Math.max(0, Math.min(newY, containerRect.height - toolbarRect.height));

                this.elems.toolbar.style.left = `${newX}px`;
                this.elems.toolbar.style.top = `${newY}px`;
            });

            document.addEventListener('mouseup', () => {
                this.toolbarDragging = false;
            });
        }
    }

    bindCanvasEvents() {
        const {
            scribbleColor, scribbleIndicator, scribbleWidth, scribbleAlpha,
            scribbleSoftness, eraserButton, drawingCanvas, imageContainer, image
        } = this.elems;

        // Scribble color
        if (scribbleColor) {
            scribbleColor.addEventListener('input', () => {
                this.scribbleColor = scribbleColor.value;
                if (scribbleIndicator) {
                    scribbleIndicator.style.borderColor = this.scribbleColor;
                }
            });
        }

        // Scribble width
        if (scribbleWidth) {
            scribbleWidth.addEventListener('input', () => {
                this.scribbleWidth = parseInt(scribbleWidth.value, 10);
                const newSize = this.scribbleWidth * 20;
                if (scribbleIndicator) {
                    scribbleIndicator.style.width = `${newSize}px`;
                    scribbleIndicator.style.height = `${newSize}px`;
                }
            });
        }

        // Scribble alpha
        if (scribbleAlpha) {
            scribbleAlpha.addEventListener('input', () => {
                this.scribbleAlpha = parseInt(scribbleAlpha.value, 10);
            });
        }

        // Scribble softness
        if (scribbleSoftness) {
            scribbleSoftness.addEventListener('input', () => {
                this.scribbleSoftness = parseInt(scribbleSoftness.value, 10);
            });
        }

        // Eraser button
        if (eraserButton) {
            eraserButton.addEventListener('click', () => {
                this.currentTool = this.currentTool === 'eraser' ? 'brush' : 'eraser';
                if (this.mask && this.currentTool === 'brush' && this.drawingCanvas) {
                    this.drawingCtx.globalCompositeOperation = 'source-over';
                    this.drawingCtx.strokeStyle = this.contrastPattern;
                }
                eraserButton.classList.toggle('active');
            });
        }

        // We'll collect pointerdown and pointermove, but actual drawing
        // will happen via requestAnimationFrame in 'startDrawingLoop()'

        // Canvas pointer events
        if (drawingCanvas) {
            drawingCanvas.addEventListener('pointerdown', e => {
                if (!this.img || e.button !== 0 || this.noScribbles || this.isZooming || (this.currentTool !== 'brush' && this.currentTool !== 'eraser')) return;

                this.drawing = true;
                if (!this.isZooming) {
                    drawingCanvas.style.cursor = 'crosshair';
                }
                if (this.elems.scribbleIndicator) {
                    this.elems.scribbleIndicator.style.display = 'none';
                }
                this.tempDrawPoints = [];
                // Remove saveState from here - we'll save when drawing ends
                this.handlePointerMoveCanvas(e); // record first position
            });

            drawingCanvas.addEventListener('pointermove', e => {
                this.handlePointerMoveCanvas(e);
            });

            drawingCanvas.addEventListener('pointerup', () => {
                if (this.drawing) {
                    this.drawing = false;
                    this.lastErasePoint = null;
                    if (!this.isZooming) {
                        drawingCanvas.style.cursor = '';
                    }
                    if (this.eraseChanged) {
                        this.saveState(); // Save state when drawing ends
                        this.eraseChanged = false;
                    }
                    // Restore toolbar visibility when drawing ends
                    if (this.elems.toolbar) {
                        this.elems.toolbar.style.opacity = '1';
                        this.elems.toolbar.style.pointerEvents = 'auto';
                    }
                }
            });

            drawingCanvas.addEventListener('pointerleave', () => {
                this.drawing = false;
                this.lastErasePoint = null;
                if (this.isZooming) {
                    drawingCanvas.style.cursor = 'zoom-in';
                }
            });

            // Handle cursor for zoom mode on drawing canvas
            drawingCanvas.addEventListener('pointerover', () => {
                if (this.isZooming) {
                    drawingCanvas.style.cursor = 'zoom-in';
                } else if (this.img && !this.noScribbles && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
                    drawingCanvas.style.cursor = 'crosshair';
                }
            });
            drawingCanvas.addEventListener('pointerout', () => {
                if (this.isZooming) {
                    drawingCanvas.style.cursor = 'zoom-in';
                } else {
                    drawingCanvas.style.cursor = '';
                }
            });
        }

        // Image dragging inside container
        if (imageContainer) {
            imageContainer.addEventListener('pointerdown', e => this.onPointerDownImageContainer(e));
            imageContainer.addEventListener('pointermove', e => this.onPointerMoveImageContainer(e), { passive: true });
            imageContainer.addEventListener('pointerup', e => this.onPointerUpImageContainer(e), { passive: true });
            imageContainer.addEventListener('pointerleave', e => this.onPointerLeaveImageContainer(e), { passive: true });
            imageContainer.addEventListener('wheel', e => this.onWheelImageContainer(e), { passive: false });
            imageContainer.addEventListener('contextmenu', e => {
                e.preventDefault();
                this.draggedJustNow = false;
            });
            imageContainer.addEventListener('pointerover', () => {
                if (this.elems.toolbar) this.elems.toolbar.style.opacity = '1';
                if (!this.img && !this.noUpload && imageContainer) {
                    imageContainer.style.cursor = 'pointer';
                } else if (this.isZooming) {
                    imageContainer.style.cursor = 'zoom-in';
                    if (drawingCanvas) drawingCanvas.style.cursor = 'zoom-in';
                }
            });
            imageContainer.addEventListener('pointerout', () => {
                if (this.elems.toolbar) this.elems.toolbar.style.opacity = '0';
                if (image) image.style.cursor = '';
                if (drawingCanvas) {
                    if (this.isZooming) {
                        drawingCanvas.style.cursor = 'zoom-in';
                    } else {
                        drawingCanvas.style.cursor = '';
                    }
                }
                if (imageContainer) {
                    if (this.isZooming) {
                        imageContainer.style.cursor = 'zoom-in';
                    } else {
                        imageContainer.style.cursor = '';
                    }
                }
                if (scribbleIndicator) scribbleIndicator.style.display = 'none';
            });
        }

        // Resize line
        if (this.elems.resizeLine) {
            this.elems.resizeLine.addEventListener('pointerdown', e => {
                this.resizing = true;
                e.preventDefault();
                e.stopPropagation();
            });
        }
        document.addEventListener('pointermove', e => {
            if (this.resizing) {
                this.resizeContainer(e);
                e.preventDefault();
                e.stopPropagation();
            }
        }, { passive: false });
        document.addEventListener('pointerup', () => {
            this.resizing = false;
        }, { passive: true });
        document.addEventListener('pointerleave', () => {
            this.resizing = false;
        }, { passive: true });
    }

    bindDragDropEvents() {
        const { imageContainer, image, drawingCanvas } = this.elems;
        if (!imageContainer) return;

        ['dragenter', 'dragover'].forEach(eventType => {
            imageContainer.addEventListener(eventType, e => e.preventDefault(), false);
        });

        // Visual feedback on drag
        imageContainer.addEventListener('dragenter', () => {
            if (this.isZooming) {
                if (image) image.style.cursor = 'zoom-in';
                if (drawingCanvas) drawingCanvas.style.cursor = 'zoom-in';
            } else {
                if (image) image.style.cursor = 'copy';
                if (drawingCanvas) drawingCanvas.style.cursor = 'copy';
            }
        });
        imageContainer.addEventListener('dragleave', () => {
            if (this.isZooming) {
                if (image) image.style.cursor = 'zoom-in';
                if (drawingCanvas) drawingCanvas.style.cursor = 'zoom-in';
            } else {
                if (image) image.style.cursor = '';
                if (drawingCanvas) drawingCanvas.style.cursor = '';
            }
        });

        // File drop
        imageContainer.addEventListener('drop', e => {
            e.preventDefault();
            const { dataTransfer } = e;
            const { files } = dataTransfer;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
    }

    bindGlobalEvents() {
        // Keyboard shortcuts
        document.addEventListener('keydown', e => {
            if (!this.pointerInsideContainer || !this.enableKeyboardShortcuts) return;
            // Don't trigger hotkeys if user is typing in an input/textarea
            const target = e.target;
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

            if (e.key === 'b' || e.key === 'B') {
                this.setTool('brush');
                this.showToolToast('brush');
            }
            if (e.key === 'e' || e.key === 'E') {
                this.setTool('eraser');
                this.showToolToast('eraser');
            }
            if (e.key === '[') {
                this.adjustBrushSize(-1);
            }
            if (e.key === ']') {
                this.adjustBrushSize(1);
            }
            if (e.ctrlKey && e.key === 'z' && !e.shiftKey) {
                e.preventDefault();
                this.undo();
            }
            if ((e.ctrlKey && e.key === 'y') || (e.ctrlKey && e.shiftKey && e.key === 'Z')) {
                e.preventDefault();
                this.redo();
            }
        });

        // Zoom hotkey (Z) - toggle zoom mode
        document.addEventListener('keydown', e => {
            if (e.key === 'z' && !e.ctrlKey && !e.repeat) { // Avoid conflict with Ctrl+Z and repeats
                // Don't trigger hotkey if user is typing in an input/textarea
                const target = e.target;
                if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

                this.isZooming = !this.isZooming;
                this.isHandTool = false; // Disable hand tool when switching to zoom
                // Disable drawing tools when switching to zoom
                if (this.isZooming && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
                    this.currentTool = '';
                }
                this.updateCursor();
                this.updateScribbleIndicator();
                this.showZoomToast();
                e.preventDefault();
            }
        });

        // Hand tool hotkey (H) - toggle hand tool mode
        document.addEventListener('keydown', e => {
            if (e.key === 'h' && !e.ctrlKey && !e.repeat) { // Avoid conflicts and repeats
                // Don't trigger hotkey if user is typing in an input/textarea
                const target = e.target;
                if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

                this.isHandTool = !this.isHandTool;
                this.isZooming = false; // Disable zoom tool when switching to hand
                // Disable drawing tools when switching to hand
                if (this.isHandTool && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
                    this.currentTool = '';
                }
                this.updateCursor();
                this.updateScribbleIndicator();
                this.showHandToolToast();
                e.preventDefault();
            }
        });        // Spacebar panning (Photoshop-style)
        let spacebarPressed = false;
        let previousTool = '';
        let previousZooming = false;
        let previousHandTool = false;

        document.addEventListener('keydown', e => {
            if (e.code === 'Space' && !e.repeat && !spacebarPressed) {
                // Don't trigger if user is typing in an input/textarea
                const target = e.target;
                if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

                spacebarPressed = true;
                e.preventDefault();

                // Store current state
                previousTool = this.currentTool;
                previousZooming = this.isZooming;
                previousHandTool = this.isHandTool;

                // Switch to hand tool for panning
                this.isHandTool = true;
                this.isZooming = false;
                this.currentTool = ''; // Disable drawing tools
                this.updateCursor();
                this.updateScribbleIndicator();
            }
        });

        document.addEventListener('keyup', e => {
            if (e.code === 'Space' && spacebarPressed) {
                spacebarPressed = false;
                e.preventDefault();

                // Restore previous state
                this.currentTool = previousTool;
                this.isZooming = previousZooming;
                this.isHandTool = previousHandTool;
                this.updateCursor();
                this.updateScribbleIndicator();
            }
        });

        // Pasting images
        document.addEventListener('paste', e => {
            if (this.pointerInsideContainer) {
                e.preventDefault();
                e.stopPropagation();
                this.handlePaste(e);
            }
        });

        // Track pointer inside container
        if (this.elems.imageContainer) {
            this.elems.imageContainer.addEventListener('pointerenter', () => {
                this.pointerInsideContainer = true;
            });
            this.elems.imageContainer.addEventListener('pointerleave', () => {
                this.pointerInsideContainer = false;
            });
        }
    }

    observeContainerResize() {
        if (!this.elems.container) return;
        const resizeObserver = new ResizeObserver(() => {
            this.adjustInitialPositionAndScale();
            this.drawImage();
        });
        resizeObserver.observe(this.elems.container);
    }

    // ============================================================
    // 3) DRAWING OPERATIONS & ANIMATION LOOP
    // ============================================================
    
    /**
     * Starts the requestAnimationFrame-based drawing loop for performance
     */
    startDrawingLoop() {
        if (this.isDrawingLoopActive) return;
        this.isDrawingLoopActive = true;

        let lastTime = performance.now();

        const drawFrame = (currentTime) => {
            if (!this.drawPending || this.brushStrokes.length === 0) {
                // No new draws, skip
                this.drawPending = false;
            } else {
                // Perform the actual line drawing or erasing
                for (const stroke of this.brushStrokes) {
                    if (stroke.type === 'eraser') {
                        this.drawEraserLine(stroke.x0, stroke.y0, stroke.x1, stroke.y1);
                        this.eraseChanged = true;
                    } else if (stroke.type === 'brush') {
                        this.drawBrushLine(stroke.x0, stroke.y0, stroke.x1, stroke.y1);
                        this.eraseChanged = true;
                    }
                }
                this.brushStrokes = []; // Clear
            }

            lastTime = currentTime;
            requestAnimationFrame(drawFrame);
        };

        requestAnimationFrame(drawFrame);
    }

    /**
     * Handles pointer movement on canvas for drawing operations
     * @param {PointerEvent} e - The pointer event
     */
    handlePointerMoveCanvas(e) {
        if (!this.drawing || !this.img || this.noScribbles || this.isZooming || this.isHandTool || (this.currentTool !== 'brush' && this.currentTool !== 'eraser')) return;
        const rect = this.drawingCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.imgScale;
        const y = (e.clientY - rect.top) / this.imgScale;

        // We'll connect the last point to this point
        if (this.tempDrawPoints.length) {
            const [x0, y0] = this.tempDrawPoints[this.tempDrawPoints.length - 1];
            // Instead of immediate draw, queue it
            this.brushStrokes.push({
                x0, y0, x1: x, y1: y,
                type: this.currentTool
            });
            this.drawPending = true;
        }
        this.tempDrawPoints.push([x, y]);
    }

    /**
     * Single line approach for brush drawing
     */
    drawBrushLine(x0, y0, x1, y1) {
        const ctx = this.drawingCtx;
        ctx.save();

        // For mask mode
        if (this.mask) {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = this.contrastPattern || this.contrastPatternCanvas;
            ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            ctx.restore();
            return;
        }

        // Use shadow for softness
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = this.scribbleColor;
        ctx.globalAlpha = this.scribbleAlpha / 100;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20;

        // Use shadowBlur to simulate softness
        ctx.shadowColor = this.scribbleColor;
        ctx.shadowBlur = this.scribbleSoftness;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        ctx.restore();
    }

    /**
     * Single line approach for eraser
     */
    drawEraserLine(x0, y0, x1, y1) {
        const ctx = this.drawingCtx;
        ctx.save();
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0,0,0,1)';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        ctx.restore();
    }

    /**
     * Single line approach for pencil (thin, sharp brush)
     */
    drawPencilLine(x0, y0, x1, y1) {
        const ctx = this.drawingCtx;
        ctx.save();

        // For mask mode
        if (this.mask) {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = this.contrastPattern || this.contrastPatternCanvas;
            ctx.lineWidth = Math.max(1, (this.scribbleWidth / this.imgScale) * 5); // Thinner
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            ctx.restore();
            return;
        }

        // Use shadow for minimal softness
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = this.scribbleColor;
        ctx.globalAlpha = this.scribbleAlpha / 100;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = Math.max(1, (this.scribbleWidth / this.imgScale) * 5); // Thinner than brush

        // Minimal shadow for sharpness
        ctx.shadowColor = this.scribbleColor;
        ctx.shadowBlur = 0;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        ctx.restore();
    }

    /**
     * Single line approach for spray (airbrush effect)
     */
    drawSprayLine(x0, y0, x1, y1) {
        const ctx = this.drawingCtx;
        ctx.save();

        // For mask mode
        if (this.mask) {
            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = this.contrastPattern || this.contrastPatternCanvas;
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = this.scribbleColor;
            ctx.globalAlpha = this.scribbleAlpha / 100;
        }

        // Draw multiple small dots along the line
        const distance = Math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2);
        const steps = Math.max(1, Math.floor(distance / 2)); // Density
        const dx = (x1 - x0) / steps;
        const dy = (y1 - y0) / steps;

        for (let i = 0; i <= steps; i++) {
            const cx = x0 + dx * i;
            const cy = y0 + dy * i;
            const radius = (this.scribbleWidth / this.imgScale) * Math.random() * 5; // Random size
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.restore();
    }

    // ============================================================
    // 4) POINTER & MOUSE EVENT HANDLERS (Image Container)
    // ============================================================
    
    /**
     * Check if pointer is near the toolbar and hide it if drawing
     * @param {number} mouseX - Client X position relative to container
     * @param {number} mouseY - Client Y position relative to container
     */
    checkAndUpdateToolbarVisibility(mouseX, mouseY) {
        if (!this.drawing || !this.elems.toolbar) return;

        const toolbarRect = this.elems.toolbar.getBoundingClientRect();
        const containerRect = this.elems.imageContainer?.getBoundingClientRect();
        if (!containerRect) return;

        // Calculate distance from mouse to toolbar
        const toolbarLeft = toolbarRect.left - containerRect.left;
        const toolbarTop = toolbarRect.top - containerRect.top;
        const toolbarRight = toolbarLeft + toolbarRect.width;
        const toolbarBottom = toolbarTop + toolbarRect.height;

        // Find closest point on toolbar to mouse
        const closestX = Math.max(toolbarLeft, Math.min(mouseX, toolbarRight));
        const closestY = Math.max(toolbarTop, Math.min(mouseY, toolbarBottom));

        // Calculate distance
        const distX = mouseX - closestX;
        const distY = mouseY - closestY;
        const distance = Math.sqrt(distX * distX + distY * distY);

        // Hide toolbar if within threshold distance while drawing
        if (distance < this.toolbarHideDistance) {
            this.elems.toolbar.style.opacity = '0';
            this.elems.toolbar.style.pointerEvents = 'none';
        } else {
            this.elems.toolbar.style.opacity = '1';
            this.elems.toolbar.style.pointerEvents = 'auto';
        }
    }

    onPointerDownImageContainer(e) {
        const { imageContainer, image } = this.elems;
        if (!imageContainer || !this.img) {
            // If no image is loaded, possibly trigger upload
            if (!this.noUpload && e.button === 0) {
                const imageInput = document.getElementById(`imageInput_${this.uuid}`);
                if (imageInput) imageInput.click();
            }
            return;
        }

        const rect = imageContainer.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;

        if (this.isZooming && e.button === 0) {
            // Start zoom operation
            this.zoomMouseDown = true;
            e.preventDefault();
            return;
        }

        // Hand tool left-click dragging
        if (this.isHandTool && e.button === 0 && this.isInsideImage(offsetX, offsetY)) {
            this.dragging = true;
            this.offsetX = (offsetX - this.imgX) / this.panSensitivity;
            this.offsetY = (offsetY - this.imgY) / this.panSensitivity;
            if (image) image.style.cursor = 'grabbing';
            if (this.drawingCanvas) this.drawingCanvas.style.cursor = 'grabbing';
            if (this.elems.scribbleIndicator) this.elems.scribbleIndicator.style.display = 'none';
            e.preventDefault();
            return;
        }

        // Right-click dragging (fallback for when hand tool is not active)
        if (e.button === 2 && this.isInsideImage(offsetX, offsetY)) {
            this.dragging = true;
            this.offsetX = (offsetX - this.imgX) / this.panSensitivity;
            this.offsetY = (offsetY - this.imgY) / this.panSensitivity;
            if (image) image.style.cursor = 'grabbing';
            if (this.drawingCanvas) this.drawingCanvas.style.cursor = 'grabbing';
            if (this.elems.scribbleIndicator) this.elems.scribbleIndicator.style.display = 'none';
        }
    }

    onPointerMoveImageContainer(e) {
        const rect = this.elems.imageContainer.getBoundingClientRect();
        const newMousePos = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };

        if (this.isZooming && this.zoomMouseDown && this.img) {
            // Change cursor during zoom operation
            if (this.elems.imageContainer) {
                this.elems.imageContainer.style.cursor = 'zoom-in';
            }
            if (this.elems.drawingCanvas) {
                this.elems.drawingCanvas.style.cursor = 'zoom-in';
            }
            // Zoom based on mouse movement (up/right = zoom in, down/left = zoom out)
            const deltaY = newMousePos.y - this.lastMousePos.y;
            const deltaX = newMousePos.x - this.lastMousePos.x;
            const zoomFactor = (deltaY - deltaX) * -0.005 * this.zoomSensitivity; // Adjust sensitivity
            const previousScale = this.imgScale;

            this.imgScale += zoomFactor;
            this.imgScale = Math.max(0.1, Math.min(10, this.imgScale)); // Clamp zoom

            if (this.imgScale !== previousScale) {
                const scaleRatio = this.imgScale / previousScale;
                this.imgX = newMousePos.x - (newMousePos.x - this.imgX) * scaleRatio;
                this.imgY = newMousePos.y - (newMousePos.y - this.imgY) * scaleRatio;
                this.drawImage();
            }
            e.preventDefault();
        } else if (this.dragging) {
            const { imageContainer } = this.elems;
            if (!imageContainer) return;
            const { x: mouseX, y: mouseY } = newMousePos;

            this.imgX = mouseX - this.offsetX * this.panSensitivity;
            this.imgY = mouseY - this.offsetY * this.panSensitivity;
            this.drawImage();
            this.draggedJustNow = true;
        } else if (this.drawing && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
            // Check toolbar proximity while drawing
            this.checkAndUpdateToolbarVisibility(newMousePos.x, newMousePos.y);
        } else if (this.elems.scribbleIndicator && this.img && !this.noScribbles && !this.drawing && !this.isZooming && !this.isHandTool && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
            // Throttle scribble indicator updates to avoid excessive reflows
            if (!this.lastIndicatorUpdateTime || performance.now() - this.lastIndicatorUpdateTime > 16) {
                const { scribbleIndicator } = this.elems;
                const { x, y } = newMousePos;

                scribbleIndicator.style.display = 'block';
                scribbleIndicator.style.left = `${x - scribbleIndicator.offsetWidth / 2}px`;
                scribbleIndicator.style.top = `${y - scribbleIndicator.offsetHeight / 2}px`;
                this.lastIndicatorUpdateTime = performance.now();
            }
        }

        // Update last mouse position
        this.lastMousePos = newMousePos;
    }

    onPointerUpImageContainer() {
        if (this.dragging) {
            this.handleDragEnd();
        }
        if (this.zoomMouseDown) {
            this.zoomMouseDown = false;
            // Keep zoom cursor if still in zoom mode
            if (this.isZooming) {
                if (this.elems.imageContainer) {
                    this.elems.imageContainer.style.cursor = 'zoom-in';
                }
                if (this.elems.drawingCanvas) {
                    this.elems.drawingCanvas.style.cursor = 'zoom-in';
                }
            }
        }
    }

    onPointerLeaveImageContainer() {
        if (this.dragging) {
            this.handleDragEnd();
        }
        if (this.zoomMouseDown) {
            this.zoomMouseDown = false;
            // Keep zoom cursor if still in zoom mode
            if (this.isZooming) {
                if (this.elems.imageContainer) {
                    this.elems.imageContainer.style.cursor = 'zoom-in';
                }
                if (this.elems.drawingCanvas) {
                    this.elems.drawingCanvas.style.cursor = 'zoom-in';
                }
            }
        }
    }

    onWheelImageContainer(e) {
        if (e.ctrlKey) {
            // Adjust brush size with Ctrl+wheel
            e.preventDefault();
            const brushChange = e.deltaY * -0.01;
            this.scribbleWidth = Math.max(1, this.scribbleWidth + brushChange);
            if (this.elems.scribbleWidth) this.elems.scribbleWidth.value = this.scribbleWidth;
            if (this.elems.scribbleIndicator && !this.isZooming && !this.isHandTool && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
                const { imageContainer, scribbleIndicator } = this.elems;
                const newSize = this.scribbleWidth * 20;
                scribbleIndicator.style.width = `${newSize}px`;
                scribbleIndicator.style.height = `${newSize}px`;

                if (imageContainer) {
                    const rect = imageContainer.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    scribbleIndicator.style.left = `${x - newSize / 2}px`;
                    scribbleIndicator.style.top = `${y - newSize / 2}px`;
                }
            }
            return;
        }

        if (!this.img) return;
        e.preventDefault();

        const { imageContainer } = this.elems;
        if (!imageContainer) return;
        const rect = imageContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const previousScale = this.imgScale;
        const zoomFactor = e.deltaY * -0.001 * this.zoomSensitivity;

        this.imgScale += zoomFactor;
        this.imgScale = Math.max(0.1, this.imgScale);
        
        // Only update transform if scale actually changed
        if (this.imgScale !== previousScale) {
            const scaleRatio = this.imgScale / previousScale;
            this.imgX = mouseX - (mouseX - this.imgX) * scaleRatio;
            this.imgY = mouseY - (mouseY - this.imgY) * scaleRatio;
            this.drawImage();
        }
    }

    resizeContainer(e) {
        const { container } = this.elems;
        if (!container) return;
        const containerRect = container.getBoundingClientRect();
        const newHeight = e.clientY - containerRect.top;
        container.style.height = `${newHeight}px`;
    }

    handleDragEnd() {
        this.dragging = false;
        if (this.isZooming) {
            if (this.elems.image) this.elems.image.style.cursor = 'zoom-in';
            if (this.drawingCanvas) this.drawingCanvas.style.cursor = 'zoom-in';
        } else if (this.isHandTool) {
            if (this.elems.image) this.elems.image.style.cursor = 'grab';
            if (this.drawingCanvas) this.drawingCanvas.style.cursor = 'grab';
        } else {
            if (this.elems.image) this.elems.image.style.cursor = '';
            if (this.drawingCanvas) this.drawingCanvas.style.cursor = '';
        }
    }

    // ============================================================
    // 5) IMAGE / FILE / CLIPBOARD HANDLING
    // ============================================================
    handleFileUpload(file) {
        if (!file || this.noUpload) return;

        this.clearHistory();
        const reader = new FileReader();
        reader.onload = evt => this.uploadBase64(evt.target.result);
        reader.onerror = err => {
            console.error('FileReader error:', err);
        };

        try {
            reader.readAsDataURL(file);
        } catch (error) {
            console.error('Failed to read file:', error);
        }
    }

    handlePaste(e) {
        const { items } = e.clipboardData;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.indexOf('image') !== -1) {
                const file = item.getAsFile();
                this.handleFileUpload(file);
                break;
            }
        }
    }

    async uploadBase64(base64Data) {
        if (this.gradioConfig && !this.gradioConfig.version?.startsWith('4')) return;
        if (!this.gradioConfig) return;

        // Handle both inline data URLs and server-side tokens (forge-canvas://<id>)
        // If token, fetch binary and create an object URL for rendering, and store token in this.imgToken
        const isToken = typeof base64Data === 'string' && base64Data.startsWith('forge-canvas://');

        if (isToken) {
            const id = String(base64Data).replace(/^forge-canvas:\/\//, '');
            if (!id) return;
            try {
                const resp = await fetch(`./internal/forge-canvas/${encodeURIComponent(id)}`);
                if (!resp.ok) throw new Error('failed to fetch background token');
                const blob = await resp.blob();
                // Revoke prior object URL if any
                try { if (this._backgroundObjectUrl) { URL.revokeObjectURL(this._backgroundObjectUrl); this._backgroundObjectUrl = null; } } catch (e) {}
                const objectUrl = URL.createObjectURL(blob);
                this._backgroundObjectUrl = objectUrl;
                this.imgToken = base64Data; // preserve token for saving states
                const img = this.tempImage || new Image();
                this.tempImage = img;
                img.onload = () => {
                    this.img = objectUrl;
                    this.originalWidth = img.width;
                    this.originalHeight = img.height;

                    const {drawingCanvas} = this.elems;
                    if (drawingCanvas && (drawingCanvas.width !== img.width || drawingCanvas.height !== img.height)) {
                        const cssWidth = img.width;
                        const cssHeight = img.height;
                        const DPR = window.devicePixelRatio || 1;

                        drawingCanvas.width = Math.round(cssWidth * DPR);
                        drawingCanvas.height = Math.round(cssHeight * DPR);
                        drawingCanvas.style.width = `${cssWidth}px`;
                        drawingCanvas.style.height = `${cssHeight}px`;

                        this.drawingCtx = drawingCanvas.getContext('2d');
                        try { this.drawingCtx.setTransform(DPR, 0, 0, DPR, 0, 0); } catch (e) { this.drawingCtx.scale(DPR, DPR); }
                    }

                    this.adjustInitialPositionAndScale();
                    this.drawImage();
                    this.onImageUpload();
                    this.saveState();
                    this.setUploadHintVisibility(false);
                };
                img.src = objectUrl;
                return;
            } catch (e) {
                console.warn('Failed to load background token, falling back:', e);
            }
        }

        const img = this.tempImage || new Image();
        this.tempImage = img;
        img.onload = () => {
            this.img = base64Data;
            this.imgToken = null;
            this.originalWidth = img.width;
            this.originalHeight = img.height;

            const {drawingCanvas} = this.elems;
            if (drawingCanvas && (drawingCanvas.width !== img.width || drawingCanvas.height !== img.height)) {
                // Use DPR-aware sizing: store pixel buffer as cssSize * DPR and keep CSS width for layout
                const cssWidth = img.width;
                const cssHeight = img.height;
                const DPR = window.devicePixelRatio || 1;

                drawingCanvas.width = Math.round(cssWidth * DPR);
                drawingCanvas.height = Math.round(cssHeight * DPR);
                drawingCanvas.style.width = `${cssWidth}px`;
                drawingCanvas.style.height = `${cssHeight}px`;

                // re-acquire context and scale
                this.drawingCtx = drawingCanvas.getContext('2d');
                try {
                    this.drawingCtx.setTransform(DPR, 0, 0, DPR, 0, 0);
                } catch (e) {
                    this.drawingCtx.scale(DPR, DPR);
                }
            }

            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.onImageUpload();
            this.saveState();
            this.setUploadHintVisibility(false);
        };

        if (base64Data) {
            img.src = base64Data;
            return;
        }

        this.img = null;
        this.originalWidth = null;
        this.originalHeight = null;
        const {drawingCanvas} = this.elems;
        if (drawingCanvas) {
            const DPR = window.devicePixelRatio || 1;
            drawingCanvas.width = Math.round(1 * DPR);
            drawingCanvas.height = Math.round(1 * DPR);
            drawingCanvas.style.width = `1px`;
            drawingCanvas.style.height = `1px`;
            if (this.drawingCtx) {
                // clear using CSS pixels (context is scaled by DPR)
                this.drawingCtx.clearRect(0, 0, 1, 1);
            }
        }
        this.adjustInitialPositionAndScale();
        this.drawImage();
        this.onImageUpload();
        this.saveState();
        this.setUploadHintVisibility(true);
    }

    uploadBase64DrawingCanvas(base64Data) {
        // Support both data URLs and server-side tokens
        const isToken = typeof base64Data === 'string' && base64Data.startsWith('forge-canvas://');

        if (isToken) {
            const id = String(base64Data).replace(/^forge-canvas:\/\//, '');
            if (!id) return;
            (async () => {
                try {
                    const resp = await fetch(`./internal/forge-canvas/${encodeURIComponent(id)}`);
                    if (!resp.ok) throw new Error('failed to fetch foreground token');
                    const blob = await resp.blob();
                    const img = this.tempImage || new Image();
                    this.tempImage = img;
                    // createObjectURL and draw
                    const objectUrl = URL.createObjectURL(blob);
                    img.onload = () => {
                        const {drawingCanvas} = this.elems;
                        if (!drawingCanvas) return;
                        const DPR = window.devicePixelRatio || 1;
                        this.drawingCtx.clearRect(0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
                        this.drawingCtx.drawImage(img, 0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
                        this.saveState();
                        try { URL.revokeObjectURL(objectUrl); } catch (e) {}
                    };
                    img.src = objectUrl;
                } catch (e) {
                    console.warn('Failed to load foreground token:', e);
                }
            })();
            return;
        }

        const img = this.tempImage || new Image();
        img.onload = () => {
            const {drawingCanvas} = this.elems;
            if (!drawingCanvas) return;
            const DPR = window.devicePixelRatio || 1;
            // drawingCtx is scaled by DPR so operate in CSS pixels
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
            this.drawingCtx.drawImage(img, 0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
            this.saveState();
        };

        if (base64Data) {
            img.src = base64Data;
        } else {
            const {drawingCanvas} = this.elems;
            if (!drawingCanvas) return;
            const DPR = window.devicePixelRatio || 1;
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
            this.saveState();
        }
    }

    // ============================================================
    // 6) CANVAS / IMAGE OPERATIONS & HISTORY
    // ============================================================
    drawImage() {
        const { image, drawingCanvas } = this.elems;
        if (!image || !drawingCanvas) return;

        if (this.img) {
            const scaledWidth = this.originalWidth * this.imgScale;
            const scaledHeight = this.originalHeight * this.imgScale;

            image.src = this.img;
            Object.assign(image.style, {
                width: `${scaledWidth}px`,
                height: `${scaledHeight}px`,
                left: `${this.imgX}px`,
                top: `${this.imgY}px`,
                display: 'block'
            });

            Object.assign(drawingCanvas.style, {
                width: `${scaledWidth}px`,
                height: `${scaledHeight}px`,
                left: `${this.imgX}px`,
                top: `${this.imgY}px`
            });
        } else {
            image.src = '';
            image.style.display = 'none';
        }
    }

    adjustInitialPositionAndScale() {
        const { imageContainer } = this.elems;
        if (!imageContainer || !this.originalWidth || !this.originalHeight) return;

        const containerWidth = imageContainer.clientWidth - 20;
        const containerHeight = imageContainer.clientHeight - 20;

        const scaleX = containerWidth / this.originalWidth;
        const scaleY = containerHeight / this.originalHeight;
        this.imgScale = Math.min(scaleX, scaleY);

        const scaledWidth = this.originalWidth * this.imgScale;
        const scaledHeight = this.originalHeight * this.imgScale;

        this.imgX = (imageContainer.clientWidth - scaledWidth) / 2;
        this.imgY = (imageContainer.clientHeight - scaledHeight) / 2;
    }

    resetImage() {
        const { drawingCanvas } = this.elems;
        if (!drawingCanvas) return;
        this.drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);

        this.adjustInitialPositionAndScale();
        this.drawImage();
        this.saveState();
    }

    removeImage() {
        this.img = null;
        const { image, drawingCanvas } = this.elems;
        if (drawingCanvas) {
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        }
        if (image) {
            image.src = '';
            image.style.width = '0';
            image.style.height = '0';
        }

        this.saveState();
        this.setUploadHintVisibility(true);

        this.onImageUpload();
        this.clearHistory();
        this.cleanupToast(); // Clean up any active toast notifications
    }

    /**
     * Saves state for non-drawing operations (tool changes, zoom, etc.)
     */
    saveNonDrawingState() {
        // Debounce non-drawing saves more aggressively
        if (this.lastNonDrawingSave && Date.now() - this.lastNonDrawingSave < 500) return;

        this.saveState();
        this.lastNonDrawingSave = Date.now();
    }

    /**
     * Enhanced saveState with separate background/drawing layers for massive memory savings
     */
    saveState() {
        const {drawingCanvas} = this.elems;
        if (!drawingCanvas || !this.drawingCtx) return;

        // Skip if no significant change (100ms debounce)
        if (this.lastSavedState && Date.now() - this.lastSavedState < 100) return;

        try {
            // Only save the drawing layer - background is handled separately
            const state = {
                timestamp: Date.now(),
                // We'll attach a drawingBitmap asynchronously where supported to save memory/CPU
                width: drawingCanvas.width,
                height: drawingCanvas.height,
                // Store background info separately (only if it changed)
                // Prefer storing a small token when available to avoid capturing object URLs
                backgroundToken: this.imgToken || null,
                backgroundImage: this.imgToken ? null : this.img,
                backgroundScale: this.imgScale,
                backgroundX: this.imgX,
                backgroundY: this.imgY
            };

            // Push an initial lightweight state; attach a bitmap async when available
            this.undoManager.pushState(state);

            // Helper: try to upload a blob to server and return a small token (forge-canvas://<id>)
            const uploadBlobForState = async (blob) => {
                try {
                    if (!blob) return null;
                    const fd = new FormData();
                    fd.append('file', blob, 'canvas.png');
                    const resp = await fetch('./internal/forge-canvas/upload', { method: 'POST', body: fd });
                    if (!resp.ok) throw new Error('upload failed');
                    const js = await resp.json();
                    if (js && js.id) return `forge-canvas://${js.id}`;
                } catch (e) {
                    console.warn('Canvas upload for undo state failed:', e);
                }
                return null;
            };

            // Try to capture an ImageBitmap for efficient undo storage (async, non-blocking)
            if (typeof createImageBitmap === 'function') {
                try {
                    // createImageBitmap can accept an HTMLCanvasElement directly
                    createImageBitmap(drawingCanvas).then(bitmap => {
                        // attach bitmap to the same state object (by reference)
                        state.drawingBitmap = bitmap;
                    }).catch(async err => {
                        // Attempt to upload the drawing as a binary to avoid storing big data URIs
                        try {
                            const blob = await this.getNormalizedBlob();
                            if (!blob) {
                                // nothing we can do - leave state without drawing data
                                return;
                            }
                            try {
                                const token = await uploadBlobForState(blob);
                                if (token) {
                                    state.drawingToken = token;
                                }
                            } catch (e) {
                                // upload failed - leave state without drawing data
                            }
                        } catch (e) {
                            // ignore
                        }
                    });
                } catch (e) {
                    // Try uploading a normalized blob first (non-blocking via Promise)
                    this.getNormalizedBlob().then(async (blob) => {
                        if (!blob) return;
                        try {
                            const token = await uploadBlobForState(blob);
                            if (token) state.drawingToken = token;
                        } catch (err) {
                            // ignore
                        }
                    }).catch(err => {
                        // ignore
                    });
                }
                // If bitmap doesn't arrive within a short window, try uploading the blob as a fallback
                setTimeout(() => {
                    try {
                        if (!state.drawingBitmap && !state.drawingToken) {
                            // Try uploading the blob; do not create data URLs
                            drawingCanvas.toBlob(async (blob) => {
                                if (!blob) return;
                                try {
                                    const token = await uploadBlobForState(blob);
                                    if (token) state.drawingToken = token;
                                } catch (e) {
                                    // ignore
                                }
                            }, 'image/png');
                        }
                    } catch (e) {
                        // ignore
                    }
                }, 250);
            } else {
                // Environments without createImageBitmap: attempt to upload normalized blob
                try {
                    this.getNormalizedBlob().then(async (blob) => {
                        if (!blob) return;
                        try {
                            const token = await uploadBlobForState(blob);
                            if (token) state.drawingToken = token;
                        } catch (e) {
                            // ignore
                        }
                    }).catch(e => {
                        // ignore
                    });
                } catch (e) {
                    // ignore
                }
            }
            this.lastSavedState = Date.now();
            this.updateUndoRedoButtons();

            // Debounce dataURL generation
            if (this.uploadDebounceTimer) {
                clearTimeout(this.uploadDebounceTimer);
            }
            this.uploadDebounceTimer = setTimeout(() => {
                this.onDrawingCanvasUpload();
            }, this.uploadDebounceDelay);
        } catch (error) {
            console.warn('Failed to save canvas state:', error);
        }
    }

    undo() {
        // Check if undo is possible before attempting
        if (!this.undoManager.canUndo()) return;

        // Track consecutive undos
        if (this.lastActionType === 'undo') {
            this.consecutiveUndos++;
        } else {
            this.consecutiveUndos = 1;
            this.consecutiveRedos = 0; // Reset redo counter
        }
        this.lastActionType = 'undo';

        // Show toast before the action
        this.showUndoRedoToast('undo', this.consecutiveUndos);

        const state = this.undoManager.undo();
        if (state) {
            this.applyState(state);
            this.updateUndoRedoButtons();
            // Force immediate dataURL
            this.onDrawingCanvasUpload(true);
        }
    }

    redo() {
        // Check if redo is possible before attempting
        if (!this.undoManager.canRedo()) return;

        // Track consecutive redos
        if (this.lastActionType === 'redo') {
            this.consecutiveRedos++;
        } else {
            this.consecutiveRedos = 1;
            this.consecutiveUndos = 0; // Reset undo counter
        }
        this.lastActionType = 'redo';

        // Show toast before the action
        this.showUndoRedoToast('redo', this.consecutiveRedos);

        const state = this.undoManager.redo();
        if (state) {
            this.applyState(state);
            this.updateUndoRedoButtons();
            // Force immediate dataURL
            this.onDrawingCanvasUpload(true);
        }
    }

    updateUndoRedoButtons() {
        const { undoButton, redoButton } = this.elems;

        if (undoButton) {
            undoButton.disabled = !this.undoManager.canUndo();
            undoButton.style.opacity = undoButton.disabled ? '0.5' : '1';
        }

        if (redoButton) {
            redoButton.disabled = !this.undoManager.canRedo();
            redoButton.style.opacity = redoButton.disabled ? '0.5' : '1';
        }
    }

    applyState(state) {
        const {drawingCanvas, image} = this.elems;
        if (!drawingCanvas || !state) return;

        // Prefer ImageBitmap if available in state
        const DPR = window.devicePixelRatio || 1;

        // If a background token is present, fetch and set up the background image first
        if (state.backgroundToken) {
            const bgToken = state.backgroundToken;
            const bgId = String(bgToken).replace(/^forge-canvas:\/\//, '');
            if (bgId) {
                (async () => {
                    try {
                        const resp = await fetch(`./internal/forge-canvas/${encodeURIComponent(bgId)}`);
                        if (!resp.ok) throw new Error('failed to fetch background token');
                        const blob = await resp.blob();
                        try { if (this._backgroundObjectUrl) { URL.revokeObjectURL(this._backgroundObjectUrl); this._backgroundObjectUrl = null; } } catch (e) {}
                        const objectUrl = URL.createObjectURL(blob);
                        this._backgroundObjectUrl = objectUrl;
                        this.imgToken = bgToken;
                        const img = new Image();
                        img.onload = () => {
                            this.img = objectUrl;
                            this.originalWidth = img.width;
                            this.originalHeight = img.height;

                            // Ensure drawing canvas buffer matches image size (DPR-aware)
                            const {drawingCanvas} = this.elems;
                            if (drawingCanvas) {
                                const cssWidth = img.width;
                                const cssHeight = img.height;
                                const DPRloc = window.devicePixelRatio || 1;
                                drawingCanvas.width = Math.round(cssWidth * DPRloc);
                                drawingCanvas.height = Math.round(cssHeight * DPRloc);
                                drawingCanvas.style.width = `${cssWidth}px`;
                                drawingCanvas.style.height = `${cssHeight}px`;
                                this.drawingCtx = drawingCanvas.getContext('2d');
                                try { this.drawingCtx.setTransform(DPRloc, 0, 0, DPRloc, 0, 0); } catch (e) { this.drawingCtx.scale(DPRloc, DPRloc); }
                            }

                            // Restore positional metadata
                            this.imgScale = state.backgroundScale || this.imgScale;
                            this.imgX = state.backgroundX || this.imgX;
                            this.imgY = state.backgroundY || this.imgY;
                            this.drawImage();
                        };
                        img.src = objectUrl;
                    } catch (e) {
                        console.warn('Failed to load background token during applyState:', e);
                    }
                })();
            }
        }

        const applyBitmap = (bitmap) => {
            // Resize internal buffer & CSS size to match stored state if provided
            if (state.width && state.height) {
                // state.width/height are pixel buffer sizes; convert to CSS px for styling
                const cssWidth = Math.round((state.width || drawingCanvas.width) / DPR);
                const cssHeight = Math.round((state.height || drawingCanvas.height) / DPR);
                drawingCanvas.width = Math.round(cssWidth * DPR);
                drawingCanvas.height = Math.round(cssHeight * DPR);
                drawingCanvas.style.width = `${cssWidth}px`;
                drawingCanvas.style.height = `${cssHeight}px`;
                this.drawingCtx = drawingCanvas.getContext('2d');
                try { this.drawingCtx.setTransform(DPR, 0, 0, DPR, 0, 0); } catch (e) { this.drawingCtx.scale(DPR, DPR); }
            }
            // Clear and draw bitmap (bitmap draws in device pixels so we draw using CSS size)
            try {
                this.drawingCtx.clearRect(0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
                this.drawingCtx.drawImage(bitmap, 0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
            } catch (e) {
                // fallback: convert bitmap to data URL
                try {
                    const canvasForBitmap = document.createElement('canvas');
                    canvasForBitmap.width = drawingCanvas.width / DPR;
                    canvasForBitmap.height = drawingCanvas.height / DPR;
                    const ctx = canvasForBitmap.getContext('2d');
                    ctx.drawImage(bitmap, 0, 0, canvasForBitmap.width, canvasForBitmap.height);
                    this.drawingCtx.drawImage(canvasForBitmap, 0, 0);
                } catch (err) {
                    console.warn('Failed to draw bitmap during applyState:', err);
                }
            }

            // Restore background if available in state
            if (state.backgroundImage && image) {
                this.img = state.backgroundImage;
                this.imgScale = state.backgroundScale || this.imgScale;
                this.imgX = state.backgroundX || this.imgX;
                this.imgY = state.backgroundY || this.imgY;
                this.drawImage();
            }
        };

        if (state.drawingBitmap) {
            // drawingBitmap may be an ImageBitmap
            applyBitmap(state.drawingBitmap);
            return;
        }

        // If a remote token exists (uploaded to server), fetch it and render
        if (state.drawingToken) {
            const token = state.drawingToken;
            const id = String(token).replace(/^forge-canvas:\/\//, '');
            if (id) {
                (async () => {
                    try {
                        const resp = await fetch(`./internal/forge-canvas/${encodeURIComponent(id)}`);
                        if (!resp.ok) throw new Error('failed to fetch drawing token');
                        const blob = await resp.blob();
                        if (typeof createImageBitmap === 'function') {
                            try {
                                const bitmap = await createImageBitmap(blob);
                                applyBitmap(bitmap);
                                return;
                            } catch (e) {
                                // fall through to object URL path
                            }
                        }
                        const objectUrl = URL.createObjectURL(blob);
                        const img = new Image();
                        img.onload = () => {
                            try {
                                const cssWidth = state.width ? Math.round(state.width / DPR) : img.width;
                                const cssHeight = state.height ? Math.round(state.height / DPR) : img.height;
                                drawingCanvas.width = Math.round(cssWidth * DPR);
                                drawingCanvas.height = Math.round(cssHeight * DPR);
                                drawingCanvas.style.width = `${cssWidth}px`;
                                drawingCanvas.style.height = `${cssHeight}px`;
                                this.drawingCtx = drawingCanvas.getContext('2d');
                                try { this.drawingCtx.setTransform(DPR, 0, 0, DPR, 0, 0); } catch (e) { this.drawingCtx.scale(DPR, DPR); }

                                this.drawingCtx.clearRect(0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
                                this.drawingCtx.drawImage(img, 0, 0);

                                if (state.backgroundImage && image) {
                                    this.img = state.backgroundImage;
                                    this.imgScale = state.backgroundScale || this.imgScale;
                                    this.imgX = state.backgroundX || this.imgX;
                                    this.imgY = state.backgroundY || this.imgY;
                                    this.drawImage();
                                }
                            } catch (err) {
                                console.warn('Failed to draw token image during applyState:', err);
                            } finally {
                                URL.revokeObjectURL(objectUrl);
                            }
                        };
                        img.src = objectUrl;
                        return;
                    } catch (e) {
                        console.warn('Failed to load drawing token during applyState:', e);
                        // fall through to dataURL path below
                    }
                })();
            }
        }

        // Fallback to dataURL if bitmap not available
        const drawingDataURL = state.drawingDataURL || state.dataURL;
        if (!drawingDataURL) return;
        const img = new Image();
        img.onload = () => {
            // Determine css sizes from state or image
            const cssWidth = state.width ? Math.round(state.width / DPR) : img.width;
            const cssHeight = state.height ? Math.round(state.height / DPR) : img.height;
            drawingCanvas.width = Math.round(cssWidth * DPR);
            drawingCanvas.height = Math.round(cssHeight * DPR);
            drawingCanvas.style.width = `${cssWidth}px`;
            drawingCanvas.style.height = `${cssHeight}px`;
            this.drawingCtx = drawingCanvas.getContext('2d');
            try { this.drawingCtx.setTransform(DPR, 0, 0, DPR, 0, 0); } catch (e) { this.drawingCtx.scale(DPR, DPR); }

            // Clear and restore drawing layer
            this.drawingCtx.clearRect(0, 0, drawingCanvas.width / DPR, drawingCanvas.height / DPR);
            this.drawingCtx.drawImage(img, 0, 0);

            // Restore background if available in state
            if (state.backgroundImage && image) {
                this.img = state.backgroundImage;
                this.imgScale = state.backgroundScale || this.imgScale;
                this.imgX = state.backgroundX || this.imgX;
                this.imgY = state.backgroundY || this.imgY;
                this.drawImage();
            }
        };
        img.src = drawingDataURL;
    }

    clearHistory() {
        this.undoManager.clear();
        this.updateUndoRedoButtons();
    }

    onImageUpload() {
        if (!this.img) {
            this.backgroundGradioBind.setValue('');
            return;
        }

        const { image } = this.elems;
        if (!image) return;

        const { tempCanvas } = this;
        const ctx = tempCanvas.getContext('2d');
        tempCanvas.width = this.originalWidth;
        tempCanvas.height = this.originalHeight;
        ctx.drawImage(image, 0, 0, this.originalWidth, this.originalHeight);

        // Try to upload as binary to avoid large base64 strings in JS memory.
        const uploadBlob = async (blob) => {
            try {
                const fd = new FormData();
                fd.append('file', blob, 'canvas.png');
                const resp = await fetch('./internal/forge-canvas/upload', { method: 'POST', body: fd });
                if (!resp.ok) throw new Error('upload failed');
                const js = await resp.json();
                if (js && js.id) return js.id;
                throw new Error('invalid upload response');
            } catch (e) {
                console.warn('Canvas upload failed, falling back to data URI:', e);
                return null;
            }
        };

        tempCanvas.toBlob(async (blob) => {
            if (!blob) {
                console.warn('Failed to generate image blob for background upload');
                this.backgroundGradioBind.setValue('');
                return;
            }

            const id = await uploadBlob(blob);
            if (id) {
                this.backgroundGradioBind.setValue(`forge-canvas://${id}`);
            } else {
                // upload failed — do not emit base64, clear bind
                console.warn('Background upload failed');
                this.backgroundGradioBind.setValue('');
            }
        }, 'image/png');
    }

    onDrawingCanvasUpload(forceImmediate = false) {
        if (!this.img) {
            this.foregroundGradioBind.setValue('');
            return;
        }
        const {drawingCanvas} = this.elems;
        if (!drawingCanvas) return;
        // Upload drawing canvas as binary to reduce base64 memory pressure.
        const uploadBlob = async (blob) => {
            try {
                const fd = new FormData();
                fd.append('file', blob, 'canvas.png');
                const resp = await fetch('./internal/forge-canvas/upload', { method: 'POST', body: fd });
                if (!resp.ok) throw new Error('upload failed');
                const js = await resp.json();
                if (js && js.id) return js.id;
                throw new Error('invalid upload response');
            } catch (e) {
                console.warn('Canvas upload failed, falling back to data URI:', e);
                return null;
            }
        };

        const emitCanvasAsBlob = (cb) => {
            try {
                // Create a normalized canvas sized to the background's natural pixel size
                // so that mask pixels align with the background image when server crops "Only masked".
                const DPR = window.devicePixelRatio || 1;
                const normCanvas = document.createElement('canvas');
                const targetW = this.originalWidth || Math.round(drawingCanvas.width / DPR);
                const targetH = this.originalHeight || Math.round(drawingCanvas.height / DPR);
                normCanvas.width = targetW;
                normCanvas.height = targetH;
                const nctx = normCanvas.getContext('2d');

                // Draw the drawingCanvas content into normalized canvas, scaling as needed
                try {
                    nctx.drawImage(drawingCanvas, 0, 0, drawingCanvas.width, drawingCanvas.height, 0, 0, normCanvas.width, normCanvas.height);
                } catch (e) {
                    // fallback to drawing without explicit source/dest sizes
                    try { nctx.drawImage(drawingCanvas, 0, 0); } catch (e2) {}
                }

                normCanvas.toBlob(async (blob) => {
                    if (!blob) {
                        console.warn('Failed to generate normalized canvas blob');
                        if (typeof cb === 'function') cb();
                        return;
                    }

                    const id = await uploadBlob(blob);
                    if (id) {
                        this.foregroundGradioBind.setValue(`forge-canvas://${id}`);
                    } else {
                        console.warn('Foreground upload failed');
                        this.foregroundGradioBind.setValue('');
                    }

                    if (typeof cb === 'function') cb();
                }, 'image/png');
            } catch (err) {
                // Final fallback: we couldn't produce a blob — clear the bind and warn
                console.warn('Failed to generate normalized canvas blob:', err);
                this.foregroundGradioBind.setValue('');
                if (typeof cb === 'function') cb();
            }
        };

        // Optionally skip the debounce if forceImmediate
        if (!forceImmediate) {
            if (this.uploadDebounceTimer) {
                clearTimeout(this.uploadDebounceTimer);
            }
            this.uploadDebounceTimer = setTimeout(() => {
                emitCanvasAsBlob();
            }, this.uploadDebounceDelay);
        } else {
            emitCanvasAsBlob();
        }
    }

    /**
     * Generate a normalized PNG blob matching the background natural pixel size
     * Returns a Promise resolving to a Blob or null on failure
     */
    getNormalizedBlob() {
        const { drawingCanvas } = this.elems;
        return new Promise((resolve) => {
            if (!drawingCanvas) return resolve(null);
            const DPR = window.devicePixelRatio || 1;
            const targetW = this.originalWidth || Math.round(drawingCanvas.width / DPR);
            const targetH = this.originalHeight || Math.round(drawingCanvas.height / DPR);
            const normCanvas = document.createElement('canvas');
            normCanvas.width = targetW;
            normCanvas.height = targetH;
            const nctx = normCanvas.getContext('2d');

            try {
                nctx.drawImage(drawingCanvas, 0, 0, drawingCanvas.width, drawingCanvas.height, 0, 0, normCanvas.width, normCanvas.height);
            } catch (e) {
                try { nctx.drawImage(drawingCanvas, 0, 0); } catch (e2) {}
            }

            try {
                normCanvas.toBlob((blob) => {
                    resolve(blob || null);
                }, 'image/png');
            } catch (e) {
                resolve(null);
            }
        });
    }

    // ============================================================
    // 7) UI MAXIMIZE / MINIMIZE & TOOL SELECTION
    // ============================================================
    maximize() {
        if (this.maximized) return;

        const { container, maxButton, minButton } = this.elems;
        if (!container || !maxButton || !minButton) return;

        this.originalState = {
            width: container.style.width,
            height: container.style.height,
            top: container.style.top,
            left: container.style.left,
            position: container.style.position,
            zIndex: container.style.zIndex
        };

        Object.assign(container.style, {
            width: '100vw',
            height: '100vh',
            top: '0',
            left: '0',
            position: 'fixed',
            zIndex: '1000'
        });

        maxButton.style.display = 'none';
        minButton.style.display = 'inline-block';
        this.maximized = true;
    }

    minimize() {
        if (!this.maximized) return;

        const { container, maxButton, minButton } = this.elems;
        if (!container || !maxButton || !minButton) return;

        Object.assign(container.style, {
            width: this.originalState.width,
            height: this.originalState.height,
            top: this.originalState.top,
            left: this.originalState.left,
            position: this.originalState.position,
            zIndex: this.originalState.zIndex
        });

        maxButton.style.display = 'inline-block';
        minButton.style.display = 'none';
        this.maximized = false;
    }

    // ============================================================
    // 8) UTILITY METHODS
    // ============================================================

    /**
     * Checks if the given coordinates are inside the image bounds
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {boolean} True if inside image bounds
     */
    isInsideImage(x, y) {
        const scaledWidth = this.originalWidth * this.imgScale;
        const scaledHeight = this.originalHeight * this.imgScale;
        return (
            x > this.imgX &&
            x < this.imgX + scaledWidth &&
            y > this.imgY &&
            y < this.imgY + scaledHeight
        );
    }

    /**
     * Adjusts brush size and shows toast feedback
     * @param {number} delta - Size change amount
     */
    adjustBrushSize(delta) {
        const oldSize = this.scribbleWidth;
        this.scribbleWidth = Math.max(1, Math.min(50, this.scribbleWidth + delta));

        if (this.scribbleWidth !== oldSize) {
            // Update UI elements
            if (this.elems.scribbleWidth) {
                this.elems.scribbleWidth.value = this.scribbleWidth;
            }

            // Update scribble indicator size
            if (this.elems.scribbleIndicator && !this.isZooming && !this.isHandTool && (this.currentTool === 'brush' || this.currentTool === 'eraser')) {
                const newSize = this.scribbleWidth * 20;
                this.elems.scribbleIndicator.style.width = `${newSize}px`;
                this.elems.scribbleIndicator.style.height = `${newSize}px`;
            }

            // Show toast notification
            this.showBrushSizeToast(delta > 0);
        }
    }
    /**
     * Toggles the empty-state overlay visibility with a smooth fade.
     * @param {boolean} shouldShow - Whether the hint should be visible.
     */
    setUploadHintVisibility(shouldShow) {
        const { uploadHint } = this.elems;
        if (!uploadHint) return;
        if (this.noUpload) {
            uploadHint.style.display = 'none';
            return;
        }
        uploadHint.classList.toggle('forge-upload-hidden', !shouldShow);
    }
    setTool(tool) {
        if (tool === this.currentTool) return;
        this.currentTool = tool;

        // Disable zoom and hand tool when switching to drawing tools
        if (tool === 'brush' || tool === 'eraser') {
            this.isZooming = false;
            this.isHandTool = false;
        }

        // Update dropdown item active states
        const { brushDropdown } = this.elems;
        if (brushDropdown) {
            const items = brushDropdown.querySelectorAll('.forge-dropdown-item');
            items.forEach(item => {
                item.classList.toggle('active', item.dataset.tool === tool);
            });
        }

        // Update brush button active state and appearance
        const { brushButton } = this.elems;
        if (brushButton) {
            brushButton.classList.toggle('active', tool === 'brush' || tool === 'eraser');
            if (tool === 'brush') {
                brushButton.textContent = '🖌️';
                brushButton.title = 'Brush Tool (B)';
            } else if (tool === 'eraser') {
                brushButton.textContent = '🧽';
                brushButton.title = 'Eraser Tool (E)';
            } else {
                // No tool selected - show default brush icon
                brushButton.textContent = '🖌️';
                brushButton.title = 'Brush Tool (B) - Hover for more options';
            }
        }

        const { eraserButton } = this.elems;
        if (eraserButton) {
            eraserButton.classList.toggle('active', tool === 'eraser');
        }

        // Add visual feedback animation
        this.showToolSwitchFeedback(tool);

        // Update cursor and scribble indicator after tool change
        this.updateCursor();
        this.updateScribbleIndicator();
    }

    /**
     * Hides all dropdown menus
     */
    hideAllDropdowns() {
        const { brushDropdown } = this.elems;
        if (brushDropdown) {
            brushDropdown.classList.remove('show');
        }
    }

    /**
     * Updates the cursor based on current tool state
     */
    updateCursor() {
        let cursor = '';

        if (this.isZooming) {
            cursor = 'zoom-in';
        } else if (this.isHandTool) {
            cursor = 'grab';
        }

        if (this.elems.imageContainer) {
            this.elems.imageContainer.style.cursor = cursor;
        }
        if (this.elems.drawingCanvas) {
            this.elems.drawingCanvas.style.cursor = cursor;
        }
    }

    /**
     * Updates the scribble indicator visibility based on current tool state
     */
    updateScribbleIndicator() {
        if (this.elems.scribbleIndicator) {
            if (this.isZooming || this.isHandTool || (this.currentTool !== 'brush' && this.currentTool !== 'eraser')) {
                this.elems.scribbleIndicator.style.display = 'none';
            } else if (this.img && !this.noScribbles && !this.drawing) {
                this.elems.scribbleIndicator.style.display = 'block';
            }
        }
    }

    /**
     * Shows visual feedback when switching tools
     * @param {string} tool - The tool that was switched to
     */
    showToolSwitchFeedback(tool) {
        const { brushButton, scribbleIndicator, toolbar } = this.elems;

        // Clear any existing animation timeouts
        this.animationTimeouts.forEach(timeoutId => clearTimeout(timeoutId));
        this.animationTimeouts.clear();

        // Flash animation on brush button
        if (brushButton) {
            brushButton.classList.remove('forge-tool-switch-flash'); // Reset state
            // Force reflow
            brushButton.offsetHeight;
            brushButton.classList.add('forge-tool-switch-flash');
            const timeoutId = setTimeout(() => {
                brushButton.classList.remove('forge-tool-switch-flash');
                this.animationTimeouts.delete(timeoutId);
            }, 200);
            this.animationTimeouts.add(timeoutId);
        }

        // Subtle flash on toolbar background
        if (toolbar) {
            toolbar.classList.remove('forge-toolbar-flash'); // Reset state
            // Force reflow
            toolbar.offsetHeight;
            toolbar.classList.add('forge-toolbar-flash');
            const timeoutId = setTimeout(() => {
                toolbar.classList.remove('forge-toolbar-flash');
                this.animationTimeouts.delete(timeoutId);
            }, 150);
            this.animationTimeouts.add(timeoutId);
        }

        // Pulse animation on scribble indicator
        if (scribbleIndicator && this.img && !this.noScribbles) {
            scribbleIndicator.classList.remove('forge-indicator-pulse'); // Reset state
            // Force reflow
            scribbleIndicator.offsetHeight;
            scribbleIndicator.classList.add('forge-indicator-pulse');
            const timeoutId = setTimeout(() => {
                scribbleIndicator.classList.remove('forge-indicator-pulse');
                this.animationTimeouts.delete(timeoutId);
            }, 400);
            this.animationTimeouts.add(timeoutId);
        }

        // Toast notification is now handled by the hotkey handler
    }

    /**
     * Helper method to show toast notifications with proper cleanup
     * @param {string} message - The message to display
     * @param {number} duration - How long to show the toast (ms)
     */
    showToast(message, duration = 1500) {
        const { toolbarMessage } = this.elems;
        if (!toolbarMessage) return;

        // Clear any existing timeouts
        if (this.toastTimeoutId) {
            clearTimeout(this.toastTimeoutId);
            this.toastTimeoutId = null;
        }
        if (this.toastVisibilityTimeoutId) {
            clearTimeout(this.toastVisibilityTimeoutId);
            this.toastVisibilityTimeoutId = null;
        }

        // Check if this is a different message or if enough time has passed
        const now = Date.now();
        const isDifferentMessage = toolbarMessage.innerHTML !== message;
        const shouldShowImmediately = isDifferentMessage || (now - this.lastToastTime >= this.toastDebounceDelay);

        if (!shouldShowImmediately) {
            // Schedule to show this message after debounce delay
            if (this.pendingToastMessage !== message) {
                this.pendingToastMessage = message;
                this.pendingToastDuration = duration;
                if (this.pendingToastTimeout) {
                    clearTimeout(this.pendingToastTimeout);
                }
                this.pendingToastTimeout = setTimeout(() => {
                    this.showToast(this.pendingToastMessage, this.pendingToastDuration);
                    this.pendingToastMessage = null;
                    this.pendingToastTimeout = null;
                }, this.toastDebounceDelay - (now - this.lastToastTime));
            }
            return;
        }

        // Clear any pending toast
        if (this.pendingToastTimeout) {
            clearTimeout(this.pendingToastTimeout);
            this.pendingToastTimeout = null;
            this.pendingToastMessage = null;
        }

        this.lastToastTime = now;

        // Reset element state
        toolbarMessage.classList.remove('show');
        toolbarMessage.style.visibility = 'hidden';
        toolbarMessage.innerHTML = '';

        // Force reflow to ensure clean state
        toolbarMessage.offsetHeight;

        // Set new message and show
        toolbarMessage.innerHTML = message;
        toolbarMessage.style.visibility = 'visible';
        toolbarMessage.classList.add('show');

        // Set up cleanup
        this.toastTimeoutId = setTimeout(() => {
            toolbarMessage.classList.remove('show');
            this.toastVisibilityTimeoutId = setTimeout(() => {
                toolbarMessage.style.visibility = 'hidden';
                toolbarMessage.innerHTML = '';
            }, 300);
        }, duration);
    }

    /**
     * Cleans up toast timeouts and state
     */
    cleanupToast() {
        if (this.toastTimeoutId) {
            clearTimeout(this.toastTimeoutId);
            this.toastTimeoutId = null;
        }
        if (this.toastVisibilityTimeoutId) {
            clearTimeout(this.toastVisibilityTimeoutId);
            this.toastVisibilityTimeoutId = null;
        }
        if (this.pendingToastTimeout) {
            clearTimeout(this.pendingToastTimeout);
            this.pendingToastTimeout = null;
        }
        this.lastToastTime = 0;
        this.pendingToastMessage = null;
        this.pendingToastDuration = 0;

        // Clean up animation timeouts
        this.animationTimeouts.forEach(timeoutId => clearTimeout(timeoutId));
        this.animationTimeouts.clear();
    }

    /**
     * Shows a toast notification for brush size changes
     * @param {boolean} increased - Whether size was increased or decreased
     */
    showBrushSizeToast(increased) {
        const message = increased ? 'Brush Size Increased' : 'Brush Size Decreased';
        const icon = '📏';
        this.showToast(`${icon} ${message}`, 1000);
    }

    /**
     * Shows a toast notification for zoom mode toggle
     */
    showZoomToast() {
        const message = this.isZooming ? 'Zoom Mode Enabled' : 'Zoom Mode Disabled';
        const icon = '🔍';
        this.showToast(`${icon} ${message}`, 1500);
    }

    /**
     * Shows a toast notification for hand tool toggle
     */
    showHandToolToast() {
        const message = this.isHandTool ? 'Hand Tool Enabled' : 'Hand Tool Disabled';
        const icon = '🖐️';
        this.showToast(`${icon} ${message}`, 1500);
    }

    /**
     * Shows a toast notification for undo/redo actions
     * @param {string} action - 'undo' or 'redo'
     * @param {number} count - Number of consecutive actions
     */
    showUndoRedoToast(action, count = 1) {
        const baseMessage = action === 'undo' ? 'Undid Last Action' : 'Redid Last Action';
        const message = count > 1 ? `${baseMessage} (x${count})` : baseMessage;
        const icon = action === 'undo' ? '↶' : '↷';
        this.showToast(`${icon} ${message}`, 2000);
    }



    /**
     * Shows a toast notification for tool switching
     * @param {string} tool - The tool that was switched to
     */
    showToolToast(tool) {
        const toolName = tool === 'brush' ? 'Brush Tool' : 'Eraser Tool';
        const icon = tool === 'brush' ? '🖌️' : '🧽';
        this.showToast(`${icon} ${toolName}`, 1500);
    }

    /**
     * Updates a setting value from the canvas
     * @param {string} key - The setting key
     * @param {*} value - The new value
     * @param {boolean} applyImmediately - Whether to apply the setting immediately
     * 
     * Example usage:
     * this.updateSetting('sd_model_checkpoint', 'model_name.safetensors');
     * this.updateSettings({ 'steps': 20, 'cfg_scale': 7.5 });
     */
    updateSetting(key, value, applyImmediately = true) {
        const elem = document.getElementById(`setting_${key}`);
        if (!elem) {
            console.warn(`Setting element for key '${key}' not found`);
            return;
        }

        // Set the value based on element type
        if (elem.type === 'checkbox') {
            elem.checked = value;
        } else if (elem.tagName === 'TEXTAREA' || elem.tagName === 'INPUT') {
            elem.value = value;
        } else {
            console.warn(`Unsupported element type for setting '${key}'`);
            return;
        }

        // Dispatch events to trigger Gradio's change handlers
        elem.dispatchEvent(new Event('input', { bubbles: true }));
        elem.dispatchEvent(new Event('change', { bubbles: true }));

        if (applyImmediately) {
            // Find and click the apply settings button
            const submitButton = document.getElementById('settings_submit');
            if (submitButton) {
                submitButton.click();
            } else {
                console.warn('Apply settings button not found');
            }
        }
    }

    /**
     * Updates multiple settings at once
     * @param {Object} settings - Object with key-value pairs
     * @param {boolean} applyImmediately - Whether to apply all settings immediately
     */
    updateSettings(settings, applyImmediately = true) {
        Object.entries(settings).forEach(([key, value]) => {
            this.updateSetting(key, value, false); // Don't apply individually
        });

        if (applyImmediately) {
            const submitButton = document.getElementById('settings_submit');
            if (submitButton) {
                submitButton.click();
            }
        }
    }
}

// ------------------------------------------------------------
// Modular System Compatibility
// ------------------------------------------------------------

// Check if modular system is available
const UseModularSystem = typeof window.ForgeCanvas !== 'undefined' &&
                        window.ForgeCanvas.prototype &&
                        window.ForgeCanvas.prototype.use;

// If modular system is loaded, extend it with legacy functionality
if (UseModularSystem) {
    // Store reference to modular system
    const ModularForgeCanvas = window.ForgeCanvas;

    // Extend with legacy compatibility
    class LegacyForgeCanvas extends ModularForgeCanvas {
        // Add any legacy-specific methods here if needed
        legacyMethod() {
            console.log('Legacy method called');
        }
    }

    // Replace with legacy-compatible version
    window.ModularForgeCanvas = ModularForgeCanvas;
    window.ForgeCanvas = LegacyForgeCanvas;
}

// Constants
const True = true;
const False = false;
