class GradioTextAreaBind {
    constructor(id, className) {
        this.target = document.querySelector(`#${id}.${className} textarea`);
        this.sync_lock = false;
        this.previousValue = "";
    }

    set_value(value) {
        if (this.sync_lock) return;
        this.sync_lock = true;
        this.target.value = value;
        this.previousValue = value;
        const event = new Event("input", { bubbles: true });
        Object.defineProperty(event, "target", { value: this.target });
        this.target.dispatchEvent(event);
        this.previousValue = value;
        this.sync_lock = false;
    }

    listen(callback) {
        setInterval(() => {
            if (this.target.value !== this.previousValue) {
                this.previousValue = this.target.value;
                if (this.sync_lock) return;
                this.sync_lock = true;
                callback(this.target.value);
                this.sync_lock = false;
            }
        }, 100);
    }
}

class ForgeCanvas {
    constructor(
        uuid,
        no_upload = false,
        no_scribbles = false,
        contrast_scribbles = false,
        initial_height = 512,
        scribbleColor = "#000000",
        scribbleColorFixed = false,
        scribbleWidth = 20,
        scribbleWidthFixed = false,
        scribbleWidthConsistent = false,
        scribbleAlpha = 100,
        scribbleAlphaFixed = false,
        scribbleSoftness = 0,
        scribbleSoftnessFixed = false,
    ) {
        this.gradio_config = gradio_config;
        this.uuid = uuid;

        this.no_upload = no_upload;
        this.no_scribbles = no_scribbles;
        this.contrast_scribbles = contrast_scribbles;

        this.img = null;
        this.imgX = 0;
        this.imgY = 0;
        this.orgWidth = 0;
        this.orgHeight = 0;
        this.imgScale = 1.0;
        this.initial_height = initial_height;

        this.dragging = false;
        this.dragged_just_now = false;
        this.drawing = false;
        this.contrast_pattern = null;

        this.scribbleColor = scribbleColor;
        this.scribbleColorFixed = scribbleColorFixed;
        this.scribbleWidth = scribbleWidth;
        this.scribbleWidthFixed = scribbleWidthFixed;
        this.scribbleWidthConsistent = scribbleWidthConsistent;
        this.scribbleAlpha = scribbleAlpha;
        this.scribbleAlphaFixed = scribbleAlphaFixed;
        this.scribbleSoftness = scribbleSoftness;
        this.scribbleSoftnessFixed = scribbleSoftnessFixed;

        this.history = [];
        this.historyIndex = -1;
        this.maximized = false;
        this.originalState = {};
        this.pointerInsideContainer = false;
        this.temp_canvas = document.createElement("canvas");
        this.temp_draw_points = [];
        this.temp_draw_bg = null;

        this.background_gradio_bind = new GradioTextAreaBind(this.uuid, "logical_image_background");
        this.foreground_gradio_bind = new GradioTextAreaBind(this.uuid, "logical_image_foreground");
        this.init();

        this._held_W = false;
        this._held_A = false;
        this._held_S = false;

        this._original_alpha = null;
    }

    init() {
        const self = this;

        const container = document.getElementById(`container_${self.uuid}`);
        const imageContainer = document.getElementById(`imageContainer_${self.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${self.uuid}`);
        const toolbar = document.getElementById(`toolbar_${self.uuid}`);

        const maxButton = document.getElementById(`maxButton_${self.uuid}`);
        const minButton = document.getElementById(`minButton_${self.uuid}`);
        const uploadButton = document.getElementById(`uploadButton_${self.uuid}`);
        const removeButton = document.getElementById(`removeButton_${self.uuid}`);
        const centerButton = document.getElementById(`centerButton_${self.uuid}`);
        const resetButton = document.getElementById(`resetButton_${self.uuid}`);
        const undoButton = document.getElementById(`undoButton_${self.uuid}`);
        const redoButton = document.getElementById(`redoButton_${self.uuid}`);

        const uploadHint = document.getElementById(`uploadHint_${self.uuid}`);
        const scribbleIndicator = document.getElementById(`scribbleIndicator_${self.uuid}`);

        minButton.style.display = "none";
        this.maximized = false;

        const scribbleColorBlock = document.getElementById(`scribbleColorBlock_${self.uuid}`);
        if (self.scribbleColorFixed) scribbleColorBlock.style.display = "none";
        const scribbleColor = document.getElementById(`scribbleColor_${self.uuid}`);
        scribbleColor.value = self.scribbleColor;

        const scribbleWidthBlock = document.getElementById(`scribbleWidthBlock_${self.uuid}`);
        if (self.scribbleWidthFixed) scribbleWidthBlock.style.display = "none";
        const scribbleWidth = document.getElementById(`scribbleWidth_${self.uuid}`);
        const scribbleWidthLabel = document.getElementById(`widthLabel_${self.uuid}`);
        scribbleWidth.value = self.scribbleWidth;
        scribbleWidthLabel.textContent = `Brush Width (${self.scribbleWidth})`;

        const scribbleAlphaBlock = document.getElementById(`scribbleAlphaBlock_${self.uuid}`);
        if (self.scribbleAlphaFixed) scribbleAlphaBlock.style.display = "none";
        const scribbleAlpha = document.getElementById(`scribbleAlpha_${self.uuid}`);
        const scribbleAlphaLabel = document.getElementById(`alphaLabel_${self.uuid}`);
        scribbleAlpha.value = self.scribbleAlpha;
        scribbleAlphaLabel.textContent = `Brush Opacity (${self.scribbleAlpha})`;

        const scribbleSoftnessBlock = document.getElementById(`scribbleSoftnessBlock_${self.uuid}`);
        if (self.scribbleSoftnessFixed) scribbleSoftnessBlock.style.display = "none";
        const scribbleSoftness = document.getElementById(`scribbleSoftness_${self.uuid}`);
        const scribbleSoftnessLabel = document.getElementById(`softnessLabel_${self.uuid}`);
        scribbleSoftness.value = self.scribbleSoftness;
        scribbleSoftnessLabel.textContent = `Brush Softness (${self.scribbleSoftness})`;

        const indicatorSize = self.scribbleWidth * 4;
        scribbleIndicator.style.width = `${indicatorSize}px`;
        scribbleIndicator.style.height = `${indicatorSize}px`;

        container.style.height = `${self.initial_height}px`;
        drawingCanvas.width = imageContainer.clientWidth;
        drawingCanvas.height = imageContainer.clientHeight;

        const drawContext = drawingCanvas.getContext("2d");
        self.drawingCanvas_ = drawingCanvas;

        if (self.no_scribbles) {
            toolbar.querySelector(".forge-toolbar-box-b").style.display = "none";
            toolbar.removeAttribute("title");
            resetButton.style.display = "none";
            undoButton.style.display = "none";
            redoButton.style.display = "none";
        }

        if (self.no_upload) {
            uploadButton.style.display = "none";
            uploadHint.style.display = "none";
        }

        if (self.contrast_scribbles) {
            const size = 10;
            const tempCanvas = self.temp_canvas;
            tempCanvas.width = size * 2;
            tempCanvas.height = size * 2;
            const tempCtx = tempCanvas.getContext("2d");
            tempCtx.fillStyle = "#ffffff";
            tempCtx.fillRect(0, 0, size, size);
            tempCtx.fillRect(size, size, size, size);
            tempCtx.fillStyle = "#000000";
            tempCtx.fillRect(size, 0, size, size);
            tempCtx.fillRect(0, size, size, size);
            self.contrast_pattern = drawContext.createPattern(tempCanvas, "repeat");
            drawingCanvas.style.opacity = "0.5";
        }

        function resetScribble(e, rect) {
            const indicatorSize = self.scribbleWidth * (self.scribbleWidthConsistent ? 1.0 : self.imgScale) * 4;
            scribbleIndicator.style.width = `${indicatorSize}px`;
            scribbleIndicator.style.height = `${indicatorSize}px`;
            scribbleIndicator.style.left = `${e.clientX - rect.left - indicatorSize / 2}px`;
            scribbleIndicator.style.top = `${e.clientY - rect.top - indicatorSize / 2}px`;
        }

        const resizeObserver = new ResizeObserver(() => {
            self.adjustInitialPositionAndScale();
            self.drawImage();
        });
        resizeObserver.observe(container);

        document.getElementById(`imageInput_${self.uuid}`).addEventListener("change", (e) => {
            self.handleFileUpload(e.target.files[0]);
        });

        uploadButton.addEventListener("click", () => {
            if (self.no_upload) return;
            document.getElementById(`imageInput_${self.uuid}`).click();
        });

        removeButton.addEventListener("click", () => {
            self.resetImage();
            self.removeImage();
        });

        centerButton.addEventListener("click", () => {
            self.adjustInitialPositionAndScale();
            self.drawImage();
        });

        resetButton.addEventListener("click", () => {
            self.resetImage();
        });

        undoButton.addEventListener("click", () => {
            self.undo();
        });

        redoButton.addEventListener("click", () => {
            self.redo();
        });

        scribbleColor.addEventListener("input", (e) => {
            self.scribbleColor = e.target.value;
            scribbleIndicator.style.borderColor = self.scribbleColor;
        });

        scribbleWidth.addEventListener("input", (e) => {
            self.scribbleWidth = e.target.value;
            scribbleWidthLabel.textContent = `Brush Width (${self.scribbleWidth})`;
            const indicatorSize = self.scribbleWidth * (self.scribbleWidthConsistent ? 1.0 : self.imgScale) * 4;
            scribbleIndicator.style.width = `${indicatorSize}px`;
            scribbleIndicator.style.height = `${indicatorSize}px`;
        });

        scribbleAlpha.addEventListener("input", (e) => {
            self.scribbleAlpha = e.target.value;
            scribbleAlphaLabel.textContent = `Brush Opacity (${self.scribbleAlpha})`;
        });

        scribbleSoftness.addEventListener("input", (e) => {
            self.scribbleSoftness = e.target.value;
            scribbleSoftnessLabel.textContent = `Brush Softness (${self.scribbleSoftness})`;
        });

        drawingCanvas.addEventListener("pointerdown", (e) => {
            if (!self.img || e.button !== 0 || self.no_scribbles) return;
            const rect = drawingCanvas.getBoundingClientRect();
            self.drawing = true;
            drawingCanvas.style.cursor = "crosshair";
            scribbleIndicator.style.display = "none";
            self.temp_draw_points = [[(e.clientX - rect.left) / self.imgScale, (e.clientY - rect.top) / self.imgScale]];
            self.temp_draw_bg = drawContext.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height);
            self.handleDraw(e);
        });

        drawingCanvas.addEventListener("pointermove", (e) => {
            if (self.drawing) self.handleDraw(e);
            if (self.img && !self.drawing && !self.dragging && !self.no_scribbles) {
                const rect = container.getBoundingClientRect();
                resetScribble(e, rect);
                scribbleIndicator.style.display = "inline-block";
            }
        });

        toolbar.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
        });

        drawingCanvas.addEventListener("pointerup", () => {
            self.drawing = false;
            drawingCanvas.style.cursor = "";
            self.saveState();
        });

        drawingCanvas.addEventListener("pointerout", () => {
            self.drawing = false;
            drawingCanvas.style.cursor = "";
            scribbleIndicator.style.display = "none";
        });

        container.addEventListener("pointerdown", (e) => {
            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (e.button === 2 && self.isInsideImage(x, y)) {
                self.dragging = true;
                self.offsetX = x - self.imgX;
                self.offsetY = y - self.imgY;
                imageContainer.style.cursor = "grabbing";
                drawingCanvas.style.cursor = "grabbing";
                scribbleIndicator.style.display = "none";
            } else if (e.button === 0 && !self.img && !self.no_upload) {
                document.getElementById(`imageInput_${self.uuid}`).click();
            }
        });

        container.addEventListener("pointermove", (e) => {
            if (self.dragging) {
                const rect = container.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                self.imgX = x - self.offsetX;
                self.imgY = y - self.offsetY;
                self.drawImage();
                self.dragged_just_now = true;
            }
        });

        container.addEventListener("pointerup", (e) => {
            if (self.dragging) self.handleDragEnd(e, false);
        });

        container.addEventListener("pointerout", (e) => {
            if (self.dragging) self.handleDragEnd(e, true);
        });

        container.addEventListener("wheel", (e) => {
            if (!self.img) return;
            e.preventDefault();
            const delta = e.deltaY * -0.001;
            let scale = true;

            if (this._held_W) {
                // Width
                scribbleWidth.value = parseInt(scribbleWidth.value) - Math.sign(e.deltaY) * 3;
                updateInput(scribbleWidth);
                const rect = container.getBoundingClientRect();
                resetScribble(e, rect);
                scale = false;
            }
            if (this._held_A) {
                // Alpha (Opacity)
                scribbleAlpha.value = parseInt(scribbleAlpha.value) - Math.sign(e.deltaY) * 5;
                updateInput(scribbleAlpha);
                scale = false;
            }
            if (this._held_S) {
                // Softness
                scribbleSoftness.value = parseInt(scribbleSoftness.value) - Math.sign(e.deltaY) * 5;
                updateInput(scribbleSoftness);
                scale = false;
            }

            if (!scale) return;

            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const oldScale = self.imgScale;
            self.imgScale += delta;
            self.imgScale = Math.max(0.1, self.imgScale);
            const newScale = self.imgScale / oldScale;
            self.imgX = x - (x - self.imgX) * newScale;
            self.imgY = y - (y - self.imgY) * newScale;
            self.drawImage();
            resetScribble(e, rect);
        });

        container.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            self.dragged_just_now = false;
            return false;
        });

        container.addEventListener("dragleave", () => {
            toolbar.style.opacity = "0";
            imageContainer.style.cursor = "";
            drawingCanvas.style.cursor = "";
            container.style.cursor = "";
            scribbleIndicator.style.display = "none";
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        for (const e of ["dragenter", "dragover", "dragleave", "drop"]) {
            container.addEventListener(e, preventDefaults, false);
        }

        container.addEventListener("dragenter", () => {
            imageContainer.style.cursor = "copy";
            drawingCanvas.style.cursor = "copy";
        });

        container.addEventListener("dragleave", () => {
            imageContainer.style.cursor = "";
            drawingCanvas.style.cursor = "";
        });

        container.addEventListener("drop", (e) => {
            imageContainer.style.cursor = "";
            drawingCanvas.style.cursor = "";
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) self.handleFileUpload(files[0]);
        });

        container.addEventListener("pointerenter", () => {
            self.pointerInsideContainer = true;
            toolbar.style.opacity = "1";
            if (!self.img && !self.no_upload) container.style.cursor = "pointer";
        });

        container.addEventListener("pointerleave", () => {
            self.pointerInsideContainer = false;
            toolbar.style.opacity = "0";
        });

        document.addEventListener("paste", (e) => {
            if (self.pointerInsideContainer) self.handlePaste(e);
        });

        document.addEventListener("keydown", (e) => {
            if (!self.pointerInsideContainer) return;
            if (e.shiftKey) {
                e.preventDefault();
                if (this._original_alpha === null)
                    this._original_alpha = scribbleAlpha.value;
                scribbleAlpha.value = 0.0;
                updateInput(scribbleAlpha);
                scribbleIndicator.style.border = "2px dotted";
                return;
            }
            if (e.ctrlKey && e.key === "z") {
                e.preventDefault();
                this.undo();
            }
            if (e.ctrlKey && e.key === "y") {
                e.preventDefault();
                this.redo();
            }
            if (e.ctrlKey && e.key === "x") {
                e.preventDefault();
                this.resetImage();
            }
            if (e.key === "e") {
                scribbleColor.click();
            }
            if (e.key === "r") {
                centerButton.click();
            }
            if (e.key === "f") {
                if (maxButton.style.display === "none")
                    minButton.click();
                else
                    maxButton.click();
            }

            if (e.key === "w") this._held_W = true;
            if (e.key === "a") this._held_A = true;
            if (e.key === "s") this._held_S = true;
        });

        document.addEventListener("keyup", () => {
            this._held_W = false;
            this._held_A = false;
            this._held_S = false;

            if (this._original_alpha !== null) {
                scribbleAlpha.value = this._original_alpha;
                this._original_alpha = null;
                updateInput(scribbleAlpha);
                scribbleIndicator.style.border = "1px solid";
            }
        });

        maxButton.addEventListener("click", () => {
            self.maximize();
        });

        minButton.addEventListener("click", () => {
            self.minimize();
        });

        self.updateUndoRedoButtons();

        self.background_gradio_bind.listen((value) => {
            self.loadImage(value);
        });

        self.foreground_gradio_bind.listen((value) => {
            self.loadDrawing(value);
        });
    }

    handleDraw(e) {
        const canvas = this.drawingCanvas_;
        const ctx = canvas.getContext("2d");
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.imgScale;
        const y = (e.clientY - rect.top) / this.imgScale;

        this.temp_draw_points.push([x, y]);
        ctx.putImageData(this.temp_draw_bg, 0, 0);
        ctx.beginPath();
        ctx.moveTo(this.temp_draw_points[0][0], this.temp_draw_points[0][1]);

        for (let i = 1; i < this.temp_draw_points.length; i++) {
            ctx.lineTo(this.temp_draw_points[i][0], this.temp_draw_points[i][1]);
        }

        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = this.scribbleWidth / (this.scribbleWidthConsistent ? this.imgScale : 1.0) * 4;

        if (this.scribbleAlpha <= 0) {
            ctx.globalCompositeOperation = "destination-out";
            ctx.globalAlpha = 1.0;
            ctx.stroke();
            return;
        }

        ctx.globalCompositeOperation = "source-over";

        if (this.contrast_scribbles) {
            ctx.strokeStyle = this.contrast_pattern;
            ctx.stroke();
            return;
        }

        ctx.strokeStyle = this.scribbleColor;

        canvas.style.opacity = 1.0;
        let drawingAlpha = this.scribbleAlpha;

        if (this.scribbleAlphaFixed) {
            canvas.style.opacity = this.scribbleAlpha / 100.0;
            drawingAlpha = 100.0;
        }

        if (this.scribbleSoftness <= 0) {
            ctx.save();
            ctx.globalCompositeOperation = "destination-out";
            ctx.globalAlpha = 1.0;
            ctx.stroke();
            ctx.restore();

            ctx.globalCompositeOperation = "source-over";
            ctx.globalAlpha = drawingAlpha / 100.0;
            ctx.stroke();
            return;
        }

        const innerWidth = ctx.lineWidth * (1 - this.scribbleSoftness / 96);
        const outerWidth = ctx.lineWidth * (1 + this.scribbleSoftness / 96);
        const steps = Math.round(5 + this.scribbleSoftness / 5);
        const stepWidth = (outerWidth - innerWidth) / (steps - 1);

        ctx.globalAlpha = 1.0 - Math.pow(1.0 - Math.min(drawingAlpha / 100, 0.95), 1.0 / steps);

        for (let i = 0; i < steps; i++) {
            ctx.lineWidth = innerWidth + stepWidth * i;
            ctx.stroke();
        }
    }

    handleFileUpload(file) {
        if (file && !this.no_upload) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.loadImage(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    }

    handlePaste(e) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.type.indexOf("image") !== -1) {
                const file = item.getAsFile();
                this.handleFileUpload(file);
                break;
            }
        }
    }

    loadImage(base64) {
        if (typeof this.gradio_config !== "undefined") {
            if (!this.gradio_config.version.startsWith("4.")) return;
        } else {
            return;
        }

        const image = new Image();
        image.onload = () => {
            this.img = base64;
            this.orgWidth = image.width;
            this.orgHeight = image.height;
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            if (canvas.width !== image.width || canvas.height !== image.height) {
                canvas.width = image.width;
                canvas.height = image.height;
            }
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.updateBackgroundImageData();
            this.saveState();
            this.updateUndoRedoButtons();
            document.getElementById(`imageInput_${this.uuid}`).value = null;
            document.getElementById(`uploadHint_${this.uuid}`).style.display = "none";
        };

        if (base64) {
            image.src = base64;
        } else {
            this.img = null;
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            canvas.width = 1;
            canvas.height = 1;
            this.adjustInitialPositionAndScale();
            this.drawImage();
            this.saveState();
            this.updateUndoRedoButtons();
        }
    }

    loadDrawing(base64) {
        const image = new Image();
        image.onload = () => {
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0);
            this.saveState();
        };
        if (base64) {
            image.src = base64;
        } else {
            const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            this.saveState();
        }
    }

    isInsideImage(x, y) {
        const scaledWidth = this.orgWidth * this.imgScale;
        const scaledHeight = this.orgHeight * this.imgScale;
        return x > this.imgX && x < this.imgX + scaledWidth && y > this.imgY && y < this.imgY + scaledHeight;
    }

    drawImage() {
        const image = document.getElementById(`image_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        if (this.img) {
            const scaledWidth = this.orgWidth * this.imgScale;
            const scaledHeight = this.orgHeight * this.imgScale;
            image.src = this.img;
            image.style.width = `${scaledWidth}px`;
            image.style.height = `${scaledHeight}px`;
            image.style.left = `${this.imgX}px`;
            image.style.top = `${this.imgY}px`;
            image.style.display = "block";
            drawingCanvas.style.width = `${scaledWidth}px`;
            drawingCanvas.style.height = `${scaledHeight}px`;
            drawingCanvas.style.left = `${this.imgX}px`;
            drawingCanvas.style.top = `${this.imgY}px`;
        } else {
            image.src = "";
            image.style.display = "none";
        }
    }

    adjustInitialPositionAndScale() {
        const container = document.getElementById(`container_${this.uuid}`);
        const containerWidth = container.clientWidth - 20;
        const containerHeight = container.clientHeight - 20;
        const scaleX = containerWidth / this.orgWidth;
        const scaleY = containerHeight / this.orgHeight;
        this.imgScale = Math.min(scaleX, scaleY);
        const scaledWidth = this.orgWidth * this.imgScale;
        const scaledHeight = this.orgHeight * this.imgScale;
        this.imgX = (container.clientWidth - scaledWidth) / 2;
        this.imgY = (container.clientHeight - scaledHeight) / 2;
    }

    resetImage() {
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.adjustInitialPositionAndScale();
        this.drawImage();
        this.saveState();
    }

    removeImage() {
        this.img = null;
        const image = document.getElementById(`image_${this.uuid}`);
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        image.src = "";
        image.style.width = "0";
        image.style.height = "0";
        this.saveState();
        if (!this.no_upload) {
            document.getElementById(`uploadHint_${this.uuid}`).style.display = "inline-block";
        }
        this.loadImage(null);
    }

    saveState() {
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(imageData);
        this.historyIndex++;
        this.updateUndoRedoButtons();
        this.updateDrawingData();
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.restoreState();
            this.updateUndoRedoButtons();
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.restoreState();
            this.updateUndoRedoButtons();
        }
    }

    restoreState() {
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const ctx = canvas.getContext("2d");
        const imageData = this.history[this.historyIndex];
        ctx.putImageData(imageData, 0, 0);
        this.updateDrawingData();
    }

    updateUndoRedoButtons() {
        const undoButton = document.getElementById(`undoButton_${this.uuid}`);
        const redoButton = document.getElementById(`redoButton_${this.uuid}`);
        undoButton.disabled = this.historyIndex <= 0;
        redoButton.disabled = this.historyIndex >= this.history.length - 1;
        undoButton.style.opacity = undoButton.disabled ? "0.5" : "1";
        redoButton.style.opacity = redoButton.disabled ? "0.5" : "1";
    }

    updateBackgroundImageData() {
        if (!this.img) {
            this.background_gradio_bind.set_value("");
            return;
        }
        const image = document.getElementById(`image_${this.uuid}`);
        const tempCanvas = this.temp_canvas;
        const tempCtx = tempCanvas.getContext("2d");
        tempCanvas.width = this.orgWidth;
        tempCanvas.height = this.orgHeight;
        tempCtx.drawImage(image, 0, 0, this.orgWidth, this.orgHeight);
        const dataUrl = tempCanvas.toDataURL("image/png");
        this.background_gradio_bind.set_value(dataUrl);
    }

    updateDrawingData() {
        if (!this.img) {
            this.foreground_gradio_bind.set_value("");
            return;
        }
        const canvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        const dataUrl = canvas.toDataURL("image/png");
        this.foreground_gradio_bind.set_value(dataUrl);
    }

    maximize() {
        if (this.maximized) return;
        const container = document.getElementById(`container_${this.uuid}`);
        const maxButton = document.getElementById(`maxButton_${this.uuid}`);
        const minButton = document.getElementById(`minButton_${this.uuid}`);

        this.originalState = {
            width: container.style.width,
            height: container.style.height,
            top: container.style.top,
            left: container.style.left,
            position: container.style.position,
            zIndex: container.style.zIndex,
        };

        container.style.width = "100vw";
        container.style.height = "100vh";
        container.style.top = "0";
        container.style.left = "0";
        container.style.position = "fixed";
        container.style.zIndex = "1000";
        maxButton.style.display = "none";
        minButton.style.display = "inline-block";
        this.maximized = true;
    }

    minimize() {
        if (!this.maximized) return;
        const container = document.getElementById(`container_${this.uuid}`);
        const maxButton = document.getElementById(`maxButton_${this.uuid}`);
        const minButton = document.getElementById(`minButton_${this.uuid}`);

        container.style.width = this.originalState.width;
        container.style.height = this.originalState.height;
        container.style.top = this.originalState.top;
        container.style.left = this.originalState.left;
        container.style.position = this.originalState.position;
        container.style.zIndex = this.originalState.zIndex;
        maxButton.style.display = "inline-block";
        minButton.style.display = "none";
        this.maximized = false;
    }

    handleDragEnd(e, isPointerOut) {
        const image = document.getElementById(`image_${this.uuid}`);
        const drawingCanvas = document.getElementById(`drawingCanvas_${this.uuid}`);
        this.dragging = false;
        image.style.cursor = "grab";
        drawingCanvas.style.cursor = "grab";
    }
}

const True = true;
const False = false;
