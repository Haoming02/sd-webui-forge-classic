(function () {
    const editTimers = {};

    const delay = 600;
    const limit = 16;

    class TextboxHistory {
        /** @param {string} id */
        constructor(id) {
            /** @type {HTMLTextAreaElement} */
            this.textarea = document.querySelector(`#${id} textarea`);
            /** @type {string[]} */
            this.undoStack = [];
            /** @type {string[]} */
            this.redoStack = [];

            this.#snapshot();

            this.textarea.addEventListener("keydown", (e) => {
                const prev = editTimers[id];

                if ((e.ctrlKey || e.metaKey) && e.key === "z") {
                    e.preventDefault();
                    if (prev) clearTimeout(prev);
                    this.#undo();
                    return false;
                }
                if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "Z") {
                    e.preventDefault();
                    if (prev) clearTimeout(prev);
                    this.#redo();
                    return false;
                }

                if (["Control", "Meta", "Shift", "Alt"].includes(e.key)) return true;

                if (prev) clearTimeout(prev);
                editTimers[id] = setTimeout(() => this.#snapshot(), delay);
            });
        }

        #undo() {
            this.#snapshot(false);

            if (this.undoStack.length < 2) return;

            const current = this.undoStack.pop();
            this.redoStack.push(current);

            const prev = this.undoStack.at(-1);
            this.textarea.value = prev;
            updateInput(this.textarea);
        }

        #redo() {
            if (this.redoStack.length < 1) return;

            const current = this.textarea.value;
            this.undoStack.push(current);

            const prev = this.redoStack.pop();
            this.textarea.value = prev;
            updateInput(this.textarea);
        }

        #snapshot(reset = true) {
            const current = this.textarea.value;
            if (current === this.undoStack.at(-1)) return;

            this.undoStack.push(current);
            if (this.undoStack.length > limit) this.undoStack.shift();
            if (reset) this.redoStack.length = 0;
        }
    }

    function setup() {
        const IDs = [
            "txt2img_prompt",
            "txt2img_neg_prompt",
            "img2img_prompt",
            "img2img_neg_prompt",
            "hires_prompt",
            "hires_neg_prompt",
        ];

        for (const id of IDs) new TextboxHistory(id);
    }

    onUiLoaded(() => {
        function checkSettings() {
            if (Object.keys(opts).length === 0) {
                setTimeout(checkSettings, 100);
                return;
            }

            if (opts.undo_redo) setup();
        }

        checkSettings();
    });
})();
