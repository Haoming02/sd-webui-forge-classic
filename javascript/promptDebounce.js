(function () {
    /** @type {Map<HTMLTextAreaElement, number>} */
    const pending = new Map();

    /** @type {WeakMap<HTMLTextAreaElement, InputEventInit>} */
    const lastInput = new WeakMap();

    /** @type {WeakSet<HTMLTextAreaElement>} */
    const dispatching = new WeakSet();

    class DebounceWatcher {
        /** @param {string} id @param {number} delay */
        constructor(id, delay) {
            /** @type {HTMLTextAreaElement} */
            this.textarea = document.querySelector(`#${id} textarea`);
            this.delay = delay;

            this.textarea.addEventListener("input", (e) => this.#onInput(e), true);
            this.textarea.addEventListener("blur", () => this.#flush(), true);
        }

        #onInput(event) {
            if (dispatching.has(this.textarea)) return;

            const timer = pending.get(this.textarea);
            if (timer !== undefined) {
                clearTimeout(timer);
                pending.delete(this.textarea);
            }

            if (!event.inputType) return;
            event.stopImmediatePropagation();

            lastInput.set(this.textarea, {
                inputType: event.inputType,
                data: event.data,
                isComposing: event.isComposing,
            });

            pending.set(
                this.textarea,
                setTimeout(() => this.#flush(), this.delay),
            );
        }

        #flush() {
            const timer = pending.get(this.textarea);
            if (timer === undefined) return;

            clearTimeout(timer);
            pending.delete(this.textarea);

            const init = lastInput.get(this.textarea);
            lastInput.delete(this.textarea);

            const event = init?.inputType
                ? new InputEvent("input", {
                    bubbles: true,
                    ...init,
                })
                : new Event("input", {
                    bubbles: true,
                });

            dispatching.add(this.textarea);
            try {
                this.textarea.dispatchEvent(event);
            } finally {
                dispatching.delete(this.textarea);
            }
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

        for (const id of IDs) new DebounceWatcher(id, opts.prompt_debounce);
    }

    onOptionsAvailable(() => { if (opts.prompt_debounce) setup(); });
})();
