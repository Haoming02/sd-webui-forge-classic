(function () {
    /**
     * Gradio re-evaluates its whole component tree on every value change, which costs ~75 ms
     * per keystroke in a build this size, regardless of which field is edited or whether it is
     * even visible. Typing in a prompt therefore runs at ~11 fps.
     *
     * Only the `input` event is withheld from the listeners below, and only until the typing
     * pauses; `keydown`, `keyup`, `paste`, `focus` and `blur` are never touched, and the value
     * of the textarea itself is always current, so anything reading it directly still sees
     * every character. The pending value is flushed before it could be read by the backend:
     * on blur, on any pointer press, on a modifier shortcut, and before the page unloads.
     */

    const IDs = [
        "txt2img_prompt",
        "txt2img_neg_prompt",
        "img2img_prompt",
        "img2img_neg_prompt",
        "hires_prompt",
        "hires_neg_prompt",
    ];

    /** @type {Map<HTMLTextAreaElement, number>} */
    const pending = new Map();
    /** @type {WeakMap<HTMLTextAreaElement, InputEventInit>} */
    const lastInput = new WeakMap();
    /** @type {Set<HTMLTextAreaElement>} */
    let targets = new Set();
    let delay = 0;

    /** @param {HTMLTextAreaElement} textarea */
    function flush(textarea) {
        const timer = pending.get(textarea);
        if (timer === undefined) return;

        clearTimeout(timer);
        pending.delete(textarea);

        // re-emit with the fields of the last real keystroke: listeners such as tag autocompletion
        // ignore `input` events without an `inputType` (that is how they skip programmatic updates)
        const init = lastInput.get(textarea);
        const event = init?.inputType ? new InputEvent("input", { bubbles: true, ...init }) : new Event("input", { bubbles: true });

        textarea.dataset.debouncedInput = "1";
        textarea.dispatchEvent(event);
    }

    function flushAll() {
        for (const textarea of Array.from(pending.keys())) flush(textarea);
    }

    function onInput(event) {
        const textarea = event.target;
        if (!targets.has(textarea)) return;

        if (textarea.dataset.debouncedInput) {
            // the event this module dispatched itself; let every listener handle it
            delete textarea.dataset.debouncedInput;
            return;
        }

        if (!event.inputType) {
            // programmatic edit (`updateInput()`: Ctrl+Up/Down, undo, extra networks cards...):
            // one event per action, let it through; it also carries any keystrokes still pending
            const timer = pending.get(textarea);
            if (timer !== undefined) clearTimeout(timer);
            pending.delete(textarea);
            return;
        }

        event.stopPropagation();
        lastInput.set(textarea, { inputType: event.inputType, data: event.data, isComposing: event.isComposing });

        const timer = pending.get(textarea);
        if (timer !== undefined) clearTimeout(timer);
        pending.set(textarea, setTimeout(() => flush(textarea), delay));
    }

    function setup() {
        targets = new Set(IDs.map((id) => document.querySelector(`#${id} textarea`)).filter(Boolean));
        if (targets.size === 0) return;

        document.addEventListener("input", onInput, true);
        document.addEventListener("blur", (e) => flush(e.target), true);
        document.addEventListener("pointerdown", flushAll, true);
        document.addEventListener(
            "keydown",
            (e) => {
                if (e.ctrlKey || e.altKey || e.metaKey) flushAll();
            },
            true,
        );
        window.addEventListener("beforeunload", flushAll);
    }

    onUiLoaded(() => {
        onOptionsAvailable(() => {
            delay = opts.prompt_debounce ?? 0;
            if (delay > 0) setup();
        });
    });
})();
