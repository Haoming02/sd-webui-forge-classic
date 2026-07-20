// Stable Diffusion WebUI - Bracket Checker
// By @Bwin4L, @akx, @w-e-w, @Haoming02
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red, and if you hover on it, a tooltip tells you what's wrong.

(function () {
    const pairs = Object.freeze([
        ["(", ")", "round brackets"],
        ["[", "]", "square brackets"],
        ["{", "}", "curly brackets"],
    ]);

    function checkBrackets(textArea, counterElem) {
        const counts = [0, 0, 0];
        const errors = [];
        const text = textArea.value;
        let i = 0;

        while (i < text.length) {
            let char = text[i];
            let backslash = 0;

            while (char === "\\" && i + 1 < text.length) {
                i++;
                backslash++;
                char = text[i];
            }

            if (backslash % 2 === 1) {
                i++;
                continue;
            }

            for (const [idx, [open, close, label]] of pairs.entries()) {
                if (char === open) {
                    counts[idx]++;
                } else if (char === close) {
                    counts[idx]--;
                    if (counts[idx] < 0) errors.push(`Incorrect order of ${label}.`);
                }
            }

            i++;
        }

        for (const [idx, [open, close, label]] of pairs.entries()) {
            if (counts[idx] === 0) continue;
            else if (counts[idx] > 0)
                errors.push(`${open} ... ${close} - Detected ${counts[idx]} more opening than closing ${label}.`);
            else if (counts[idx] < 0)
                errors.push(`${open} ... ${close} - Detected ${-counts[idx]} more closing than opening ${label}.`);
        }

        counterElem.title = [...errors].join("\n");
        counterElem.classList.toggle("error", errors.length > 0);
    }

    function setupBracketChecking(id_prompt, id_counter) {
        const textarea = gradioApp().querySelector(`#${id_prompt} > label > textarea`);
        const counter = gradioApp().getElementById(id_counter);

        if (textarea && counter)
            onEdit(`${id_prompt}_BracketChecking`, textarea, 1000, () => checkBrackets(textarea, counter));
    }

    onUiLoaded(() => {
        setupBracketChecking("txt2img_prompt", "txt2img_token_counter");
        setupBracketChecking("txt2img_neg_prompt", "txt2img_negative_token_counter");
        setupBracketChecking("img2img_prompt", "img2img_token_counter");
        setupBracketChecking("img2img_neg_prompt", "img2img_negative_token_counter");
    });
})();
