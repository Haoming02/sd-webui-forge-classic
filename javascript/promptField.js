function shushTextfield() {
    const IDs = [
        "txt2img_prompt",
        "txt2img_neg_prompt",
        "img2img_prompt",
        "img2img_neg_prompt",
        "hires_prompt",
        "hires_neg_prompt",
    ];

    for (const id of IDs) {
        const textArea = document.getElementById(id)?.querySelector("textarea");
        if (textArea == null) continue;

        textArea.setAttribute("autocorrect", "off");
        textArea.setAttribute("autocapitalize", "off");
        textArea.setAttribute("autocomplete", "off");
        textArea.setAttribute("spellcheck", false);
    }
}

onUiLoaded(() => {
    function checkSettings() {
        if (Object.keys(opts).length === 0) {
            setTimeout(checkSettings, 100);
            return;
        }

        if (opts.no_spellcheck) shushTextfield();
    }

    checkSettings();
});
