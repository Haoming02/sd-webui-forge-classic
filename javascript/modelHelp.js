(function () {
    let link = null;

    onUiLoaded(() => {
        const modules = document.getElementById("setting_sd_modules");
        const title = modules.querySelector("span");

        link = document.createElement("a");
        link.href = "https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models";
        link.title = "Where to get the modules?";
        link.textContent = "?";

        link.style.marginLeft = "0.5em";
        link.style.padding = "0em 0.5em";
        link.style.border = "var(--input-border-width) solid var(--input-border-color)";
        link.style.borderRadius = "var(--input-radius)";
        link.style.background = "var(--input-background-fill)";

        link.style.visibility = "hidden";
        title.appendChild(link);
    });

    onAfterUiUpdate(() => {
        if (link == null) return;

        const preset = document.getElementById("forge_ui_preset")?.querySelector("input");
        if (preset == null) return;
        if (["sd", "xl"].includes(preset.value)) {
            link.style.visibility = "hidden";
            return;
        }

        const modules = document.getElementById("setting_sd_modules");
        const input = modules?.querySelector("div.wrap-inner");
        if (input == null) return;
        link.style.visibility = input.querySelector("div.token") ? "hidden" : "visible";
    });
})();
