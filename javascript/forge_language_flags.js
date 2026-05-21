// Inject SVG flag icons into the #forge_ui_language quicksettings dropdown.
//
// Gradio 4 Dropdown labels are plain text and cannot hold raw HTML, so we
// can't put an <img> directly into a choice tuple. This script:
//   1. On page load, pins a flag indicator next to the closed dropdown
//      input matching the currently-selected locale.
//   2. When the user opens the dropdown, prepends a flag <img> to each
//      generated <li> option.
//
// No document-wide MutationObserver: tab switches in Forge produce a lot
// of DOM churn, and watching everything would tank scroll/repaint perf.
// We only touch the DOM during page load and when the user actually
// clicks the language dropdown.

(function () {
    "use strict";

    // Recognised by displayed dropdown label text (LANGUAGE_DISPLAY in
    // modules_forge/main_entry.py). "English" gets the UK Union Jack —
    // the most internationally recognised indicator for the English
    // language. (The previous globe icon was technically not a flag.)
    const LABEL_TO_CODE = {
        "English": "uk",
        "Italiano": "it",
        "Español": "es",
        "Français": "fr",
        "Deutsch": "de",
        "简体中文": "cn",
        "日本語": "jp",
    };

    // Each entry is a compact SVG with a roughly 3:2 aspect ratio,
    // encoded as a data URI so no static file fetch is required.
    const FLAG_SVG = {
        uk:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 30">' +
                    '<clipPath id="t"><path d="M30,15 h30 v15 z v15 h-30 z h-30 v-15 z v-15 h30 z"/></clipPath>' +
                    '<path d="M0,0 v30 h60 v-30 z" fill="#012169"/>' +
                    '<path d="M0,0 L60,30 M60,0 L0,30" stroke="#fff" stroke-width="6"/>' +
                    '<path d="M0,0 L60,30 M60,0 L0,30" clip-path="url(#t)" stroke="#C8102E" stroke-width="4"/>' +
                    '<path d="M30,0 v30 M0,15 h60" stroke="#fff" stroke-width="10"/>' +
                    '<path d="M30,0 v30 M0,15 h60" stroke="#C8102E" stroke-width="6"/>' +
                    "</svg>",
            ),
        globe:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">' +
                    '<circle cx="12" cy="12" r="10" fill="#4a90e2"/>' +
                    '<path d="M2 12h20 M12 2c4 5 4 15 0 20 M12 2c-4 5 -4 15 0 20" stroke="#fff" stroke-width="1.2" fill="none"/>' +
                    "</svg>",
            ),
        it:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3 2">' +
                    '<rect width="1" height="2" fill="#009246"/>' +
                    '<rect x="1" width="1" height="2" fill="#fff"/>' +
                    '<rect x="2" width="1" height="2" fill="#ce2b37"/>' +
                    "</svg>",
            ),
        es:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3 2">' +
                    '<rect width="3" height="2" fill="#aa151b"/>' +
                    '<rect y="0.5" width="3" height="1" fill="#f1bf00"/>' +
                    "</svg>",
            ),
        fr:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3 2">' +
                    '<rect width="1" height="2" fill="#0055a4"/>' +
                    '<rect x="1" width="1" height="2" fill="#fff"/>' +
                    '<rect x="2" width="1" height="2" fill="#ef4135"/>' +
                    "</svg>",
            ),
        de:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 5 3">' +
                    '<rect width="5" height="1" fill="#000"/>' +
                    '<rect y="1" width="5" height="1" fill="#dd0000"/>' +
                    '<rect y="2" width="5" height="1" fill="#ffce00"/>' +
                    "</svg>",
            ),
        cn:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">' +
                    '<rect width="30" height="20" fill="#de2910"/>' +
                    '<polygon points="6,3 7.18,5.78 10,5.78 7.71,7.55 8.71,10.33 6,8.7 3.29,10.33 4.29,7.55 2,5.78 4.82,5.78" fill="#ffde00"/>' +
                    "</svg>",
            ),
        jp:
            "data:image/svg+xml;utf8," +
            encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">' +
                    '<rect width="30" height="20" fill="#fff"/>' +
                    '<circle cx="15" cy="10" r="6" fill="#bc002d"/>' +
                    "</svg>",
            ),
    };

    function codeFromLabel(text) {
        if (!text) return null;
        const trimmed = text.trim();
        if (LABEL_TO_CODE[trimmed]) return LABEL_TO_CODE[trimmed];
        // Fallback: substring match — but ONLY against the known autoglottonym
        // set, so we never inject flags into random UI text.
        for (const label of Object.keys(LABEL_TO_CODE)) {
            if (trimmed.indexOf(label) !== -1) return LABEL_TO_CODE[label];
        }
        return null;
    }

    function makeFlagImg(code) {
        const img = document.createElement("img");
        img.src = FLAG_SVG[code];
        img.className = "forge-flag";
        img.alt = "";
        img.setAttribute("aria-hidden", "true");
        return img;
    }

    function decorateOption(el) {
        if (!el || el.dataset.forgeFlagApplied) return;
        const code = codeFromLabel(el.textContent);
        if (!code) return;
        el.insertBefore(makeFlagImg(code), el.firstChild);
        el.dataset.forgeFlagApplied = "1";
    }

    // Decorate the option <li> elements inside the open dropdown menu.
    // Gradio 4 generates a ul.options keyed by the input's aria-controls
    // attribute; sometimes it is mounted as a child of the dropdown root,
    // sometimes it is detached and lives elsewhere in the body. We also
    // tag the ul with `.forge-language-options` so style.css can scope
    // its panel padding without affecting any other dropdown on the page.
    function decorateOpenList(dropdown) {
        if (!dropdown) return;
        const input = dropdown.querySelector(
            "input[role='listbox'], input[role='combobox']",
        );
        const controlledId = input && input.getAttribute("aria-controls");
        let ul = null;
        if (controlledId) {
            ul = document.getElementById(controlledId);
        }
        if (!ul) {
            ul = dropdown.querySelector("ul.options");
        }
        if (!ul) return;
        ul.classList.add("forge-language-options");
        ul.querySelectorAll("li.item, [role='option']").forEach(decorateOption);
    }

    function updateInputIndicator(dropdown) {
        if (!dropdown) return;
        const input = dropdown.querySelector(
            "input[role='listbox'], input[role='combobox']",
        );
        if (!input) return;
        // Anchor the indicator to the input's immediate row, not to the
        // outer dropdown container — the container also wraps the
        // "Lingua" label above the input, which would offset any
        // absolute positioning. By appending inside `.wrap` we get a
        // box whose vertical centre matches the input's vertical centre.
        const row = input.closest(".wrap") || input.parentElement;
        if (!row) return;
        const code = codeFromLabel(input.value);
        const existing = row.querySelector(":scope > .forge-flag-indicator");
        if (existing && existing.dataset.forgeFlagCode === code) return;
        if (existing) existing.remove();
        if (!code) return;
        const indicator = document.createElement("span");
        indicator.className = "forge-flag-indicator";
        indicator.dataset.forgeFlagCode = code;
        indicator.appendChild(makeFlagImg(code));
        row.appendChild(indicator);
    }

    function bindDropdown(dropdown) {
        if (!dropdown || dropdown.dataset.forgeFlagBound) return;
        dropdown.dataset.forgeFlagBound = "1";

        updateInputIndicator(dropdown);

        // Re-decorate the options list each time the user opens or types
        // into the dropdown. A small delay gives Svelte time to mount the ul.
        const refreshOpen = () => {
            // Run twice (next frame + a short delay) because Gradio sometimes
            // remounts the ul a beat after the click registers.
            requestAnimationFrame(() => decorateOpenList(dropdown));
            setTimeout(() => decorateOpenList(dropdown), 120);
        };
        dropdown.addEventListener("click", refreshOpen);
        const input = dropdown.querySelector(
            "input[role='listbox'], input[role='combobox']",
        );
        if (input) {
            input.addEventListener("focus", refreshOpen);
            input.addEventListener("input", refreshOpen);
            // Update the input-side indicator if the value ever changes
            // without a full restart_reload.
            input.addEventListener("change", () =>
                updateInputIndicator(dropdown),
            );
        }
    }

    function tryBind() {
        const dropdown = document.querySelector("#forge_ui_language");
        if (dropdown) {
            bindDropdown(dropdown);
            return true;
        }
        return false;
    }

    function arm() {
        if (tryBind()) return;
        // The Svelte UI mounts after DOMContentLoaded — poll briefly
        // until the dropdown shows up, then stop. No long-running observer.
        let attempts = 0;
        const maxAttempts = 80; // ~8 seconds at 100ms
        const handle = setInterval(() => {
            attempts++;
            if (tryBind() || attempts >= maxAttempts) {
                clearInterval(handle);
            }
        }, 100);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", arm, { once: true });
    } else {
        arm();
    }
})();
