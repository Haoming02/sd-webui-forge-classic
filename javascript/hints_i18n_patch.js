// Make hints.js tooltip lookup survive label translation.
//
// hints.js builds a `titles` dict keyed by the *English* UI label
// (e.g. titles["Sampling Method"] = "The algorithm used to refine
// each step of the image"). At runtime `updateTooltip(el)` reads
// `el.textContent` and indexes into `titles` to find the tooltip.
//
// localization.js, however, replaces `el.textContent` with the
// translated label *before* hints.js gets a chance to look it up —
// so the index miss leaves the element with no tooltip on every
// non-English locale.
//
// This script runs after hints.js (alphabetical script load order:
// `hints.js` < `hints_i18n_patch.js`) and augments `titles` with the
// translated variant of every key, pointing at the same tooltip:
//
//     titles["Sampling Method"]    = "The algorithm used..."   (already there)
//     titles["Metodo di Sampling"] = "The algorithm used..."   (added by us)
//
// updateTooltip() then resolves the tooltip even when the visible
// text is translated. The Italian tooltip itself is still produced
// by the existing `localization[titles[text]]` chain inside hints.js,
// no further changes needed.
//
// Idempotent. Re-runs cheaply if window.localization is repopulated
// after a restart_reload — the new translated keys just overwrite
// the previous ones with the same value.

(function () {
    "use strict";

    const MAX_WAIT_MS = 10000;
    const POLL_MS = 100;
    let elapsed = 0;

    function attempt() {
        // hints.js declares `const titles` at top-level script scope, so it
        // is reachable from this sibling classic script.
        const dict = typeof titles !== "undefined" ? titles : null;
        const loc = window.localization;

        if (!dict || !loc || Object.keys(loc).length === 0) {
            elapsed += POLL_MS;
            if (elapsed >= MAX_WAIT_MS) return;
            setTimeout(attempt, POLL_MS);
            return;
        }

        let added = 0;
        // Snapshot the original English keys before we start mutating, so we
        // never feed a previously-added translated key back through the loop.
        const englishKeys = Object.keys(dict);
        for (const englishLabel of englishKeys) {
            const translatedLabel = loc[englishLabel];
            if (!translatedLabel) continue;
            if (translatedLabel === englishLabel) continue;
            if (dict[translatedLabel]) continue;
            dict[translatedLabel] = dict[englishLabel];
            added++;
        }

        if (added > 0) {
            console.info(
                "[forge i18n] augmented hints.js titles with " +
                    added +
                    " localized label aliases",
            );
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attempt, { once: true });
    } else {
        attempt();
    }
})();
