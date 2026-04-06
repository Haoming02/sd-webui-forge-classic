(function () {
    const boundTabs = new WeakSet();
    const scheduledTabs = new WeakSet();

    function getDropdownValue(tab, selector) {
        const dropdown = tab.querySelector(selector);
        if (!dropdown) return "";

        const input = dropdown.querySelector("input");
        if (input?.value) return input.value.trim();

        const selected = dropdown.querySelector(".single-select, .selected, .checked");
        if (selected?.textContent) return selected.textContent.trim();

        return dropdown.textContent?.trim() || "";
    }

    function isReadOnlyPreview(tab) {
        const moduleValue = getDropdownValue(tab, '[id$="_controlnet_preprocessor_dropdown"]');
        const modelValue = getDropdownValue(tab, '[id$="_controlnet_model_dropdown"]');
        return /openpose/i.test(`${moduleValue} ${modelValue}`);
    }

    function syncPreviewState(tab) {
        const generatedGroup = tab.querySelector(".cnet-generated-image-group");
        if (!generatedGroup) return;

        const readOnly = isReadOnlyPreview(tab);
        generatedGroup.classList.toggle("cnet-preview-readonly", readOnly);

        const hint = generatedGroup.querySelector(".cnet-preview-edit-hint");
        if (hint) {
            hint.textContent = readOnly
                ? "OpenPose preview is read-only here. Use Edit to adjust the pose."
                : "Paint or erase directly on this preview before enabling Preview as Input.";
        }
    }

    function scheduleSync(tab) {
        if (scheduledTabs.has(tab)) return;
        scheduledTabs.add(tab);
        requestAnimationFrame(() => {
            scheduledTabs.delete(tab);
            syncPreviewState(tab);
        });
    }

    function bindTab(tab) {
        if (boundTabs.has(tab)) return;
        boundTabs.add(tab);

        const observer = new MutationObserver(() => scheduleSync(tab));
        observer.observe(tab, {
            subtree: true,
            childList: true,
            attributes: true,
            characterData: true,
        });

        for (const input of tab.querySelectorAll('[id$="_controlnet_preprocessor_dropdown"] input, [id$="_controlnet_model_dropdown"] input')) {
            for (const eventName of ["input", "change", "blur"]) {
                input.addEventListener(eventName, () => scheduleSync(tab));
            }
        }

        tab.addEventListener("click", () => scheduleSync(tab));
        scheduleSync(tab);
    }

    function scanTabs() {
        for (const tab of document.querySelectorAll("#txt2img_controlnet #controlnet .tabitem, #img2img_controlnet #controlnet .tabitem")) {
            bindTab(tab);
            scheduleSync(tab);
        }
    }

    onUiLoaded(scanTabs);
    onUiUpdate(scanTabs);
})();