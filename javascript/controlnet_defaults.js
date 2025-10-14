// Auto-set ControlNet defaults on page load
// This script sets default preprocessor and model for all ControlNet units

(function() {
    'use strict';

    const DEFAULT_PREPROCESSOR = "lineart_standard (from white bg & black line)";
    const DEFAULT_MODEL = "mistoline_v10 [0f320932]";

    console.log("[ControlNet Defaults] Script loaded");

    function updateGradioDropdown(dropdown, value) {
        // Find the input element
        const input = dropdown.querySelector('input');
        if (!input) return false;

        // Set the value
        input.value = value;

        // Trigger all necessary events to update Gradio's internal state
        const events = ['input', 'change', 'blur'];
        events.forEach(eventType => {
            const event = new Event(eventType, { bubbles: true, cancelable: true });
            input.dispatchEvent(event);
        });

        // Also try triggering with InputEvent for better compatibility
        const inputEvent = new InputEvent('input', { bubbles: true, cancelable: true, data: value });
        input.dispatchEvent(inputEvent);

        // Focus and blur to ensure Gradio processes the change
        input.focus();
        setTimeout(() => input.blur(), 50);

        return true;
    }

    function setControlNetDefaults() {
        console.log("[ControlNet Defaults] Starting to set defaults...");

        // Find all ControlNet preprocessor and model dropdowns
        const preprocessorDropdowns = document.querySelectorAll('[id*="controlnet_preprocessor_dropdown"]');
        const modelDropdowns = document.querySelectorAll('[id*="controlnet_model_dropdown"]');

        console.log(`[ControlNet Defaults] Found ${preprocessorDropdowns.length} preprocessor dropdowns`);
        console.log(`[ControlNet Defaults] Found ${modelDropdowns.length} model dropdowns`);

        // Set preprocessor defaults first
        preprocessorDropdowns.forEach((dropdown, index) => {
            const input = dropdown.querySelector('input');
            if (input && (input.value === "None" || input.value === "")) {
                console.log(`[ControlNet Defaults] Setting preprocessor ${index} to: ${DEFAULT_PREPROCESSOR}`);
                updateGradioDropdown(dropdown, DEFAULT_PREPROCESSOR);
            }
        });

        // Wait a bit before setting models to ensure preprocessor changes are processed
        setTimeout(() => {
            modelDropdowns.forEach((dropdown, index) => {
                const input = dropdown.querySelector('input');
                if (input && (input.value === "None" || input.value === "")) {
                    console.log(`[ControlNet Defaults] Setting model ${index} to: ${DEFAULT_MODEL}`);
                    updateGradioDropdown(dropdown, DEFAULT_MODEL);
                }
            });
            console.log("[ControlNet Defaults] Defaults set complete");
        }, 500);
    }

    // Try setting defaults after page load with multiple attempts
    window.addEventListener('load', function() {
        console.log("[ControlNet Defaults] Page loaded, setting defaults...");

        // Initial attempt
        setTimeout(setControlNetDefaults, 2000);

        // Backup attempt in case first one fails
        setTimeout(setControlNetDefaults, 4000);
    });

    // Also try after Gradio finishes initializing
    if (window.gradioApp) {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes.length > 0) {
                    const controlnetElements = document.querySelectorAll('[id*="controlnet"]');
                    if (controlnetElements.length > 0) {
                        console.log("[ControlNet Defaults] ControlNet elements detected, setting defaults...");
                        setTimeout(setControlNetDefaults, 1500);
                        observer.disconnect();
                    }
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
})();
