(function () {
    /** @param {HTMLDivElement} gradioImage */
    function patchDragAndDrop(gradioImage) {
        gradioImage.addEventListener("dragover", (e) => {
            const dt = e.dataTransfer;
            const isDroppingImage = dt.types.includes("text/uri-list") || dt.types.includes("text/html");
            if (!isDroppingImage) return;

            const closeButton = gradioImage.querySelector('button[aria-label="Remove Image"]');
            if (closeButton) closeButton.click();
        });
    }

    function setup() {
        if (opts.remove_image_on_hover === false) return;
        if (opts.remove_image_on_hover === undefined) {
            setTimeout(setup, 50);
            return;
        }

        for (const id of ["extras_image", "pnginfo_image"]) patchDragAndDrop(document.getElementById(id));
    }

    onUiLoaded(() => {
        setTimeout(setup, 100);
    });
})();
