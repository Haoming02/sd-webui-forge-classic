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
        for (const id of ["extras_image", "pnginfo_image"]) patchDragAndDrop(document.getElementById(id));
    }

    onOptionsAvailable(() => { if (opts.remove_image_on_hover) setup(); });
})();
