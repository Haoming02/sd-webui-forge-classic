(function () {
    function extractImageFromHTML(html) {
        if (!html) return null;
        const match = html.match(/<img[^>]+src="([^">]+)"/i);
        return match ? match[1] : null;
    }

    /** @param {HTMLDivElement} gradioImage */
    function patchDragAndDrop(gradioImage) {
        let isDroppingImage = false;

        gradioImage.addEventListener("dragover", (e) => {
            const dt = e.dataTransfer;
            isDroppingImage = dt.types.includes("text/uri-list") || dt.types.includes("text/html");
            if (isDroppingImage) e.preventDefault();
        });

        gradioImage.addEventListener("drop", async (e) => {
            if (!isDroppingImage) return;

            const dt = e.dataTransfer;

            const url = dt.getData("text/uri-list") || extractImageFromHTML(dt.getData("text/html"));
            if (!url) return;

            e.preventDefault();

            try {
                const closeButton = gradioImage.querySelector('button[aria-label="Remove Image"]');
                closeButton.click();

                const res = await fetch(url);
                const blob = await res.blob();

                const file = new File([blob], "dropped-image", {
                    type: blob.type || "image/png",
                });

                const newDT = new DataTransfer();
                newDT.items.add(file);

                const fileInput = gradioImage.querySelector('input[type="file"]');
                fileInput.files = newDT.files;

                fileInput.dispatchEvent(new Event("change", { bubbles: true }));
            } catch {
                console.log("Failed to replace image...");
            }
        });
    }

    onUiLoaded(() => {
        for (const id of ["extras_image", "pnginfo_image"]) patchDragAndDrop(document.getElementById(id));
    });
})();
