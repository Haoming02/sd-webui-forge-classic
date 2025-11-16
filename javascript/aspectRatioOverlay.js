let currentWidth;
let currentHeight;
let arFrameTimeout;

function dimensionChange(e, is_width, is_height) {
    if (is_width) {
        currentWidth = e.target.value * 1.0;
    }
    if (is_height) {
        currentHeight = e.target.value * 1.0;
    }

    let inImg2img = gradioApp().querySelector("#tab_img2img").style.display == "block";

    if (!inImg2img) {
        return;
    }

    let targetElement = null;

    let tabIndex = get_tab_index("mode_img2img");
    if (tabIndex == 0) {
        // img2img
        targetElement = gradioApp().querySelector(
            "#img2img_image div.forge-image-container img",
        );
    } else if (tabIndex == 1) {
        // Sketch
        targetElement = gradioApp().querySelector(
            "#img2img_sketch div.forge-image-container img",
        );
    } else if (tabIndex == 2) {
        // Inpaint
        targetElement = gradioApp().querySelector(
            "#img2maskimg div.forge-image-container img",
        );
    } else if (tabIndex == 3) {
        // Inpaint sketch
        targetElement = gradioApp().querySelector(
            "#inpaint_sketch div.forge-image-container img",
        );
    } else if (tabIndex == 4) {
        // Inpaint upload
        targetElement = gradioApp().querySelector(
            "#img_inpaint_base div[data-testid=image] img",
        );
    }

    if (targetElement) {
        let arPreviewRect = gradioApp().querySelector("#imageARPreview");
        if (!arPreviewRect) {
            arPreviewRect = document.createElement("div");
            arPreviewRect.id = "imageARPreview";
            gradioApp().appendChild(arPreviewRect);
        }

        let viewportOffset = targetElement.getBoundingClientRect();
        let viewportscale = Math.min(
            targetElement.clientWidth / targetElement.width,
            targetElement.clientHeight / targetElement.height,
        );

        let scaledx = targetElement.width * viewportscale;
        let scaledy = targetElement.height * viewportscale;

        let clientRectTop = viewportOffset.top + window.scrollY;
        let clientRectLeft = viewportOffset.left + window.scrollX;
        let clientRectCentreY = clientRectTop + targetElement.clientHeight / 2;
        let clientRectCentreX = clientRectLeft + targetElement.clientWidth / 2;

        let arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
        let arscaledx = currentWidth * arscale;
        let arscaledy = currentHeight * arscale;

        let arRectTop = clientRectCentreY - arscaledy / 2;
        let arRectLeft = clientRectCentreX - arscaledx / 2;
        let arRectWidth = arscaledx;
        let arRectHeight = arscaledy;

        arPreviewRect.style.top = arRectTop + "px";
        arPreviewRect.style.left = arRectLeft + "px";
        arPreviewRect.style.width = arRectWidth + "px";
        arPreviewRect.style.height = arRectHeight + "px";

        clearTimeout(arFrameTimeout);
        arFrameTimeout = setTimeout(function () {
            arPreviewRect.style.display = "none";
        }, 2000);

        arPreviewRect.style.display = "block";
    }
}

onAfterUiUpdate(function () {
    let arPreviewRect = gradioApp().querySelector("#imageARPreview");
    if (arPreviewRect) {
        arPreviewRect.style.display = "none";
    }

    let tabImg2img = gradioApp().querySelector("#tab_img2img");
    if (tabImg2img) {
        if (tabImg2img.style.display == "block") {
            let inputs = gradioApp().querySelectorAll("input");
            inputs.forEach(function (e) {
                let is_width =
                    (e.parentElement.id == "img2img_width" && e.type == "range") ||
                    (e.parentElement.parentElement.parentElement.id == "img2img_width" &&
                        e.type == "number");
                let is_height =
                    (e.parentElement.id == "img2img_height" && e.type == "range") ||
                    (e.parentElement.parentElement.parentElement.id == "img2img_height" &&
                        e.type == "number");

                if ((is_width || is_height) && !e.classList.contains("scrollwatch")) {
                    e.addEventListener("input", function (e) {
                        dimensionChange(e, is_width, is_height);
                    });
                    e.classList.add("scrollwatch");
                }
                if (is_width) {
                    currentWidth = e.value * 1.0;
                }
                if (is_height) {
                    currentHeight = e.value * 1.0;
                }
            });
        }
    }
});
