let currentWidth;
let currentHeight;
let arFrameTimeout;
let arPreviewExpiresAt = 0;

function hideArPreview() {
    let arPreviewRect = gradioApp().querySelector("#imageARPreview");
    if (arPreviewRect) {
        arPreviewRect.style.display = "none";
    }
}

function isVisible(element) {
    return !!(element && element.offsetParent !== null);
}

function getSliderValue(sliderId) {
    let sliderRoot = gradioApp().getElementById(sliderId);
    if (!sliderRoot) {
        return null;
    }

    let numberInput = sliderRoot.querySelector('input[type="number"]');
    let rangeInput = sliderRoot.querySelector('input[type="range"]');
    let value = parseFloat(
        (numberInput && numberInput.value) ||
            (rangeInput && rangeInput.value) ||
            "",
    );

    return Number.isFinite(value) ? value : null;
}

function getDisplayedMediaRect(targetElement) {
    if (!targetElement) {
        return null;
    }

    let sourceWidth =
        targetElement.naturalWidth || targetElement.videoWidth || targetElement.width;
    let sourceHeight =
        targetElement.naturalHeight || targetElement.videoHeight || targetElement.height;

    if (
        !sourceWidth ||
        !sourceHeight ||
        !targetElement.clientWidth ||
        !targetElement.clientHeight
    ) {
        return null;
    }

    let boxWidth = targetElement.clientWidth;
    let boxHeight = targetElement.clientHeight;
    let drawWidth = boxWidth;
    let drawHeight = boxHeight;

    if (targetElement.tagName !== "CANVAS") {
        let sourceAspect = sourceWidth / sourceHeight;
        let boxAspect = boxWidth / boxHeight;

        if (sourceAspect > boxAspect) {
            drawHeight = boxWidth / sourceAspect;
        } else {
            drawWidth = boxHeight * sourceAspect;
        }
    }

    let viewportOffset = targetElement.getBoundingClientRect();

    return {
        top: viewportOffset.top + window.scrollY + (boxHeight - drawHeight) / 2,
        left: viewportOffset.left + window.scrollX + (boxWidth - drawWidth) / 2,
        width: drawWidth,
        height: drawHeight,
    };
}

function getActiveImg2imgTargetElement() {
    let tabImg2img = gradioApp().querySelector("#tab_img2img");
    if (!isVisible(tabImg2img)) {
        return null;
    }

    let tabSelectors = [
        "#img2img_image",
        "#img2img_sketch",
        "#img2maskimg",
        "#inpaint_sketch",
        "#img_inpaint_base",
    ];
    let activeTabSelector = tabSelectors[get_tab_index("mode_img2img")];
    if (!activeTabSelector) {
        return null;
    }

    let activeTabRoot = gradioApp().querySelector(activeTabSelector);
    if (!activeTabRoot) {
        return null;
    }

    let candidates = activeTabRoot.querySelectorAll(
        "div.forge-image-container img, div.forge-image-container canvas, div[data-testid=image] img, div[data-testid=image] canvas",
    );

    for (let candidate of candidates) {
        let sourceWidth =
            candidate.naturalWidth || candidate.videoWidth || candidate.width;
        let sourceHeight =
            candidate.naturalHeight || candidate.videoHeight || candidate.height;

        if (
            isVisible(candidate) &&
            candidate.clientWidth > 0 &&
            candidate.clientHeight > 0 &&
            sourceWidth > 0 &&
            sourceHeight > 0
        ) {
            return candidate;
        }
    }

    return null;
}

function showAspectRatioPreview() {
    currentWidth = getSliderValue("img2img_width") ?? currentWidth;
    currentHeight = getSliderValue("img2img_height") ?? currentHeight;

    if (!currentWidth || !currentHeight) {
        hideArPreview();
        return;
    }

    let targetElement = getActiveImg2imgTargetElement();
    let displayedRect = getDisplayedMediaRect(targetElement);
    if (!displayedRect) {
        hideArPreview();
        return;
    }

    let arPreviewRect = gradioApp().querySelector("#imageARPreview");
    if (!arPreviewRect) {
        arPreviewRect = document.createElement("div");
        arPreviewRect.id = "imageARPreview";
        gradioApp().appendChild(arPreviewRect);
    }

    let arscale = Math.min(
        displayedRect.width / currentWidth,
        displayedRect.height / currentHeight,
    );
    let arscaledx = currentWidth * arscale;
    let arscaledy = currentHeight * arscale;

    arPreviewRect.style.top = displayedRect.top + (displayedRect.height - arscaledy) / 2 + "px";
    arPreviewRect.style.left = displayedRect.left + (displayedRect.width - arscaledx) / 2 + "px";
    arPreviewRect.style.width = arscaledx + "px";
    arPreviewRect.style.height = arscaledy + "px";
    arPreviewRect.style.display = "block";

    arPreviewExpiresAt = Date.now() + 2000;
    clearTimeout(arFrameTimeout);
    arFrameTimeout = setTimeout(function () {
        if (Date.now() >= arPreviewExpiresAt) {
            hideArPreview();
        }
    }, 2000);
}

function dimensionChange(e, is_width, is_height) {
    if (is_width) {
        currentWidth = e.target.value * 1.0;
    }
    if (is_height) {
        currentHeight = e.target.value * 1.0;
    }

    let inImg2img = isVisible(gradioApp().querySelector("#tab_img2img"));

    if (!inImg2img) {
        return;
    }

    showAspectRatioPreview();
}

onAfterUiUpdate(function () {
    [
        ["img2img_width", true, false],
        ["img2img_height", false, true],
    ].forEach(function ([sliderId, isWidth, isHeight]) {
        let sliderRoot = gradioApp().getElementById(sliderId);
        if (!sliderRoot) {
            return;
        }

        let inputs = sliderRoot.querySelectorAll(
            'input[type="range"], input[type="number"]',
        );
        inputs.forEach(function (input) {
            if (input.dataset.arPreviewBound === "true") {
                return;
            }

            input.addEventListener("input", function (event) {
                dimensionChange(event, isWidth, isHeight);
            });

            if (input.type == "number") {
                input.addEventListener("change", function (event) {
                    dimensionChange(event, isWidth, isHeight);
                });
            }

            input.dataset.arPreviewBound = "true";
        });
    });

    currentWidth = getSliderValue("img2img_width") ?? currentWidth;
    currentHeight = getSliderValue("img2img_height") ?? currentHeight;

    if (arPreviewExpiresAt > Date.now()) {
        showAspectRatioPreview();
    } else if (!isVisible(gradioApp().querySelector("#tab_img2img"))) {
        hideArPreview();
    }
});
