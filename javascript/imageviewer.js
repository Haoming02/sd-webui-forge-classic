// A full size 'lightbox' preview modal shown when left clicking on gallery previews
let modalWheelZoom = 1;

function getAppRoot() {
    return gradioApp();
}

function getLightboxElement(id) {
    const appRoot = getAppRoot();
    return appRoot.getElementById?.(id) || document.getElementById(id);
}

function mountLightboxModal(modal) {
    const appRoot = getAppRoot();

    if (modal.parentElement === appRoot) return;

    if (appRoot && typeof appRoot.appendChild === "function") {
        appRoot.appendChild(modal);
        return;
    }

    document.body.appendChild(modal);
}

function getSelectedMainTabName() {
    const selectedTab = gradioApp().querySelector(
        '#tabs div button.selected, #tabs div button[aria-selected="true"]',
    );
    return selectedTab?.innerText?.trim()?.toLowerCase() || "";
}

function getGalleryImageFromNode(node) {
    if (!node) {
        return null;
    }

    if (node.tagName === "IMG") {
        return node;
    }

    return node.querySelector?.("img[src]") || null;
}

function getImageSourceUrl(image) {
    return image?.currentSrc || image?.src || "";
}

function getSelectedGalleryImageFromNode(node) {
    const gallery = node?.closest?.('div[id$="_gallery"]');
    if (!gallery) {
        return null;
    }

    const selectedButton = gallery.querySelector(
        '.thumbnails > .thumbnail-item.selected, .thumbnails > button.thumbnail-item.selected, .thumbnails > .thumbnail-item[aria-selected="true"], .thumbnails > button.thumbnail-item[aria-selected="true"]',
    );

    return getGalleryImageFromNode(selectedButton);
}

function getVisibleLivePreviewImage() {
    return Array.from(
        gradioApp().querySelectorAll('div[id$="_results"] .livePreview img[src]'),
    )
        .filter((img) => img.offsetParent !== null)
        .at(-1);
}

function applyModalWheelZoom() {
    const modalImage = getLightboxElement("modalImage");
    if (!modalImage) return;

    modalImage.style.transform = `scale(${modalWheelZoom})`;
}

function resetModalWheelZoom() {
    modalWheelZoom = 1;
    const modalImage = getLightboxElement("modalImage");
    if (!modalImage) return;

    modalImage.style.transform = "";
    modalImage.style.transformOrigin = "center center";
}

function modalWheelZoomHandler(event) {
    const modalImage = getLightboxElement("modalImage");
    if (!modalImage || modalImage.style.display === "none") return;

    event.preventDefault();
    event.stopPropagation();

    const zoomStep = 0.1;
    const zoomDelta = event.deltaY < 0 ? zoomStep : -zoomStep;
    const nextZoom = Math.max(0.2, Math.min(8, modalWheelZoom + zoomDelta));

    if (nextZoom === modalWheelZoom) return;

    const rect = modalImage.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
        const x = ((event.clientX - rect.left) / rect.width) * 100;
        const y = ((event.clientY - rect.top) / rect.height) * 100;
        modalImage.style.transformOrigin = `${x}% ${y}%`;
    }

    modalWheelZoom = nextZoom;
    applyModalWheelZoom();
}

function closeModal() {
    resetModalWheelZoom();
    const modal = getLightboxElement("lightboxModal");
    if (modal) modal.style.display = "none";
}

function ensureLightboxModal() {
    const existingModal = getLightboxElement("lightboxModal");
    if (existingModal) {
        mountLightboxModal(existingModal);
        return;
    }

    const modal = document.createElement("div");
    modal.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex = 0;
    modal.addEventListener("keydown", modalKeyHandler, true);

    const modalControls = document.createElement("div");
    modalControls.className = "modalControls gradio-container";
    modal.append(modalControls);

    const modalZoom = document.createElement("span");
    modalZoom.className = "modalZoom cursor";
    modalZoom.innerHTML = "&#10529;";
    modalZoom.addEventListener("click", modalZoomToggle, true);
    modalZoom.title = "Toggle zoomed view";
    modalControls.appendChild(modalZoom);

    const modalTileImage = document.createElement("span");
    modalTileImage.className = "modalTileImage cursor";
    modalTileImage.innerHTML = "&#8862;";
    modalTileImage.addEventListener("click", modalTileImageToggle, true);
    modalTileImage.title = "Preview tiling";
    modalControls.appendChild(modalTileImage);

    const modalSave = document.createElement("span");
    modalSave.className = "modalSave cursor";
    modalSave.id = "modal_save";
    modalSave.innerHTML = "&#x1F5AB;";
    modalSave.addEventListener("click", modalSaveImage, true);
    modalSave.title = "Save Image(s)";
    modalControls.appendChild(modalSave);

    const modalToggleLivePreview = document.createElement("span");
    modalToggleLivePreview.className = "modalToggleLivePreview cursor";
    modalToggleLivePreview.id = "modal_toggle_live_preview";
    modalToggleLivePreview.innerHTML = "&#x1F5C6;";
    modalToggleLivePreview.onclick = modalLivePreviewToggle;
    modalToggleLivePreview.title = "Toggle live preview";
    modalControls.appendChild(modalToggleLivePreview);

    const modalClose = document.createElement("span");
    modalClose.className = "modalClose cursor";
    modalClose.innerHTML = "&times;";
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    modalControls.appendChild(modalClose);

    const modalImage = document.createElement("img");
    modalImage.id = "modalImage";
    modalImage.onclick = closeModal;
    modalImage.tabIndex = 0;
    modalImage.addEventListener("keydown", modalKeyHandler, true);
    modalImage.addEventListener("wheel", modalWheelZoomHandler, {
        passive: false,
        capture: true,
    });
    modal.appendChild(modalImage);

    const modalPrev = document.createElement("a");
    modalPrev.className = "modalPrev";
    modalPrev.innerHTML = "&#10094;";
    modalPrev.tabIndex = 0;
    modalPrev.addEventListener("click", modalPrevImage, true);
    modalPrev.addEventListener("keydown", modalKeyHandler, true);
    modal.appendChild(modalPrev);

    const modalNext = document.createElement("a");
    modalNext.className = "modalNext";
    modalNext.innerHTML = "&#10095;";
    modalNext.tabIndex = 0;
    modalNext.addEventListener("click", modalNextImage, true);
    modalNext.addEventListener("keydown", modalKeyHandler, true);
    modal.appendChild(modalNext);

    mountLightboxModal(modal);
}

function showModal(event) {
    ensureLightboxModal();
    const source = event.target || event.srcElement;
    const sourceUrl = getImageSourceUrl(source);
    const modalImage = getLightboxElement("modalImage");
    const modalToggleLivePreviewBtn = getLightboxElement(
        "modal_toggle_live_preview",
    );
    const lb = getLightboxElement("lightboxModal");
    if (!sourceUrl || !modalImage || !lb || !modalToggleLivePreviewBtn) return;
    modalToggleLivePreviewBtn.innerHTML = opts.js_live_preview_in_modal_lightbox
        ? "&#x1F5C7;"
        : "&#x1F5C6;";
    resetModalWheelZoom();
    modalImage.src = sourceUrl;
    if (modalImage.style.display === "none") {
        lb.style.setProperty("background-image", "url(" + sourceUrl + ")");
    }
    lb.style.display = "flex";
    lb.focus();

    const selectedTab = getSelectedMainTabName();
    if (selectedTab === "txt2img" || selectedTab === "img2img") {
        getLightboxElement("modal_save").style.display = "inline";
    } else {
        getLightboxElement("modal_save").style.display = "none";
    }
    event.stopPropagation();
}

function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    const modalImage = getLightboxElement("modalImage");
    if (modalImage && modalImage.offsetParent) {
        let currentButton = selected_gallery_button();
        let preview = getVisibleLivePreviewImage();
        if (opts.js_live_preview_in_modal_lightbox && preview?.src) {
            modalImage.src = getImageSourceUrl(preview);
        } else if (
            getImageSourceUrl(getGalleryImageFromNode(currentButton)) &&
            modalImage.src != getImageSourceUrl(getGalleryImageFromNode(currentButton))
        ) {
            modalImage.src = getImageSourceUrl(
                getGalleryImageFromNode(currentButton),
            );
            if (modalImage.style.display === "none") {
                const modal = getLightboxElement("lightboxModal");
                modal.style.setProperty("background-image", `url(${modalImage.src})`);
            }
        }
    }
}

function modalImageSwitch(offset) {
    let galleryButtons = all_gallery_buttons();

    if (galleryButtons.length > 1) {
        let result = selected_gallery_index();

        if (result != -1) {
            let nextButton =
                galleryButtons[negmod(result + offset, galleryButtons.length)];
            const nextImage = getGalleryImageFromNode(nextButton);
            if (!nextImage?.src) {
                return;
            }
            nextButton.click();
            const modalImage = getLightboxElement("modalImage");
            const modal = getLightboxElement("lightboxModal");
            resetModalWheelZoom();
            modalImage.src = nextImage.src;
            if (modalImage.style.display === "none") {
                modal.style.setProperty("background-image", `url(${modalImage.src})`);
            }
            setTimeout(function () {
                modal.focus();
            }, 10);
        }
    }
}

function saveImage() {
    const saveTxt2Img = "save_txt2img";
    const saveImg2Img = "save_img2img";
    const selectedTab = getSelectedMainTabName();
    if (selectedTab === "txt2img") {
        gradioApp().getElementById(saveTxt2Img).click();
    } else if (selectedTab === "img2img") {
        gradioApp().getElementById(saveImg2Img).click();
    } else {
        console.error("missing implementation for saving modal of this type");
    }
}

function modalSaveImage(event) {
    saveImage();
    event.stopPropagation();
}

function modalNextImage(event) {
    modalImageSwitch(1);
    event.stopPropagation();
}

function modalPrevImage(event) {
    modalImageSwitch(-1);
    event.stopPropagation();
}

const LIGHTBOX_PREVIEW_SELECTOR =
    'div[id$="_gallery"] button.preview, div[id$="_gallery"] .thumbnail-item, div[id$="_gallery"] .thumbnail-item img, div[id$="_results"] .livePreview, div[id$="_results"] .livePreview img';

let delegatedLightboxHandlersInstalled = false;

function modalKeyHandler(event) {
    switch (event.key) {
        case "s":
            saveImage();
            break;
        case "ArrowLeft":
            modalPrevImage(event);
            break;
        case "ArrowRight":
            modalNextImage(event);
            break;
        case "Escape":
            closeModal();
            break;
    }
}

function openSourceInLightbox(source, evt) {
    if (!getImageSourceUrl(source) || !opts.js_modal_lightbox) {
        return;
    }

    modalZoomSet(
        getLightboxElement("modalImage"),
        opts.js_modal_lightbox_initially_zoomed,
    );

    if (evt) {
        evt.preventDefault();
    }

    showModal({ target: source, stopPropagation: () => {} });
}

function resolveSourceFromNode(node, evt) {
    return (
        evt.target?.closest?.("img") ||
        getGalleryImageFromNode(node) ||
        getSelectedGalleryImageFromNode(node) ||
        getVisibleLivePreviewImage() ||
        node
    );
}

function queueLightboxOpen(previewNode, evt, attempt = 0) {
    const source = resolveSourceFromNode(previewNode, evt);
    if (getImageSourceUrl(source)) {
        evt.lightboxHandled = true;
        openSourceInLightbox(source, evt);
        return;
    }

    if (attempt >= 4) {
        return;
    }

    requestAnimationFrame(() => {
        queueLightboxOpen(previewNode, evt, attempt + 1);
    });
}

function findLightboxPreviewNode(evt) {
    const path = evt.composedPath?.() || [];

    for (const node of path) {
        if (node?.matches?.(LIGHTBOX_PREVIEW_SELECTOR)) {
            return node;
        }
    }

    return evt.target?.closest?.(LIGHTBOX_PREVIEW_SELECTOR) || null;
}

function handleDelegatedLightboxMouseDown(evt) {
    const previewNode = findLightboxPreviewNode(evt);
    if (!previewNode) return;

    const source = resolveSourceFromNode(previewNode, evt);
    if (!source?.src || evt.button != 1) return;

    open(source.src);
    evt.preventDefault();
}

function handleDelegatedLightboxClick(evt) {
    if (evt.lightboxHandled) return;
    const previewNode = findLightboxPreviewNode(evt);
    if (!previewNode || !opts.js_modal_lightbox || evt.button != 0) return;

    queueLightboxOpen(previewNode, evt);
}

function installDelegatedLightboxHandlers() {
    if (delegatedLightboxHandlersInstalled) return;

    const appRoot = getAppRoot();
    if (!appRoot?.addEventListener) return;

    appRoot.addEventListener("mousedown", handleDelegatedLightboxMouseDown, true);
    appRoot.addEventListener("click", handleDelegatedLightboxClick, true);
    delegatedLightboxHandlersInstalled = true;
}

function setupImageForLightbox(e) {
    if (e.dataset.modded) {
        return;
    }

    e.dataset.modded = true;
    e.style.cursor = "pointer";
    e.style.userSelect = "none";

    e.addEventListener(
        "mousedown",
        function (evt) {
            const source = resolveSourceFromNode(e, evt);
            if (!source?.src || evt.button != 1) return;

            evt.lightboxHandled = true;
            open(source.src);
            evt.preventDefault();
            evt.stopPropagation();
        },
        true,
    );

    e.addEventListener(
        "click",
        function (evt) {
            if (evt.button != 0) return;

            queueLightboxOpen(e, evt);
        },
        true,
    );
}

function modalZoomSet(modalImage, enable) {
    if (modalImage) modalImage.classList.toggle("modalImageFullscreen", !!enable);
}

function modalZoomToggle(event) {
    let modalImage = getLightboxElement("modalImage");
    modalZoomSet(
        modalImage,
        !modalImage.classList.contains("modalImageFullscreen"),
    );
    event.stopPropagation();
}

function modalLivePreviewToggle(event) {
    const modalToggleLivePreview = getLightboxElement(
        "modal_toggle_live_preview",
    );
    opts.js_live_preview_in_modal_lightbox =
        !opts.js_live_preview_in_modal_lightbox;
    modalToggleLivePreview.innerHTML = opts.js_live_preview_in_modal_lightbox
        ? "&#x1F5C7;"
        : "&#x1F5C6;";
    event.stopPropagation();
}

function modalTileImageToggle(event) {
    const modalImage = getLightboxElement("modalImage");
    const modal = getLightboxElement("lightboxModal");
    const isTiling = modalImage.style.display === "none";
    if (isTiling) {
        modalImage.style.display = "block";
        modal.style.setProperty("background-image", "none");
        resetModalWheelZoom();
    } else {
        modalImage.style.display = "none";
        modal.style.setProperty("background-image", `url(${modalImage.src})`);
    }

    event.stopPropagation();
}

onAfterUiUpdate(function () {
    ensureLightboxModal();
    installDelegatedLightboxHandlers();
    let fullImg_preview = gradioApp().querySelectorAll(
        [
            LIGHTBOX_PREVIEW_SELECTOR,
            ".gradio-gallery > button > button > img",
            ".gradio-gallery > .livePreview",
        ].join(", "),
    );
    if (fullImg_preview != null) {
        fullImg_preview.forEach(setupImageForLightbox);
    }
    updateOnBackgroundChange();
});
