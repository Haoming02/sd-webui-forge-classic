// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() { }

function getGallerySelectedIndex() { }

function request(url, data, handler, errorHandler) {
    let xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    let js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    let js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? "0" + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return (
            pad2(Math.floor(secs / 60 / 60)) +
            ":" +
            pad2(Math.floor(secs / 60) % 60) +
            ":" +
            pad2(Math.floor(secs) % 60)
        );
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}

let originalAppTitle = undefined;

onUiLoaded(function () {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    let title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = "[" + progress.trim() + "] " + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}

function randomId() {
    return (
        "task(" +
        Math.random().toString(36).slice(2, 7) +
        Math.random().toString(36).slice(2, 7) +
        Math.random().toString(36).slice(2, 7) +
        ")"
    );
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(
    id_task,
    progressbarContainer,
    gallery,
    atEnd,
    onProgress,
    inactivityTimeout = 40,
) {
    let dateStart = new Date();
    let wasEverActive = false;
    let parentProgressbar = progressbarContainer.parentNode;
    let wakeLock = null;

    if (gallery && gallery.classList.contains("hidden"))
        gallery = gallery.parentElement.querySelector(".gradio-video");

    let requestWakeLock = async function () {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock) return;
        try {
            wakeLock = await navigator.wakeLock.request("screen");
        } catch (err) {
            console.error("Wake Lock is not supported.");
        }
    };

    let releaseWakeLock = async function () {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error("Wake Lock release failed", err);
        }
    };

    let divProgress = document.createElement("div");
    divProgress.className = "progressDiv";
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    let divInner = document.createElement("div");
    divInner.className = "progress";

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    let livePreview = null;

    let removeProgressBar = function () {
        releaseWakeLock();
        if (!divProgress) return;

        setTitle("");
        parentProgressbar.removeChild(divProgress);
        if (gallery && livePreview) {
            // revoke any blob URLs created for previews to avoid memory leaks
            try {
                for (let i = 0; i < livePreview.childElementCount; ++i) {
                    let el = livePreview.children[i];
                    if (el && el.dataset && el.dataset.blobUrl) {
                        try { URL.revokeObjectURL(el.dataset.blobUrl); } catch (e) {}
                    }
                }
            } catch (e) {}
            gallery.removeChild(livePreview);
        }
        atEnd();

        divProgress = null;
    };

    let funProgress = function (id_task) {
        requestWakeLock();
        request(
            "./internal/progress",
            { id_task: id_task, live_preview: false },
            function (res) {
                if (res.completed) {
                    removeProgressBar();
                    return;
                }

                let progressText = "";

                divInner.style.width = (res.progress || 0) * 100.0 + "%";
                divInner.style.background = res.progress ? "" : "transparent";

                if (res.progress > 0) {
                    progressText = ((res.progress || 0) * 100.0).toFixed(0) + "%";
                }

                if (res.eta) {
                    progressText += " ETA: " + formatTime(res.eta);
                }

                setTitle(progressText);

                if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
                    progressText = res.textinfo + " " + progressText;
                }

                divInner.textContent = progressText;

                let elapsedFromStart = (new Date() - dateStart) / 1000;

                if (res.active) wasEverActive = true;

                if (!res.active && wasEverActive) {
                    removeProgressBar();
                    return;
                }

                if (
                    elapsedFromStart > inactivityTimeout &&
                    !res.queued &&
                    !res.active
                ) {
                    removeProgressBar();
                    return;
                }

                if (onProgress) {
                    onProgress(res);
                }

                setTimeout(() => {
                    funProgress(id_task, res.id_live_preview);
                }, opts.live_preview_refresh_period || 500);
            },
            function () {
                removeProgressBar();
            },
        );
    };

    let funLivePreview = function (id_task, id_live_preview) {
        request(
            "./internal/progress",
            { id_task: id_task, id_live_preview: id_live_preview },
            function (res) {
                if (!divProgress) {
                    return;
                }

                if (res.live_preview && gallery) {
                    // If backend returned a data URI (legacy), use it directly.
                    if (typeof res.live_preview === 'string' && res.live_preview.startsWith('data:')) {
                        let img = new Image();
                        img.onload = function () {
                            if (!livePreview) {
                                livePreview = document.createElement("div");
                                livePreview.className = "livePreview";
                                gallery.insertBefore(livePreview, gallery.firstElementChild);
                            }

                            livePreview.appendChild(img);
                            if (livePreview.childElementCount > 2) {
                                // remove oldest without blob revocation (data URI)
                                livePreview.removeChild(livePreview.firstElementChild);
                            }
                        };
                        img.src = res.live_preview;
                    } else {
                        // New flow: res.live_preview is a URL to fetch binary data from.
                        fetch(res.live_preview)
                            .then(function (r) {
                                if (!r.ok) throw new Error('Failed to fetch preview');
                                return r.blob();
                            })
                            .then(function (blob) {
                                let img = new Image();
                                let objectUrl = URL.createObjectURL(blob);
                                img.dataset.blobUrl = objectUrl;
                                img.onload = function () {
                                    if (!livePreview) {
                                        livePreview = document.createElement("div");
                                        livePreview.className = "livePreview";
                                        gallery.insertBefore(livePreview, gallery.firstElementChild);
                                    }

                                    livePreview.appendChild(img);
                                    if (livePreview.childElementCount > 2) {
                                        let first = livePreview.firstElementChild;
                                        // revoke blob URL if present
                                        if (first && first.dataset && first.dataset.blobUrl) {
                                            try { URL.revokeObjectURL(first.dataset.blobUrl); } catch (e) {}
                                        }
                                        livePreview.removeChild(first);
                                    }
                                };
                                img.src = objectUrl;
                            })
                            .catch(function (e) {
                                // on error, just ignore this preview
                                console.error('Live preview fetch error', e);
                            });
                    }
                }

                setTimeout(() => {
                    funLivePreview(id_task, res.id_live_preview);
                }, opts.live_preview_refresh_period || 500);
            },
            function () {
                removeProgressBar();
            },
        );
    };

    funProgress(id_task, 0);

    if (gallery) {
        funLivePreview(id_task, 0);
    }
}
