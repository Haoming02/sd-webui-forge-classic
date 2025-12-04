// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme) {
    let gradioURL = window.location.href;
    if (!gradioURL.includes("?__theme=")) {
        window.location.replace(gradioURL + "?__theme=" + theme);
    }
}

function all_gallery_buttons() {
    let allGalleryButtons = gradioApp().querySelectorAll(
        '[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small',
    );
    let visibleGalleryButtons = [];
    allGalleryButtons.forEach(function (elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    return (
        all_gallery_buttons().find((elem) => elem.classList.contains("selected")) ??
        null
    );
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex((elem) =>
        elem.classList.contains("selected"),
    );
}

function gallery_container_buttons(gallery_container) {
    return gradioApp().querySelectorAll(
        `#${gallery_container} .thumbnail-item.thumbnail-small`,
    );
}

function selected_gallery_index_id(gallery_container) {
    return Array.from(gallery_container_buttons(gallery_container)).findIndex(
        (elem) => elem.classList.contains("selected"),
    );
}

function extract_image_from_gallery(gallery) {
    if (gallery.length == 0) {
        return [null];
    }

    let index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // Use the first image in the gallery as the default
        index = 0;
    }

    return [[gallery[index]]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img() {
    gradioApp().querySelector("#tabs").querySelectorAll("button")[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
    gradioApp().querySelector("#tabs").querySelectorAll("button")[1].click();
    gradioApp()
        .getElementById("mode_img2img")
        .querySelectorAll("button")
    [no].click();
}
function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    gradioApp().querySelector("#tabs").querySelectorAll("button")[2].click();

    return Array.from(arguments);
}

function get_tab_index(tabId) {
    let buttons = gradioApp()
        .getElementById(tabId)
        .querySelector("div")
        .querySelectorAll("button");
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].classList.contains("selected")) {
            return i;
        }
    }
    return 0;
}

function create_tab_index_args(tabId, args) {
    let res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    let res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index("mode_img2img");
    return res;
}

function create_submit_args(args) {
    // Currently, txt2img and img2img also send the output args (gallery / player / generation_info / infotext / html_log) whenever you generate a new image.
    let res = Array.from(args);

    if (Array.isArray(res[res.length - 5]))
        res = res.slice(0, res.length - 5);
    else if (Array.isArray(res[res.length - 4]))
        res = res.slice(0, res.length - 4);
    else if (Array.isArray(res[res.length - 3]))
        res = res.slice(0, res.length - 3);

    // NOTE: If gradio at some point stops sending outputs, this may break something
    return res;
}

function setSubmitButtonsVisibility(
    tabname,
    showInterrupt,
    showSkip,
    showInterrupting,
) {
    gradioApp().getElementById(tabname + "_interrupt").style.display =
        showInterrupt ? "block" : "none";
    gradioApp().getElementById(tabname + "_skip").style.display = showSkip
        ? "block"
        : "none";
    gradioApp().getElementById(tabname + "_interrupting").style.display =
        showInterrupting ? "block" : "none";
}

function showSubmitButtons(tabname, show) {
    setSubmitButtonsVisibility(tabname, !show, !show, false);
}

function showSubmitInterruptingPlaceholder(tabname) {
    setSubmitButtonsVisibility(tabname, false, true, true);
}

function showRestoreProgressButton(tabname, show) {
    let button = gradioApp().getElementById(tabname + "_restore_progress");
    if (!button) return;
    button.style.setProperty("display", show ? "flex" : "none", "important");
}

function submit() {
    showSubmitButtons("txt2img", false);

    let id = randomId();
    localSet("txt2img_task_id", id);

    requestProgress(
        id,
        gradioApp().getElementById("txt2img_gallery_container"),
        gradioApp().getElementById("txt2img_gallery"),
        function () {
            showSubmitButtons("txt2img", true);
            localRemove("txt2img_task_id");
            showRestoreProgressButton("txt2img", false);
        },
    );

    let res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function submit_txt2img_upscale() {
    let res = submit(...arguments);

    res[2] = selected_gallery_index();

    return res;
}

function submit_img2img() {
    showSubmitButtons("img2img", false);

    let id = randomId();
    localSet("img2img_task_id", id);

    requestProgress(
        id,
        gradioApp().getElementById("img2img_gallery_container"),
        gradioApp().getElementById("img2img_gallery"),
        function () {
            showSubmitButtons("img2img", true);
            localRemove("img2img_task_id");
            showRestoreProgressButton("img2img", false);
        },
    );

    let res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function submit_extras() {
    showSubmitButtons("extras", false);

    let id = randomId();

    requestProgress(
        id,
        gradioApp().getElementById("extras_gallery_container"),
        gradioApp().getElementById("extras_gallery"),
        function () {
            showSubmitButtons("extras", true);
        },
    );

    let res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function restoreProgressTxt2img() {
    showRestoreProgressButton("txt2img", false);
    let id = localGet("txt2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder("txt2img");
        requestProgress(
            id,
            gradioApp().getElementById("txt2img_gallery_container"),
            gradioApp().getElementById("txt2img_gallery"),
            function () {
                showSubmitButtons("txt2img", true);
            },
            null,
            0,
        );
    }

    return id;
}

function restoreProgressImg2img() {
    showRestoreProgressButton("img2img", false);

    let id = localGet("img2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder("img2img");
        requestProgress(
            id,
            gradioApp().getElementById("img2img_gallery_container"),
            gradioApp().getElementById("img2img_gallery"),
            function () {
                showSubmitButtons("img2img", true);
            },
            null,
            0,
        );
    }

    return id;
}

/**
 * Configure the width and height elements on `tabname` to accept
 * pasting of resolutions in the form of "width x height".
 */
function setupResolutionPasting(tabname) {
    let width = gradioApp().querySelector(`#${tabname}_width input[type=number]`);
    let height = gradioApp().querySelector(
        `#${tabname}_height input[type=number]`,
    );
    for (const el of [width, height]) {
        el.addEventListener("paste", function (event) {
            let pasteData = event.clipboardData.getData("text/plain");
            let parsed = pasteData.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (parsed) {
                width.value = parsed[1];
                height.value = parsed[2];
                updateInput(width);
                updateInput(height);
                event.preventDefault();
            }
        });
    }
}

/**
 * Allow the user to click on the Style name in order to deselect it just like Gradio 3
 */
function restoreStyleDeselection(tabname) {
    const dropdown = document.getElementById(`${tabname}_styles`);
    dropdown.addEventListener("click", (e) => {
        const remove = e.target.closest("div.token-remove");
        if (remove) return;
        const style = e.target.closest("div.token");
        if (style) {
            style.querySelector("div.token-remove").click();
            e.preventDefault();
            e.stopPropagation();
        }
    });
}

onUiLoaded(function () {
    showRestoreProgressButton("txt2img", localGet("txt2img_task_id"));
    showRestoreProgressButton("img2img", localGet("img2img_task_id"));
    setupResolutionPasting("txt2img");
    setupResolutionPasting("img2img");
    restoreStyleDeselection("txt2img");
    restoreStyleDeselection("img2img");
});

function modelmerger() {
    let id = randomId();
    requestProgress(
        id,
        gradioApp().getElementById("modelmerger_results_panel"),
        null,
        function () { },
    );

    let res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    let name_ = prompt("Style name:");
    return [name_, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}

let opts = {};
onAfterUiUpdate(function () {
    if (Object.keys(opts).length != 0) return;

    let json_elem = gradioApp().getElementById("settings_json");
    if (json_elem == null) return;

    let textarea = json_elem.querySelector("textarea");
    let jsdata = textarea.value;
    opts = JSON.parse(jsdata);

    executeCallbacks(
        optionsAvailableCallbacks,
    ); /*global optionsAvailableCallbacks*/
    executeCallbacks(optionsChangedCallbacks); /*global optionsChangedCallbacks*/

    Object.defineProperty(textarea, "value", {
        set: function (newValue) {
            let valueProp = Object.getOwnPropertyDescriptor(
                HTMLTextAreaElement.prototype,
                "value",
            );
            let oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function () {
            let valueProp = Object.getOwnPropertyDescriptor(
                HTMLTextAreaElement.prototype,
                "value",
            );
            return valueProp.get.call(textarea);
        },
    });

    json_elem.parentElement.style.display = "none";
});

onOptionsChanged(function () {
    let elem = gradioApp().getElementById("sd_checkpoint_hash");
    let sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    let shorthash = sd_checkpoint_hash.substring(0, 10);

    if (elem && elem.textContent != shorthash) {
        elem.textContent = shorthash;
        elem.title = sd_checkpoint_hash;
        elem.href = "https://civitai.com/search/models?query=" + sd_checkpoint_hash;
    }
});

let txt2img_textarea,
    img2img_textarea = undefined;

function restart_reload() {
    document.body.style.backgroundColor = "var(--background-fill-primary)";
    document.body.innerHTML =
        '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    let requestPing = function () {
        requestGet(
            "./internal/ping",
            {},
            function (data) {
                location.reload();
            },
            function () {
                setTimeout(requestPing, 500);
            },
        );
    };

    setTimeout(requestPing, 2000);

    return [];
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target) {
    let e = new Event("input", { bubbles: true });
    Object.defineProperty(e, "target", { value: target });
    target.dispatchEvent(e);
}

let desiredCheckpointName = null;
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    gradioApp().getElementById("change_checkpoint").click();
}
let desiredVAEName = 0;
function selectVAE(vae) {
    desiredVAEName = vae;
}

function currentImg2imgSourceResolution(w, h, r) {
    let img = gradioApp().querySelector(
        '#mode_img2img > div[style="display: block;"] :is(img, canvas)',
    );
    return img
        ? [img.naturalWidth || img.width, img.naturalHeight || img.height, r]
        : [0, 0, r];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    // At the time this is called from gradio, the image has no yet been replaced.
    // There may be a better solution, but this is simple and straightforward so I'm going with it.

    setTimeout(function () {
        gradioApp().getElementById("img2img_update_resize_to").click();
    }, 500);

    return [];
}

function setRandomSeed(elem_id) {
    let input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}

function switchWidthHeight(tabname) {
    let width = gradioApp().querySelector(
        "#" + tabname + "_width input[type=number]",
    );
    let height = gradioApp().querySelector(
        "#" + tabname + "_height input[type=number]",
    );
    if (!width || !height) return [];

    let tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    updateInput(width);
    updateInput(height);
    return [];
}

let onEditTimers = {};

// calls func after afterMs milliseconds has passed since the input elem has been edited by user
function onEdit(editId, elem, afterMs, func) {
    let edited = function () {
        let existingTimer = onEditTimers[editId];
        if (existingTimer) clearTimeout(existingTimer);

        onEditTimers[editId] = setTimeout(func, afterMs);
    };

    elem.addEventListener("input", edited);

    return edited;
}
