
// localization = {} -- the dict with translations is created by the backend

var ignore_ids_for_localization = {
    setting_sd_model_checkpoint: 'OPTION',
    txt2img_styles: 'OPTION',
    img2img_styles: 'OPTION',
    setting_random_artist_categories: 'OPTION',
    setting_face_restoration_model: 'OPTION',
    setting_realesrgan_enabled_models: 'OPTION',
    extras_upscaler_1: 'OPTION',
    extras_upscaler_2: 'OPTION',
};

const re_num = /^[.\d]+$/;
const re_emoji = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F9B0}-\u{1F9B3}]/u;

const original_lines = {};
const translated_lines = {};

function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

function textNodesUnder(el) {
    let n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;

    let parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;

    if (parentType == 'OPTION' || parentType == 'SPAN') {
        let pnode = node;
        for (let level = 0; level < 4; level++) {
            pnode = pnode.parentElement;
            if (!pnode) break;

            if (ignore_ids_for_localization[pnode.id] == parentType) return false;
        }
    }

    if (re_num.test(text)) return false;
    if (re_emoji.test(text)) return false;
    return true;
}

function getTranslation(text) {
    if (!text) return undefined;

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }

    let tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }

    return tl;
}

function processTextNode(node) {
    let text = node.textContent.trim();

    if (!canBeTranslated(node, text)) return;

    let tl = getTranslation(text);
    if (tl !== undefined) {
        node.textContent = tl;
    }
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) {
            node.title = tl;
        }
    }

    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) {
            node.placeholder = tl;
        }
    }

    textNodesUnder(node).forEach(function (node) {
        processTextNode(node);
    });
}

function localizeWholePage() {
    processNode(gradioApp());

    function elem(comp) {
        let elem_id = comp.props.elem_id ? comp.props.elem_id : "component-" + comp.id;
        return gradioApp().getElementById(elem_id);
    }

    for (let comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip) {
            let e = elem(comp);

            let tl = e ? getTranslation(e.title) : undefined;
            if (tl !== undefined) {
                e.title = tl;
            }
        }
        if (comp.props.placeholder) {
            let e = elem(comp);
            let textbox = e ? e.querySelector('[placeholder]') : null;

            let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
            if (tl !== undefined) {
                textbox.placeholder = tl;
            }
        }
    }
}

function dumpTranslations() {
    if (!hasLocalization()) {
        // If we don't have any localization,
        // we will not have traversed the app to find
        // original_lines, so do that now.
        localizeWholePage();
    }
    let dumped = {};
    if (localization.rtl) {
        dumped.rtl = true;
    }

    for (const text in original_lines) {
        if (dumped[text] !== undefined) continue;
        dumped[text] = localization[text] || text;
    }

    return dumped;
}

function download_localization() {
    let text = JSON.stringify(dumpTranslations(), null, 4);

    let element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', "localization.json");
    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

document.addEventListener("DOMContentLoaded", function () {
    if (!hasLocalization()) {
        return;
    }

    onUiUpdate(function (m) {
        m.forEach(function (mutation) {
            mutation.addedNodes.forEach(function (node) {
                processNode(node);
            });
        });
    });

    localizeWholePage();

    if (localization.rtl) { // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        for (const x of node.sheet.rules) { // find all rtl media rules
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all'); // enable them
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), { childList: true });
    }
});
