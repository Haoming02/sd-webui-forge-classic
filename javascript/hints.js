// mouseover tooltips for various UI elements

const titles = {
    "Sampling Method": "The algorithm used to refine each step of the image",
    "Schedule Type": "The algorithm used to adjust the magnitude of refinement",
    "Sampling Steps": "The number of times the image is iteratively refined",

    "Batch Count": "How many batches of images to generate (in sequence)",
    "Batch Size": "How many images to generate in a single batch (in parallel)",

    "CFG Scale": "The strength used to calculate conditionings",
    "Rescale CFG": "Reduce the high-contrast burnt-color effects (mainly for v-pred checkpoints)",
    "MaHiRo": "An alternative algorithm used for CFG calculation",

    "Seed": 'Given the same prompts and parameters, you "should" generate the same image if the Seed is also the same',

    "Just resize": "Resize input image directly to target resolution",
    "Crop and resize": "Resize the image while maintaining the aspect ratio; crop the excessive parts",
    "Resize and fill": "Resize the image while maintaining the aspect ratio; fill the empty parts with neighboring colors",

    "Mask blur": "How much feathering to apply to the mask (in pixels)",
    "fill": "Fill the masked areas with neighboring colors",
    "original": "Keep whatever was within the masked areas",
    "latent noise": "Fill the masked areas with noise (requires high Denoising strength)",
    "latent nothing": "Fill the masked areas with zero values (requires high Denoising strength)",

    "Denoising Strength": "How strong should the image be changed",

    "Hires. fix": "Automatically perform an additional pass of img2img",
    "Hires steps": "Sampling Steps for the img2img pass; use original if 0",
    "Upscale by": "Multiply the txt2img dimension by this ratio, to serve as the target dimension",
    "Resize width to": 'Resize image to this width; use "Upscale by" if 0',
    "Resize height to": 'Resize image to this height; use "Upscale by" if 0',
};

function updateTooltip(element) {
    if (element.title) return;

    const text = element.textContent || element.value;
    let tooltip = localization[titles[text]] || titles[text];

    if (!tooltip) return;
    element.title = tooltip;

    try {
        for (let i = 0; i < 5; i++) {
            element = element.parentNode;
            if (element.classList.contains("block")) break;
        }
        const fields = element.querySelectorAll("input");
        for (const field of fields)
            field.title = tooltip;
    } catch { };
}

const tooltipCheckNodes = new Set();
let tooltipCheckTimer = null;

function processTooltipCheckNodes() {
    for (const node of tooltipCheckNodes) updateTooltip(node);
    tooltipCheckNodes.clear();
}

onUiUpdate(function (mutationRecords) {
    for (const record of mutationRecords) {
        for (const node of record.addedNodes) {
            if (
                node.nodeType === Node.ELEMENT_NODE &&
                !node.classList.contains("hide")
            ) {
                if (!node.title) {
                    if (["SPAN", "BUTTON", "P"].includes(node.tagName))
                        tooltipCheckNodes.add(node);
                }
                node
                    .querySelectorAll("span, button, p")
                    .forEach((n) => tooltipCheckNodes.add(n));
            }
        }
    }
    if (tooltipCheckNodes.size) {
        clearTimeout(tooltipCheckTimer);
        tooltipCheckTimer = setTimeout(processTooltipCheckNodes, 1000);
    }
});

onUiLoaded(() => {
    for (const comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip && comp.props.elem_id) {
            const elem = gradioApp().getElementById(comp.props.elem_id);
            if (elem) elem.title = comp.props.webui_tooltip;
        }
    }
});
