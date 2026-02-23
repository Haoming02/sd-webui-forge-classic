function fetchLicense(id, url) {
    fetch(url).then((response) => {
        response.text().then(
            licenseText => document.getElementById(id).textContent = licenseText
        );
    })
}

function populateLicense() {
    const pairs = [
        ["comfy", "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/LICENSE"],
        ["sd1", "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/LICENSE"],
        ["sdxl", "https://raw.githubusercontent.com/Stability-AI/generative-models/main/LICENSE-CODE"],
        ["flux", "https://raw.githubusercontent.com/black-forest-labs/flux/main/LICENSE"],
        ["flux2", "https://raw.githubusercontent.com/black-forest-labs/flux2/main/LICENSE.md"],
        ["qwen", "https://raw.githubusercontent.com/QwenLM/Qwen-Image/refs/heads/main/LICENSE"],
        ["lumina", "https://raw.githubusercontent.com/Alpha-VLLM/Lumina-Image-2.0/main/LICENSE"],
        ["zimage", "https://raw.githubusercontent.com/Tongyi-MAI/Z-Image/main/LICENSE"],
        ["wan", "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/LICENSE.txt"],
        ["chain", "https://raw.githubusercontent.com/chaiNNer-org/chaiNNer/main/LICENSE"],
        ["tfm", "https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE"],
        ["dot", "https://raw.githubusercontent.com/huggingface/diffusers/main/LICENSE"],
        ["invoke", "https://raw.githubusercontent.com/invoke-ai/InvokeAI/main/LICENSE"],
        ["taesd", "https://raw.githubusercontent.com/madebyollin/taesd/main/LICENSE"],
    ];

    for (const [id, url] of pairs)
        fetchLicense(`${id}-license-content`, url);
}
