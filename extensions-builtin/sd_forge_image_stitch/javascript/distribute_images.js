const STITCH_IDS = Array.from({length: 50}, (_, i) => `img_stitch_ref${i+1}`);

function handleMultifile(files) {
    if (!files || files.length === 0) return false;
    
    // Extract only image files
    const imageFiles = [];
    for (let i = 0; i < files.length; i++) {
        if (files[i].type.startsWith("image/")) {
            imageFiles.push(files[i]);
        }
    }
    
    if (imageFiles.length === 0) return false;

    // Distribute them sequentially to first available empty slot
    let fileIdx = 0;
    
    for (const id of STITCH_IDS) {
        const container = document.getElementById(id);
        if (!container) continue;
        
        // A populated Gradio Image component gets an <img> tag for the preview
        const hasImage = container.querySelector('img');
        
        if (!hasImage && fileIdx < imageFiles.length) {
            const fileInput = container.querySelector('input[type="file"]');
            if (fileInput) {
                const dt = new DataTransfer();
                dt.items.add(imageFiles[fileIdx]);
                fileInput.files = dt.files;
                
                const event = new Event("change", { bubbles: true });
                fileInput.dispatchEvent(event);
                
                fileIdx++;
            }
        }
    }
    
    return fileIdx > 0;
}

document.addEventListener("click", (e) => {
    const dropzone = e.target.closest("#custom_stitch_dropzone");
    if (dropzone) {
        const fileInput = dropzone.querySelector("#custom_stitch_file_input");
        if (fileInput) fileInput.click();
    }
});

document.addEventListener("change", (e) => {
    if (e.target.id === "custom_stitch_file_input") {
        handleMultifile(e.target.files);
        setTimeout(() => { e.target.value = ""; }, 100);
    }
});

document.addEventListener("dragover", (e) => {
    const dropzone = e.target.closest("#custom_stitch_dropzone");
    if (dropzone) {
        e.preventDefault();
        dropzone.style.borderColor = "#fff";
        dropzone.style.backgroundColor = "rgba(255, 255, 255, 0.1)";
    }
});

document.addEventListener("dragleave", (e) => {
    const dropzone = e.target.closest("#custom_stitch_dropzone");
    if (dropzone) {
        e.preventDefault();
        dropzone.style.borderColor = "#777";
        dropzone.style.backgroundColor = "rgba(0,0,0,0.2)";
    }
});

document.addEventListener("drop", (e) => {
    // Only capture drops explicitly hitting our top-level container area
    const container = e.target.closest("#image_stitch_container");
    if (!container) return;
    
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) { 
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        const internalDropzone = container.querySelector("#custom_stitch_dropzone");
        if (internalDropzone) {
            internalDropzone.style.borderColor = "#777";
            internalDropzone.style.backgroundColor = "rgba(0,0,0,0.2)";
        }
        
        handleMultifile(files);
    }
}, true);

document.addEventListener("paste", (e) => {
    // Determine target via active element or hover
    const active = document.activeElement;
    const container = active ? active.closest("#image_stitch_container") : null;
    
    let hoverContainer = false;
    try {
        hoverContainer = !!document.querySelector("#image_stitch_container:hover");
    } catch(err) {}

    if (!container && !hoverContainer) return;
    
    const files = e.clipboardData?.files;
    
    if (files && files.length > 0) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        handleMultifile(files);
    }
}, true);
