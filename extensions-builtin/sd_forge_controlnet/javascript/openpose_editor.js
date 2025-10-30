(function () {
    const cnetOpenposeEditorRegisteredElements = new Set();
    function loadOpenposeEditor() {
        // Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
        // will only visible on web page and not sent to python.
        function updateInput(target) {
            let e = new Event("input", { bubbles: true })
            Object.defineProperty(e, "target", { value: target })
            target.dispatchEvent(e);
        }

        const tabs = gradioApp().querySelectorAll('#controlnet .input-accordion');
        tabs.forEach(tab => {
            if (cnetOpenposeEditorRegisteredElements.has(tab)) return;
            cnetOpenposeEditorRegisteredElements.add(tab);

            const generatedImageGroup = tab.querySelector('.cnet-generated-image-group');
            /*
            * Writes the pose data URL to an link element on input image group.
            * Click a hidden button to trigger a backend rendering of the pose JSON.
            *
            * The backend should:
            * - Set the rendered pose image as preprocessor generated image.
            */
            function updatePreviewPose(poseURL) {
                const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
                const renderButton = generatedImageGroup.querySelector('.cnet-render-pose');
                const poseTextbox = generatedImageGroup.querySelector('.cnet-pose-json textarea');
                const allowPreviewCheckbox = tab.querySelector('.cnet-allow-preview input');

                if (!allowPreviewCheckbox.checked)
                    allowPreviewCheckbox.click();

                // Only set href when download link exists and needs an update. `downloadLink`
                // can be null when user closes preview and click `Upload JSON` button again.
                // https://github.com/Mikubill/sd-webui-controlnet/issues/2308
                if (downloadLink !== null)
                    downloadLink.href = poseURL;

                poseTextbox.value = poseURL;
                updateInput(poseTextbox);
                renderButton.click();
            }

            const inputImageGroup = tab.querySelector('.cnet-input-image-group');
            const uploadButton = inputImageGroup.querySelector('.cnet-upload-pose input');
            // Updates preview image when JSON file is uploaded.
            uploadButton.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (!file)
                    return;

                const reader = new FileReader();
                reader.onload = function (e) {
                    const contents = e.target.result;
                    const poseURL = `data:application/json;base64,${btoa(contents)}`;
                    updatePreviewPose(poseURL);
                };
                reader.readAsText(file);
                // Reset the file input value so that uploading the same file still triggers callback.
                event.target.value = '';
            });
        });
    }

    onUiLoaded(loadOpenposeEditor);
})();
