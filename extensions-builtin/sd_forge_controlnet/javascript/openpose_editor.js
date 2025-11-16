(function () {
    const AllCnetTabs = new Set();

    function setupOpenPoseEditor() {
        const tabs = document.querySelectorAll("#controlnet .tabitem");

        for (const tab of tabs) {
            if (AllCnetTabs.has(tab)) continue;
            AllCnetTabs.add(tab);

            const generatedImageGroup = tab.querySelector(".cnet-generated-image-group");
            const allowPreviewCheckbox = tab.querySelector(".cnet-allow-preview input");

            const downloadLink = generatedImageGroup.querySelector(".cnet-download-pose a");
            const poseTextbox = generatedImageGroup.querySelector(".cnet-pose-json textarea");
            const renderButton = generatedImageGroup.querySelector(".cnet-render-pose");

            function updatePreviewPose(poseURL) {
                if (!allowPreviewCheckbox.checked) allowPreviewCheckbox.click();

                if (downloadLink != null) downloadLink.href = poseURL;

                poseTextbox.value = poseURL;
                updateInput(poseTextbox);

                renderButton.click();
            }

            const inputImageGroup = tab.querySelector(".cnet-input-image-group");
            const uploadButton = inputImageGroup.querySelector(".cnet-upload-pose input");

            uploadButton.addEventListener("change", (event) => {
                const file = event.target.files[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = function (e) {
                    const contents = e.target.result;
                    const poseURL = `data:application/json;base64,${btoa(contents)}`;
                    updatePreviewPose(poseURL);
                };
                reader.readAsText(file);

                event.target.value = "";
            });
        }
    }

    onUiLoaded(setupOpenPoseEditor);
})();
