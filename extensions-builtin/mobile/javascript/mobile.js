(function () {
    let isSetupForMobile = false;

    function isMobile() {
        for (const tab of ["txt2img", "img2img"]) {
            const imageTab = gradioApp().getElementById(tab + "_results");
            if (imageTab && imageTab.offsetParent && imageTab.offsetLeft === 0) {
                return true;
            }
        }

        return false;
    }

    function reportWindowSize() {
        // not applicable for compact prompt layout
        if (gradioApp().querySelector(".toprow-compact-tools")) return;

        const currentlyMobile = isMobile();
        if (currentlyMobile === isSetupForMobile) return;
        isSetupForMobile = currentlyMobile;

        for (const tab of ["txt2img", "img2img"]) {
            const button = gradioApp().getElementById(tab + "_generate_box");
            const target = gradioApp().getElementById(currentlyMobile ? tab + "_results" : tab + "_actions_column");
            target.insertBefore(button, target.firstElementChild);

            gradioApp()
                .getElementById(tab + "_results")
                .classList.toggle("mobile", currentlyMobile);
        }
    }

    window.addEventListener("resize", reportWindowSize);

    onUiLoaded(reportWindowSize);
})();
