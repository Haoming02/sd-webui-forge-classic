(function () {
    const AllControlnet = new Set();

    class ControlNetAccordion {
        constructor(ext) {
            const badge = document.createElement("span");
            badge.classList.add("cnet-badge");
            badge.style.visibility = "hidden";

            const label = ext.querySelector("button.label-wrap").querySelector("span");
            label.appendChild(badge);
            label.style.display = "flex";

            this.badge = badge;
            this.count = 0;
        }

        increase() {
            this.count += 1;
            this.badge.textContent = `${this.count}x Unit`;
            this.badge.style.visibility = "visible";
        }

        decrease() {
            this.count -= 1;
            this.badge.textContent = `${this.count}x Unit`;
            if (this.count === 0) this.badge.style.visibility = "hidden";
        }
    }

    class ControlNetUnitTab {
        constructor(cnet, tab, index) {
            this.cnet = cnet;
            this.unitHeader = tab.parentNode.querySelector(".tab-nav").querySelectorAll("button")[index];

            this.enabledCheckbox = tab.querySelector(".cnet-unit-enabled input");
            this.inputImage = tab.querySelector('.cnet-input-image-group .cnet-image input[type="file"]');
            this.controlTypeRadios = tab.querySelectorAll('.controlnet_control_type_filter_group input[type="radio"]');

            this.attachEnabledButtonListener();
            this.attachControlTypeRadioListener();
            this.attachImageUploadListener();
            this.attachA1111SendInfoObserver();
        }

        attachEnabledButtonListener() {
            this.enabledCheckbox.addEventListener("change", () => {
                this.updateActiveState();
            });
        }

        attachControlTypeRadioListener() {
            for (const radio of this.controlTypeRadios) {
                radio.addEventListener("change", () => {
                    this.updateActiveControlType();
                });
            }
        }

        attachImageUploadListener() {
            this.inputImage.addEventListener("change", (event) => {
                if (!event.target.files) return;
                if (!this.enabledCheckbox.checked) this.enabledCheckbox.click();
            });
        }

        attachA1111SendInfoObserver() {
            const pasteButtons = document.querySelectorAll("#paste");
            const pngButtons = document.querySelectorAll(this.isImg2Img ? "#img2img_tab, #inpaint_tab" : "#txt2img_tab");

            for (const button of [...pasteButtons, ...pngButtons]) {
                button.addEventListener("click", () => {
                    setTimeout(() => {
                        this.updateActiveState();
                    }, 2500);
                });
            }
        }

        updateActiveState() {
            if (this.enabledCheckbox.checked) {
                this.unitHeader.classList.add("cnet-unit-active");
                this.cnet.increase();
            }
            else {
                this.unitHeader.classList.remove("cnet-unit-active");
                this.cnet.decrease();
            }
        }

        updateActiveControlType() {
            const controlTypeSuffix = this.unitHeader.querySelector(".control-type-suffix");
            if (controlTypeSuffix) controlTypeSuffix.remove();

            const controlType = this.getActiveControlType();
            if (controlType === "All") return;

            const span = document.createElement("span");
            span.innerHTML = `[${controlType}]`;
            span.classList.add("control-type-suffix");
            this.unitHeader.appendChild(span);
        }

        getActiveControlType() {
            for (const radio of this.controlTypeRadios) if (radio.checked) return radio.value;
        }
    }

    onUiLoaded(() => {
        for (const tab of ["txt2img", "img2img"]) {
            const ext = document.getElementById(`${tab}_controlnet`).querySelector("#controlnet");

            if (AllControlnet.has(ext)) return;
            AllControlnet.add(ext);

            const cnet = new ControlNetAccordion(ext);
            for (const [i, tab] of ext.querySelectorAll(".tabitem").entries())
                new ControlNetUnitTab(cnet, tab, i);
        }
    });
})();
