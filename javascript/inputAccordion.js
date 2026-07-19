function inputAccordionChecked(id, checked) {
    let accordion = gradioApp().getElementById(id);
    accordion.visibleCheckbox.checked = checked;
    accordion.onVisibleCheckboxChange();
}

function setupAccordion(accordion) {
    let labelWrap = accordion.querySelector(".label-wrap");
    let gradioCheckbox = gradioApp().querySelector("#" + accordion.id + "-checkbox input");
    let extra = gradioApp().querySelector("#" + accordion.id + "-extra");
    let span = labelWrap.querySelector("span");
    let linked = true;

    let isOpen = function () {
        return labelWrap.classList.contains("open");
    };

    let observerAccordionOpen = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutationRecord) {
            accordion.classList.toggle("input-accordion-open", isOpen());

            if (linked) {
                accordion.visibleCheckbox.checked = isOpen();
                accordion.onVisibleCheckboxChange();
            }
        });
    });
    observerAccordionOpen.observe(labelWrap, {
        attributes: true,
        attributeFilter: ["class"],
    });

    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    accordion.onChecked = function (checked) {
        if (isOpen() != checked) {
            labelWrap.click();
        }
    };

    let visibleCheckbox = document.createElement("INPUT");
    visibleCheckbox.type = "checkbox";
    visibleCheckbox.checked = isOpen();
    visibleCheckbox.id = accordion.id + "-visible-checkbox";
    visibleCheckbox.className = gradioCheckbox.className + " input-accordion-checkbox";
    span.insertBefore(visibleCheckbox, span.firstChild);

    accordion.visibleCheckbox = visibleCheckbox;
    accordion.onVisibleCheckboxChange = function () {
        if (linked && isOpen() != visibleCheckbox.checked) {
            labelWrap.click();
        }

        gradioCheckbox.checked = visibleCheckbox.checked;
        updateInput(gradioCheckbox);
    };

    visibleCheckbox.addEventListener("click", function (event) {
        linked = false;
        event.stopPropagation();
    });
    visibleCheckbox.addEventListener("input", accordion.onVisibleCheckboxChange);
}

onUiLoaded(function () {
    for (let accordion of gradioApp().querySelectorAll(".input-accordion")) {
        setupAccordion(accordion);
    }
});
