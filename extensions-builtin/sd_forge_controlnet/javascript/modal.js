(function () {
    const cnetModalRegisteredElements = new Set();

    onUiUpdate(() => {
        const btns = gradioApp().querySelectorAll(".cnet-modal-open");
        const spans = document.querySelectorAll(".cnet-modal-close");

        btns.forEach((btn) => {
            if (cnetModalRegisteredElements.has(btn)) return;
            cnetModalRegisteredElements.add(btn);

            const modalId = btn.id.replace("cnet-modal-open-", "");
            const modal = document.getElementById("cnet-modal-" + modalId);
            btn.addEventListener("click", () => {
                modal.style.display = "block";
            });
        });

        spans.forEach((span) => {
            if (cnetModalRegisteredElements.has(span)) return;
            cnetModalRegisteredElements.add(span);

            const modal = span.parentNode;
            span.addEventListener("click", () => {
                modal.style.display = "none";
            });
        });
    });
})();
