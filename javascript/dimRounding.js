/** @param {number} val @param {int} step @returns {int} */
function sRound(val, step) {
    return Math.floor(val / step + 0.5) * step;
}

(function () {
    function roundDimension() {
        const _step = parseInt(opts.res_step);

        const IDs = ["txt2img_width", "txt2img_height", "img2img_width", "img2img_height"];

        for (const id of IDs) {
            const input = document.getElementById(id)?.querySelector('input:not([type="range"])');
            if (input == null) continue;

            input.onblur = () => {
                input.value = sRound(input.value, _step);
                updateInput(input);
            };
        }
    }

    onOptionsAvailable(() => {
        roundDimension();
    });
})();
