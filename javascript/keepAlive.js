// https://github.com/Haoming02/sd-webui-forge-classic/issues/1051#issuecomment-4414473943

(function () {
    function keepAlive() {
        const _raf = window.requestAnimationFrame.bind(window);
        const _caf = window.cancelAnimationFrame.bind(window);
        const bgPending = new Map();
        let bgId = 1 << 24;

        window.requestAnimationFrame = function (cb) {
            if (!document.hidden) return _raf(cb);
            const id = bgId++;
            const ch = new MessageChannel();
            ch.port1.onmessage = () => { bgPending.delete(id) && cb(performance.now()); };
            bgPending.set(id, true);
            ch.port2.postMessage(null);
            return id;
        };

        window.cancelAnimationFrame = function (id) {
            bgPending.has(id) ? bgPending.delete(id) : _caf(id);
        };
    }

    onOptionsAvailable(() => { if (opts.keep_alive) keepAlive(); });
})();
