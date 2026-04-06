(function () {
    const OPENPOSE_BRIDGE_DEBUG = window.localStorage?.getItem('openposeBridgeDebug') === '1';

    function debugLog(...args) {
        if (OPENPOSE_BRIDGE_DEBUG) {
            console.log(...args);
        }
    }

    function debugWarn(...args) {
        if (OPENPOSE_BRIDGE_DEBUG) {
            console.warn(...args);
        }
    }

    async function checkEditorAvailable() {
        const LOCAL_EDITOR_PATH = '/openpose_editor_index';
        const REMOTE_EDITOR_PATH = 'https://huchenlei.github.io/sd-webui-openpose-editor/';

        try {
            debugLog('[OpenPose bridge] probing local editor', LOCAL_EDITOR_PATH);
            const res = await fetch(LOCAL_EDITOR_PATH, { cache: 'no-store' });
            if (res.status === 200) {
                debugLog('[OpenPose bridge] using local editor', LOCAL_EDITOR_PATH);
                return LOCAL_EDITOR_PATH;
            }
            debugWarn('[OpenPose bridge] local editor probe returned', res.status);
        } catch (e) {
            debugWarn('[OpenPose bridge] local editor probe failed', e);
        }

        debugWarn('[OpenPose bridge] falling back to remote editor', REMOTE_EDITOR_PATH);
        return REMOTE_EDITOR_PATH;
    }

    let editorURL = null;
    const boundButtons = new WeakSet();
    let lastScanCount = null;
    debugLog('[OpenPose bridge] script loaded');

    function updateInput(target) {
        const e = new Event('input', { bubbles: true });
        Object.defineProperty(e, 'target', { value: target });
        target.dispatchEvent(e);
    }

    function getTabForButton(button) {
        return button.closest('.input-accordion, .controlnet');
    }

    function getGeneratedGroup(button) {
        return button.closest('.cnet-generated-image-control-group')?.parentElement || button.closest('.cnet-generated-image-group') || button.closest('.input-accordion')?.querySelector('.cnet-generated-image-group');
    }

    function hasPoseData(generatedImageGroup) {
        const poseTextbox = generatedImageGroup?.querySelector('.cnet-pose-json textarea');
        const poseLink = generatedImageGroup?.querySelector('.cnet-download-pose a');
        return Boolean(
            (poseTextbox?.value && poseTextbox.value.startsWith('data:application/json;base64,'))
            || (poseLink?.href && poseLink.href.startsWith('data:application/json;base64,'))
        );
    }

    function ensureInitStore() {
        if (!window.__openposeEditorInitPayloads) {
            window.__openposeEditorInitPayloads = {};
        }
        return window.__openposeEditorInitPayloads;
    }

    function decodeBase64Unicode(base64) {
        const binary = atob(base64);
        const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        return new TextDecoder().decode(bytes);
    }

    function encodeBase64Unicode(text) {
        const bytes = new TextEncoder().encode(text);
        let binary = '';
        for (const byte of bytes) {
            binary += String.fromCharCode(byte);
        }
        return btoa(binary);
    }

    function parsePoseDataURL(poseURL) {
        if (!poseURL || !poseURL.startsWith('data:application/json;base64,')) return null;
        const base64 = poseURL.split(',', 2)[1];
        return JSON.parse(decodeBase64Unicode(base64));
    }

    function getPoseCanvasSizeFromURL(poseURL) {
        const poseJson = parsePoseDataURL(poseURL);
        if (!poseJson) return null;

        const width = Number(poseJson.canvas_width || poseJson.width);
        const height = Number(poseJson.canvas_height || poseJson.height);

        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
            return null;
        }

        return { width, height };
    }

    function encodePoseDataURL(payload) {
        return `data:application/json;base64,${encodeBase64Unicode(JSON.stringify(payload))}`;
    }

    function getResizeMode(tab) {
        const checkedInput = tab?.querySelector('.controlnet_resize_mode_radio input[type="radio"]:checked');
        const checkedLabel = checkedInput?.closest('label');
        const checkedText = checkedLabel?.textContent?.trim();
        if (checkedText) return checkedText;

        const selectedButton = tab?.querySelector('.controlnet_resize_mode_radio .selected, .controlnet_resize_mode_radio .checked');
        const selectedText = selectedButton?.textContent?.trim();
        if (selectedText) return selectedText;

        return 'Crop and Resize';
    }

    function loadImageElement(imageURL) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            if (imageURL && !imageURL.startsWith('data:')) {
                image.crossOrigin = 'anonymous';
            }
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error(`Failed to load image: ${imageURL}`));
            image.src = imageURL;
        });
    }

    async function createAlignedBackgroundImage(imageURL, poseURL, resizeMode) {
        if (!imageURL || !poseURL) return imageURL;

        const canvasSize = getPoseCanvasSizeFromURL(poseURL);
        if (!canvasSize) return imageURL;

        const image = await loadImageElement(imageURL);
        const targetWidth = canvasSize.width;
        const targetHeight = canvasSize.height;

        if (!targetWidth || !targetHeight || !image.naturalWidth || !image.naturalHeight) {
            return imageURL;
        }

        const canvas = document.createElement('canvas');
        canvas.width = targetWidth;
        canvas.height = targetHeight;

        const context = canvas.getContext('2d');
        if (!context) return imageURL;

        context.clearRect(0, 0, targetWidth, targetHeight);
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = 'high';

        const sourceWidth = image.naturalWidth;
        const sourceHeight = image.naturalHeight;

        if (resizeMode === 'Just Resize') {
            context.drawImage(image, 0, 0, targetWidth, targetHeight);
            return canvas.toDataURL('image/png');
        }

        const containScale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
        const coverScale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
        const scale = resizeMode === 'Resize and Fill' ? containScale : coverScale;
        const drawWidth = sourceWidth * scale;
        const drawHeight = sourceHeight * scale;
        const drawX = (targetWidth - drawWidth) / 2;
        const drawY = (targetHeight - drawHeight) / 2;

        context.drawImage(image, drawX, drawY, drawWidth, drawHeight);
        return canvas.toDataURL('image/png');
    }

    function clonePoint(point) {
        return point ? { x: point.x, y: point.y } : null;
    }

    function midpoint(pointA, pointB) {
        if (!pointA || !pointB) return null;
        return { x: (pointA.x + pointB.x) / 2, y: (pointA.y + pointB.y) / 2 };
    }

    function distance(pointA, pointB) {
        if (!pointA || !pointB) return 0;
        const dx = pointA.x - pointB.x;
        const dy = pointA.y - pointB.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function clampPoint(point) {
        if (!point) return null;
        return {
            x: Math.min(0.98, Math.max(0.02, point.x)),
            y: Math.min(0.98, Math.max(0.02, point.y)),
        };
    }

    function readBodyPoint(numbers, index) {
        const offset = index * 3;
        if (!Array.isArray(numbers) || numbers.length < offset + 3) return null;
        const x = Number(numbers[offset]);
        const y = Number(numbers[offset + 1]);
        const confidence = Number(numbers[offset + 2]);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(confidence) || confidence <= 0) return null;
        return { x, y };
    }

    function writeBodyPoint(numbers, index, point) {
        const offset = index * 3;
        while (numbers.length < 54) {
            numbers.push(0, 0, 0);
        }
        const clamped = clampPoint(point);
        if (!clamped) return;
        numbers[offset] = clamped.x;
        numbers[offset + 1] = clamped.y;
        numbers[offset + 2] = 1.0;
    }

    function inferBodyAnchors(numbers) {
        const points = Array.from({ length: 18 }, (_, index) => readBodyPoint(numbers, index));

        const rightShoulder = points[2];
        const leftShoulder = points[5];
        const rightHip = points[8];
        const leftHip = points[11];
        const neck = points[1] || midpoint(rightShoulder, leftShoulder);
        const shoulderMid = midpoint(rightShoulder, leftShoulder) || clonePoint(neck);
        const hipMid = midpoint(rightHip, leftHip);
        const shoulderWidth = distance(rightShoulder, leftShoulder) || 0.14;
        const torsoHeight = distance(neck, hipMid) || Math.max(shoulderWidth * 1.45, 0.18);
        const defaultHipMid = hipMid || (neck ? { x: neck.x, y: neck.y + torsoHeight } : { x: 0.5, y: 0.62 });
        const hipOffset = Math.max(shoulderWidth * 0.34, 0.045);
        const rightHipSeed = rightHip || { x: defaultHipMid.x - hipOffset, y: defaultHipMid.y };
        const leftHipSeed = leftHip || { x: defaultHipMid.x + hipOffset, y: defaultHipMid.y };
        const legSegment = Math.max(torsoHeight * 0.92, 0.16);
        const armSegment = Math.max(shoulderWidth * 0.9, 0.11);
        const defaultNeck = neck || (hipMid ? { x: hipMid.x, y: hipMid.y - torsoHeight } : { x: 0.5, y: 0.34 });
        const shoulderOffset = Math.max(shoulderWidth * 0.5, 0.07);
        const rightShoulderSeed = rightShoulder
            || (leftShoulder && defaultNeck ? { x: defaultNeck.x - Math.abs(leftShoulder.x - defaultNeck.x), y: leftShoulder.y } : null)
            || { x: defaultNeck.x - shoulderOffset, y: defaultNeck.y };
        const leftShoulderSeed = leftShoulder
            || (rightShoulder && defaultNeck ? { x: defaultNeck.x + Math.abs(rightShoulder.x - defaultNeck.x), y: rightShoulder.y } : null)
            || { x: defaultNeck.x + shoulderOffset, y: defaultNeck.y };

        return {
            points,
            neck: defaultNeck,
            shoulderMid,
            shoulderWidth,
            torsoHeight,
            rightShoulderSeed,
            leftShoulderSeed,
            rightHipSeed,
            leftHipSeed,
            legSegment,
            armSegment,
        };
    }

    function seedBodyChain(numbers, startIndex, middleIndex, endIndex, fallbackStart, segmentLength, sideX = 0) {
        const start = readBodyPoint(numbers, startIndex) || clampPoint(fallbackStart);
        let middle = readBodyPoint(numbers, middleIndex);
        let end = readBodyPoint(numbers, endIndex);

        if (!start) return;

        if (!middle && end) {
            middle = { x: start.x + (end.x - start.x) * 0.5, y: start.y + (end.y - start.y) * 0.5 };
        }
        if (!middle) {
            middle = { x: start.x + sideX * 0.01, y: start.y + segmentLength };
        }
        if (!end) {
            const dx = middle.x - start.x;
            const dy = middle.y - start.y;
            end = { x: middle.x + dx, y: middle.y + dy };
        }

        if (!readBodyPoint(numbers, middleIndex)) writeBodyPoint(numbers, middleIndex, middle);
        if (!readBodyPoint(numbers, endIndex)) writeBodyPoint(numbers, endIndex, end);
    }

    function seedMissingBodyPointsInJson(poseJson) {
        if (!poseJson || !Array.isArray(poseJson.people)) return poseJson;

        poseJson.people.forEach((person) => {
            const numbers = Array.isArray(person.pose_keypoints_2d) ? person.pose_keypoints_2d.slice() : [];
            while (numbers.length < 54) {
                numbers.push(0, 0, 0);
            }

            const anchors = inferBodyAnchors(numbers);

            if (!readBodyPoint(numbers, 1) && anchors.neck) {
                writeBodyPoint(numbers, 1, anchors.neck);
            }
            if (!readBodyPoint(numbers, 2)) {
                writeBodyPoint(numbers, 2, anchors.rightShoulderSeed);
            }
            if (!readBodyPoint(numbers, 5)) {
                writeBodyPoint(numbers, 5, anchors.leftShoulderSeed);
            }
            if (!readBodyPoint(numbers, 8)) {
                writeBodyPoint(numbers, 8, anchors.rightHipSeed);
            }
            if (!readBodyPoint(numbers, 11)) {
                writeBodyPoint(numbers, 11, anchors.leftHipSeed);
            }

            seedBodyChain(numbers, 8, 9, 10, anchors.rightHipSeed, anchors.legSegment, -1);
            seedBodyChain(numbers, 11, 12, 13, anchors.leftHipSeed, anchors.legSegment, 1);

            const rightShoulder = readBodyPoint(numbers, 2);
            const leftShoulder = readBodyPoint(numbers, 5);
            if (rightShoulder) {
                seedBodyChain(numbers, 2, 3, 4, rightShoulder, anchors.armSegment, -1);
            }
            if (leftShoulder) {
                seedBodyChain(numbers, 5, 6, 7, leftShoulder, anchors.armSegment, 1);
            }

            person.pose_keypoints_2d = numbers;
        });

        return poseJson;
    }

    function seedMissingBodyPointsInPoseURL(poseURL) {
        const poseJson = parsePoseDataURL(poseURL);
        if (!poseJson) return null;
        return encodePoseDataURL(seedMissingBodyPointsInJson(poseJson));
    }

    function getPoseControlsByModalId(modalId) {
        const editButton = gradioApp().querySelector(`#cnet-modal-open-${modalId}`);
        const generatedImageGroup = editButton ? getGeneratedGroup(editButton) : null;
        return {
            editButton,
            generatedImageGroup,
            poseTextbox: generatedImageGroup?.querySelector('.cnet-pose-json textarea') || null,
            renderButton: generatedImageGroup?.querySelector('.cnet-render-pose') || null,
            downloadLink: generatedImageGroup?.querySelector('.cnet-download-pose a') || null,
        };
    }

    function postPayloadToIframe(modalId) {
        const modal = document.getElementById(`cnet-modal-${modalId}`);
        const iframe = modal?.querySelector('iframe');
        const payload = ensureInitStore()[modalId];
        if (!iframe?.contentWindow || !payload) return false;
        iframe.contentWindow.postMessage(payload, '*');
        return true;
    }

    function seedMissingBodyPoints(modalId) {
        const { poseTextbox, renderButton, downloadLink } = getPoseControlsByModalId(modalId);
        const currentPoseURL = poseTextbox?.value || downloadLink?.href;
        const seededPoseURL = seedMissingBodyPointsInPoseURL(currentPoseURL);
        if (!seededPoseURL || !poseTextbox || !renderButton) {
            alert('Forge could not prepare additional body points from the current pose JSON.');
            return;
        }

        if (downloadLink) downloadLink.href = seededPoseURL;
        poseTextbox.value = seededPoseURL;
        updateInput(poseTextbox);
        renderButton.click();

        const payloads = ensureInitStore();
        payloads[modalId] = {
            ...(payloads[modalId] || { modalId }),
            poseURL: seededPoseURL,
        };

        postPayloadToIframe(modalId);
    }

    function isSameOriginIframe(iframe) {
        try {
            return !!(iframe?.contentWindow?.document && iframe.contentWindow.location);
        } catch (e) {
            return false;
        }
    }

    function findIframeSendButton(iframe) {
        try {
            const doc = iframe?.contentDocument || iframe?.contentWindow?.document;
            if (!doc) return null;
            const buttons = Array.from(doc.querySelectorAll('#control-panel button'));
            if (!buttons.length) return null;
            return buttons.find((button) => /controlnet/i.test(button.textContent || '')) || buttons[0] || null;
        } catch (e) {
            return null;
        }
    }

    function setApplyButtonState(button, iframe) {
        if (!button) return;
        const sameOrigin = isSameOriginIframe(iframe);
        const sendButton = sameOrigin ? findIframeSendButton(iframe) : null;
        const ready = !!sendButton;
        button.disabled = !ready;
        button.classList.toggle('is-disabled', !ready);
        button.title = ready
            ? 'Send the edited pose back to ControlNet.'
            : sameOrigin
                ? 'Waiting for the editor controls to finish loading.'
                : 'Direct send is only available with the local OpenPose editor.';
    }

    function refreshApplyButton(modalId) {
        const modal = document.getElementById(`cnet-modal-${modalId}`);
        const button = modal?.querySelector('.cnet-openpose-apply');
        const iframe = modal?.querySelector('iframe');
        setApplyButtonState(button, iframe);
    }

    function triggerIframeSend(modalId) {
        const modal = document.getElementById(`cnet-modal-${modalId}`);
        const iframe = modal?.querySelector('iframe');
        const sendButton = findIframeSendButton(iframe);
        if (!sendButton) return false;
        sendButton.click();
        return true;
    }

    function ensureModalApplyButton(modalId) {
        const modal = document.getElementById(`cnet-modal-${modalId}`);
        const modalContent = modal?.querySelector('.cnet-modal-content');
        const iframe = modalContent?.querySelector('iframe');
        if (!modalContent || !iframe) return;

        let actions = modalContent.querySelector('.cnet-openpose-modal-actions');
        if (!actions) {
            actions = document.createElement('div');
            actions.className = 'cnet-openpose-modal-actions';

            const applyButton = document.createElement('button');
            applyButton.type = 'button';
            applyButton.className = 'cnet-openpose-apply';
            applyButton.textContent = 'Use in ControlNet';
            applyButton.addEventListener('click', () => {
                if (triggerIframeSend(modalId)) return;
                alert('Forge could not directly reach the editor send action yet. If you are using the remote editor, use its built-in "Send pose to ControlNet" button inside the iframe.');
            });

            const seedButton = document.createElement('button');
            seedButton.type = 'button';
            seedButton.className = 'cnet-openpose-seed';
            seedButton.textContent = 'Add Missing Body Points';
            seedButton.title = 'Seed missing body joints into the pose so you can drag them in the editor.';
            seedButton.addEventListener('click', () => seedMissingBodyPoints(modalId));

            actions.appendChild(seedButton);
            actions.appendChild(applyButton);
            modalContent.appendChild(actions);
        }

        if (!iframe.dataset.cnetApplyBound) {
            iframe.dataset.cnetApplyBound = 'true';
            iframe.addEventListener('load', () => {
                window.setTimeout(() => refreshApplyButton(modalId), 100);
                window.setTimeout(() => refreshApplyButton(modalId), 500);
                window.setTimeout(() => refreshApplyButton(modalId), 1200);
            });
        }

        refreshApplyButton(modalId);
    }

    function navigateIframe(iframe, targetEditorURL, modalId) {
        function getPathname(rawURL) {
            try {
                return new URL(rawURL).pathname;
            } catch (e) {
                return rawURL;
            }
        }

        return new Promise((resolve) => {
            const params = new URLSearchParams();
            if (document.body.classList.contains('dark')) {
                params.set('theme', 'dark');
            }
            params.set('modalId', modalId);
            params.set('_ts', Date.now().toString());
            const finalURL = `${targetEditorURL}?${params.toString()}`;

            let resolved = false;
            let timeoutId = null;
            const onMessage = (event) => {
                const message = event.data;
                if (event.source !== iframe.contentWindow) return;
                if (!message || !message.ready) return;
                if (message.modalId && message.modalId !== modalId) return;
                if (timeoutId) clearTimeout(timeoutId);
                window.removeEventListener('message', onMessage);
                if (!resolved) {
                    resolved = true;
                    debugLog('[OpenPose bridge] iframe reported ready');
                    resolve();
                }
            };

            window.addEventListener('message', onMessage);

            const currentPath = targetEditorURL.startsWith('http') ? iframe.src : getPathname(iframe.src);
            iframe.src = finalURL;
            debugLog('[OpenPose bridge] iframe src set', iframe.src, { previousSrc: currentPath });
            timeoutId = setTimeout(() => {
                window.removeEventListener('message', onMessage);
                if (!resolved) {
                    resolved = true;
                    debugWarn('[OpenPose bridge] iframe ready timeout');
                    resolve();
                }
            }, 5000);
        });
    }

    async function handleEditClick(editButton) {
        debugLog('[OpenPose bridge] edit click', editButton.id);
        const tab = getTabForButton(editButton);
        const generatedImageGroup = getGeneratedGroup(editButton);
        debugLog('[OpenPose bridge] resolved tab/group', { hasTab: !!tab, hasGeneratedGroup: !!generatedImageGroup });
        if (!tab || !generatedImageGroup) return;

        const inputImage = tab.querySelector('.cnet-input-image-group .cnet-image img');
        const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
        const modalIframe = generatedImageGroup.querySelector('.cnet-modal iframe');
        const modalId = editButton.id.replace('cnet-modal-open-', '');

        debugLog('[OpenPose bridge] resolved modal elements', { hasInputImage: !!inputImage, hasDownloadLink: !!downloadLink, hasIframe: !!modalIframe, modalId });
        if (!modalIframe) return;

        if (!editorURL) {
            editorURL = await checkEditorAvailable();
            debugLog('[OpenPose bridge] selected editor URL', editorURL);
            if (!editorURL) {
                alert('No openpose editor available.');
                return;
            }
        }

        const poseURL = downloadLink ? downloadLink.href : undefined;
        const resizeMode = getResizeMode(tab);
        let alignedImageURL = inputImage ? inputImage.src : undefined;

        try {
            alignedImageURL = await createAlignedBackgroundImage(alignedImageURL, poseURL, resizeMode);
        } catch (error) {
            debugWarn('[OpenPose bridge] failed to align background image to pose canvas', error);
        }

        const payload = {
            modalId,
            imageURL: alignedImageURL,
            poseURL,
        };
        ensureInitStore()[modalId] = payload;
        debugLog('[OpenPose bridge] stored init payload', payload);

        ensureModalApplyButton(modalId);
        await navigateIframe(modalIframe, editorURL, modalId);
        postPayloadToIframe(modalId);
        refreshApplyButton(modalId);
        modalIframe.contentWindow.focus();
    }

    function bindEditButton(editButton) {
        if (!editButton || boundButtons.has(editButton)) return;
        boundButtons.add(editButton);
        debugLog('[OpenPose bridge] binding edit button', editButton.id);
        editButton.addEventListener('click', () => {
            handleEditClick(editButton).catch((e) => console.error('[OpenPose bridge] edit handler failed', e));
        });
    }

    function ensurePersistentEditLauncher(editButton) {
        const tab = getTabForButton(editButton);
        const inputGroup = tab?.querySelector('.cnet-input-image-group');
        const generatedImageGroup = getGeneratedGroup(editButton);
        if (!inputGroup || !generatedImageGroup) return;

        const modalId = editButton.id.replace('cnet-modal-open-', '');
        let launcher = inputGroup.querySelector(`.cnet-openpose-persistent[data-modal-id="${modalId}"]`);

        if (!launcher) {
            launcher = document.createElement('button');
            launcher.type = 'button';
            launcher.className = 'cnet-openpose-persistent';
            launcher.dataset.modalId = modalId;
            launcher.textContent = 'Edit Pose';
            launcher.title = 'Open the OpenPose editor for the current pose.';
            launcher.addEventListener('click', () => editButton.click());
            inputGroup.appendChild(launcher);
        }

        launcher.style.display = hasPoseData(generatedImageGroup) ? 'flex' : 'none';
    }

    function refreshPersistentEditLaunchers() {
        gradioApp().querySelectorAll('.cnet-edit-pose').forEach(ensurePersistentEditLauncher);
    }

    function bindButtons() {
        const buttons = gradioApp().querySelectorAll('.cnet-edit-pose');
        if (buttons.length !== lastScanCount) {
            debugLog('[OpenPose bridge] scanning edit buttons', buttons.length);
            lastScanCount = buttons.length;
        }
        buttons.forEach((button) => {
            bindEditButton(button);
            ensurePersistentEditLauncher(button);
        });

        const uploadButtons = gradioApp().querySelectorAll('.cnet-input-image-group .cnet-upload-pose input');
        uploadButtons.forEach((uploadButton) => {
            if (boundButtons.has(uploadButton)) return;
            boundButtons.add(uploadButton);
            uploadButton.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (!file) return;
                const tab = uploadButton.closest('.input-accordion, .controlnet');
                const generatedImageGroup = tab?.querySelector('.cnet-generated-image-group');
                const downloadLink = generatedImageGroup?.querySelector('.cnet-download-pose a');
                const renderButton = generatedImageGroup?.querySelector('.cnet-render-pose');
                const poseTextbox = generatedImageGroup?.querySelector('.cnet-pose-json textarea');
                if (!renderButton || !poseTextbox) return;
                const reader = new FileReader();
                reader.onload = function (e) {
                    const contents = e.target.result;
                    const poseURL = `data:application/json;base64,${btoa(contents)}`;
                    if (downloadLink) downloadLink.href = poseURL;
                    poseTextbox.value = poseURL;
                    updateInput(poseTextbox);
                    renderButton.click();
                };
                reader.readAsText(file);
                event.target.value = '';
            });
        });
    }

    window.addEventListener('message', (event) => {
        const message = event.data;
        if (!message || !message.modalId || !message.poseURL) return;
        debugLog('[OpenPose bridge] iframe message', message);
        const editButton = gradioApp().querySelector(`#cnet-modal-open-${message.modalId}`);
        const generatedImageGroup = editButton ? getGeneratedGroup(editButton) : null;
        const tab = editButton ? getTabForButton(editButton) : null;
        const downloadLink = generatedImageGroup?.querySelector('.cnet-download-pose a');
        const renderButton = generatedImageGroup?.querySelector('.cnet-render-pose');
        const poseTextbox = generatedImageGroup?.querySelector('.cnet-pose-json textarea');
        const allowPreviewCheckbox = tab?.querySelector('.cnet-allow-preview input');
        if (allowPreviewCheckbox && !allowPreviewCheckbox.checked) allowPreviewCheckbox.click();
        if (downloadLink) downloadLink.href = message.poseURL;
        if (poseTextbox && renderButton) {
            poseTextbox.value = message.poseURL;
            updateInput(poseTextbox);
            renderButton.click();
        }
        refreshPersistentEditLaunchers();
        refreshApplyButton(message.modalId);
        const closeModalButton = generatedImageGroup?.querySelector('.cnet-modal .cnet-modal-close');
        if (closeModalButton) closeModalButton.click();
    });

    onUiUpdate(bindButtons);
})();
