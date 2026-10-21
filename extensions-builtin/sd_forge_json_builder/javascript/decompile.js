// Reconstructs an editable Layout draft from a compiled MOSS-Image prompt

function decompilePrompt(input, width, height) {
    function require(condition, message) {
        if (!condition) {
            throw new Error(message);
        }
    }

    function normalizeDraft(value) {
        require(value !== null && typeof value === "object" && !Array.isArray(value), "Draft must be an object");

        const draftWidth = value.width;
        const draftHeight = value.height;

        for (const [name, dim] of [
            ["width", draftWidth],
            ["height", draftHeight],
        ]) {
            require(typeof dim === "number" && Number.isFinite(dim) && dim > 0, `${name} must be a positive number`);
        }

        const summary = value.summary;
        require(summary !== null && typeof summary === "object" && !Array.isArray(summary), "summary must be an object");
        require(typeof summary.caption === "string" && summary.caption.trim().length > 0, "summary.caption must be nonempty text");

        const canvas = summary.canvas === undefined ? "single" : summary.canvas;
        require(["single", "multi_panel", "collage"].includes(canvas), "summary.canvas must be single, multi_panel or collage");

        let meta = [];
        if (summary.meta !== undefined) {
            require(Array.isArray(summary.meta), "summary.meta must be an array");
            meta = summary.meta.map((entry, index) => {
                require(typeof entry === "string" && entry.trim().length > 0, `summary.meta[${index}] must be nonempty text`);
                return entry;
            });
        }

        const items = value.items;
        require(Array.isArray(items) && items.length > 0, "items must be a nonempty array");

        const cleanItems = items.map((item, index) => {
            const location = `items[${index}]`;

            require(item !== null && typeof item === "object" && !Array.isArray(item), `${location} must be an object`);
            require(["object", "character", "text"].includes(item.label), `${location}.label must be object, character or text`);
            require(typeof item.caption === "string" && item.caption.trim().length > 0, `${location}.caption must be nonempty text`);

            const box = item.bbox;
            require(Array.isArray(box) && box.length === 4, `${location}.bbox must be [x1,y1,x2,y2]`);
            require(box.every((v) => typeof v === "number" && Number.isFinite(v)), `${location}.bbox must contain four finite numbers`);
            require(box[0] < box[2] && box[1] < box[3], `${location}.bbox must have positive width and height`);

            const clean = { label: item.label, caption: item.caption, bbox: box.slice() };

            if (item.label === "text") {
                require(typeof item.text === "string" && item.text.trim().length > 0, `${location}.text must be nonempty text`);
                clean.text = item.text;
            }

            return clean;
        });

        return {
            width: draftWidth,
            height: draftHeight,
            summary: { caption: summary.caption, meta, canvas },
            items: cleanItems,
        };
    }

    // escaped parenthesis is not valid JSON
    function restoreJsonParens(prompt) {
        return prompt.replace(/(?<!\\)((?:\\\\)*)\\([()])/g, (match, runs, paren) => runs + "\\\\" + paren);
    }

    const draft = typeof input === "string" ? JSON.parse(restoreJsonParens(input)) : input;

    require(draft !== null && typeof draft === "object" && !Array.isArray(draft), "Prompt must be a JSON object");

    if (typeof draft.width === "number" && typeof draft.height === "number") {
        return normalizeDraft(draft);
    }

    require(Number.isFinite(width) && width > 0, "A valid width is required to decompile a prompt");
    require(Number.isFinite(height) && height > 0, "A valid height is required to decompile a prompt");
    require(draft.bbox_coord_space === "abs_centralized", "Prompt bbox_coord_space must be abs_centralized");
    require(draft.bbox_format === "yxyx", "Prompt bbox_format must be yxyx");
    require(Array.isArray(draft.items), "Prompt items must be an array");

    const halfWidth = Math.floor(width / 2);
    const halfHeight = Math.floor(height / 2);

    const items = draft.items.map((item, index) => {
        require(item !== null && typeof item === "object" && !Array.isArray(item), `items[${index}] must be an object`);

        const box = item.bbox;
        require(Array.isArray(box) && box.length === 4, `items[${index}].bbox must be [y1,x1,y2,x2]`);
        require(box.every((v) => typeof v === "number" && Number.isFinite(v)), `items[${index}].bbox must contain four finite numbers`);

        const [y1, x1, y2, x2] = box;
        return { ...item, bbox: [x1 + halfWidth, y1 + halfHeight, x2 + halfWidth, y2 + halfHeight] };
    });

    return normalizeDraft({
        width,
        height,
        bbox_coord_space: "abs",
        bbox_format: "xyxy",
        summary: draft.summary,
        items,
    });
}
