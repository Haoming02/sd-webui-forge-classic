function createRow(table, cellName, items) {
    let tr = document.createElement("tr");
    let res = [];

    items.forEach(function (x, i) {
        if (x === undefined) {
            res.push(null);
            return;
        }

        let td = document.createElement(cellName);
        td.textContent = x;
        tr.appendChild(td);
        res.push(td);

        let colspan = 1;
        for (let n = i + 1; n < items.length; n++) {
            if (items[n] !== undefined) {
                break;
            }

            colspan += 1;
        }

        if (colspan > 1) {
            td.colSpan = colspan;
        }
    });

    table.appendChild(tr);

    return res;
}

function createVisualizationTable(data, cutoff = 0, sort = "") {
    let table = document.createElement("table");
    table.className = "popup-table";

    let keys = Object.keys(data);
    if (sort === "number") {
        keys = keys.sort(function (a, b) {
            return data[b] - data[a];
        });
    } else {
        keys = keys.sort();
    }
    let items = keys.map(function (x) {
        return { key: x, parts: x.split("/"), value: data[x] };
    });
    let maxLength = items.reduce(function (a, b) {
        return Math.max(a, b.parts.length);
    }, 0);

    let cols = createRow(table, "th", [
        cutoff === 0 ? "key" : "record",
        cutoff === 0 ? "value" : "seconds",
    ]);
    cols[0].colSpan = maxLength;

    function arraysEqual(a, b) {
        return !(a < b || b < a);
    }

    let addLevel = function (level, parent, hide) {
        let matching = items.filter(function (x) {
            return (
                x.parts[level] &&
                !x.parts[level + 1] &&
                arraysEqual(x.parts.slice(0, level), parent)
            );
        });
        if (sort === "number") {
            matching = matching.sort(function (a, b) {
                return b.value - a.value;
            });
        } else {
            matching = matching.sort();
        }
        let othersTime = 0;
        let othersList = [];
        let othersRows = [];
        let childrenRows = [];
        matching.forEach(function (x) {
            let visible = (cutoff === 0 && !hide) || (x.value >= cutoff && !hide);

            let cells = [];
            for (let i = 0; i < maxLength; i++) {
                cells.push(x.parts[i]);
            }
            cells.push(cutoff === 0 ? x.value : x.value.toFixed(3));
            let cols = createRow(table, "td", cells);
            for (let i = 0; i < level; i++) {
                cols[i].className = "muted";
            }

            let tr = cols[0].parentNode;
            if (!visible) {
                tr.classList.add("hidden");
            }

            if (cutoff === 0 || x.value >= cutoff) {
                childrenRows.push(tr);
            } else {
                othersTime += x.value;
                othersList.push(x.parts[level]);
                othersRows.push(tr);
            }

            let children = addLevel(level + 1, parent.concat([x.parts[level]]), true);
            if (children.length > 0) {
                let cell = cols[level];
                let onclick = function () {
                    cell.classList.remove("link");
                    cell.removeEventListener("click", onclick);
                    children.forEach(function (x) {
                        x.classList.remove("hidden");
                    });
                };
                cell.classList.add("link");
                cell.addEventListener("click", onclick);
            }
        });

        if (othersTime > 0) {
            let cells = [];
            for (let i = 0; i < maxLength; i++) {
                cells.push(parent[i]);
            }
            cells.push(othersTime.toFixed(3));
            cells[level] = "others";
            let cols = createRow(table, "td", cells);
            for (let i = 0; i < level; i++) {
                cols[i].className = "muted";
            }

            let cell = cols[level];
            let tr = cell.parentNode;
            let onclick = function () {
                tr.classList.add("hidden");
                cell.classList.remove("link");
                cell.removeEventListener("click", onclick);
                othersRows.forEach(function (x) {
                    x.classList.remove("hidden");
                });
            };

            cell.title = othersList.join(", ");
            cell.classList.add("link");
            cell.addEventListener("click", onclick);

            if (hide) {
                tr.classList.add("hidden");
            }

            childrenRows.push(tr);
        }

        return childrenRows;
    };

    addLevel(0, []);

    return table;
}

function showProfile(path, cutoff = 0.05) {
    requestGet(path, {}, function (data) {
        data.records["total"] = data.total;
        const table = createVisualizationTable(data.records, cutoff, "number");
        popup(table);
    });
}
