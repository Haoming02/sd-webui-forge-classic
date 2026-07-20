// various hints and extra info for the settings tab

let settingsHintsSetup = false;

onOptionsChanged(function () {
    if (settingsHintsSetup) return;
    settingsHintsSetup = true;

    gradioApp()
        .querySelectorAll("#settings [id^=setting_]")
        .forEach(function (div) {
            let name = div.id.substr(8);
            let commentBefore = opts._comments_before[name];
            let commentAfter = opts._comments_after[name];

            if (!commentBefore && !commentAfter) return;

            let span = null;
            if (div.classList.contains("gradio-checkbox")) {
                span = div.querySelector("label span");
            } else if (div.classList.contains("gradio-checkboxgroup")) {
                span = div.querySelector("span").firstChild;
            } else if (div.classList.contains("gradio-radio")) {
                span = div.querySelector("span").firstChild;
            } else {
                let elem = div.querySelector("label span");
                if (elem) span = elem.firstChild;
            }

            if (!span) return;

            if (commentBefore) {
                let comment = document.createElement("DIV");
                comment.className = "settings-comment";
                comment.innerHTML = commentBefore;
                span.parentElement.insertBefore(document.createTextNode("\xa0"), span);
                span.parentElement.insertBefore(comment, span);
                span.parentElement.insertBefore(document.createTextNode("\xa0"), span);
            }
            if (commentAfter) {
                comment = document.createElement("DIV");
                comment.className = "settings-comment";
                comment.innerHTML = commentAfter;
                span.parentElement.insertBefore(comment, span.nextSibling);
                span.parentElement.insertBefore(document.createTextNode("\xa0"), span.nextSibling);
            }
        });
});

function settingsHintsShowQuicksettings() {
    requestGet("./internal/quicksettings-hint", {}, function (data) {
        let table = document.createElement("table");
        table.className = "popup-table";

        data.forEach(function (obj) {
            let tr = document.createElement("tr");
            let td = document.createElement("td");
            td.textContent = obj.name;
            tr.appendChild(td);

            td = document.createElement("td");
            td.textContent = obj.label;
            tr.appendChild(td);

            table.appendChild(tr);
        });

        popup(table);
    });
}
