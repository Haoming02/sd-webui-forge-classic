let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

function settingsShowAllTabs() {
    gradioApp()
        .querySelectorAll("#settings > div")
        .forEach(function (elem) {
            if (settingsExcludeTabsFromShowAll[elem.id]) return;

            elem.style.display = "block";
        });
}

function settingsShowOneTab() {
    gradioApp().querySelector("#settings_show_one_page").click();
}

onUiLoaded(function () {
    let edit = gradioApp().querySelector("#settings_search");
    let editTextarea = gradioApp().querySelector(
        "#settings_search > label > input",
    );
    let buttonShowAllPages = gradioApp().getElementById(
        "settings_show_all_pages",
    );
    let settings_tabs = gradioApp().querySelector("#settings div");

    onEdit("settingsSearch", editTextarea, 300, function () {
        let searchText = (editTextarea.value || "").trim().toLowerCase();

        gradioApp()
            .querySelectorAll(
                "#settings > div[id^=settings_] div[id^=column_settings_] > *",
            )
            .forEach(function (elem) {
                let visible =
                    elem.textContent.trim().toLowerCase().indexOf(searchText) != -1;
                elem.style.display = visible ? "" : "none";
            });

        if (searchText != "") {
            settingsShowAllTabs();
        } else {
            settingsShowOneTab();
        }
    });

    settings_tabs.insertBefore(edit, settings_tabs.firstChild);
    settings_tabs.appendChild(buttonShowAllPages);

    buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
});

onOptionsChanged(function () {
    if (gradioApp().querySelector("#settings .settings-category")) return;

    let sectionMap = {};
    gradioApp()
        .querySelectorAll("#settings > div > button")
        .forEach(function (x) {
            sectionMap[x.textContent.trim()] = x;
        });

    opts._categories.forEach(function (x) {
        let section = localization[x[0]] ?? x[0];
        let category = localization[x[1]] ?? x[1];

        let span = document.createElement("SPAN");
        span.textContent = category;
        span.className = "settings-category";

        let sectionElem = sectionMap[section];
        if (!sectionElem) return;

        sectionElem.parentElement.insertBefore(span, sectionElem);
    });
});
