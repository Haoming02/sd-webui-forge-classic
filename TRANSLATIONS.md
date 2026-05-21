# Translations Guide

The WebUI ships with a multilingual interface. The language dropdown in the
top-right of the WebUI (in the same row as **UI Preset** / **Checkpoint** /
**VAE**) swaps the language on the fly (a page reload is triggered automatically).

| File | Language | Status |
|---|---|---|
| (no file) | English | default — no JSON loaded |
| `localizations/it_IT.json` | Italiano | seeded |
| `localizations/es_ES.json` | Español | seeded |
| `localizations/fr_FR.json` | Français | seeded |
| `localizations/de_DE.json` | Deutsch | seeded |
| `localizations/zh_CN.json` | 简体中文 | seeded |
| `localizations/ja_JP.json` | 日本語 | seeded |

Each non-source language is seeded with a full machine-assisted translation.
Quality varies — **native-speaker review and refinement via PR is welcome and
expected**.

---

## How the localization system works

Inherited from AUTOMATIC1111 / Forge:

1. At startup, `modules.localization.list_localizations()` scans
   `localizations/*.json` and registers each file as a selectable language.
2. When the user picks a language, the backend serializes the JSON into
   `window.localization = {…}` and injects it into the page.
3. `javascript/localization.js` walks the DOM and replaces every text node
   whose value matches a key with the translated value.

**Keys are the literal English strings as they appear in the UI.** There is no
abstract key system — `"Generate"` translates to `"Genera"` because the JSON
contains `"Generate": "Genera"`.

---

## How to contribute a translation

1. Pick a target file, e.g. `localizations/fr_FR.json`.
2. Open it in any text editor.
3. For each entry, replace the English value with the French translation.
   **Do not change the keys.**

   ```diff
   - "Generate": "Generate",
   + "Generate": "Générer",
   ```

4. Save and reload the WebUI. Pick your language from the dropdown.
5. Open a Pull Request.

### Translation guidelines

- **Preserve technical terms.** Stable Diffusion vocabulary that is universal
  in English (CFG, VAE, LoRA, UNet, SDXL, ControlNet, Hires, fp16/bf16/fp32,
  sigma, eta, Karras, RNG, txt2img, img2img, infotext, MaHiRo, Flux, Wan,
  Spandrel, COCO, ONNX) is **not** translated.
- **Preserve HTML markup.** Tags like `<b>`, `<a href="…">`, `<ins>` and the
  text inside `class="…"` attributes must remain intact.
- **Preserve format specifiers.** `%s`, `%d`, `%.2f` and template tokens like
  `{prompt}` must remain unchanged and in the same position.
- **Preserve emoji and special characters.** `↙️`, `📂`, `✨` and similar must
  be kept where they appear.
- **Match the upstream tone.** The original UI is concise and somewhat
  informal — your translation should be the same.

---

## How to regenerate the JSON files from scratch

If upstream adds new UI strings, regenerate the template and rebuild:

```bash
# 1. Extract every English UI string into the template
python tools/extract_ui_strings.py

# 2. Merge per-language dictionaries into localizations/*.json
python tools/build_translations.py
```

`extract_ui_strings.py` walks `modules/`, `modules_forge/`, and
`extensions-builtin/`, parses each Python file with `ast`, and harvests every
literal string passed as `label=`, `info=`, `placeholder=`, `tooltip=`, or
`title=` on a `gr.*` component, plus the second positional argument of
`OptionInfo(…)`. The result is written to `tools/translations/_template.json`.

`build_translations.py` reads the template and, for each language `<lang>`,
applies `tools/translations/<lang>.py`'s `TRANSLATIONS` dict over the
template. Missing keys fall through to the English source.

### Adding a new language

1. Add the locale code to the `LANGUAGES` list in `tools/build_translations.py`
   (e.g. `"es_ES"` for Spanish).
2. Create `tools/translations/es_ES.py` with a `TRANSLATIONS = {...}` dict.
3. Run `python tools/build_translations.py`. A `localizations/es_ES.json` is
   generated automatically.

---

## How the in-UI language dropdown works

Defined in `modules_forge/main_entry.py::make_language_selector_ui`, called
from `modules/ui_settings.py::add_quicksettings` after the standard quicksettings
loop. Selecting a value:

1. Writes the new value to `shared.opts.localization`.
2. Persists `shared.opts` to disk.
3. Reloads the page via JavaScript so Gradio re-renders with the new
   translations injected.

The selector is pinned to the right of the quicksettings row by the CSS rule
`#quicksettings>div#forge_ui_language { margin-left: auto; ... }` in
`style.css`.

The same value is also reflected in **Settings → User Interface → Localization**
(both edit `shared.opts.localization`).
