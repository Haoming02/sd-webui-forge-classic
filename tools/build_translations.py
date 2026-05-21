"""
Build localization JSON files for all supported languages.

Reads the master template at `tools/translations/_template.json` (produced by
`extract_ui_strings.py`) and produces six files in `localizations/`:

    en_US.json  identity (English to English)
    it_IT.json  Italian, fully translated
    fr_FR.json  French, placeholder (English keys, English values)
    de_DE.json  German, placeholder
    zh_CN.json  Simplified Chinese, placeholder
    ja_JP.json  Japanese, placeholder

The placeholders are intentional: they keep the file structure intact so
contributors can fill in translations one entry at a time. The A1111/Forge
localization system tolerates identity entries (it just renders the original
string).

For Italian, the per-string mapping lives in `tools/translations/it_IT.py`
so this script stays small and the translation file is easy to review/diff.

Usage:
    python tools/build_translations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"
LOCALIZATIONS_DIR = REPO_ROOT / "localizations"

TEMPLATE_PATH = TRANSLATIONS_DIR / "_template.json"

LANGUAGES = ["it_IT", "es_ES", "fr_FR", "de_DE", "zh_CN", "ja_JP"]


def load_translation_map(lang: str) -> dict[str, str]:
    """Load `tools/translations/<lang>.py` which must define IT_TRANSLATIONS dict."""
    py_path = TRANSLATIONS_DIR / f"{lang}.py"
    if not py_path.is_file():
        return {}
    namespace: dict = {}
    exec(py_path.read_text(encoding="utf-8"), namespace)
    return namespace.get("TRANSLATIONS", {})


def main() -> int:
    if not TEMPLATE_PATH.is_file():
        print(
            f"Template not found at {TEMPLATE_PATH}\n"
            f"Run `python tools/extract_ui_strings.py` first.",
            file=sys.stderr,
        )
        return 1

    template: dict[str, str] = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded template with {len(template)} strings")

    LOCALIZATIONS_DIR.mkdir(exist_ok=True)

    for lang in LANGUAGES:
        out_path = LOCALIZATIONS_DIR / f"{lang}.json"
        mapping = load_translation_map(lang)
        result = {}
        translated = 0
        for key in sorted(template.keys()):
            if key in mapping:
                result[key] = mapping[key]
                translated += 1
            else:
                result[key] = key
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        pct = (100 * translated // len(template)) if template else 0
        print(f"  {lang}: {translated}/{len(template)} translated ({pct}%) -> {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
