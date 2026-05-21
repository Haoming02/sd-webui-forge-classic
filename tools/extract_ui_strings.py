"""
Extract user-facing UI strings from the codebase to seed localization JSON files.

Walks selected source directories, parses each Python file with the `ast` module,
and harvests string literals passed as `label`, `info`, `placeholder`, `value`,
`tooltip`, and `title` keyword arguments on `gr.*` component constructors, plus
positional text in `gr.Markdown(...)` / `gr.HTML(...)` short calls.

The output is a JSON file mapping each English string to itself (placeholder),
ready to be copied per-language and translated. The A1111/Forge localization
system matches DOM text against these keys verbatim, so the key IS the source
string.

Usage:
    python tools/extract_ui_strings.py
    python tools/extract_ui_strings.py --out localizations/_template.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCAN_DIRS = [
    REPO_ROOT / "modules",
    REPO_ROOT / "modules_forge",
    REPO_ROOT / "extensions-builtin",
    REPO_ROOT / "scripts",
]

# Plain-JS files that hold object literals of user-visible strings the
# Forge localization system looks up at runtime (the `titles` dict in
# hints.js is the canonical example — its values become tooltip text).
JS_SCAN_FILES = [
    REPO_ROOT / "javascript" / "hints.js",
]

TARGET_KWARGS = {"label", "info", "placeholder", "tooltip", "title"}

# Top-level scripts that hand-curate strings the source can't reach
# (Gradio-native UI text like the file-upload widget, tab labels stored
# as positional tuple members in modules/ui.py, f-string-constructed
# button labels, etc.). Each entry is added to the template with itself
# as the placeholder value so the per-language dicts can override it.
MANUAL_STRINGS = [
    # Gradio file/image upload widget — not in Forge source
    "Drop Image Here",
    "Click to Upload",
    "or",
    # modules/ui.py interfaces[] positional tuple labels
    "txt2img",
    "img2img",
    "Extras",
    "PNG Info",
    "Checkpoint Merger",
    "Settings",
    "Extensions",
    # modules/infotext_utils.py::create_buttons — f-string `Send to {tab}`,
    # called with ["txt2img", "img2img", "inpaint", "extras"] from ui.py
    "Send to txt2img",
    "Send to img2img",
    "Send to inpaint",
    "Send to extras",
    # modules/ui.py::add_copy_image_controls — list literal iterated with zip
    "to img2img",
    "to sketch",
    "to inpaint",
    "to inpaint sketch",
    # modules/ui_common.py — ToolButton tooltip f"Save the image to a dedicated directory ({shared.opts.outdir_save})."
    # Default outdir_save is "output\\images" on Windows / "output/images" on Linux.
    "Save the image to a dedicated directory (output\\images).",
    "Save the image to a dedicated directory (output/images).",
    "Save zip archive with images to a dedicated directory (output\\images)",
    "Save zip archive with images to a dedicated directory (output/images)",
    # modules/ui_common.py::create_refresh_button — tooltip f"{label}: refresh"
    "Upscaler 1: refresh",
    "Upscaler 2: refresh",
]

# Sections used by options_section((id, label, category)) — the second
# tuple element is the human-readable section title shown in Settings.
SECTION_LABEL_FUNCTION = "options_section"

GR_COMPONENTS_WITH_LABEL = {
    "Button", "Textbox", "Dropdown", "Checkbox", "CheckboxGroup", "Radio",
    "Slider", "Number", "ColorPicker", "Image", "Gallery", "File", "Files",
    "Code", "JSON", "Audio", "Video", "Tab", "TabItem", "Accordion", "Group",
    "Row", "Column", "Markdown", "HTML", "DataFrame", "Label", "State",
    "UploadButton", "ClearButton", "DownloadButton", "FormColorPicker",
    "DropdownMulti", "ToolButton",
}

COMPONENTS_WITH_POSITIONAL_LABEL = {
    "Tab", "TabItem", "Accordion", "Button", "UploadButton", "DownloadButton",
    "ClearButton", "ToolButton",
}

SKIP_STRINGS = {
    "", " ", "\n", "/", "|", ".", ",", ":", ";", "-", "_", "*", "x",
    "True", "False", "None", "null", "undefined",
}


def is_skippable(s: str) -> bool:
    if not s or len(s.strip()) == 0:
        return True
    if s in SKIP_STRINGS:
        return True
    if s.startswith("/") or s.startswith("\\"):
        return True
    if s.startswith("http://") or s.startswith("https://"):
        return True
    if s.startswith("#") and len(s) <= 8:
        return True
    if s.replace(".", "").replace("-", "").isdigit():
        return True
    if all(not c.isalpha() for c in s):
        return True
    if "{" in s and "}" in s and any(c in s for c in ("$", "%")):
        return True
    return False


def get_call_name(node: ast.Call) -> str | None:
    """Return the rightmost attribute name of a Call's func, e.g. gr.Button -> Button."""
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def is_gr_call(node: ast.Call) -> bool:
    fn = node.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        return fn.value.id in {"gr", "gradio"}
    return False


JS_DICT_KV = re.compile(
    r"""
    ^[ \t]*                                # leading indent
    (?P<kq>["'])                           # key quote
    (?P<key>(?:\\.|(?!(?P=kq)).)*)         # key body
    (?P=kq)
    [ \t]*:[ \t]*
    (?P<vq>["'])                           # value quote
    (?P<value>(?:\\.|(?!(?P=vq)).)*)       # value body
    (?P=vq)
    [ \t]*,?[ \t]*$                        # optional comma
    """,
    re.VERBOSE | re.MULTILINE,
)


def _unescape_js_string(s: str) -> str:
    """Resolve the small set of JS escape sequences we expect in hints.js."""
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\'", "'")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def harvest_js_file(path: Path) -> dict[str, list[tuple[str, int]]]:
    """Extract every "key": "value" pair (object-literal style) from a JS file.

    Forge's hints.js maps UI element text to tooltip text using this exact
    style, and the runtime translator looks up `localization[value]` — so
    each *value* is a string that has to land in the localization JSONs.
    """
    findings: dict[str, list[tuple[str, int]]] = {}
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings
    rel = path.relative_to(REPO_ROOT).as_posix()
    for match in JS_DICT_KV.finditer(source):
        value = _unescape_js_string(match.group("value"))
        if is_skippable(value):
            continue
        line = source.count("\n", 0, match.start()) + 1
        findings.setdefault(value, []).append((rel, line))
    return findings


def harvest_file(path: Path) -> dict[str, list[tuple[str, int]]]:
    """Return mapping: english_string -> list of (file, line) where it appears."""
    findings: dict[str, list[tuple[str, int]]] = {}
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return findings

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return findings

    rel = path.relative_to(REPO_ROOT).as_posix()

    # Scan `def title(self): return "..."` patterns. These are the
    # human-readable names that script subclasses return for the
    # Scripts dropdown (Prompt Matrix, X/Y/Z plot, …).
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "title":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant):
                if isinstance(sub.value.value, str):
                    s = sub.value.value
                    if not is_skippable(s):
                        findings.setdefault(s, []).append((rel, sub.lineno))

    # Scan top-level dict assignments named like `tooltips = {…}` or
    # `titles = {…}` — these are runtime lookup tables that hold UI
    # strings the JS / Python side later resolves through localization.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        target_names = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                target_names.append(tgt.id)
            elif isinstance(tgt, ast.Attribute):
                target_names.append(tgt.attr)
        if not any(n in {"tooltips", "titles", "hints", "tips"} for n in target_names):
            continue
        for v in node.value.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                s = v.value
                if not is_skippable(s):
                    findings.setdefault(s, []).append((rel, node.lineno))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = get_call_name(node)
        if call_name is None:
            continue

        is_relevant = is_gr_call(node) and call_name in GR_COMPONENTS_WITH_LABEL

        # Forge wraps several gradio components (ToolButton, FormRow, …) and
        # invokes them without the `gr.` prefix; treat any call whose tail
        # name is in our known-good set as relevant so their tooltip=/label=
        # kwargs are harvested too.
        if not is_relevant and call_name in GR_COMPONENTS_WITH_LABEL:
            is_relevant = True

        if is_relevant:
            for kw in node.keywords:
                if kw.arg not in TARGET_KWARGS:
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    s = kw.value.value
                    if is_skippable(s):
                        continue
                    findings.setdefault(s, []).append((rel, node.lineno))

            if call_name in {"Markdown", "HTML"}:
                # First positional argument
                if node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        s = first.value
                        if not is_skippable(s) and len(s) < 300:
                            findings.setdefault(s, []).append((rel, node.lineno))
                # `value=` kwarg (e.g. gr.HTML(value="<p>...</p>"))
                for kw in node.keywords:
                    if kw.arg != "value":
                        continue
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        s = kw.value.value
                        if not is_skippable(s) and len(s) < 300:
                            findings.setdefault(s, []).append((rel, node.lineno))

            if call_name in COMPONENTS_WITH_POSITIONAL_LABEL and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    s = first.value
                    if not is_skippable(s):
                        findings.setdefault(s, []).append((rel, node.lineno))

            # Radio / CheckboxGroup choices are almost always user-facing
            # short labels (Just resize, Weighted sum, A, B or C, …).
            # Don't scan Dropdown choices — those can be model names,
            # checkpoint paths, locale codes, etc.
            if call_name in {"Radio", "CheckboxGroup"}:
                for kw in node.keywords:
                    if kw.arg != "choices":
                        continue
                    seq = kw.value
                    if isinstance(seq, (ast.List, ast.Tuple)):
                        for elt in seq.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                s = elt.value
                                # Skip single-character placeholders like "B"/"C"
                                # — too ambiguous to translate safely.
                                if len(s) >= 2 and not is_skippable(s):
                                    findings.setdefault(s, []).append((rel, node.lineno))

            # Button / ToolButton / ClearButton etc. — `value=` kwarg is
            # the displayed label when no positional arg is given.
            if call_name in {"Button", "ToolButton", "UploadButton", "ClearButton", "DownloadButton"}:
                for kw in node.keywords:
                    if kw.arg != "value":
                        continue
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        s = kw.value.value
                        if not is_skippable(s):
                            findings.setdefault(s, []).append((rel, node.lineno))

        # options_section((section_id, "Human Section Title", category_id), {…})
        # — the second tuple element is the localizable label.
        if call_name == SECTION_LABEL_FUNCTION and node.args:
            first = node.args[0]
            if isinstance(first, ast.Tuple) and len(first.elts) >= 2:
                second = first.elts[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    s = second.value
                    if not is_skippable(s):
                        findings.setdefault(s, []).append((rel, node.lineno))

        if call_name == "OptionInfo":
            if len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    s = second.value
                    if not is_skippable(s):
                        findings.setdefault(s, []).append((rel, node.lineno))

            for kw in node.keywords:
                if kw.arg in {"infotext", "comment_after", "comment_before"}:
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        s = kw.value.value
                        if not is_skippable(s):
                            findings.setdefault(s, []).append((rel, node.lineno))

        if call_name == "info" and isinstance(node.func, ast.Attribute):
            if node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    s = first.value
                    if not is_skippable(s) and len(s) < 300:
                        findings.setdefault(s, []).append((rel, node.lineno))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "tools" / "translations" / "_template.json"),
        help="Output JSON path (default: tools/translations/_template.json)",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "tools" / "translations" / "_extraction_report.txt"),
        help="Text report with per-file counts and unmatched candidates",
    )
    args = parser.parse_args()

    all_strings: dict[str, list[tuple[str, int]]] = {}
    per_file: dict[str, int] = {}

    for base in SCAN_DIRS:
        if not base.is_dir():
            print(f"[skip] {base} not found", file=sys.stderr)
            continue
        for py in base.rglob("*.py"):
            file_findings = harvest_file(py)
            if file_findings:
                per_file[py.relative_to(REPO_ROOT).as_posix()] = sum(
                    len(v) for v in file_findings.values()
                )
            for s, refs in file_findings.items():
                all_strings.setdefault(s, []).extend(refs)

    for js_path in JS_SCAN_FILES:
        if not js_path.is_file():
            print(f"[skip] {js_path} not found", file=sys.stderr)
            continue
        file_findings = harvest_js_file(js_path)
        if file_findings:
            per_file[js_path.relative_to(REPO_ROOT).as_posix()] = sum(
                len(v) for v in file_findings.values()
            )
        for s, refs in file_findings.items():
            all_strings.setdefault(s, []).extend(refs)

    for s in MANUAL_STRINGS:
        all_strings.setdefault(s, []).append(("(manual)", 0))

    template = {s: s for s in sorted(all_strings.keys())}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report_lines = [
        f"UI string extraction report",
        f"=" * 60,
        f"Total unique strings: {len(template)}",
        f"Files with hits: {len(per_file)}",
        f"",
        f"Top 30 files by hit count:",
    ]
    for fname, n in sorted(per_file.items(), key=lambda x: -x[1])[:30]:
        report_lines.append(f"  {n:5d}  {fname}")
    report_path = Path(args.report)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {len(template)} strings to {out_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
