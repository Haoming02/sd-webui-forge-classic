"""Ideogram 4.0 JSON caption verifier.

Faithful re-implementation of the official ``CaptionVerifier``
(github.com/ideogram-oss/ideogram4 ``src/ideogram4/caption_verifier.py``,
Apache-2.0), following the rules described in the integration spec §4.5 / §4.4.

Behaviour mirrors the official tool: verification produces *warnings only* and
never blocks generation (Ideogram explicitly tolerates schema deviations).  Each
``verify*`` method returns a ``list[str]`` of warnings; an empty list means the
caption is clean.

NOTE: when running against real weights, reconcile this with the upstream
``src/ideogram4/caption_verifier.py`` — this is a spec-driven re-implementation.

Pure-Python and dependency-free for unit testing without the model.
"""

import json
import re

HEX_RE = re.compile(r"^#[0-9A-F]{6}$")
# \uXXXX escape for a non-ASCII codepoint (>= 0080) — sign of ensure_ascii=True misuse
NON_ASCII_ESCAPE_RE = re.compile(r"\\u(?!00[0-7][0-9a-fA-F])[0-9a-fA-F]{4}")

TOP_LEVEL_ORDER = ["high_level_description", "style_description", "compositional_deconstruction"]
TOP_LEVEL_ALLOWED = set(TOP_LEVEL_ORDER)
REQUIRED_TOP_LEVEL = ["compositional_deconstruction"]

STYLE_ALLOWED = {"aesthetics", "lighting", "medium", "photo", "art_style", "color_palette"}
STYLE_REQUIRED = ["aesthetics", "lighting", "medium"]
STYLE_ORDER_PHOTO = ["aesthetics", "lighting", "photo", "medium", "color_palette"]
STYLE_ORDER_ART = ["aesthetics", "lighting", "medium", "art_style", "color_palette"]

COMP_ALLOWED = {"background", "elements"}

ELEMENT_ALLOWED = {"type", "bbox", "text", "desc", "color_palette"}
ELEMENT_ORDER_OBJ = ["type", "bbox", "desc", "color_palette"]
ELEMENT_ORDER_TEXT = ["type", "bbox", "text", "desc", "color_palette"]
ELEMENT_TYPES = {"obj", "text"}

MAX_STYLE_COLORS = 16
MAX_ELEMENT_COLORS = 5


def _check_order(actual_keys, expected_order, context, warnings):
    """Warn if the known keys appear in a different relative order than expected."""
    known_actual = [k for k in actual_keys if k in expected_order]
    expected = [k for k in expected_order if k in known_actual]
    if known_actual != expected:
        warnings.append(
            f"{context}: key order should be {expected} but got {known_actual}"
        )


def _check_palette(colors, limit, context, warnings):
    if not isinstance(colors, (list, tuple)):
        warnings.append(f"{context}: color_palette must be a list")
        return
    if len(colors) > limit:
        warnings.append(f"{context}: color_palette has {len(colors)} colors (max {limit})")
    for c in colors:
        if not isinstance(c, str) or not HEX_RE.match(c):
            warnings.append(f'{context}: invalid color "{c}" (must be upper-case #RRGGBB)')


def _check_bbox(bbox, context, warnings):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        warnings.append(f"{context}: bbox must be [y_min, x_min, y_max, x_max]")
        return
    for v in bbox:
        if not isinstance(v, int) or isinstance(v, bool):
            warnings.append(f"{context}: bbox values must be integers, got {v!r}")
        elif not (0 <= v <= 1000):
            warnings.append(f"{context}: bbox value {v} out of range 0–1000")


class CaptionVerifier:
    def verify(self, caption: dict) -> list[str]:
        """Verify an already-parsed caption dict. Returns a list of warnings."""
        warnings: list[str] = []

        if not isinstance(caption, dict):
            return ["caption must be a JSON object"]

        keys = list(caption.keys())

        for k in keys:
            if k not in TOP_LEVEL_ALLOWED:
                warnings.append(f"unknown top-level key: {k}")

        for req in REQUIRED_TOP_LEVEL:
            if req not in caption:
                warnings.append(f"missing required top-level key: {req}")

        _check_order(keys, TOP_LEVEL_ORDER, "top-level", warnings)

        if "style_description" in caption:
            self._verify_style(caption["style_description"], warnings)

        if "compositional_deconstruction" in caption:
            self._verify_comp(caption["compositional_deconstruction"], warnings)

        return warnings

    def _verify_style(self, style, warnings):
        if not isinstance(style, dict):
            warnings.append("style_description must be an object")
            return

        keys = list(style.keys())
        for k in keys:
            if k not in STYLE_ALLOWED:
                warnings.append(f"style_description: unknown key {k}")

        for req in STYLE_REQUIRED:
            if req not in style:
                warnings.append(f"style_description: missing required key {req}")

        has_photo = "photo" in style
        has_art = "art_style" in style
        if has_photo and has_art:
            warnings.append("style_description: 'photo' and 'art_style' are mutually exclusive")

        expected = STYLE_ORDER_PHOTO if has_photo else STYLE_ORDER_ART
        _check_order(keys, expected, "style_description", warnings)

        if "color_palette" in style:
            _check_palette(style["color_palette"], MAX_STYLE_COLORS, "style_description", warnings)

    def _verify_comp(self, comp, warnings):
        if not isinstance(comp, dict):
            warnings.append("compositional_deconstruction must be an object")
            return

        for k in comp.keys():
            if k not in COMP_ALLOWED:
                warnings.append(f"compositional_deconstruction: unknown key {k}")

        if "background" not in comp:
            warnings.append("compositional_deconstruction: missing required key background")
        elif not isinstance(comp["background"], str):
            warnings.append("compositional_deconstruction: background must be a string")

        if "elements" not in comp:
            warnings.append("compositional_deconstruction: missing required key elements")
            return

        elements = comp["elements"]
        if not isinstance(elements, list):
            warnings.append("compositional_deconstruction: elements must be a list")
            return

        for i, el in enumerate(elements):
            self._verify_element(el, i, warnings)

    def _verify_element(self, el, index, warnings):
        ctx = f"elements[{index}]"
        if not isinstance(el, dict):
            warnings.append(f"{ctx}: must be an object")
            return

        keys = list(el.keys())
        for k in keys:
            if k not in ELEMENT_ALLOWED:
                warnings.append(f"{ctx}: unknown key {k}")

        etype = el.get("type")
        if etype not in ELEMENT_TYPES:
            warnings.append(f'{ctx}: type must be "obj" or "text", got {etype!r}')

        if etype == "text" and "text" not in el:
            warnings.append(f"{ctx}: text element missing required 'text'")
        if etype == "obj" and "text" in el:
            warnings.append(f"{ctx}: obj element should not have a 'text' field")

        expected = ELEMENT_ORDER_TEXT if etype == "text" else ELEMENT_ORDER_OBJ
        _check_order(keys, expected, ctx, warnings)

        if "bbox" in el:
            _check_bbox(el["bbox"], ctx, warnings)
        if "color_palette" in el:
            _check_palette(el["color_palette"], MAX_ELEMENT_COLORS, ctx, warnings)

    def verify_raw(self, raw: str) -> list[str]:
        """Verify a raw JSON string: checks parseability + \\uXXXX misuse, then schema."""
        warnings: list[str] = []
        try:
            caption = json.loads(raw)
        except (ValueError, TypeError) as e:
            return [f"invalid JSON: {e}"]

        # \uXXXX escapes for non-ASCII with no literal non-ASCII present → likely ensure_ascii=True
        if NON_ASCII_ESCAPE_RE.search(raw) and raw.isascii():
            warnings.append(
                "caption contains \\uXXXX escapes for non-ASCII characters; "
                "serialize with ensure_ascii=False (spec §4.4.4)"
            )

        warnings.extend(self.verify(caption))
        return warnings

    def verify_file(self, path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            return self.verify_raw(f.read())
