"""Ideogram 4.0 structured JSON caption assembly.

Builds the structured JSON caption that Ideogram 4.0 is trained on, with the
*exact* key ordering the model expects.  The ordering is guaranteed by explicit
ordered builders here, NOT by relying on the insertion order of the caller's
input dict (spec §4.4, "実装方針").

Top-level order:           high_level_description → style_description → compositional_deconstruction
style_description (photo): aesthetics → lighting → photo → medium → color_palette
style_description (other): aesthetics → lighting → medium → art_style → color_palette
element (obj):             type → bbox → desc → color_palette
element (text):            type → bbox → text → desc → color_palette

All pure-Python and dependency-free so it can be unit-tested without the model.
"""

import json

_HEX_DIGITS = set("0123456789ABCDEF")

# value of `medium` that selects the photographic key ordering / `photo` field
PHOTO_MEDIUM = "photograph"

MEDIUM_CHOICES = ["photograph", "illustration", "3d_render", "painting", "graphic_design"]

MAX_STYLE_COLORS = 16
MAX_ELEMENT_COLORS = 5


def normalize_hex(value: str) -> str:
    """Normalize a colour to upper-case ``#RRGGBB`` (spec §4.4.4).

    ``#1b1b2f`` → ``#1B1B2F``.  A 3-digit shorthand like ``#fff`` is expanded to
    ``#FFFFFF`` for convenience; anything that is not a clean hex string is
    returned upper-cased/``#``-prefixed unchanged so the CaptionVerifier can flag
    it rather than silently corrupting the value.
    """
    if value is None:
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    if not s.startswith("#"):
        s = "#" + s
    body = s[1:]
    if len(body) == 3 and all(c in _HEX_DIGITS for c in body):
        body = "".join(c * 2 for c in body)
    return "#" + body


def normalize_palette(colors, limit: int) -> list[str]:
    """Normalize a list of colours, dropping blanks and capping at ``limit``."""
    if not colors:
        return []
    out = []
    for c in colors:
        h = normalize_hex(c)
        if h:
            out.append(h)
    return out[:limit]


def normalize_bbox(bbox):
    """Coerce a bbox to ``[y_min, x_min, y_max, x_max]`` ints clamped to 0–1000.

    Returns ``None`` when the input is empty / not 4 numbers (bbox is optional).
    """
    if bbox is None:
        return None
    try:
        vals = list(bbox)
    except TypeError:
        return None
    if len(vals) != 4:
        return None
    out = []
    for v in vals:
        if v is None or v == "":
            return None
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        out.append(max(0, min(1000, iv)))
    return out


def _clean(value) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value))


def build_style_description(style: dict) -> dict:
    """Build the ``style_description`` block in canonical key order, or ``{}``."""
    if not style:
        return {}

    aesthetics = _clean(style.get("aesthetics"))
    lighting = _clean(style.get("lighting"))
    medium = _clean(style.get("medium"))
    photo = _clean(style.get("photo"))
    art_style = _clean(style.get("art_style"))
    palette = normalize_palette(style.get("color_palette"), MAX_STYLE_COLORS)

    is_photo = medium.lower() == PHOTO_MEDIUM

    sd: dict = {}
    if aesthetics:
        sd["aesthetics"] = aesthetics
    if lighting:
        sd["lighting"] = lighting

    if is_photo:
        if photo:
            sd["photo"] = photo
        if medium:
            sd["medium"] = medium
    else:
        if medium:
            sd["medium"] = medium
        if art_style:
            sd["art_style"] = art_style

    if palette:
        sd["color_palette"] = palette

    return sd


def build_element(element: dict) -> dict:
    """Build one ``elements[]`` entry in canonical key order."""
    etype = _clean(element.get("type")) or "obj"
    bbox = normalize_bbox(element.get("bbox"))
    text = _clean(element.get("text"))
    desc = _clean(element.get("desc"))
    palette = normalize_palette(element.get("color_palette"), MAX_ELEMENT_COLORS)

    e: dict = {"type": etype}
    if bbox is not None:
        e["bbox"] = bbox
    if etype == "text" and text:
        e["text"] = text
    if desc:
        e["desc"] = desc
    if palette:
        e["color_palette"] = palette
    return e


def build_compositional_deconstruction(comp: dict) -> dict:
    """Build the (required) ``compositional_deconstruction`` block."""
    comp = comp or {}
    background = _clean(comp.get("background"))
    raw_elements = comp.get("elements") or []

    elements = []
    for el in raw_elements:
        if not el:
            continue
        built = build_element(el)
        # skip completely empty cards (only a default type and nothing else)
        if len(built) == 1 and not _clean(el.get("desc")) and not _clean(el.get("text")):
            continue
        elements.append(built)

    cd: dict = {}
    if background:
        cd["background"] = background
    if elements:
        cd["elements"] = elements
    return cd


def assemble_caption(data: dict) -> dict:
    """Assemble the full ordered caption dict from a loose input dict.

    ``data`` keys (all optional except ``compositional_deconstruction`` content):
        high_level_description: str
        style_description: dict
        compositional_deconstruction: dict {background, elements: [..]}
    """
    data = data or {}
    caption: dict = {}

    hld = _clean(data.get("high_level_description"))
    if hld:
        caption["high_level_description"] = hld

    style = build_style_description(data.get("style_description") or {})
    if style:
        caption["style_description"] = style

    comp = build_compositional_deconstruction(data.get("compositional_deconstruction") or {})
    if comp:
        caption["compositional_deconstruction"] = comp

    return caption


def dumps(caption: dict) -> str:
    """Serialize per spec §4.4.4: compact separators, keep non-ASCII literal."""
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def assemble_and_dump(data: dict) -> str:
    return dumps(assemble_caption(data))
