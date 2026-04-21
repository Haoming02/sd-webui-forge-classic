from typing import Final

UNION_CONTROLNET_TYPES: Final[dict[str, int]] = {
    "OpenPose": 0,
    "Depth": 1,
    "Scribble/SoftEdge/Sketch": 2,
    "Canny/Lineart/MLSD": 3,
    "NormalMap": 4,
    "Segmentation": 5,
    "Tile": 6,
    "Inpaint": 7,
}


def convert_control_type(control_type: str) -> int | None:
    for keys, v in UNION_CONTROLNET_TYPES.items():
        if control_type in keys:
            return v

    return None
