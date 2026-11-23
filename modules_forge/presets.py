import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.logging import setup_logger
from modules.paths_internal import presets_dir

logger = logging.getLogger("presets")
setup_logger(logger)


class PresetArch(Enum):
    sd = 1  # SD1
    xl = 2  # SDXL
    flux = 3  # Flux.1
    klein = 4  # Flux.2
    qwen = 5  # Qwen-Image
    lumina = 6  # Lumina-Image-2.0
    zit = 7  # Z-Image-Turbo
    wan = 8  # Wan2.2
    anima = 9  # Anima
    pid = 11  # PiD
    krea = 12  # Krea2
    qwen21 = 13  # Qwen-Image-2.1
    moss = 14  # MOSS-Image

    @staticmethod
    def choices() -> list[str]:
        return [preset.name for preset in PresetArch]


@dataclass
class Sampling:
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    step: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    batch_size: Optional[int] = None
    dcfg: Optional[float] = None
    shift: Optional[float] = None


@dataclass
class Preset:
    name: str
    arch: str
    t2i: Sampling
    hr: Sampling
    i2i: Sampling
    distill: Optional[bool] = False
    shift: Optional[bool] = False
    fps: int = 0

    def __post_init__(self):
        for field in ("t2i", "hr", "i2i"):
            value = getattr(self, field)
            if isinstance(value, dict):
                setattr(self, field, Sampling(**value))


def use_distill(preset: "Preset", default: "Preset") -> bool:
    return bool(preset.distill if preset.distill is not None else default.distill)


def use_shift(preset: "Preset", default: "Preset") -> bool:
    return bool(preset.shift if preset.shift is not None else default.shift)


def is_video(preset: "Preset", default: "Preset") -> int:
    return int(preset.fps or default.fps or 1)


def load_presets(disabled: set[str]) -> tuple[dict[str, "Preset"], dict[str, "Preset"]]:
    builtin = os.path.join(presets_dir, "builtin.json")
    assert os.path.isfile(builtin)

    with open(builtin, "r") as f:
        builtin_presets: list[dict] = json.load(f)

    assert len(builtin_presets) == len(PresetArch)
    assert {p["arch"] for p in builtin_presets} == set(PresetArch.choices())

    all_presets: dict[str, "Preset"] = dict()
    _loaded: set[str] = set()

    def register_preset(preset: dict[str, str | int | float]):
        assert isinstance(preset, dict)
        arch = preset["arch"].strip()
        assert arch in PresetArch.choices()
        name = preset["name"].strip()
        assert name not in _loaded
        if arch not in disabled:
            all_presets[name] = Preset(**preset)
            _loaded.add(name)

    for preset in builtin_presets:
        try:
            register_preset(preset)
        except Exception as e:
            logger.error(e)
            raise SystemExit

    defaults: dict[str, "Preset"] = {p.arch: p for p in all_presets.values()}

    for path in os.listdir(presets_dir):
        if not path.endswith(".json"):
            continue
        if path == "builtin.json":
            continue

        with open(os.path.join(presets_dir, path), "r") as f:
            _presets = json.load(f)
            if isinstance(_presets, dict):
                _presets = [_presets]

        for _preset in _presets:
            try:
                register_preset(_preset)
            except Exception as e:
                logger.error(e)

    logger.info(f"Loaded {len(all_presets)} Presets")
    return defaults, all_presets
