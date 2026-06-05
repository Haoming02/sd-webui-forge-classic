"""Ideogram 4.0 integration package for Forge Neo.

Public, dependency-light building blocks (pure logic — safe to import anywhere):
    caption.assemble_caption / dumps / assemble_and_dump
    caption_verifier.CaptionVerifier
    sampler_configs.PRESETS / get_preset / DEFAULT_PRESET

Heavy, model-dependent entry points (torch/diffusers imported lazily on call):
    pipeline.get_pipeline / call_pipeline
    processing.process_images_ideogram4
"""

from modules.ideogram4.caption import (
    assemble_and_dump,
    assemble_caption,
    dumps,
)
from modules.ideogram4.caption_verifier import CaptionVerifier
from modules.ideogram4.sampler_configs import (
    DEFAULT_PRESET,
    PRESETS,
    Preset,
    get_preset,
    preset_names,
)

__all__ = [
    "assemble_caption",
    "assemble_and_dump",
    "dumps",
    "CaptionVerifier",
    "PRESETS",
    "Preset",
    "get_preset",
    "preset_names",
    "DEFAULT_PRESET",
]
