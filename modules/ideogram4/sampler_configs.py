"""Ideogram 4.0 sampler presets (spec §4.6).

Mirrors the official ``src/ideogram4/sampler_configs.py`` ``PRESETS`` registry.

The two-stage guidance schedule runs the main steps at ``base_guidance`` (gw=7)
then a few "polish" steps at ``polish_guidance`` (gw=3).  Critically, the
official ``guidance_schedule`` is in **loop-INDEX order**: index 0 is the LAST
(polish) step.  We therefore emit ``[polish] * polish_steps + [base] * main``.

Pure-Python and dependency-free for unit testing without the model.
"""

from dataclasses import dataclass

BASE_GUIDANCE = 7.0
POLISH_GUIDANCE = 3.0


@dataclass(frozen=True)
class Preset:
    name: str
    steps: int
    mu: float
    std: float
    polish_steps: int
    base_guidance: float = BASE_GUIDANCE
    polish_guidance: float = POLISH_GUIDANCE

    @property
    def main_steps(self) -> int:
        return self.steps - self.polish_steps

    @property
    def guidance_schedule(self) -> list[float]:
        """Per-step guidance weights in loop-index order (index 0 = last/polish)."""
        return [self.polish_guidance] * self.polish_steps + [self.base_guidance] * self.main_steps


PRESETS: dict[str, Preset] = {
    "V4_QUALITY_48": Preset("V4_QUALITY_48", steps=48, mu=0.0, std=1.5, polish_steps=3),
    "V4_DEFAULT_20": Preset("V4_DEFAULT_20", steps=20, mu=0.0, std=1.75, polish_steps=2),
    "V4_TURBO_12": Preset("V4_TURBO_12", steps=12, mu=0.5, std=1.75, polish_steps=1),
}

DEFAULT_PRESET = "V4_QUALITY_48"


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> Preset:
    return PRESETS.get(name, PRESETS[DEFAULT_PRESET])
