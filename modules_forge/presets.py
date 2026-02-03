from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modules.options import OptionInfo

from enum import Enum


class PresetArch(Enum):
    sd = 1  # SD1
    xl = 2  # SDXL
    flux = 3  # Flux.1
    klein = 4  # Flux.2
    qwen = 5  # Qwen-Image
    lumina = 6  # Lumina-Image-2.0
    zit = 7  # Z-Image-Turbo
    wan = 8  # Wan2.2

    @staticmethod
    def choices() -> list[str]:
        return [preset.name for preset in PresetArch]


SAMPLERS = {
    PresetArch.sd: "Euler a",
    PresetArch.xl: "Euler a",
    PresetArch.flux: "Euler",
    PresetArch.klein: "Euler",
    PresetArch.qwen: "LCM",
    PresetArch.lumina: "Res Multistep",
    PresetArch.zit: "Euler",
    PresetArch.wan: "Euler",
}

SCHEDULERS = {
    PresetArch.sd: "Automatic",
    PresetArch.xl: "Automatic",
    PresetArch.flux: "Beta",
    PresetArch.klein: "Beta",
    PresetArch.qwen: "Normal",
    PresetArch.lumina: "Simple",
    PresetArch.zit: "Beta",
    PresetArch.wan: "Simple",
}

STEPS = {
    PresetArch.sd: 32,
    PresetArch.xl: 24,
    PresetArch.flux: 20,
    PresetArch.klein: 4,
    PresetArch.qwen: 8,
    PresetArch.lumina: 32,
    PresetArch.zit: 9,
    PresetArch.wan: 4,
}

CFG = {
    PresetArch.sd: 6.0,
    PresetArch.xl: 4.5,
    PresetArch.flux: 1.0,
    PresetArch.klein: 1.0,
    PresetArch.qwen: 1.0,
    PresetArch.lumina: 4.0,
    PresetArch.zit: 1.0,
    PresetArch.wan: 1.0,
}

DISTILL = {
    PresetArch.flux: 3.0,
}

SHIFT = {
    PresetArch.lumina: 6.0,
    PresetArch.zit: 9.0,
    PresetArch.wan: 5.0,
}


def register(options_templates: dict, options_section: Callable, OptionInfo: "OptionInfo"):
    from gradio import Dropdown, Slider

    from modules.shared_items import list_samplers, list_schedulers

    for arch in PresetArch:
        name = arch.name

        options_templates.update(
            options_section(
                (None, "Forge Hidden Options"),
                {
                    f"forge_checkpoint_{name}": OptionInfo(None),
                    f"forge_additional_modules_{name}": OptionInfo([]),
                },
            )
        )

        sampler, scheduler = SAMPLERS[arch], SCHEDULERS[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_sampler": OptionInfo(sampler, "txt2img sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_t2i_scheduler": OptionInfo(scheduler, "txt2img scheduler", Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{name}_i2i_sampler": OptionInfo(sampler, "img2img sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_i2i_scheduler": OptionInfo(scheduler, "img2img scheduler", Dropdown, lambda: {"choices": list_schedulers()}),
                },
            )
        )

        cfg = CFG[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_cfg": OptionInfo(cfg, "txt2img CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                    f"{name}_t2i_hr_cfg": OptionInfo(cfg, "txt2img Hires. CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                    f"{name}_i2i_cfg": OptionInfo(cfg, "img2img CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                },
            )
        )

        if (distill := DISTILL.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_t2i_dcfg": OptionInfo(distill, "txt2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(distill, "txt2img Hires. Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(distill, "img2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                    },
                )
            )

        if (shift := SHIFT.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_t2i_dcfg": OptionInfo(shift, "txt2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(shift, "txt2img Hires. Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(shift, "img2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                    },
                )
            )
