from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modules.options import OptionInfo

from enum import Enum

import gradio as gr

from backend.memory_management import total_vram
from modules.shared_items import list_samplers, list_schedulers


class PresetArch(Enum):
    sd = 1
    xl = 2
    flux = 3
    qwen = 4
    lumina = 5
    wan = 6


SAMPLERS = {
    PresetArch.sd: "Euler a",
    PresetArch.xl: "DPM++ 2M SDE",
    PresetArch.flux: "Euler",
    PresetArch.qwen: "LCM",
    PresetArch.lumina: "Res Multistep",
    PresetArch.wan: "Euler",
}

SCHEDULERS = {
    PresetArch.sd: "Automatic",
    PresetArch.xl: "Karras",
    PresetArch.flux: "Beta",
    PresetArch.qwen: "Normal",
    PresetArch.lumina: "Linear Quadratic",
    PresetArch.wan: "Simple",
}

WIDTH = {
    PresetArch.sd: 512,
    PresetArch.xl: 896,
    PresetArch.flux: 896,
    PresetArch.qwen: 896,
    PresetArch.lumina: 1024,
    PresetArch.wan: 1152,
}

HEIGHT = {
    PresetArch.sd: 512,
    PresetArch.xl: 1152,
    PresetArch.flux: 1152,
    PresetArch.qwen: 1152,
    PresetArch.lumina: 1024,
    PresetArch.wan: 896,
}

CFG = {
    PresetArch.sd: 6.0,
    PresetArch.xl: 4.0,
    PresetArch.flux: 1.0,
    PresetArch.qwen: 1.0,
    PresetArch.lumina: 4.5,
    PresetArch.wan: 1.0,
}


def register(options_templates: dict, options_section: Callable, OptionInfo: "OptionInfo"):
    inference_vram = int(total_vram - (1024 if total_vram < 8200 else 2048))

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
                    f"{name}_t2i_sampler": OptionInfo(sampler, "txt2img sampler", gr.Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_t2i_scheduler": OptionInfo(scheduler, "txt2img scheduler", gr.Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{name}_i2i_sampler": OptionInfo(sampler, "img2img sampler", gr.Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_i2i_scheduler": OptionInfo(scheduler, "img2img scheduler", gr.Dropdown, lambda: {"choices": list_schedulers()}),
                },
            )
        )

        w, h, cfg = WIDTH[arch], HEIGHT[arch], CFG[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_width": OptionInfo(w, "txt2img Width", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
                    f"{name}_t2i_height": OptionInfo(h, "txt2img Height", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
                    f"{name}_t2i_cfg": OptionInfo(cfg, "txt2img CFG", gr.Slider, {"minimum": 1, "maximum": 30, "step": 0.1}),
                    f"{name}_t2i_hr_cfg": OptionInfo(cfg, "txt2img Hires. CFG", gr.Slider, {"minimum": 1, "maximum": 30, "step": 0.1}),
                    f"{name}_i2i_width": OptionInfo(w, "img2img Width", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
                    f"{name}_i2i_height": OptionInfo(h, "img2img Height", gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
                    f"{name}_i2i_cfg": OptionInfo(cfg, "img2img CFG", gr.Slider, {"minimum": 1, "maximum": 30, "step": 0.1}),
                    f"{name}_gpu_mb": OptionInfo(inference_vram, "GPU Weights (MB)", gr.Slider, {"visible": (arch is not PresetArch.sd), "minimum": 0, "maximum": total_vram, "step": 1}),
                },
            )
        )

    options_templates.update(
        options_section(
            ("ui_flux", "FLUX", "presets"),
            {
                "flux_t2i_d_cfg": OptionInfo(3.0, "txt2img Distilled CFG", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
                "flux_t2i_hr_d_cfg": OptionInfo(3.0, "txt2img Distilled Hires. CFG", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
                "flux_i2i_d_cfg": OptionInfo(3.0, "img2img Distilled CFG", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
            },
        )
    )

    options_templates.update(
        options_section(
            ("ui_lumina", "LUMINA", "presets"),
            {
                "lumina_t2i_d_cfg": OptionInfo(6.0, "txt2img Shift", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
                "lumina_t2i_hr_d_cfg": OptionInfo(6.0, "txt2img Hires. Shift", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
                "lumina_i2i_d_cfg": OptionInfo(6.0, "img2img Shift", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
            },
        )
    )

    options_templates.update(
        options_section(
            ("ui_wan", "WAN", "presets"),
            {
                "wan_t2i_d_cfg": OptionInfo(8.0, "txt2img Shift", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
                "wan_t2i_hr_d_cfg": OptionInfo(8.0, "txt2img Hires. Shift", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
                "wan_i2i_d_cfg": OptionInfo(8.0, "img2img Shift", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
            },
        )
    )
