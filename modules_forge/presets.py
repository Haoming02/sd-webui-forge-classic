from enum import Enum
import gradio as gr

from modules import shared


class PresetArch(Enum):
    sd = 1
    xl = 2
    flux = 3
    # sd3 = 4

    @staticmethod
    def choices():
        return tuple(arch.name for arch in PresetArch)


def register_ui_presets(total_vram):
    inference_vram = int(total_vram - 1024)

    for arch, w, h, cfg in zip(PresetArch.choices(), (512, 896, 896), (512, 1152, 1152), (7, 6, 1)):
        shared.options_templates.update(shared.options_section((f"ui_{arch}", arch.upper(), "presets"), {
            f"{arch}_t2i_width":  shared.OptionInfo(w,   "txt2img Width",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
            f"{arch}_t2i_height": shared.OptionInfo(h,   "txt2img Height",    gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
            f"{arch}_t2i_cfg":    shared.OptionInfo(cfg, "txt2img CFG",       gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
            f"{arch}_t2i_hr_cfg": shared.OptionInfo(cfg, "txt2img HiRes CFG", gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
            f"{arch}_i2i_width":  shared.OptionInfo(w,   "img2img Width",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
            f"{arch}_i2i_height": shared.OptionInfo(h,   "img2img Height",    gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
            f"{arch}_i2i_cfg":    shared.OptionInfo(cfg, "img2img CFG",       gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
            f"{arch}_gpu_mb":     shared.OptionInfo(inference_vram, "GPU Weights (MB)", gr.Slider, {"minimum": 0, "maximum": total_vram, "step": 1}),
        }))

    shared.options_templates.update(shared.options_section(("ui_flux", "FLUX", "presets"), {
        "flux_t2i_d_cfg": shared.OptionInfo(3.5, "txt2img Distilled CFG", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
        "flux_i2i_d_cfg": shared.OptionInfo(3.5, "img2img Distilled CFG", gr.Slider, {"minimum": 1, "maximum": 10, "step": 0.1}),
    }))


def register_sampler_presets(sampler_names, scheduler_names):
    for arch, sampler, scheduler in zip(PresetArch.choices(), ("Euler a", "DPM++ 2M SDE", "Euler"), ("Automatic", "Karras", "Simple")):
        shared.options_templates.update(shared.options_section((f"ui_{arch}", arch.upper(), "presets"), {
            f"{arch}_t2i_sampler":     shared.OptionInfo(sampler,   "txt2img sampler",   gr.Dropdown, {"choices": sampler_names}),
            f"{arch}_t2i_scheduler":   shared.OptionInfo(scheduler, "txt2img scheduler", gr.Dropdown, {"choices": scheduler_names}),
            f"{arch}_i2i_sampler":     shared.OptionInfo(sampler,   "img2img sampler",   gr.Dropdown, {"choices": sampler_names}),
            f"{arch}_i2i_scheduler":   shared.OptionInfo(scheduler, "img2img scheduler", gr.Dropdown, {"choices": scheduler_names}),
        }))
