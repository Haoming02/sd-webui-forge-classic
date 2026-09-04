import logging
import re

from networks import available_network_aliases

from backend.args import dynamic_args
from backend.logging import setup_logger
from backend.patcher.base import ModelPatcher, OnlineLoRAPatch
from modules import scripts, shared
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser

logger = logging.getLogger("lora_ctl")
setup_logger(logger)

lora_ctl = re.compile(r"<lora:([^\:\>]+):\[([^\]\>]+)\]>")


class LoRAControl(scripts.Script):
    mapping: dict[str, list[float]] = {}

    def title(self):
        return "LoRA Control Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def before_process(self, p: StableDiffusionProcessing):
        self.mapping.clear()

        matches = re.finditer(lora_ctl, p.prompt)
        ctl = False

        for m in matches:
            p.prompt = p.prompt.replace(f"[{m.group(2)}]", "-1.0")
            ctl = True

            alias: str = m.group(1)
            lora: str = available_network_aliases[alias].filename
            schedule: list[float] = [float(val) for val in m.group(2).split(":")]

            if len(schedule) not in (2, 3):
                logger.error(f'Invalid Syntax for "{alias}"\nOnly "[start:end]" or "[from:to:when]" are supported')
                continue

            self.mapping[lora] = schedule + [None] * (3 - len(schedule))

        if ctl and not dynamic_args.online_lora:
            logger.error("LoRA Control requires on-the-fly Patching")
            self.mapping.clear()

    @classmethod
    def adjust_lora(cls, params: CFGDenoiserParams):
        if not cls.mapping:
            return

        t: float = params.sampling_step / params.total_sampling_steps

        m: ModelPatcher = shared.sd_model.forge_objects.unet
        assert m.has_online_lora()

        patches: list[list[OnlineLoRAPatch]] = list(m.weight_wrapper_patches.values())
        for loras in patches:
            for lora in loras:
                for name, (from_, to_, switch_) in cls.mapping.items():
                    if lora.name != name:
                        continue

                    if switch_ is None:
                        lora.patch[0][0] = (1 - t) * from_ + t * to_
                    elif switch_ < 1.0:
                        lora.patch[0][0] = from_ if t < switch_ else to_
                    else:
                        lora.patch[0][0] = from_ if params.sampling_step < switch_ else to_


on_cfg_denoiser(LoRAControl.adjust_lora)
