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
    mapping: dict[str, list[tuple[float, float]]] = {}

    def title(self):
        return "LoRA Control Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    @staticmethod
    def _parse_schedule(syntax: str) -> list[tuple[float, float]]:
        return [tuple(map(float, chunk.split("@"))) for chunk in syntax.split(",")]

    def before_process(self, p: StableDiffusionProcessing):
        self.mapping.clear()

        matches = re.finditer(lora_ctl, p.prompt)
        ctl = False

        for m in matches:
            p.prompt = p.prompt.replace(f"[{m.group(2)}]", "-1.0")
            ctl = True

            alias: str = m.group(1)
            lora: str = available_network_aliases[alias].filename

            try:
                self.mapping[lora] = self._parse_schedule(m.group(2))
                assert len(self.mapping[lora]) > 1
            except Exception:
                logger.error(f'Invalid Syntax for "{alias}": "{m.group(2)}"')
                continue

        if ctl and not dynamic_args.online_lora:
            logger.error("LoRA Control requires on-the-fly Patching")
            self.mapping.clear()

    @classmethod
    def adjust_lora(cls, params: CFGDenoiserParams):
        if not cls.mapping:
            return

        m: ModelPatcher = shared.sd_model.forge_objects.unet
        assert m.has_online_lora()

        patches: list[list[OnlineLoRAPatch]] = m.weight_wrapper_patches.values()
        t: float = params.sampling_step / params.total_sampling_steps

        for loras in patches:
            for lora in loras:
                for name, schedule in cls.mapping.items():
                    if lora.name != name:
                        continue

                    if schedule[-1][1] <= t:
                        w = schedule[-1][0]
                    else:
                        for i in range(len(schedule) - 1):
                            w1, t1 = schedule[i]
                            w2, t2 = schedule[i + 1]

                            if t1 <= t < t2:
                                ratio = (t - t1) / (t2 - t1)
                                w = w1 + (w2 - w1) * ratio
                                break

                    lora.patch[0][0] = w


on_cfg_denoiser(LoRAControl.adjust_lora)
