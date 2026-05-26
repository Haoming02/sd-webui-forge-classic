import torch
import torch.nn.functional as F
from lib_controllllite.lib_controllllite import LLLiteLoader
from lib_controllllite.lib_controllllite_anima import (
    ControlNetLLLiteDiT,
    infer_anima_config,
    load_lllite_weights_from_dict,
)

from backend.utils import load_torch_file
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher


class ControlLLLiteAnimaPatcher(ControlModelPatcher):

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any(k.startswith("lllite_dit") for k in state_dict):
            return None
        _, metadata = load_torch_file(ckpt_path, return_metadata=True)
        inpaint_masked_input: bool = (metadata or {}).get("lllite.inpaint_masked_input", None) == "true"
        return ControlLLLiteAnimaPatcher(state_dict, inpaint=inpaint_masked_input)

    def __init__(self, state_dict: dict[str, torch.Tensor], inpaint: bool):
        super().__init__()
        self.state_dict = state_dict
        self._is_inpaint = inpaint
        self._lllite_net: ControlNetLLLiteDiT = None

    @staticmethod
    def _is_black_on_white(image: torch.Tensor) -> bool:
        grey = image.mean(dim=1).squeeze(0)
        white_ratio = (grey > 0.9).float().mean().item()
        return white_ratio > 0.85

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet
        device, dtype = unet.load_device, unet.model.computation_dtype

        if self._lllite_net is None:
            dit = unet.model.diffusion_model
            cfg = infer_anima_config(self.state_dict)
            self._lllite_net = ControlNetLLLiteDiT(dit, **cfg)
            load_lllite_weights_from_dict(self._lllite_net, self.state_dict)
            self._lllite_net = self._lllite_net.eval().to(device=device, dtype=dtype)
            del self.state_dict

        if mask is not None and getattr(process, "inpainting_mask_invert", False):
            mask = 1.0 - mask

        if (not self._is_inpaint) and (mask is not None):
            if inv := self._is_black_on_white(cond):
                cond = 1.0 - cond
            cond *= mask
            if inv:
                cond = 1.0 - cond

        cond_image = cond * 2.0 - 1.0
        if self._is_inpaint:
            assert isinstance(mask, torch.Tensor)
            if mask.shape != cond_image.shape:
                mask = F.interpolate(mask, size=(cond_image.shape[2], cond_image.shape[3]), mode="nearest")
            inpaint_mask = mask.to(device=cond_image.device, dtype=cond_image.dtype)
            cond_image = cond_image * (inpaint_mask < 0.5)
            inpaint_mask = inpaint_mask * 2.0 - 1.0
            cond_image = torch.cat([cond_image, inpaint_mask], dim=1)

        self._lllite_net.set_cond_image(cond_image.to(device=device, dtype=dtype))
        self._lllite_net.set_multiplier(self.strength)
        self._lllite_net.set_step_range(num_steps=process.steps, start_percent=self.start_percent, end_percent=self.end_percent)
        self._lllite_net.apply_to()

    def process_after_every_sampling(self, *args, **kwargs):
        if self._lllite_net is not None:
            self._lllite_net.restore()


class ControlLLLitePatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any(k.startswith("lllite") for k in state_dict):
            return None
        return ControlLLLitePatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        if mask is not None:
            if getattr(process, "inpainting_mask_invert", False):
                mask = 1.0 - mask
            cond *= mask

        unet = LLLiteLoader.load_lllite(model=unet, state_dict=self.state_dict, cond_image=cond.movedim(1, -1), strength=self.strength, steps=process.steps, start_percent=self.start_percent, end_percent=self.end_percent)

        process.sd_model.forge_objects.unet = unet


add_supported_control_model(ControlLLLiteAnimaPatcher)

add_supported_control_model(ControlLLLitePatcher)
