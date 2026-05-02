from lib_controllllite.lib_controllllite import LLLiteLoader
from lib_controllllite.lib_controllllite_anima import (
    ControlNetLLLiteDiT,
    infer_anima_config,
    load_lllite_weights_from_dict,
)

from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher


class ControlLLLiteAnimaPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any(k.startswith("lllite_dit") for k in state_dict):
            return None
        return ControlLLLiteAnimaPatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self._lllite_net = None

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet
        device, dtype = unet.load_device, unet.model.computation_dtype

        if self._lllite_net is None:
            dit = unet.model.diffusion_model
            cfg = infer_anima_config(self.state_dict)
            self._lllite_net = ControlNetLLLiteDiT(dit, **cfg)
            load_lllite_weights_from_dict(self._lllite_net, self.state_dict)
            self._lllite_net = self._lllite_net.eval().to(device=device, dtype=dtype)

        cond_image = cond * 2.0 - 1.0
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

        unet = LLLiteLoader.load_lllite(model=unet, state_dict=self.state_dict, cond_image=cond.movedim(1, -1), strength=self.strength, steps=process.steps, start_percent=self.start_percent, end_percent=self.end_percent)

        process.sd_model.forge_objects.unet = unet


add_supported_control_model(ControlLLLiteAnimaPatcher)

add_supported_control_model(ControlLLLitePatcher)
