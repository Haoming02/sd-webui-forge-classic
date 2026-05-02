from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_controllllite.lib_controllllite import LLLiteLoader
from lib_controllllite.lib_controllllite_anima import (
    ControlNetLLLiteDiT,
    infer_anima_config,
    load_lllite_weights_from_dict,
)


opLLLiteLoader = LLLiteLoader().load_lllite


class ControlLLLiteAnimaPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if any("lllite_dit" in k for k in state_dict):
            return ControlLLLiteAnimaPatcher(state_dict)
        return None

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self._lllite_net = None

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet
        dit = unet.model.diffusion_model

        if self._lllite_net is None:
            cfg = infer_anima_config(self.state_dict)
            self._lllite_net = ControlNetLLLiteDiT(dit, **cfg)
            load_lllite_weights_from_dict(self._lllite_net, self.state_dict)

        device = unet.load_device
        dtype = unet.model.computation_dtype
        self._lllite_net.to(device=device, dtype=dtype)

        # cond is (B, 3, H, W) in [0, 1]; conditioning1 expects [-1, 1]
        cond_image = cond * 2.0 - 1.0
        self._lllite_net.set_cond_image(cond_image.to(device=device, dtype=dtype))
        self._lllite_net.set_multiplier(self.strength)

        # Timestep range
        num_steps = process.steps
        start_step = round(num_steps * self.start_percent)
        end_step = round(num_steps * self.end_percent)
        self._lllite_net.set_step_range(num_steps, start_step, end_step)

        self._lllite_net.apply_to()

        process.sd_model.forge_objects.unet = unet

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        if self._lllite_net is not None:
            self._lllite_net.restore()
            self._lllite_net.clear_cond_image()


class ControlLLLitePatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if any("lllite_dit" in k for k in state_dict):
            return None  # handled by ControlLLLiteAnimaPatcher
        if not any("lllite" in k for k in state_dict):
            return None
        return ControlLLLitePatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = opLLLiteLoader(
            model=unet,
            state_dict=self.state_dict,
            cond_image=cond.movedim(1, -1),
            strength=self.strength,
            steps=process.steps,
            start_percent=self.start_percent,
            end_percent=self.end_percent
        )[0]

        process.sd_model.forge_objects.unet = unet
        return


# Anima must be registered first so its try_build_from_state_dict is checked first
add_supported_control_model(ControlLLLiteAnimaPatcher)
add_supported_control_model(ControlLLLitePatcher)
