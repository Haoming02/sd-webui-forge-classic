import torch

from backend.modules.k_model import KModel
from backend.patcher.base import ModelPatcher

try:
    from backend.nn.svdq import NunchakuModelMixin
except ImportError:
    NunchakuModelMixin = type(None)


class UnetPatcher(ModelPatcher):
    @classmethod
    def from_model(cls, model, diffusers_scheduler, config, k_predictor=None):
        model = KModel(model=model, diffusers_scheduler=diffusers_scheduler, k_predictor=k_predictor, config=config)
        if isinstance(model.diffusion_model, NunchakuModelMixin):
            return NunchakuPatcher(model, load_device=model.diffusion_model.load_device, offload_device=model.diffusion_model.offload_device, current_device=model.diffusion_model.initial_device)
        else:
            return UnetPatcher(model, load_device=model.diffusion_model.load_device, offload_device=model.diffusion_model.offload_device, current_device=model.diffusion_model.initial_device)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet_linked_list = None
        self.extra_preserved_memory_during_sampling = 0
        self.extra_model_patchers_during_sampling = []
        self.extra_concat_condition = None

    def clone(self):
        n = super().clone()
        n.controlnet_linked_list = self.controlnet_linked_list
        n.extra_preserved_memory_during_sampling = self.extra_preserved_memory_during_sampling
        n.extra_model_patchers_during_sampling = self.extra_model_patchers_during_sampling.copy()
        n.extra_concat_condition = self.extra_concat_condition
        return n

    def add_extra_preserved_memory_during_sampling(self, memory_in_bytes: int):
        """Use this to ask Forge to preserve a certain amount of memory during sampling"""
        self.extra_preserved_memory_during_sampling += memory_in_bytes

    def add_extra_model_patcher_during_sampling(self, model_patcher: ModelPatcher):
        """Use this to ask Forge to move extra model patchers to GPU during sampling"""
        self.extra_model_patchers_during_sampling.append(model_patcher)

    def add_extra_torch_module_during_sampling(self, m: torch.nn.Module, cast_to_unet_dtype: bool = True):
        """
        Use this method to bind an extra torch.nn.Module to this UNet during sampling.
        This model `m` will be delegated to the memory management system.
        - `m` will be loaded to GPU everytime when sampling starts
        - `m` will be unloaded if necessary
        - `m` will influence Forge's judgement about use GPU memory or capacity
        - Use `cast_to_unet_dtype` if you want `m` to have same dtype with unet during sampling
        """
        if cast_to_unet_dtype:
            m.to(self.model.diffusion_model.dtype)

        patcher = ModelPatcher(model=m, load_device=self.load_device, offload_device=self.offload_device)
        self.add_extra_model_patcher_during_sampling(patcher)
        return patcher

    def add_patched_controlnet(self, cnet):
        cnet.set_previous_controlnet(self.controlnet_linked_list)
        self.controlnet_linked_list = cnet

    def list_controlnets(self):
        results = []
        pointer = self.controlnet_linked_list
        while pointer is not None:
            results.append(pointer)
            pointer = pointer.previous_controlnet
        return results

    def append_model_option(self, k, v, ensure_uniqueness=False):
        if k not in self.model_options:
            self.model_options[k] = []
        if ensure_uniqueness and v in self.model_options[k]:
            return
        self.model_options[k].append(v)

    def append_transformer_option(self, k, v, ensure_uniqueness=False):
        if "transformer_options" not in self.model_options:
            self.model_options["transformer_options"] = {}
        to = self.model_options["transformer_options"]
        if k not in to:
            to[k] = []
        if ensure_uniqueness and v in to[k]:
            return
        to[k].append(v)

    def set_transformer_option(self, k, v):
        if "transformer_options" not in self.model_options:
            self.model_options["transformer_options"] = {}
        self.model_options["transformer_options"][k] = v

    def add_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option("conditioning_modifiers", modifier, ensure_uniqueness)

    def add_sampler_pre_cfg_function(self, modifier, ensure_uniqueness=False):
        self.append_model_option("sampler_pre_cfg_function", modifier, ensure_uniqueness)

    def add_alphas_cumprod_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option("alphas_cumprod_modifiers", modifier, ensure_uniqueness)

    def add_block_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option("block_modifiers", modifier, ensure_uniqueness)

    def add_block_inner_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option("block_inner_modifiers", modifier, ensure_uniqueness)

    def add_controlnet_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option("controlnet_conditioning_modifiers", modifier, ensure_uniqueness)

    def set_group_norm_wrapper(self, wrapper):
        self.set_transformer_option("group_norm_wrapper", wrapper)

    def set_controlnet_model_function_wrapper(self, wrapper):
        self.set_transformer_option("controlnet_model_function_wrapper", wrapper)

    def set_model_replace_all(self, patch, target="attn1"):
        for block_name in ["input", "middle", "output"]:
            for number in range(16):
                for transformer_index in range(16):
                    self.set_model_patch_replace(patch, target, block_name, number, transformer_index)

    def load_frozen_patcher(self, filename, state_dict, strength):
        patch_dict = {}
        for k, w in state_dict.items():
            model_key, patch_type, weight_index = k.split("::")
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        self.add_patches(filename=filename, patches=patch_flat, strength_patch=float(strength), strength_model=1.0)


class NunchakuPatcher(UnetPatcher):

    def load(self, device_to=None, *args, **kwargs):
        self.model.diffusion_model.to(device_to)

    def detach(self, *args, **kwargs):
        self.model.diffusion_model.to(self.offload_device)
