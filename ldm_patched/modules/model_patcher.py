"""
Credit: ComfyUI
https://github.com/comfyanonymous/ComfyUI

- Edited by. Forge Official
- Edited by. Haoming02
"""

import copy
import inspect
import logging
import time

import torch

import ldm_patched.modules.model_management
import ldm_patched.modules.utils
from ldm_patched.modules.args_parser import args
from ldm_patched.modules.lora import pad_tensor_to_shape, weight_decompose

logger = logging.getLogger(__name__)
extra_weight_calculators = {}  # backward compatibility


PERSISTENT_PATCHES = args.persistent_patches
if PERSISTENT_PATCHES:
    print("[Experimental] Persistent Patches:", PERSISTENT_PATCHES)


class PatchStatus:
    def __init__(self):
        self.t_apply: float = 0
        """the last `time` a Patch was actually **applied**"""
        self.t_added: float = 0
        """the last `time` a new Patch was **added**"""
        self.l_cache: list[tuple[str, float, float]] = None
        """the Patches that are **currently** applied"""

    def require_patch(self) -> bool:
        """whether a new Patch was added after the last application"""
        return self.t_apply < self.t_added

    def require_unpatch(self) -> bool:
        """whether the current Patches do not match the target Patches"""
        from modules.shared import cached_lora_hash

        return self.l_cache != cached_lora_hash

    def patch(self):
        """update the time when Patches are applied"""
        self.t_apply = time.time()
        self.sync()

    def sync(self):
        """update the current Patches to match the system Patches"""
        from modules.shared import cached_lora_hash

        self.l_cache = cached_lora_hash

    def on_add_patches(self):
        """signal that a new Patch was added"""
        self.t_added = time.time()


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, weight_inplace_update=False):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.current_device = self.offload_device if current_device is None else current_device
        self.weight_inplace_update = weight_inplace_update

        self.patch_status = PatchStatus()

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = ldm_patched.modules.model_management.module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self):
        n = ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            self.weight_inplace_update,
        )

        for k in self.patches:
            n.patches[k] = self.patches[k][:]

        n.backup = self.backup
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.patch_status = self.patch_status

        return n

    def is_clone(self, other):
        return getattr(other, "model", None) is self.model

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"])
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options["sampler_post_cfg_function"] = self.model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_vae_encode_wrapper(self, wrapper_function):
        self.model_options["model_vae_encode_wrapper"] = wrapper_function

    def set_model_vae_decode_wrapper(self, wrapper_function):
        self.model_options["model_vae_decode_wrapper"] = wrapper_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        to = self.model_options["transformer_options"]
        if "patches_replace" not in to:
            to["patches_replace"] = {}
        if name not in to["patches_replace"]:
            to["patches_replace"][name] = {}
        if transformer_index is not None:
            block = (block_name, number, transformer_index)
        else:
            block = (block_name, number)
        to["patches_replace"][name][block] = patch

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def model_patches_to(self, device, *, dtype=None):
        to: dict[str, dict[str, list["torch.Tensor"] | dict[str, "torch.Tensor"]]] = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device=device, dtype=dtype)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device=device, dtype=dtype)
        if "model_function_wrapper" in self.model_options:
            wrap_func: "torch.Tensor" = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device=device, dtype=dtype)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        self.patch_status.on_add_patches()
        return list(p)

    def get_key_patches(self, filter_prefix=None):
        ldm_patched.modules.model_management.unload_model_clones(self)
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            if k in self.patches:
                p[k] = [model_sd[k]] + self.patches[k]
            else:
                p[k] = (model_sd[k],)
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = ldm_patched.modules.utils.get_attr(self.model, k)
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old
            ldm_patched.modules.utils.set_attr_raw(self.model, k, self.object_patches[k])

        if not patch_weights:
            return self.model

        if PERSISTENT_PATCHES and self.patch_status.require_unpatch():
            self.unpatch_model(move=True)

        if self.patches and ((not PERSISTENT_PATCHES) or self.patch_status.require_patch()):
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    print(f'Could not patch as "{key}" does not exist in model...')
                    continue

                weight = model_sd[key]

                inplace_update = self.weight_inplace_update

                if key not in self.backup:
                    self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)

                if device_to is not None:
                    temp_weight = ldm_patched.modules.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
                else:
                    temp_weight = weight.to(torch.float32, copy=True)
                out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
                if inplace_update:
                    ldm_patched.modules.utils.copy_to_param(self.model, key, out_weight)
                else:
                    ldm_patched.modules.utils.set_attr(self.model, key, out_weight)
                del temp_weight

            self.patch_status.patch()
            logger.debug("Patch Model")

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            strength = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0][1](ldm_patched.modules.model_management.cast_to_device(v[0][0], weight.device, torch.float32, copy=True), inplace=True), key),)

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                diff: torch.Tensor = v[0]
                do_pad_weight = len(v) > 1 and v[1]["pad_weight"]
                if do_pad_weight and diff.shape != weight.shape:
                    logger.debug(f'Padding Weight "{key}" ({weight.shape} -> {diff.shape})')
                    weight = pad_tensor_to_shape(weight, diff.shape)

                if strength != 0.0:
                    if diff.shape != weight.shape:
                        logger.warning(f'SHAPE MISMATCH "{key}" WEIGHT NOT MERGED ({diff.shape} != {weight.shape})')
                    else:
                        weight += strength * ldm_patched.modules.model_management.cast_to_device(diff, weight.device, weight.dtype)

            elif patch_type == "lora":  # lora/locon
                mat1 = ldm_patched.modules.model_management.cast_to_device(v[0], weight.device, torch.float32)
                mat2 = ldm_patched.modules.model_management.cast_to_device(v[1], weight.device, torch.float32)
                dora_scale = v[4]
                reshape = v[5]

                if reshape is not None:
                    weight = pad_tensor_to_shape(weight, reshape)
                if v[2] is not None:
                    alpha = v[2] / mat2.shape[0]
                else:
                    alpha = 1.0
                if v[3] is not None:
                    mat3 = ldm_patched.modules.model_management.cast_to_device(v[3], weight.device, torch.float32)
                    final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                    mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)

                try:
                    lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, torch.float32)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logger.error(f"Failed to apply {patch_type} to {key}\n{e}")

            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dora_scale = v[8]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1_a, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w1_b, weight.device, torch.float32))
                else:
                    w1 = ldm_patched.modules.model_management.cast_to_device(w1, weight.device, torch.float32)

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, torch.float32))
                    else:
                        w2 = torch.einsum("i j k l, j r, i p -> p r k l", ldm_patched.modules.model_management.cast_to_device(t2, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w2_b, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w2_a, weight.device, torch.float32))
                else:
                    w2 = ldm_patched.modules.model_management.cast_to_device(w2, weight.device, torch.float32)

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha = v[2] / dim
                else:
                    alpha = 1.0

                try:
                    lora_diff = torch.kron(w1, w2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, torch.float32)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logger.error(f"Failed to apply {patch_type} to {key}\n{e}")

            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha = v[2] / w1b.shape[0]
                else:
                    alpha = 1.0

                w2a = v[3]
                w2b = v[4]
                dora_scale = v[7]
                if v[5] is not None:  # cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum("i j k l, j r, i p -> p r k l", ldm_patched.modules.model_management.cast_to_device(t1, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, torch.float32))
                    m2 = torch.einsum("i j k l, j r, i p -> p r k l", ldm_patched.modules.model_management.cast_to_device(t2, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, torch.float32))
                else:
                    m1 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w1a, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w1b, weight.device, torch.float32))
                    m2 = torch.mm(ldm_patched.modules.model_management.cast_to_device(w2a, weight.device, torch.float32), ldm_patched.modules.model_management.cast_to_device(w2b, weight.device, torch.float32))

                try:
                    lora_diff = (m1 * m2).reshape(weight.shape)
                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, torch.float32)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logger.error(f"Failed to apply {patch_type} to {key}\n{e}")

            elif patch_type == "glora":
                dora_scale = v[5]

                old_glora = False
                if v[3].shape[1] == v[2].shape[0] == v[0].shape[0] == v[1].shape[1]:
                    rank = v[0].shape[0]
                    old_glora = True

                if v[3].shape[0] == v[2].shape[1] == v[0].shape[1] == v[1].shape[0]:
                    if old_glora and v[1].shape[0] == weight.shape[0] and weight.shape[0] == weight.shape[1]:
                        pass
                    else:
                        old_glora = False
                        rank = v[1].shape[0]

                a1 = ldm_patched.modules.model_management.cast_to_device(v[0].flatten(start_dim=1), weight.device, torch.float32)
                a2 = ldm_patched.modules.model_management.cast_to_device(v[1].flatten(start_dim=1), weight.device, torch.float32)
                b1 = ldm_patched.modules.model_management.cast_to_device(v[2].flatten(start_dim=1), weight.device, torch.float32)
                b2 = ldm_patched.modules.model_management.cast_to_device(v[3].flatten(start_dim=1), weight.device, torch.float32)

                if v[4] is not None:
                    alpha = v[4] / rank
                else:
                    alpha = 1.0

                try:
                    if old_glora:
                        lora_diff = (torch.mm(b2, b1) + torch.mm(torch.mm(weight.flatten(start_dim=1).to(dtype=torch.float32), a2), a1)).reshape(weight.shape)  # old lycoris glora
                    else:
                        if weight.dim() > 2:
                            lora_diff = torch.einsum("o i ..., i j -> o j ...", torch.einsum("o i ..., i j -> o j ...", weight.to(dtype=torch.float32), a1), a2).reshape(weight.shape)
                        else:
                            lora_diff = torch.mm(torch.mm(weight.to(dtype=torch.float32), a1), a2).reshape(weight.shape)
                        lora_diff += torch.mm(b1, b2).reshape(weight.shape)

                    if dora_scale is not None:
                        weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, torch.float32)
                    else:
                        weight += ((strength * alpha) * lora_diff).type(weight.dtype)
                except Exception as e:
                    logger.error(f"Failed to apply {patch_type} to {key}\n{e}")

            else:
                logger.warning(f'Unrecognized/Unsupported Patch Type "{patch_type}"...')

        return weight

    def unpatch_model(self, device_to=None, *, move: bool = (not PERSISTENT_PATCHES)):
        if self.backup and move:
            keys = list(self.backup.keys())

            if self.weight_inplace_update:
                for k in keys:
                    ldm_patched.modules.utils.copy_to_param(self.model, k, self.backup[k])
            else:
                for k in keys:
                    ldm_patched.modules.utils.set_attr(self.model, k, self.backup[k])

            self.backup.clear()
            self.patch_status.sync()
            logger.debug("Unpatch Model")

        if device_to is not None:
            self.model.to(device_to)
            self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            ldm_patched.modules.utils.set_attr_raw(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()
