# https://github.com/Comfy-Org/ComfyUI/blob/v0.27.1/comfy/model_patcher.py

"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import collections
import inspect
import logging
import uuid
from typing import Callable

import torch

from backend import memory_management, utils
from backend.float import stochastic_rounding
from backend.patcher.lora import merge_lora_to_weight, string_to_seed
from backend.quant_ops import QuantizedTensor


def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options


def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_pre_cfg_function"] = model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


def move_weight_functions(m, device):
    if device is None:
        return 0

    memory = 0
    if hasattr(m, "weight_function"):
        for f in m.weight_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)

    if hasattr(m, "bias_function"):
        for f in m.bias_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)
    return memory


class LowVramPatch:
    is_lowvram_patch = True

    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.convert_func = convert_func  # TODO: remove
        self.set_func = set_func
        self.prepared_patches = None

    def memory_required(self):
        counter = [0]
        for patch in self.patches[self.key]:
            prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, destination, stream, copy=True, commit=True):
        counter = [0]
        prepared_patches = [(patch[0], prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4]) for patch in self.patches[self.key]]
        if commit:
            self.prepared_patches = prepared_patches
        return prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
        return merge_lora_to_weight(patches, weight, self.key, intermediate_dtype=weight.dtype)


LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR = 2


def low_vram_patch_estimate_vram(model, key):
    weight, set_func, convert_func = get_key_weight(model, key)
    if weight is None:
        return 0
    model_dtype = getattr(model, "manual_cast_dtype", torch.float32)
    if model_dtype is None:
        model_dtype = weight.dtype

    return weight.numel() * model_dtype.itemsize * LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR


def get_key_weight(model, key):
    set_func = None
    convert_func = None
    op_keys = key.rsplit(".", 1)
    if len(op_keys) < 2:
        weight = utils.get_attr(model, key)
    else:
        op = utils.get_attr(model, op_keys[0])
        try:
            set_func = getattr(op, "set_{}".format(op_keys[1]))
        except AttributeError:
            pass

        try:
            convert_func = getattr(op, "convert_{}".format(op_keys[1]))
        except AttributeError:
            pass

        weight = getattr(op, op_keys[1])
        if convert_func is not None:
            weight = utils.get_attr(model, key)

    return weight, set_func, convert_func


def key_param_name_to_key(key, param):
    if len(key) == 0:
        return param
    return "{}.{}".format(key, param)


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        self.size = size
        self.model = model
        if not hasattr(self.model, "device"):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        self.patches = {}
        self.backup = {}
        self.backup_buffers = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        self.model_options = {"transformer_options": {}}
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self.force_cast_weights = False
        self.patches_uuid = uuid.uuid4()
        self.parent = None
        self.pinned = set()

        self.cached_patcher_init: tuple[Callable, tuple] | tuple[Callable, tuple, int] | None = None
        self.clone_base_uuid = uuid.uuid4()

        if not hasattr(self.model, "model_loaded_weight_memory"):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, "lowvram_patch_counter"):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, "model_lowvram"):
            self.model.model_lowvram = False

        if not hasattr(self.model, "current_weight_patches_uuid"):
            self.model.current_weight_patches_uuid = None

        if not hasattr(self.model, "model_offload_buffer_memory"):
            self.model.model_offload_buffer_memory = 0

    def is_dynamic(self):
        return False

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = memory_management.module_size(self.model)
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def lowvram_patch_counter(self):
        return self.model.lowvram_patch_counter

    def get_free_memory(self, device):
        return memory_management.get_free_memory(device)

    def get_clone_model_override(self):
        return self.model, (self.backup, self.backup_buffers, self.object_patches_backup, self.pinned)

    def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
        class_ = self.__class__
        if self.is_dynamic() and disable_dynamic or force_deepcopy:
            if self.is_dynamic() and disable_dynamic:
                class_ = ModelPatcher
            if model_override is None:
                if self.cached_patcher_init is None:
                    raise RuntimeError("Cannot create non-dynamic delegate: cached_patcher_init is not initialized.")
                temp_model_patcher = self.cached_patcher_init[0](*self.cached_patcher_init[1], disable_dynamic=True)
                if len(self.cached_patcher_init) > 2:
                    temp_model_patcher = temp_model_patcher[self.cached_patcher_init[2]]
                model_override = temp_model_patcher.get_clone_model_override()
        if model_override is None:
            model_override = self.get_clone_model_override()

        n = class_(model_override[0], self.load_device, self.offload_device, self.model_size(), weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.model_options = deepcopy_list_dict(self.model_options)
        n.parent = self

        n.force_cast_weights = self.force_cast_weights

        n.backup, n.backup_buffers, n.object_patches_backup, n.pinned = model_override[1]

        n.cached_patcher_init = self.cached_patcher_init
        n.clone_base_uuid = self.clone_base_uuid

        return n

    def is_clone(self, other):
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone: ModelPatcher):
        if not self.is_clone(clone):
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logging.warning("WARNING: something went wrong, same patch uuid but different length of patches.")
            else:
                return True

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def disable_model_cfg1_optimization(self):
        self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"])  # Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.disable_model_cfg1_optimization()

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, sampler_calc_cond_batch_function):
        self.model_options["sampler_calc_cond_batch_function"] = sampler_calc_cond_batch_function

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

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

    def set_model_emb_patch(self, patch):
        self.set_model_patch(patch, "emb_patch")

    def set_model_forward_timestep_embed_patch(self, patch):
        self.set_model_patch(patch, "forward_timestep_embed_patch")

    def set_model_double_block_patch(self, patch):
        self.set_model_patch(patch, "double_block")

    def set_model_post_input_patch(self, patch):
        self.set_model_patch(patch, "post_input")

    def set_model_noise_refiner_patch(self, patch):
        self.set_model_patch(patch, "noise_refiner")

    def set_model_middle_block_after_patch(self, patch):
        self.set_model_patch(patch, "middle_block_after_patch")

    def set_model_rope_options(self, scale_x, shift_x, scale_y, shift_y, scale_t, shift_t, **kwargs):
        rope_options = self.model_options["transformer_options"].get("rope_options", {})
        rope_options["scale_x"] = scale_x
        rope_options["scale_y"] = scale_y
        rope_options["scale_t"] = scale_t

        rope_options["shift_x"] = shift_x
        rope_options["shift_y"] = shift_y
        rope_options["shift_t"] = shift_t

        self.model_options["transformer_options"]["rope_options"] = rope_options

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_compute_dtype(self, dtype):
        self.add_object_patch("manual_cast_dtype", dtype)
        if dtype is not None:
            self.force_cast_weights = True
        self.patches_uuid = uuid.uuid4()  # TODO: optimize by preventing a full model reload for this

    def add_weight_wrapper(self, name, function):
        self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]
        self.patches_uuid = uuid.uuid4()

    def get_model_object(self, name: str) -> torch.nn.Module:
        """Retrieves a nested attribute from an object using dot notation considering
        object patches.

        Args:
            name (str): The attribute path using dot notation (e.g. "model.layer.weight")

        Returns:
            The value of the requested attribute

        Example:
            patcher = ModelPatcher()
            weight = patcher.get_model_object("layer1.conv.weight")
        """
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_patches_models(self):
        to = self.model_options["transformer_options"]
        models = []
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "models"):
                        models += patch_list[i].models()
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "models"):
                        models += patch_list[k].models()
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "models"):
                models += wrap_func.models()

        return models

    def model_patches_call_function(self, function_name="cleanup", arguments={}):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], function_name):
                        getattr(patch_list[i], function_name)(**arguments)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], function_name):
                        getattr(patch_list[k], function_name)(**arguments)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, function_name):
                getattr(wrap_func, function_name)(**arguments)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        model_sd = self.model.state_dict()
        for k in patches:
            offset = None
            function = None
            if isinstance(k, str):
                key = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]

            if key in model_sd:
                p.add(k)
                current_patches = self.patches.get(key, [])
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches

        self.patches_uuid = uuid.uuid4()
        return list(p)

    def get_key_patches(self, filter_prefix=None):
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            bk = self.backup.get(k, None)
            weight, set_func, convert_func = get_key_weight(self.model, k)
            if bk is not None:
                weight = bk.weight
            if convert_func is None:
                convert_func = lambda a, **kwargs: a

            if k in self.patches:
                p[k] = [(weight, convert_func)] + self.patches[k]
            else:
                p[k] = [(weight, convert_func)]
        return p

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if key not in self.patches and not force_cast:
            return weight

        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup and not return_weight:
            self.backup[key] = collections.namedtuple("Dimension", ["weight", "inplace_update"])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        temp_dtype = memory_management.lora_compute_dtype(device_to) if key in self.patches else None
        if device_to is not None:
            temp_weight = memory_management.cast_to_device(weight, device_to, temp_dtype, copy=True)
        else:
            temp_weight = weight.to(temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = merge_lora_to_weight(self.patches[key], temp_weight, key) if key in self.patches else temp_weight
        if set_func is None:
            if key in self.patches:
                out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
            if return_weight:
                return out_weight
            elif inplace_update:
                utils.copy_to_param(self.model, key, out_weight)
            else:
                utils.set_attr(self.model, key, out_weight)
        else:
            return set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key), return_weight=return_weight)

    def pin_weight_to_device(self, key):
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if memory_management.pin_memory(weight):
            self.pinned.add(key)

    def unpin_weight(self, key):
        if key in self.pinned:
            weight, set_func, convert_func = get_key_weight(self.model, key)
            memory_management.unpin_memory(weight)
            self.pinned.remove(key)

    def unpin_all_weights(self):
        for key in list(self.pinned):
            self.unpin_weight(key)

    def _load_list(self, for_dynamic=False, default_device=None):
        loading = []
        for n, m in self.model.named_modules():
            default = False
            params = {name: param for name, param in m.named_parameters(recurse=False)}
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    default = True  # default random weights in non leaf modules
                    break
            if default and default_device is not None:
                for param_name, param in params.items():
                    param.data = param.data.to(device=default_device, dtype=getattr(m, param_name + "_comfy_model_dtype", None))
            if not default and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                module_mem = memory_management.module_size(m)
                module_offload_mem = module_mem
                if hasattr(m, "comfy_cast_weights"):

                    def check_module_offload_mem(key):
                        if key in self.patches:
                            return low_vram_patch_estimate_vram(self.model, key)
                        model_dtype = getattr(self.model, "manual_cast_dtype", None)
                        weight, _, _ = get_key_weight(self.model, key)
                        if model_dtype is None or weight is None:
                            return 0
                        if weight.dtype != model_dtype or isinstance(weight, QuantizedTensor):
                            return weight.numel() * model_dtype.itemsize
                        return 0

                    module_offload_mem += check_module_offload_mem("{}.weight".format(n))
                    module_offload_mem += check_module_offload_mem("{}.bias".format(n))
                # Dynamic: small weights (<64KB) first, then larger weights prioritized by size.
                # Non-dynamic: prioritize by module offload cost.
                if for_dynamic:
                    sort_criteria = (module_offload_mem >= 64 * 1024, -module_offload_mem)
                else:
                    sort_criteria = (module_offload_mem,)
                loading.append(sort_criteria + (module_mem, n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        lowvram_mem_counter = 0
        loading = self._load_list()

        load_completely = []
        offloaded = []
        offload_buffer = 0
        loading.sort(reverse=True)
        for i, x in enumerate(loading):
            module_offload_mem, module_mem, n, m, params = x

            lowvram_weight = False

            potential_offload = max(offload_buffer, module_offload_mem + sum([x1[1] for x1 in loading[i + 1 : i + 1 + memory_management.NUM_STREAMS]]))
            lowvram_fits = mem_counter + module_mem + potential_offload < lowvram_model_memory

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if not full_load and hasattr(m, "comfy_cast_weights"):
                if not lowvram_fits:
                    offload_buffer = potential_offload
                    lowvram_weight = True
                    lowvram_counter += 1
                    lowvram_mem_counter += module_mem
                    if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                        continue

            cast_weight = self.force_cast_weights
            m.comfy_force_cast_weights = self.force_cast_weights
            if lowvram_weight:
                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []

                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        _, set_func, convert_func = get_key_weight(self.model, weight_key)
                        m.weight_function = [LowVramPatch(weight_key, self.patches, convert_func, set_func)]
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        _, set_func, convert_func = get_key_weight(self.model, bias_key)
                        m.bias_function = [LowVramPatch(bias_key, self.patches, convert_func, set_func)]
                        patch_counter += 1

                cast_weight = True
                offloaded.append((module_mem, n, m, params))
            else:
                if hasattr(m, "comfy_cast_weights"):
                    wipe_lowvram_weight(m)

                if full_load or lowvram_fits:
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m, params))
                else:
                    offload_buffer = potential_offload

            if cast_weight and hasattr(m, "comfy_cast_weights"):
                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True

            if weight_key in self.weight_wrapper_patches:
                m.weight_function.extend(self.weight_wrapper_patches[weight_key])

            if bias_key in self.weight_wrapper_patches:
                m.bias_function.extend(self.weight_wrapper_patches[bias_key])

            mem_counter += move_weight_functions(m, device_to)

        load_completely.sort(reverse=True)
        for x in load_completely:
            n = x[1]
            m = x[2]
            params = x[3]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue

            for param in params:
                key = key_param_name_to_key(n, param)
                self.unpin_weight(key)
                self.patch_weight_to_device(key, device_to=device_to)
            if memory_management.is_device_cuda(device_to):
                torch.cuda.synchronize()

            logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
            m.comfy_patched_weights = True

        for x in load_completely:
            x[2].to(device_to)

        for x in offloaded:
            n = x[1]
            params = x[3]
            for param in params:
                self.pin_weight_to_device(key_param_name_to_key(n, param))

        usable_stat = "{:.2f} MB usable,".format(lowvram_model_memory / (1024 * 1024)) if lowvram_model_memory < 1e32 else ""
        if lowvram_counter > 0:
            logging.info("loaded partially; {} {:.2f} MB loaded, {:.2f} MB offloaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(usable_stat, mem_counter / (1024 * 1024), lowvram_mem_counter / (1024 * 1024), offload_buffer / (1024 * 1024), patch_counter))
            self.model.model_lowvram = True
        else:
            logging.info("loaded completely; {} {:.2f} MB loaded, full load: {}".format(usable_stat, mem_counter / (1024 * 1024), full_load))
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

        self.model.lowvram_patch_counter += patch_counter
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter
        self.model.model_offload_buffer_memory = offload_buffer
        self.model.current_weight_patches_uuid = self.patches_uuid

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        for k in self.object_patches:
            old = utils.set_attr_raw(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if lowvram_model_memory == 0:
            full_load = True
        else:
            full_load = False

        if load_weights:
            self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            self.unpin_all_weights()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    utils.copy_to_param(self.model, k, bk.weight)
                else:
                    utils.set_attr(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            utils.set_attr_raw(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        memory_freed = 0
        patch_counter = 0
        unload_list = self._load_list()
        unload_list.sort()

        offload_buffer = self.model.model_offload_buffer_memory
        if len(unload_list) > 0:
            NS = memory_management.NUM_STREAMS
            offload_weight_factor = [min(offload_buffer / (NS + 1), unload_list[0][1])] * NS

        for unload in unload_list:
            if memory_to_free + offload_buffer - self.model.model_offload_buffer_memory < memory_freed:
                break
            module_offload_mem, module_mem, n, m, params = unload

            potential_offload = module_offload_mem + sum(offload_weight_factor)

            lowvram_possible = hasattr(m, "comfy_cast_weights")
            if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                move_weight = True
                for param in params:
                    key = key_param_name_to_key(n, param)
                    bk = self.backup.get(key, None)
                    if bk is not None:
                        if not lowvram_possible:
                            move_weight = False
                            break

                        if bk.inplace_update:
                            utils.copy_to_param(self.model, key, bk.weight)
                        else:
                            utils.set_attr(self.model, key, bk.weight)
                        self.backup.pop(key)

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)
                if move_weight:
                    cast_weight = self.force_cast_weights
                    m.to(device_to)
                    module_mem += move_weight_functions(m, device_to)
                    if lowvram_possible:
                        if weight_key in self.patches:
                            if force_patch_weights:
                                self.patch_weight_to_device(weight_key)
                            else:
                                _, set_func, convert_func = get_key_weight(self.model, weight_key)
                                m.weight_function.append(LowVramPatch(weight_key, self.patches, convert_func, set_func))
                                patch_counter += 1
                        if bias_key in self.patches:
                            if force_patch_weights:
                                self.patch_weight_to_device(bias_key)
                            else:
                                _, set_func, convert_func = get_key_weight(self.model, bias_key)
                                m.bias_function.append(LowVramPatch(bias_key, self.patches, convert_func, set_func))
                                patch_counter += 1
                        cast_weight = True

                    if cast_weight and hasattr(m, "comfy_cast_weights"):
                        m.prev_comfy_cast_weights = m.comfy_cast_weights
                        m.comfy_cast_weights = True
                    m.comfy_patched_weights = False
                    memory_freed += module_mem
                    offload_buffer = max(offload_buffer, potential_offload)
                    offload_weight_factor.append(module_mem)
                    offload_weight_factor.pop(0)
                    logging.debug("freed {}".format(n))

                    for param in params:
                        self.pin_weight_to_device(key_param_name_to_key(n, param))

        self.model.model_lowvram = True
        self.model.lowvram_patch_counter += patch_counter
        self.model.model_loaded_weight_memory -= memory_freed
        self.model.model_offload_buffer_memory = offload_buffer
        logging.info("Unloaded partially: {:.2f} MB freed, {:.2f} MB remains loaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(memory_freed / (1024 * 1024), self.model.model_loaded_weight_memory / (1024 * 1024), offload_buffer / (1024 * 1024), self.model.lowvram_patch_counter))
        return memory_freed

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
        # TODO: force_patch_weights should not unload + reload full model
        used = self.model.model_loaded_weight_memory
        self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
        if unpatch_weights:
            extra_memory += used - self.model.model_loaded_weight_memory

        self.patch_model(load_weights=False)
        if extra_memory < 0 and not unpatch_weights:
            self.partially_unload(self.offload_device, -extra_memory, force_patch_weights=force_patch_weights)
            return 0
        full_load = False
        if self.model.model_lowvram == False and self.model.model_loaded_weight_memory > 0:
            return 0
        if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
            full_load = True
        current_used = self.model.model_loaded_weight_memory
        try:
            self.load(device_to, lowvram_model_memory=current_used + extra_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        except Exception as e:
            self.detach()
            raise e

        return self.model.model_loaded_weight_memory - current_used

    def pinned_memory_size(self):
        # Pinned memory pressure tracking is only implemented for DynamicVram loading
        return 0

    def loaded_ram_size(self):
        # Loaded RAM pressure tracking is only implemented for DynamicVram loading
        return 0

    def partially_unload_ram(self, ram_to_unload):
        return 0

    def detach(self, unpatch_all=True):
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        return self.model

    def current_loaded_device(self):
        return self.model.device

    def cleanup(self):
        self.model_patches_call_function(function_name="cleanup")
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None

    def pre_run(self):
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = self

    def __del__(self):
        self.unpin_all_weights()
        self.detach(unpatch_all=False)
