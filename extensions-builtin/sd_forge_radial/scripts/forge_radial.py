# https://github.com/woct0rdho/ComfyUI-RadialAttn/blob/main/nodes.py

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher

import gradio as gr
import torch
import torch.nn.functional as F
from lib_radial.attn_mask import RADIAL_ENABLE

from backend.nn import wan
from modules import scripts
from modules.ui_components import InputAccordion

ORIG_ATTENTION = wan.attention


def get_radial_attn_func(video_token_num, num_frame, block_size, decay_factor, allow_compile):
    from lib_radial.attn_mask import MaskMap, RadialAttention

    mask_map = MaskMap(video_token_num, num_frame)

    def radial_attn_func(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        assert q.shape == k.shape
        assert mask is None

        if skip_reshape:
            b, _, orig_seq_len, head_dim = q.shape
            q, k, v = map(lambda t: t.permute(0, 2, 1, 3).reshape(-1, heads, head_dim), (q, k, v))
        else:
            b, orig_seq_len, head_dim = q.shape
            head_dim //= heads
            q, k, v = map(lambda t: t.view(-1, heads, head_dim), (q, k, v))

        padded_len = b * video_token_num
        if q.shape[0] != padded_len:
            q, k, v = map(lambda t: F.pad(t, (0, 0, 0, 0, 0, padded_len - t.shape[0])), (q, k, v))

        out = RadialAttention(q, k, v, mask_map=mask_map, block_size=block_size, decay_factor=decay_factor, model_type="wan")

        out = out[: b * orig_seq_len, :, :]

        if skip_output_reshape:
            out = out.reshape(b, orig_seq_len, heads, head_dim).permute(0, 2, 1, 3)
        else:
            out = out.reshape(b, -1, heads * head_dim)

        return out

    if not allow_compile:
        radial_attn_func = torch.compiler.disable()(radial_attn_func)

    return radial_attn_func


class PatchRadialAttn:

    @staticmethod
    def patch_radial_attn(model: "UnetPatcher", dense_block: int, dense_timestep: int, last_dense_timestep: int, block_size: int, decay_factor: float, allow_compile: bool):
        model = model.clone()

        diffusion_model = model.get_model_object("diffusion_model")
        assert diffusion_model.__class__.__name__ == "WanModel"

        model.model_options["transformer_options"]["radial_attn"] = {
            "patch_size": diffusion_model.patch_size,
            "dense_block": dense_block,
            "dense_timestep": dense_timestep,
            "last_dense_timestep": last_dense_timestep,
            "block_size": block_size,
            "decay_factor": decay_factor,
            "allow_compile": allow_compile,
        }

        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            sigmas = c["transformer_options"]["sampling_sigmas"]

            if len(matched_step_index := (sigmas == timestep).nonzero()) > 0:
                current_step_index = matched_step_index.item()
            else:
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
                else:
                    current_step_index = 0

            ra_options = c["transformer_options"]["radial_attn"]

            if not (ra_options["dense_timestep"] <= current_step_index < len(sigmas) - 1 - ra_options["last_dense_timestep"]):
                wan.attention = ORIG_ATTENTION
            else:
                patch_size = ra_options["patch_size"]
                num_frame = (input.shape[2] - 1) // patch_size[0] + 1
                frame_size = (input.shape[3] // patch_size[1]) * (input.shape[4] // patch_size[2])
                video_token_num = frame_size * num_frame

                padded_video_token_num = video_token_num
                block_size = ra_options["block_size"]
                if video_token_num % block_size != 0:
                    padded_video_token_num = (video_token_num // block_size + 1) * block_size

                dense_block = ra_options["dense_block"]
                radial_attn_func = get_radial_attn_func(padded_video_token_num, num_frame, block_size, ra_options["decay_factor"], ra_options["allow_compile"])

                @wraps(ORIG_ATTENTION)
                def try_radial_attn(*args, **kwargs):
                    transformer_options = kwargs.get("transformer_options", {})
                    block_index = transformer_options.get("block_index", -1)

                    if block_index >= dense_block:
                        return radial_attn_func(*args, **kwargs)
                    else:
                        return ORIG_ATTENTION(*args, **kwargs)

                wan.attention = try_radial_attn

            return model_function(input, timestep, **c)

        model.set_model_unet_function_wrapper(unet_wrapper_function)

        return model


class RadialAttentionForForge(scripts.ScriptBuiltinUI):
    sorting_priority = 18137

    def title(self):
        return "RadialAttention Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if RADIAL_ENABLE else None

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            with gr.Row():
                dense_block = gr.Slider(minimum=0, maximum=40, value=1, step=1, label="Dense Block", info="Number of first few Blocks to bypass RadialAttention")
                decay_factor = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Decay Factor", info="Lower is Faster ; Higher is more Accurate")
            with gr.Row():
                dense_timestep = gr.Slider(minimum=0, maximum=100, value=1, step=1, label="Dense Timestep", info="Number of first few Steps to bypass RadialAttention")
                last_dense_timestep = gr.Slider(minimum=0, maximum=100, value=1, step=1, label="Last Dense Timestep", info="Number of last few Steps to bypass RadialAttention")
            with gr.Row():
                block_size = gr.Radio(choices=[64, 128], value=128, label="Block Size")
                allow_compile = gr.Checkbox(False, label="Allow Torch.Compile")

        for comp in (comps := (enable, dense_block, dense_timestep, last_dense_timestep, block_size, decay_factor, allow_compile)):
            comp.do_not_save_to_config = True

        return comps

    def process_before_every_sampling(self, p, enable: bool, *args, **kwargs):
        if not enable:
            return

        unet = p.sd_model.forge_objects.unet

        unet = PatchRadialAttn.patch_radial_attn(unet, *args)

        p.sd_model.forge_objects.unet = unet
