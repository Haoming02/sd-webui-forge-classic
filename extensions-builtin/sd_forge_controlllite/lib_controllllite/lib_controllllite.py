# https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI/blob/main/node_control_net_lllite.py

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.patcher.unet import UnetPatcher

import torch
import torch.nn as nn

from backend.state_dict import load_state_dict

logger = logging.getLogger("ControlNet")


def extra_options_to_module_prefix(extra_options: dict) -> str:
    block = extra_options["block"]
    block_index = extra_options["block_index"]
    if block[0] == "input":
        module_pfx = f"lllite_unet_input_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    elif block[0] == "middle":
        module_pfx = f"lllite_unet_middle_block_1_transformer_blocks_{block_index}"
    elif block[0] == "output":
        module_pfx = f"lllite_unet_output_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    else:
        raise ValueError("invalid block name")
    return module_pfx


def load_control_net_lllite_patch(ctrl_sd: dict, cond_image: torch.Tensor, multiplier: float, num_steps: int, start_percent: float, end_percent: float, *, model_dtype: torch.dtype):
    start_step = math.floor(num_steps * start_percent) if start_percent > 0 else 0
    end_step = math.floor(num_steps * end_percent) if end_percent > 0 else num_steps

    module_weights = {}
    for key, value in ctrl_sd.items():
        fragments = key.split(".")
        module_name = fragments[0]
        weight_name = ".".join(fragments[1:])

        if module_name not in module_weights:
            module_weights[module_name] = {}
        module_weights[module_name][weight_name] = value

    modules = {}
    for module_name, weights in module_weights.items():
        if "conditioning1.4.weight" in weights:
            depth = 3
        elif weights["conditioning1.2.weight"].shape[-1] == 4:
            depth = 2
        else:
            depth = 1

        module = LLLiteModule(
            name=module_name,
            is_conv2d=weights["down.0.weight"].ndim == 4,
            in_dim=weights["down.0.weight"].shape[1],
            depth=depth,
            cond_emb_dim=weights["conditioning1.0.weight"].shape[0] * 2,
            mlp_dim=weights["down.0.weight"].shape[0],
            multiplier=multiplier,
            num_steps=num_steps,
            start_step=start_step,
            end_step=end_step,
        )
        load_state_dict(module, weights)
        modules[module_name] = module.eval().to(dtype=model_dtype)
        if len(modules) == 1:
            module.is_first = True

    logger.info(f"Loaded Control-LLLite ({len(modules)} modules)")

    cond_image = cond_image.permute(0, 3, 1, 2)
    cond_image = cond_image * 2.0 - 1.0

    for module in modules.values():
        module.set_cond_image(cond_image)

    class control_net_lllite_patch:
        def __init__(self, modules: dict[str, nn.Module]):
            self.modules = modules

        def __call__(self, q, k, v, extra_options):
            module_pfx = extra_options_to_module_prefix(extra_options)

            is_attn1 = q.shape[-1] == k.shape[-1]
            if is_attn1:
                module_pfx = module_pfx + "_attn1"
            else:
                module_pfx = module_pfx + "_attn2"

            module_pfx_to_q = module_pfx + "_to_q"
            module_pfx_to_k = module_pfx + "_to_k"
            module_pfx_to_v = module_pfx + "_to_v"

            if module_pfx_to_q in self.modules:
                q = q + self.modules[module_pfx_to_q](q)
            if module_pfx_to_k in self.modules:
                k = k + self.modules[module_pfx_to_k](k)
            if module_pfx_to_v in self.modules:
                v = v + self.modules[module_pfx_to_v](v)

            return q, k, v

        def to(self, device):
            for d in self.modules.keys():
                self.modules[d] = self.modules[d].to(device)
            return self

    return control_net_lllite_patch(modules)


class LLLiteModule(nn.Module):
    def __init__(
        self,
        name: str,
        is_conv2d: bool,
        in_dim: int,
        depth: int,
        cond_emb_dim: int,
        mlp_dim: int,
        multiplier: int,
        num_steps: int,
        start_step: int,
        end_step: int,
    ):
        super().__init__()
        self.name = name
        self.is_conv2d = is_conv2d
        self.multiplier = multiplier
        self.num_steps = num_steps
        self.start_step = start_step
        self.end_step = end_step
        self.is_first = False

        modules = []
        modules.append(nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
        if depth == 1:
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
        elif depth == 2:
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
        elif depth == 3:
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

        self.conditioning1 = nn.Sequential(*modules)

        if self.is_conv2d:
            self.down = nn.Sequential(
                nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
            self.mid = nn.Sequential(
                nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
            )
            self.up = nn.Sequential(
                nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.down = nn.Sequential(
                nn.Linear(in_dim, mlp_dim),
                nn.ReLU(inplace=True),
            )
            self.mid = nn.Sequential(
                nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
                nn.ReLU(inplace=True),
            )
            self.up = nn.Sequential(
                nn.Linear(mlp_dim, in_dim),
            )

        self.depth = depth
        self.cond_image = None
        self.cond_emb = None
        self.current_step = 0

    def set_cond_image(self, cond_image):
        self.cond_image = cond_image
        self.cond_emb = None
        self.current_step = 0

    @torch.inference_mode()
    def forward(self, x):
        if self.num_steps > 0:
            if self.current_step < self.start_step:
                self.current_step += 1
                return torch.zeros_like(x)
            elif self.current_step >= self.end_step:
                if self.is_first and self.current_step == self.end_step:
                    logger.debug(f"LLLite End: step {self.current_step}")
                self.current_step += 1
                if self.current_step >= self.num_steps:
                    self.current_step = 0
                return torch.zeros_like(x)
            else:
                if self.is_first and self.current_step == self.start_step:
                    logger.debug(f"LLLite Start: step {self.current_step}")
                self.current_step += 1
                if self.current_step >= self.num_steps:
                    self.current_step = 0

        if self.cond_emb is None:
            cx = self.conditioning1(self.cond_image.to(x.device, dtype=x.dtype))
            if not self.is_conv2d:
                n, c, h, w = cx.shape
                cx = cx.view(n, c, h * w).permute(0, 2, 1)
            self.cond_emb = cx

        cx = self.cond_emb

        if x.shape[0] != cx.shape[0]:
            if self.is_conv2d:
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1, 1)
            else:
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        return cx * self.multiplier


class LLLiteLoader:

    @staticmethod
    def load_lllite(model: "UnetPatcher", state_dict, cond_image, strength, steps, start_percent, end_percent):
        m = model.clone()

        patch = load_control_net_lllite_patch(state_dict, cond_image, strength, steps, start_percent, end_percent, model_dtype=model.model.diffusion_model.computation_dtype)

        if patch is not None:
            m.set_model_attn1_patch(patch)
            m.set_model_attn2_patch(patch)

        return m
