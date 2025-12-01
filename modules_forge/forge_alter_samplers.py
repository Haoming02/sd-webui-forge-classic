import logging
from typing import Callable

import k_diffusion.sampling

from modules import sd_samplers_common, sd_samplers_kdiffusion


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name):
        sampler_function: Callable = getattr(k_diffusion.sampling, f"sample_{sampler_name}", None)
        if sampler_function is None:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        super().__init__(sampler_function, sd_model, None)

    def sample(self, p, *args, **kwargs):
        if p.cfg_scale > 2.0:
            logging.warning("CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers")
        return super().sample(p, *args, **kwargs)

    def sample_img2img(self, p, *args, **kwargs):
        if p.cfg_scale > 2.0:
            logging.warning("CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers")
        return super().sample_img2img(p, *args, **kwargs)


def build_constructor(sampler_key: str) -> Callable:
    def constructor(model):
        return AlterSampler(model, sampler_key)

    return constructor


def create_cfg_pp_sampler(sampler_name: str, sampler_key: str) -> "sd_samplers_common.SamplerData":
    config = {}
    base_name = sampler_name.removesuffix(" CFG++")
    for name, _, _, params in sd_samplers_kdiffusion.samplers_k_diffusion:
        if name == base_name:
            config = params.copy()
            break

    return sd_samplers_common.SamplerData(sampler_name, build_constructor(sampler_key=sampler_key), [sampler_key], config)


samplers_data_alter = [
    create_cfg_pp_sampler("DPM++ 2M CFG++", "dpmpp_2m_cfg_pp"),
    create_cfg_pp_sampler("Euler a CFG++", "euler_ancestral_cfg_pp"),
    create_cfg_pp_sampler("Euler CFG++", "euler_cfg_pp"),
]
