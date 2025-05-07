import logging

from modules import sd_samplers_cfgpp, sd_samplers_common, sd_samplers_kdiffusion


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, scheduler_name=None):
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.unet = sd_model.forge_objects.unet

        match self.sampler_name:
            case "euler_cfg_pp":
                sampler_function = sd_samplers_cfgpp.sample_euler_cfg_pp
            case "euler_ancestral_cfg_pp":
                sampler_function = sd_samplers_cfgpp.sample_euler_ancestral_cfg_pp
            case "dpmpp_sde_cfg_pp":
                sampler_function = sd_samplers_cfgpp.sample_dpmpp_sde_cfg_pp
            case "dpmpp_2m_cfg_pp":
                sampler_function = sd_samplers_cfgpp.sample_dpmpp_2m_cfg_pp
            case "dpmpp_3m_sde_cfg_pp":
                sampler_function = sd_samplers_cfgpp.sample_dpmpp_3m_sde_cfg_pp
            case _:
                raise ValueError(f"Unknown sampler: {sampler_name}")

        super().__init__(sampler_function, sd_model, None)

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        if p.cfg_scale > 2.0:
            logging.warning("Low CFG is recommended when using CFG++ samplers")
        self.scheduler_name = p.scheduler
        return super().sample(p, x, conditioning, unconditional_conditioning, steps, image_conditioning)

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        if p.cfg_scale > 2.0:
            logging.warning("Low CFG is recommended when using CFG++ samplers")
        self.scheduler_name = p.scheduler
        return super().sample_img2img(p, x, noise, conditioning, unconditional_conditioning, steps, image_conditioning)


def build_constructor(sampler_name):
    def constructor(model):
        return AlterSampler(model, sampler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData("DPM++ 2M CFG++", build_constructor(sampler_name="dpmpp_2m_cfg_pp"), ["dpmpp_2m_cfg_pp"], {}),
    sd_samplers_common.SamplerData("DPM++ SDE CFG++", build_constructor(sampler_name="dpmpp_sde_cfg_pp"), ["dpmpp_sde_cfg_pp"], {}),
    sd_samplers_common.SamplerData("DPM++ 3M SDE CFG++", build_constructor(sampler_name="dpmpp_3m_sde_cfg_pp"), ["dpmpp_3m_sde_cfg_pp"], {}),
    sd_samplers_common.SamplerData("Euler a CFG++", build_constructor(sampler_name="euler_ancestral_cfg_pp"), ["euler_ancestral_cfg_pp"], {}),
    sd_samplers_common.SamplerData("Euler CFG++", build_constructor(sampler_name="euler_cfg_pp"), ["euler_cfg_pp"], {}),
]
