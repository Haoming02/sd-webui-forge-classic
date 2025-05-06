import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.sd_samplers_common import SamplerData

from modules import (
    sd_samplers_kdiffusion,
    sd_samplers_lcm,
    sd_samplers_timesteps,
    sd_schedulers,
    shared,
)
from modules.sd_samplers_common import (  # noqa: F401
    sample_to_image,
    samples_to_image_grid,
)
from modules_forge import forge_alter_samplers

all_samplers = [
    *sd_samplers_kdiffusion.samplers_data_k_diffusion,
    *sd_samplers_timesteps.samplers_data_timesteps,
    *sd_samplers_lcm.samplers_data_lcm,
    *forge_alter_samplers.samplers_data_alter,
]
all_samplers_map = {x.name: x for x in all_samplers}

samplers: list["SamplerData"] = []
samplers_for_img2img: list["SamplerData"] = []
samplers_map: dict[str, str] = {}
samplers_hidden: set[str] = {}


def get_sampler_from_infotext(d: dict):
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[0]


def get_scheduler_from_infotext(d: dict):
    return get_sampler_and_scheduler(d.get("Sampler"), d.get("Schedule type"))[1]


def find_sampler_config(name):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    return config


def create_sampler(name, model):
    config = find_sampler_config(name)

    assert config is not None, f"bad sampler name: {name}"

    if model.is_sdxl and config.options.get("no_sdxl", False):
        raise Exception(f"Sampler {config.name} is not supported for SDXL")

    sampler = config.constructor(model)
    sampler.config = config

    return sampler


def set_samplers():
    global samplers, samplers_for_img2img, samplers_hidden

    samplers = all_samplers
    samplers_for_img2img = all_samplers

    _samplers_hidden = set(shared.opts.hide_samplers)
    if shared.opts.hide_samplers_invert:
        samplers_hidden = set(x.name for x in samplers if x.name not in _samplers_hidden)
    else:
        samplers_hidden = _samplers_hidden

    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name


def visible_samplers() -> list["SamplerData"]:
    return [x for x in samplers if x.name not in samplers_hidden]


def visible_sampler_names() -> list[str]:
    return [x.name for x in samplers if x.name not in samplers_hidden]


set_samplers()


def get_hr_sampler_and_scheduler(d: dict):
    hr_sampler = d.get("Hires sampler", "Use same sampler")
    sampler = d.get("Sampler") if hr_sampler == "Use same sampler" else hr_sampler

    hr_scheduler = d.get("Hires schedule type", "Use same scheduler")
    scheduler = d.get("Schedule type") if hr_scheduler == "Use same scheduler" else hr_scheduler

    sampler, scheduler = get_sampler_and_scheduler(sampler, scheduler)

    sampler = sampler if sampler != d.get("Sampler") else "Use same sampler"
    scheduler = scheduler if scheduler != d.get("Schedule type") else "Use same scheduler"

    return sampler, scheduler


def get_hr_sampler_from_infotext(d: dict):
    return get_hr_sampler_and_scheduler(d)[0]


def get_hr_scheduler_from_infotext(d: dict):
    return get_hr_sampler_and_scheduler(d)[1]


@functools.lru_cache(maxsize=10, typed=False)
def get_sampler_and_scheduler(sampler_name: str, scheduler_name: str, *, status: bool = False):
    default_sampler = samplers[0]
    found_scheduler = sd_schedulers.schedulers_map.get(scheduler_name, sd_schedulers.schedulers[0])

    name = sampler_name or default_sampler.name

    for scheduler in sd_schedulers.schedulers:
        name_options = [scheduler.label, scheduler.name, *(scheduler.aliases or [])]

        for name_option in name_options:
            if name.endswith(" " + name_option):
                found_scheduler = scheduler
                name = name[0 : -(len(name_option) + 1)]
                break

    sampler = all_samplers_map.get(name, default_sampler)

    _automatic = False
    if sampler.options.get("scheduler", None) == found_scheduler.name:
        found_scheduler = sd_schedulers.schedulers[0]
        _automatic = True

    if not status:
        return sampler.name, found_scheduler.label
    else:
        return sampler.name, found_scheduler.label, _automatic


def fix_p_invalid_sampler_and_scheduler(p):
    i_sampler_name, i_scheduler = p.sampler_name, p.scheduler
    p.sampler_name, p.scheduler, _automatic = get_sampler_and_scheduler(p.sampler_name, p.scheduler, status=True)
    if i_sampler_name != p.sampler_name:
        logging.warning(f'Sampler Correction: "{i_sampler_name}" -> "{p.sampler_name}"')
    if i_scheduler != p.scheduler and not _automatic:
        logging.warning(f'Scheduler Correction: "{i_scheduler}" -> "{p.scheduler}"')
