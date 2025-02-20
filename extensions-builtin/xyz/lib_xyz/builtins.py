from modules import sd_models, sd_samplers, sd_samplers_kdiffusion, sd_vae, shared
from modules.shared import opts

from .axis_application import (
    apply_checkpoint,
    apply_clip_skip,
    apply_face_restore,
    apply_field,
    apply_order,
    apply_override,
    apply_prompt,
    apply_styles,
    apply_uni_pc_order,
    apply_vae,
    do_nothing,
)
from .axis_format import (
    format_nothing,
    format_remove_path,
    format_value,
    format_value_join_list,
)
from .axis_validate import (
    confirm_checkpoints,
    confirm_checkpoints_or_none,
    confirm_samplers,
)
from .classes import AxisOption, AxisOptionImg2Img, AxisOptionTxt2Img
from .utils import boolean_choice, str_permutations

builtin_options = [
    AxisOption("Nothing", str, do_nothing, format_value=format_nothing),
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOptionTxt2Img("Hires steps", int, apply_field("hr_second_pass_steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOptionImg2Img("Image CFG Scale", float, apply_field("image_cfg_scale")),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value),
    AxisOption(
        "Prompt order",
        str_permutations,
        apply_order,
        format_value=format_value_join_list,
    ),
    AxisOptionTxt2Img(
        "Sampler",
        str,
        apply_field("sampler_name"),
        format_value=format_value,
        confirm=confirm_samplers,
        choices=sd_samplers.visible_sampler_names,
    ),
    AxisOptionTxt2Img(
        "Hires sampler",
        str,
        apply_field("hr_sampler_name"),
        confirm=confirm_samplers,
        choices=sd_samplers.visible_sampler_names,
    ),
    AxisOptionImg2Img(
        "Sampler",
        str,
        apply_field("sampler_name"),
        format_value=format_value,
        confirm=confirm_samplers,
        choices=sd_samplers.visible_sampler_names,
    ),
    AxisOption(
        "Checkpoint name",
        str,
        apply_checkpoint,
        format_value=format_remove_path,
        confirm=confirm_checkpoints,
        cost=1.0,
        choices=lambda: sorted(sd_models.checkpoints_list, key=str.casefold),
    ),
    AxisOption("Negative Guidance minimum sigma", float, apply_field("s_min_uncond")),
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    AxisOption(
        "Schedule type",
        str,
        apply_override("k_sched_type"),
        choices=lambda: list(sd_samplers_kdiffusion.k_diffusion_scheduler),
    ),
    AxisOption("Schedule min sigma", float, apply_override("sigma_min")),
    AxisOption("Schedule max sigma", float, apply_override("sigma_max")),
    AxisOption("Schedule rho", float, apply_override("rho")),
    AxisOption("Eta", float, apply_field("eta")),
    AxisOption("Clip skip", int, apply_clip_skip),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOption(
        "Initial noise multiplier", float, apply_field("initial_noise_multiplier")
    ),
    AxisOption("Extra noise", float, apply_override("img2img_extra_noise")),
    AxisOptionTxt2Img(
        "Hires upscaler",
        str,
        apply_field("hr_upscaler"),
        choices=lambda: [
            *shared.latent_upscale_modes,
            *[x.name for x in shared.sd_upscalers],
        ],
    ),
    AxisOptionImg2Img(
        "Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight")
    ),
    AxisOption(
        "VAE",
        str,
        apply_vae,
        cost=0.7,
        choices=lambda: ["None"] + list(sd_vae.vae_dict),
    ),
    AxisOption(
        "Styles", str, apply_styles, choices=lambda: list(shared.prompt_styles.styles)
    ),
    AxisOption("UniPC Order", int, apply_uni_pc_order, cost=0.5),
    AxisOption("Face restore", str, apply_face_restore, format_value=format_value),
    AxisOption("Token merging ratio", float, apply_override("token_merging_ratio")),
    AxisOption(
        "Token merging ratio high-res", float, apply_override("token_merging_ratio_hr")
    ),
    AxisOption(
        "Always discard next-to-last sigma",
        str,
        apply_override("always_discard_next_to_last_sigma", boolean=True),
        choices=boolean_choice(reverse=True),
    ),
    AxisOption(
        "SGM noise multiplier",
        str,
        apply_override("sgm_noise_multiplier", boolean=True),
        choices=boolean_choice(reverse=True),
    ),
    AxisOption(
        "Refiner checkpoint",
        str,
        apply_field("refiner_checkpoint"),
        format_value=format_remove_path,
        confirm=confirm_checkpoints_or_none,
        cost=1.0,
        choices=lambda: ["None"] + sorted(sd_models.checkpoints_list, key=str.casefold),
    ),
    AxisOption("Refiner switch at", float, apply_field("refiner_switch_at")),
    AxisOption(
        "RNG source",
        str,
        apply_override("randn_source"),
        choices=lambda: ["GPU", "CPU", "NV"],
    ),
]
