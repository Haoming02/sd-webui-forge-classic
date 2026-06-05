"""Dedicated txt2img generation path for Ideogram 4.0 (spec §4.1, §4.6).

``process_images_ideogram4(p)`` is dispatched from ``modules.processing.process_images``
when ``p.ideogram4_enabled`` is set (by the Ideogram UI script's ``before_process``).

It deliberately bypasses Forge's native model loading + k-diffusion sampler
(``process_images_inner`` is hard-wired to a loaded ``shared.sd_model``), driving
the official ``Ideogram4Pipeline`` instead, and reuses only the shared output-tail
helpers (``images.save_image`` / ``images.image_grid`` / ``Processed``).

All ``modules.*`` imports are done lazily inside the function to avoid an import
cycle with ``modules.processing``.
"""

import logging
import random

from modules.ideogram4 import pipeline as ig_pipeline
from modules.ideogram4.sampler_configs import DEFAULT_PRESET, get_preset

logger = logging.getLogger("ideogram4")

MIN_DIM = 256
MAX_DIM = 2048
DIM_STEP = 16
MAX_ASPECT = 6.0


def _sanitize_dim(value: int, axis: str, warnings: list[str]) -> int:
    """Clamp to 256–2048 and round to a multiple of 16 (spec §2.3, §4.8)."""
    v = int(value)
    clamped = max(MIN_DIM, min(MAX_DIM, v))
    if clamped != v:
        warnings.append(f"{axis} {v} clamped to {clamped} (allowed range {MIN_DIM}–{MAX_DIM}).")
        v = clamped
    if v % DIM_STEP != 0:
        nv = max(MIN_DIM, min(MAX_DIM, round(v / DIM_STEP) * DIM_STEP))
        warnings.append(f"{axis} {v} rounded to {nv} (must be a multiple of {DIM_STEP}).")
        v = nv
    return v


def _check_aspect(width: int, height: int, warnings: list[str]) -> None:
    long_side, short_side = max(width, height), min(width, height)
    if short_side > 0 and long_side / short_side > MAX_ASPECT + 1e-6:
        warnings.append(
            f"Aspect ratio {long_side}:{short_side} exceeds the {int(MAX_ASPECT)}:1 limit; "
            "the result may be poor."
        )


def _resolve_params(p) -> dict:
    """Merge UI-stashed params with preset defaults and global settings."""
    from modules import shared

    params = dict(getattr(p, "ideogram4_params", {}) or {})

    preset_name = params.get("preset") or DEFAULT_PRESET
    preset = get_preset(preset_name)

    resolved = {
        "preset": preset.name,
        "steps": params.get("steps") or preset.steps,
        "guidance_scale": params.get("guidance_scale") or preset.base_guidance,
        "guidance_schedule": params.get("guidance_schedule") or preset.guidance_schedule,
        "mu": params.get("mu") if params.get("mu") is not None else preset.mu,
        "std": params.get("std") if params.get("std") is not None else preset.std,
        "transparent": bool(params.get("transparent", False)),
        "model_path": params.get("model_path") or getattr(shared.opts, "ideogram4_model_path", ""),
        "quantization": params.get("quantization") or getattr(shared.opts, "ideogram4_quantization", "nf4"),
    }
    return resolved


def _build_infotext(p, caption, negative, seed, params, width, height) -> str:
    """A1111-style infotext that round-trips through PNG metadata (spec §7)."""
    from modules.infotext_utils import quote

    fields = {
        "Steps": params["steps"],
        "Sampler": f"Ideogram {params['preset']}",
        "CFG scale": params["guidance_scale"],
        "Seed": seed,
        "Size": f"{width}x{height}",
        "Model": "Ideogram 4.0",
        "Ideogram preset": params["preset"],
        "Ideogram mu": params["mu"],
        "Ideogram std": params["std"],
        "Ideogram transparent": "true" if params["transparent"] else None,
        "Ideogram quant": params["quantization"],
        **(getattr(p, "extra_generation_params", None) or {}),
    }

    text = ", ".join(f"{k}: {quote(v)}" for k, v in fields.items() if v is not None)
    negative_text = f"\nNegative prompt: {negative}" if negative else ""
    return f"{caption}{negative_text}\n{text}".strip()


def process_images_ideogram4(p):
    from modules import images, shared
    from modules.processing import Processed, get_fixed_seed
    from modules.shared import opts, state

    warnings: list[str] = list(getattr(p, "ideogram4_warnings", None) or [])

    params = _resolve_params(p)

    width = _sanitize_dim(p.width, "Width", warnings)
    height = _sanitize_dim(p.height, "Height", warnings)
    _check_aspect(width, height, warnings)
    p.width, p.height = width, height

    caption = getattr(p, "ideogram4_caption", None)
    if not caption:
        caption = p.prompt if isinstance(p.prompt, str) else (p.prompt[0] if p.prompt else "")
    p.prompt = caption  # so Processed/infotext display the caption

    negative = p.negative_prompt if isinstance(p.negative_prompt, str) else (p.negative_prompt[0] if p.negative_prompt else "")

    for message in warnings:
        p.comment(message)
        logger.warning("Ideogram 4.0: %s", message)

    # ---- seeds ----------------------------------------------------------------
    seed = get_fixed_seed(p.seed)
    if seed is None or seed == -1:
        seed = int(random.randrange(2**32))
    n_images = max(1, p.n_iter) * max(1, p.batch_size)
    all_seeds = [int(seed) + i for i in range(n_images)]

    # fields read by Processed() that are normally set inside process_images_inner
    p.all_prompts = [caption] * n_images
    p.all_negative_prompts = [negative] * n_images
    p.all_seeds = all_seeds
    p.all_subseeds = [0] * n_images
    p.sd_model_name = "Ideogram 4.0"
    p.sd_model_hash = ""
    p.sd_vae_name = None
    p.sd_vae_hash = None

    # ---- load pipeline (raises Ideogram4Error → shown in the UI) --------------
    pipe = ig_pipeline.get_pipeline(params["model_path"], params["quantization"])

    state.job_count = n_images
    state.job_no = 0

    output_images = []
    infotexts = []
    save_samples = not p.do_not_save_samples
    enable_pnginfo = getattr(opts, "enable_pnginfo", True)

    for i in range(n_images):
        if state.interrupted or state.stopping_generation:
            break
        if state.skipped:
            state.skipped = False

        img_seed = all_seeds[i]
        state.job = f"Ideogram 4.0 — image {i + 1}/{n_images}"
        state.textinfo = state.job

        produced = ig_pipeline.call_pipeline(
            pipe,
            caption,
            height=height,
            width=width,
            steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            guidance_schedule=params["guidance_schedule"],
            mu=params["mu"],
            std=params["std"],
            negative_prompt=(negative or None),
            transparent=params["transparent"],
            seed=img_seed,
        )

        for image in produced:
            infotext = _build_infotext(p, caption, negative, img_seed, params, width, height)
            if save_samples:
                images.save_image(
                    image, p.outpath_samples, "", img_seed, caption,
                    getattr(opts, "samples_format", "png"), info=infotext, p=p,
                )
            if enable_pnginfo:
                image.info["parameters"] = infotext
            output_images.append(image)
            infotexts.append(infotext)

        state.nextjob()

    # ---- grid -----------------------------------------------------------------
    index_of_first_image = 0
    if len(output_images) > 1 and getattr(opts, "return_grid", True) and not p.do_not_save_grid:
        grid = images.image_grid(output_images, p.batch_size)
        grid_text = infotexts[0] if infotexts else ""
        if getattr(opts, "grid_save", True):
            images.save_image(
                grid, p.outpath_grids, "grid", all_seeds[0], caption,
                getattr(opts, "grid_format", "png"), info=grid_text,
                short_filename=not getattr(opts, "grid_extended_filename", False),
                p=p, grid=True,
            )
        output_images.insert(0, grid)
        infotexts.insert(0, grid_text)
        index_of_first_image = 1

    return Processed(
        p,
        output_images,
        seed=all_seeds[0],
        info=infotexts[0] if infotexts else "",
        infotexts=infotexts,
        index_of_first_image=index_of_first_image,
        all_prompts=p.all_prompts,
        all_seeds=all_seeds,
        all_subseeds=p.all_subseeds,
    )
