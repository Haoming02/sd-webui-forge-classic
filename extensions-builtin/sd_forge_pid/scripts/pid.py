from unittest.mock import patch

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from backend import memory_management
from modules import devices, images, scripts
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    apply_color_correction,
    decode_latent_batch,
    logger,
    process_images,
)
from modules.sd_models import checkpoint_tiles
from modules.sd_vae import vae_dict
from modules.shared import opts, state
from modules.ui_components import InputAccordion
from modules_forge.main_entry import module_list


class PiDForForge(scripts.Script):
    results: list[tuple[Image.Image], str] = []

    def __init__(self):
        self.models: list[str] = [m for m in sorted(checkpoint_tiles(use_short=True)) if "pid" in m.lower()]

    def title(self):
        return "PiD Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if bool(self.models) else None

    def ui(self, *args, **kwargs):
        vaes: list[str] = sorted(vae_dict.keys())
        modules: list[str] = [m for m in sorted(module_list.keys()) if m not in vaes]

        with InputAccordion(False, label=self.title()) as enable:
            prompt = gr.Textbox(value=None, lines=3, max_lines=3, label="Prompt", placeholder="(leave empty to use main prompt)")
            with gr.Row():
                ckpt = gr.Dropdown(value=next(iter(self.models), None), choices=self.models, label="PiD")
                vae = gr.Dropdown(value=next(iter(vaes), None), choices=vaes, label="VAE")
                tec = gr.Dropdown(value=next(iter(modules), None), choices=modules, label="Gemma2 2B IT ELM")
            with gr.Row():
                degrade_sigma = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Degrade Sigma", info="(denoising strength)")
                cc = gr.Checkbox(True, label="Color Correction", info="(fix discoloration issue)")

        self.infotext_fields = [
            (prompt, "pid_prompt"),
            (ckpt, "pid_ckpt"),
            (vae, "pid_vae"),
            (tec, "pid_tec"),
            (degrade_sigma, "degrade_sigma"),
        ]

        return [enable, prompt, ckpt, vae, tec, degrade_sigma, cc]

    @torch.inference_mode()
    def post_sample(
        self,
        p: StableDiffusionProcessing,
        ps: scripts.PostSampleArgs,
        enable: bool,
        prompt: str,
        ckpt: str,
        vae: str,
        tec: str,
        degrade_sigma: float,
        cc: bool,
    ):
        if not enable:
            return

        if getattr(p, "enable_hr", False):
            logger.error("PiD does not support Hires. Fix")
            return

        override_settings = p.override_settings.copy()
        override_settings.update({"sd_model_checkpoint": ckpt, "forge_additional_modules": [vae, tec]})
        extra_generation_params = p.extra_generation_params.copy()
        extra_generation_params.update(
            {
                "pid_prompt": prompt or None,
                "pid_ckpt": ckpt,
                "pid_vae": vae,
                "pid_tec": tec,
                "degrade_sigma": degrade_sigma,
            }
        )

        batch_size: int = ps.samples.size(0)
        state.job_count += batch_size

        _samples = ps.samples.detach().clone()
        ps.samples = decode_latent_batch(p.sd_model, ps.samples, target_device=devices.cpu, check_for_nans=True)
        if cc:
            _samples_ddim = torch.stack(ps.samples).float().detach().clone().squeeze(1).add(1.0).div(2.0).cpu().numpy()

        for b in range(batch_size):
            if state.interrupted or state.skipped:
                continue

            memory_management.soft_empty_cache()
            latent = _samples[b].unsqueeze(0)
            cc_target = None
            state.nextjob()

            if cc:
                _sample = np.clip((np.moveaxis(_samples_ddim[b], 0, 2) * 255.0).round(), 0, 255).astype(np.uint8)
                cc_target = cv2.cvtColor(_sample, cv2.COLOR_RGB2LAB)

            with patch.dict(opts.data, {"multiple_tqdm": False}, clear=False):
                self._sample_inner(p, b, latent, degrade_sigma, prompt, override_settings, extra_generation_params, cc_target)

    @staticmethod
    def _sample_inner(
        p: StableDiffusionProcessing,
        b: int,
        latent: torch.Tensor,
        degrade_sigma: float,
        prompt: str,
        override_settings: dict,
        extra_generation_params: dict,
        cc_target: np.ndarray,
    ):
        i2i = StableDiffusionProcessingImg2Img(
            init_images=None,
            init_latent=latent,
            resize_mode=3,
            denoising_strength=degrade_sigma,
            mask=None,
            mask_blur=4,
            inpainting_fill=1,
            inpaint_full_res=False,
            inpaint_full_res_padding=16,
            inpainting_mask_invert=0,
            initial_noise_multiplier=1.0,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=prompt or p.all_prompts[b],
            negative_prompt="",
            styles=None,
            seed=p.all_seeds[b],
            subseed=p.all_subseeds[b],
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name="LCM",
            scheduler="Simple",
            batch_size=1,
            n_iter=1,
            steps=4,
            cfg_scale=1.0,
            distilled_cfg_scale=1.5,
            width=p.width * 4,
            height=p.height * 4,
            restore_faces=False,
            tiling=False,
            extra_generation_params=extra_generation_params,
            do_not_save_samples=True,
            do_not_save_grid=True,
            override_settings=override_settings,
        )

        i2i.cached_c = [None, None, None]
        i2i.cached_uc = [None, None, None]
        i2i.scripts = None
        i2i.script_args = None

        try:
            if (processed := process_images(i2i)) is not None:
                assert len(processed.images) == 1
                image: Image.Image = processed.images[0]
                info: str = processed.infotexts[0]

                if cc_target is not None:
                    image = apply_color_correction(cc_target, image)

                PiDForForge.results.append((image, info))
                images.save_image(
                    image,
                    i2i.outpath_samples,
                    "",
                    i2i.seed,
                    i2i.main_prompt,
                    opts.samples_format,
                    info=info,
                    p=i2i,
                    suffix="-pid",
                )
        except Exception as e:
            logger.error(f"Error during PiD Pass:\n{e}")
        finally:
            i2i.close()

    def postprocess(self, p, processed: Processed, *args):
        for image, info in self.results:
            processed.extra_images.append(image)
            processed.infotexts.append(info)
        self.results.clear()
