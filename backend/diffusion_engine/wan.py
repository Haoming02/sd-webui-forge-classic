import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.misc.image_resize import adaptive_resize, resize_reference
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.wan_engine import UMT5XXLEngine
from backend.utils import resize_to_batch_size

# get_learned_conditioning is not called in the Refiner pass;
# so we store the desired shift value for the low_noise model
refiner_shift: float = None


class Wan(ForgeDiffusionEngine):
    matched_guesses = [model_list.WAN21_T2V, model_list.WAN21_I2V]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"umt5xxl": huggingface_components["text_encoder"]}, tokenizer_dict={"umt5xxl": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_t5 = UMT5XXLEngine(
            text_encoder=clip.cond_stage_model.umt5xxl,
            tokenizer=clip.tokenizer.umt5xxl,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True
        self.is_wan = True

        global refiner_shift
        if refiner_shift is not None:
            super().set_shift(refiner_shift)
            refiner_shift = None

        del self.ini_latent
        del self.ref_latents

        self.start_image: torch.Tensor = None
        """first frame; cleared automatically every generation"""
        self.end_image: torch.Tensor = None
        """last frame; cleared manually by ImageStitch"""

    def set_shift(self, shift: float, *args, **kwargs):
        global refiner_shift
        super().set_shift(shift)
        refiner_shift = shift

    def clear_references(self):
        # called by ImageStitch
        self.start_image = None
        self.end_image = None
        memory_management.soft_empty_cache()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_t5.tokenize(prompt))
        return token_count, max(510, token_count)

    @torch.inference_mode()
    def image_to_video(self, length: int, latent_shape: list[int]):
        # https://github.com/Comfy-Org/ComfyUI/blob/v0.20.1/comfy_extras/nodes_wan.py#L209

        frame = self.start_image if self.start_image is not None else self.end_image
        if frame is None:
            raise SystemError("No images passed to Wan I2V...?")

        frame = resize_reference(frame)
        h, w = frame.shape[-2:]

        if self.start_image is not None:
            start_image = frame.movedim(1, -1)
        if self.end_image is not None:
            end_image = frame if self.start_image is None else adaptive_resize(self.end_image, w, h, "bilinear", "center")
            end_image = end_image.movedim(1, -1)

        if self.start_image is not None and self.end_image is not None:
            memory_management.logger.info("[Wan] FirstLastFrameToVideo")
        elif self.start_image is not None and self.end_image is None:
            memory_management.logger.info("[Wan] ImageToVideo")
        else:
            memory_management.logger.info("[Wan] LastFrameToVideo")

        image = torch.ones((length, h, w, 3), device="cpu", dtype=torch.float32).mul(0.5)
        mask = torch.ones((1, 1, latent_shape[2] * 4, latent_shape[-2], latent_shape[-1]), device="cpu", dtype=torch.float32)

        if self.start_image is not None:
            image[: start_image.shape[0]] = start_image
            mask[:, :, : start_image.shape[0] + 3] = 0.0

        if self.end_image is not None:
            image[-end_image.shape[0] :] = end_image
            mask[:, :, -end_image.shape[0] :] = 0.0

        concat_latent_image = self.forge_objects.vae.encode(image[:, :, :, :3])
        concat_mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

        # https://github.com/Comfy-Org/ComfyUI/blob/v0.20.1/comfy/model_base.py#L1291

        image: torch.Tensor = concat_latent_image
        mask: torch.Tensor = concat_mask

        extra_channels: int = 20
        latent_dim: int = 16

        for i in range(0, image.shape[1], latent_dim):
            image[:, i : i + latent_dim] = self.forge_objects.vae.first_stage_model.process_in(image[:, i : i + latent_dim])
        image = resize_to_batch_size(image, latent_shape[0])

        if image.shape[-2:] != tuple(latent_shape[-2:]):
            image = adaptive_resize(image, latent_shape[-1], latent_shape[-2], "bilinear", "disabled")

        if image.shape[1] > (extra_channels - 4):
            image = image[:, : (extra_channels - 4)]

        if mask.shape[1] != 4:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = (1.0 - mask).to(image)
        mask = adaptive_resize(mask, latent_shape[-1], latent_shape[-2], "bilinear", "center")
        if mask.shape[-3] < latent_shape[-3]:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, latent_shape[-3] - mask.shape[-3]), mode="constant", value=0)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 4, 1, 1, 1)
        mask = resize_to_batch_size(mask, latent_shape[0])

        z = torch.cat((mask, image), dim=1)

        dynamic_args.concat_latent = z.cpu()

        self.start_image = None

    def set_end_image(self, image: torch.Tensor, length: int):
        assert self.forge_objects.unet.model.diffusion_model.in_dim == 36
        self.end_image = image.cpu()
        if length > 1:
            h, w = image.shape[-2:]
            self.image_to_video(length, [1, 16, (length - 1) // 4 + 1, h // 8, w // 8])

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        b, _, h, w = x.shape
        x = x[0].mul(0.5).add(0.5).unsqueeze(0)  # enforce batch_size of 1

        if b == 1:
            # img2img
            sample = self.forge_objects.vae.encode(x.movedim(1, -1))
            sample = self.forge_objects.vae.first_stage_model.process_in(sample)
            assert self.forge_objects.unet.model.diffusion_model.in_dim == 16
            return sample.to(x)

        # FirstFrameToVideo
        self.start_image = x.cpu()
        assert self.forge_objects.unet.model.diffusion_model.in_dim == 36
        latent = torch.zeros([1, 16, ((b - 1) // 4) + 1, h // 8, w // 8], device=self.forge_objects.vae.device)
        self.image_to_video(b, list(latent.shape))
        sample = self.forge_objects.vae.first_stage_model.process_in(latent)
        return sample.to(x)
