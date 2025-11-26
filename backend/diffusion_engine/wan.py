import torch
from huggingface_guess import model_list
from huggingface_guess.utils import resize_to_batch_size

from backend import args, memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.umt5_engine import UMT5TextProcessingEngine

# get_learned_conditioning is not called in the Refiner pass;
# so we store the desired shift value for the low_noise model
refiner_shift: float = None


class Wan(ForgeDiffusionEngine):
    matched_guesses = [model_list.WAN21_T2V, model_list.WAN21_I2V]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(model_dict={"umt5xxl": huggingface_components["text_encoder"]}, tokenizer_dict={"umt5xxl": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)
        vae.first_stage_model.latent_format = self.model_config.latent_format

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_t5 = UMT5TextProcessingEngine(
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
            self.forge_objects.unet.model.predictor.set_parameters(shift=refiner_shift)
            refiner_shift = None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        global refiner_shift
        shift = getattr(prompt, "distilled_cfg_scale", 8.0)
        self.forge_objects.unet.model.predictor.set_parameters(shift=shift)
        refiner_shift = shift
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(510, token_count)

    @torch.inference_mode()
    def image_to_video(self, length: int, start_image: torch.Tensor, noise: torch.Tensor):
        _, h, w, c = start_image.shape

        _image = torch.ones((length, h, w, c), device=start_image.device, dtype=start_image.dtype) * 0.5
        _image[: start_image.shape[0]] = start_image

        concat_latent_image = self.forge_objects.vae.encode(_image[:, :, :, :3])
        mask = torch.ones((1, 1, noise.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
        mask[:, :, : ((start_image.shape[0] - 1) // 4) + 1] = 0.0

        image = concat_latent_image

        extra_channels = self.forge_objects.unet.model.diffusion_model.in_dim - 16  # 20

        for i in range(0, image.shape[1], 16):
            image[:, i : i + 16] = self.forge_objects.vae.first_stage_model.process_in(image[:, i : i + 16])
        image = resize_to_batch_size(image, noise.shape[0])

        if image.shape[1] > (extra_channels - 4):
            image = image[:, : (extra_channels - 4)]

        if mask.shape[1] != 4:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = (1.0 - mask).to(image)
        if mask.shape[-3] < noise.shape[-3]:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode="constant", value=0)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 4, 1, 1, 1)
        mask = resize_to_batch_size(mask, noise.shape[0])

        _concat_mask_index = 0  # TODO

        if _concat_mask_index != 0:
            z = torch.cat((image[:, :_concat_mask_index], mask, image[:, _concat_mask_index:]), dim=1)
        else:
            z = torch.cat((mask, image), dim=1)

        args.dynamic_args["concat_latent"] = z

    @torch.inference_mode()
    def encode_first_stage(self, x):
        length, c, h, w = x.shape
        assert c == 3
        if length > 1:
            x = x[0].unsqueeze(0)  # enforce batch_size of 1
        start_image = x.movedim(1, -1) * 0.5 + 0.5
        latent = torch.zeros([1, 16, ((length - 1) // 4) + 1, h // 8, w // 8], device=self.forge_objects.vae.device)
        self.image_to_video(length, start_image, latent)
        sample = self.forge_objects.vae.first_stage_model.process_in(latent)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
        return sample.to(x)
