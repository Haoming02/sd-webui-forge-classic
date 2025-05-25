# Reference: https://github.com/comfyanonymous/ComfyUI


import ldm_patched.modules.lora
import ldm_patched.modules.model_patcher
import ldm_patched.modules.supported_models_base
import ldm_patched.modules.utils
import ldm_patched.t2ia.adapter
import ldm_patched.taesd.taesd
import torch

from ldm_patched.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine
from ldm_patched.modules import model_management

from . import diffusers_convert


def load_model_weights(model, sd):
    m, u = model.load_state_dict(sd, strict=False)
    m = set(m)
    unexpected_keys = set(u)

    k = list(sd.keys())
    for x in k:
        if x not in unexpected_keys:
            w = sd.pop(x)
            del w
    if len(m) > 0:
        print("extra", m)
    return model


def load_lora_for_models(model, clip, lora, strength_model, strength_clip, filename="default"):
    model_flag = type(model.model).__name__ if model is not None else "default"

    unet_keys = ldm_patched.modules.lora.model_lora_keys_unet(model.model) if model is not None else {}
    clip_keys = ldm_patched.modules.lora.model_lora_keys_clip(clip.cond_stage_model) if clip is not None else {}

    lora_unmatch = lora
    lora_unet, lora_unmatch = ldm_patched.modules.lora.load_lora(lora_unmatch, unet_keys)
    lora_clip, lora_unmatch = ldm_patched.modules.lora.load_lora(lora_unmatch, clip_keys)

    if len(lora_unmatch) > 12:
        print(f"[LORA] LoRA version mismatch for {model_flag}: {filename}")
        return model, clip

    if len(lora_unmatch) > 0:
        print(f"[LORA] Loading {filename} for {model_flag} with unmatched keys {list(lora_unmatch.keys())}")

    new_model = model.clone() if model is not None else None
    new_clip = clip.clone() if clip is not None else None

    if new_model is not None and len(lora_unet) > 0:
        loaded_keys = new_model.add_patches(lora_unet, strength_model)
        skipped_keys = [item for item in lora_unet if item not in loaded_keys]
        if len(skipped_keys) > 12:
            print(f"[LORA] Mismatch {filename} for {model_flag}-UNet with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys")
        else:
            print(f"[LORA] Loaded {filename} for {model_flag}-UNet with {len(loaded_keys)} keys at weight {strength_model} (skipped {len(skipped_keys)} keys)")
            model = new_model

    if new_clip is not None and len(lora_clip) > 0:
        loaded_keys = new_clip.add_patches(lora_clip, strength_clip)
        skipped_keys = [item for item in lora_clip if item not in loaded_keys]
        if len(skipped_keys) > 12:
            print(f"[LORA] Mismatch {filename} for {model_flag}-CLIP with {len(skipped_keys)} keys mismatched in {len(loaded_keys)} keys")
        else:
            print(f"[LORA] Loaded {filename} for {model_flag}-CLIP with {len(loaded_keys)} keys at weight {strength_clip} (skipped {len(skipped_keys)} keys)")
            clip = new_clip

    return model, clip


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        params["device"] = offload_device
        params["dtype"] = model_management.text_encoder_dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ldm_patched.modules.model_patcher.ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        if self.layer_idx is not None:
            self.cond_stage_model.clip_layer(self.layer_idx)
        else:
            self.cond_stage_model.reset_clip_layer()

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd):
        return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        return self.cond_stage_model.state_dict()

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None, no_init=False):
        if no_init:
            return

        if "decoder.up_blocks.0.resnets.0.norm1.weight" in sd.keys():  # diffusers format
            sd = diffusers_convert.convert_vae_state_dict(sd)

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype)  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)
        self.downscale_ratio = 8
        self.latent_channels = 4

        if config is None:
            if "taesd_decoder.1.weight" in sd:
                self.first_stage_model = ldm_patched.taesd.taesd.TAESD()
            else:
                # default SD1 VAE parameters
                ddconfig = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }

                if "encoder.down.2.downsample.conv.weight" not in sd:  # Stable diffusion x4 upscaler VAE
                    ddconfig["ch_mult"] = [1, 2, 4]
                    self.downscale_ratio = 4

                self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=4)
        else:
            self.first_stage_model = AutoencoderKL(**(config["params"]))
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("Missing VAE keys", m)

        if len(u) > 0:
            print("Leftover VAE keys", u)

        if device is None:
            device = model_management.vae_device()
        self.device = device
        offload_device = model_management.vae_offload_device()
        if dtype is None:
            dtype = model_management.vae_dtype()
        print("VAE dtype:", dtype)
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = model_management.intermediate_device()

        self.patcher = ldm_patched.modules.model_patcher.ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        return n

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = ldm_patched.modules.utils.ProgressBar(steps, title="VAE tiled decode")

        decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()
        output = torch.clamp(
            (
                (
                    ldm_patched.modules.utils.tiled_scale(
                        samples,
                        decode_fn,
                        tile_x // 2,
                        tile_y * 2,
                        overlap,
                        upscale_amount=self.downscale_ratio,
                        output_device=self.output_device,
                        pbar=pbar,
                    )
                    + ldm_patched.modules.utils.tiled_scale(
                        samples,
                        decode_fn,
                        tile_x * 2,
                        tile_y // 2,
                        overlap,
                        upscale_amount=self.downscale_ratio,
                        output_device=self.output_device,
                        pbar=pbar,
                    )
                    + ldm_patched.modules.utils.tiled_scale(
                        samples,
                        decode_fn,
                        tile_x,
                        tile_y,
                        overlap,
                        upscale_amount=self.downscale_ratio,
                        output_device=self.output_device,
                        pbar=pbar,
                    )
                )
                / 3.0
            )
            / 2.0,
            min=0.0,
            max=1.0,
        )
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x // 2,
            tile_y * 2,
            overlap,
        )
        steps += pixel_samples.shape[0] * ldm_patched.modules.utils.get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x * 2,
            tile_y // 2,
            overlap,
        )
        pbar = ldm_patched.modules.utils.ProgressBar(steps, title="VAE tiled encode")

        encode_fn = lambda a: self.first_stage_model.encode((2.0 * a - 1.0).to(self.vae_dtype).to(self.device)).float()
        samples = ldm_patched.modules.utils.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x,
            tile_y,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples += ldm_patched.modules.utils.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x * 2,
            tile_y // 2,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples += ldm_patched.modules.utils.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x // 2,
            tile_y * 2,
            overlap,
            upscale_amount=(1 / self.downscale_ratio),
            out_channels=self.latent_channels,
            output_device=self.output_device,
            pbar=pbar,
        )
        samples /= 3.0
        return samples

    def decode_inner(self, samples_in):
        if model_management.VAE_ALWAYS_TILED:
            return self.decode_tiled(samples_in).to(self.output_device)

        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            model_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty(
                (
                    samples_in.shape[0],
                    3,
                    round(samples_in.shape[2] * self.downscale_ratio),
                    round(samples_in.shape[3] * self.downscale_ratio),
                ),
                device=self.output_device,
            )
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x : x + batch_number] = torch.clamp(
                    (self.first_stage_model.decode(samples).to(self.output_device).float() + 1.0) / 2.0,
                    min=0.0,
                    max=1.0,
                )
        except model_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            pixel_samples = self.decode_tiled_(samples_in)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode(self, samples_in):
        wrapper = self.patcher.model_options.get("model_vae_decode_wrapper", None)
        if wrapper is None:
            return self.decode_inner(samples_in)
        else:
            return wrapper(self.decode_inner, samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        model_management.load_model_gpu(self.patcher)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        return output.movedim(1, -1)

    def encode_inner(self, pixel_samples):
        if model_management.VAE_ALWAYS_TILED:
            return self.encode_tiled(pixel_samples)

        pixel_samples = pixel_samples.movedim(-1, 1)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            model_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = model_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)
            samples = torch.empty(
                (
                    pixel_samples.shape[0],
                    self.latent_channels,
                    round(pixel_samples.shape[2] // self.downscale_ratio),
                    round(pixel_samples.shape[3] // self.downscale_ratio),
                ),
                device=self.output_device,
            )
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (2.0 * pixel_samples[x : x + batch_number] - 1.0).to(self.vae_dtype).to(self.device)
                samples[x : x + batch_number] = self.first_stage_model.encode(pixels_in).to(self.output_device).float()

        except model_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode(self, pixel_samples):
        wrapper = self.patcher.model_options.get("model_vae_encode_wrapper", None)
        if wrapper is None:
            return self.encode_inner(pixel_samples)
        else:
            return wrapper(self.encode_inner, pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        model_management.load_model_gpu(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()
