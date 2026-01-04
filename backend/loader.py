import importlib
import logging
import os

import huggingface_guess
import torch
from diffusers import DiffusionPipeline
from transformers import modeling_utils

import backend.args
from backend import memory_management
from backend.diffusion_engine.chroma import Chroma
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.lumina import Lumina2
from backend.diffusion_engine.qwen import QwenImage
from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
from backend.diffusion_engine.wan import Wan
from backend.diffusion_engine.zimage import ZImage
from backend.nn.clip import IntegratedCLIP
from backend.nn.unet import IntegratedUNet2DConditionModel
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.wan_vae import WanVAE
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict, try_filter_state_dict
from backend.utils import (
    beautiful_print_gguf_state_dict_statics,
    load_torch_file,
    read_arbitrary_config,
)

possible_models = [StableDiffusion, StableDiffusionXLRefiner, StableDiffusionXL, Chroma, Flux, Wan, QwenImage, Lumina2, ZImage]


logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ["feature_extractor", "safety_checker"]:
        return None

    if lib_name in ["transformers", "diffusers"]:
        if component_name == "scheduler":
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if component_name.startswith("tokenizer"):
            cls = getattr(importlib.import_module(lib_name), cls_name)
            comp = cls.from_pretrained(os.path.join(repo_path, component_name))
            comp._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
            return comp
        if cls_name == "AutoencoderKL":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have VAE state dict!"

            config = IntegratedAutoencoderKL.load_config(config_path)

            with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                model = IntegratedAutoencoderKL.from_config(config)

            if "decoder.up_blocks.0.resnets.0.norm1.weight" in state_dict.keys():  # diffusers format
                state_dict = huggingface_guess.diffusers_convert.convert_vae_state_dict(state_dict)
            load_state_dict(model, state_dict, ignore_start="loss.")
            return model
        if cls_name in ["AutoencoderKLWan", "AutoencoderKLQwenImage"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have VAE state dict!"

            config = WanVAE.load_config(config_path)

            with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                model = WanVAE.from_config(config)

            load_state_dict(model, state_dict)
            return model
        if component_name.startswith("text_encoder") and cls_name in ["CLIPTextModel", "CLIPTextModelWithProjection"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have CLIP state dict!"

            from transformers import CLIPTextConfig, CLIPTextModel

            config = CLIPTextConfig.from_pretrained(config_path)

            to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

            with modeling_utils.no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=True):
                    model = IntegratedCLIP(CLIPTextModel, config, add_text_projection=True).to(**to_args)

            load_state_dict(model, state_dict, ignore_errors=["transformer.text_projection.weight", "transformer.text_model.embeddings.position_ids", "logit_scale"], log_name=cls_name)

            return model
        if cls_name == "Qwen2_5_VLForConditionalGeneration":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Qwen 2.5 state dict!"

            from backend.nn.llm.llama import Qwen25_7BVLI

            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected Qwen2.5 Data Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f"Using Default Qwen2.5 Data Type: {storage_dtype}")

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = Qwen25_7BVLI(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = Qwen25_7BVLI(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=["lm_head.weight"])

            return model
        if cls_name == "Gemma2Model":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Gemma2 state dict!"

            from backend.nn.llm.llama import Gemma2_2B

            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected Gemma2 Data Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f"Using Default Gemma2 Data Type: {storage_dtype}")

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = Gemma2_2B(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = Gemma2_2B(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=[])

            return model
        if cls_name == "Qwen3Model":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Qwen3 state dict!"

            from backend.nn.llm.llama import Qwen3_4B

            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected Qwen3 Data Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f"Using Default Qwen3 Data Type: {storage_dtype}")

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = Qwen3_4B(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = Qwen3_4B(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=[])

            return model
        if cls_name in ["T5EncoderModel", "UMT5EncoderModel"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have T5 state dict!"

            if filename := state_dict.get("transformer.filename", None):
                if memory_management.is_device_cpu(memory_management.text_encoder_device()):
                    raise SystemError("nunchaku T5 does not support CPU!")

                from backend.nn.svdq import SVDQT5

                print("Using Nunchaku T5")
                model = SVDQT5(filename)
                return model

            from backend.nn.t5 import IntegratedT5

            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected T5 Data Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f"Using Default T5 Data Type: {storage_dtype}")

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = IntegratedT5(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = IntegratedT5(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=["transformer.encoder.embed_tokens.weight", "logit_scale"])

            return model
        if cls_name in ["UNet2DConditionModel", "FluxTransformer2DModel", "ChromaTransformer2DModel", "WanTransformer3DModel", "QwenImageTransformer2DModel", "Lumina2Transformer2DModel", "ZImageTransformer2DModel"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have model state dict!"

            model_loader = None
            _nz = False  # Nunchaku Z-Image

            if cls_name == "UNet2DConditionModel":
                model_loader = lambda c: IntegratedUNet2DConditionModel.from_config(c)
            elif cls_name == "FluxTransformer2DModel":
                if guess.nunchaku:
                    from backend.nn.svdq import SVDQFluxTransformer2DModel

                    model_loader = lambda c: SVDQFluxTransformer2DModel(c)
                else:
                    from backend.nn.flux import IntegratedFluxTransformer2DModel

                    model_loader = lambda c: IntegratedFluxTransformer2DModel(**c)
            elif cls_name == "ChromaTransformer2DModel":
                from backend.nn.chroma import IntegratedChromaTransformer2DModel

                model_loader = lambda c: IntegratedChromaTransformer2DModel(**c)
            elif cls_name == "WanTransformer3DModel":
                from backend.nn.wan import WanModel

                model_loader = lambda c: WanModel(**c)
            elif cls_name == "QwenImageTransformer2DModel":
                if guess.nunchaku:
                    from backend.nn.svdq import NunchakuQwenImageTransformer2DModel

                    model_loader = lambda c: NunchakuQwenImageTransformer2DModel(**c)
                else:
                    from backend.nn.qwen import QwenImageTransformer2DModel

                    model_loader = lambda c: QwenImageTransformer2DModel(**c)
            elif cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel"):
                if guess.nunchaku:
                    from backend.nn.svdq import patch_nunchaku_zimage

                    guess.unet_config.pop("filename")
                    precision = guess.unet_config.pop("precision")
                    rank = guess.unet_config.pop("rank")
                    _nz = True

                from backend.nn.lumina import NextDiT

                model_loader = lambda c: NextDiT(**c)

            unet_config = guess.unet_config.copy()
            state_dict_parameters = memory_management.state_dict_parameters(state_dict)
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)

            unet_storage_dtype_overwrite = backend.args.dynamic_args.get("forge_unet_storage_dtype")

            if unet_storage_dtype_overwrite is not None:
                storage_dtype = unet_storage_dtype_overwrite
            elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected UNet Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)

            load_device = memory_management.get_torch_device()
            computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)
            offload_device = memory_management.unet_offload_device()

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
                with using_forge_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                    model = model_loader(unet_config)
            else:
                initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
                need_manual_cast = storage_dtype != computation_dtype
                to_args = dict(device=initial_device, dtype=storage_dtype)

                with using_forge_operations(operations=False if _nz else None, **to_args, manual_cast_enabled=need_manual_cast):
                    model = model_loader(unet_config).to(**to_args)

            if _nz:
                model = patch_nunchaku_zimage(model, precision, rank)
            load_state_dict(model, state_dict)

            if hasattr(model, "_internal_dict"):
                model._internal_dict = unet_config
            else:
                model.config = unet_config

            model.storage_dtype = storage_dtype
            model.computation_dtype = computation_dtype
            model.load_device = load_device
            model.initial_device = initial_device
            model.offload_device = offload_device

            return model

    print(f"Skipped: {component_name} = {lib_name}.{cls_name}")
    return None


def replace_state_dict(sd: dict[str, torch.Tensor], asd: dict[str, torch.Tensor], guess, path: os.PathLike):
    vae_key_prefix = guess.vae_key_prefix[0]
    text_encoder_key_prefix = guess.text_encoder_key_prefix[0]

    if "enc.blk.0.attn_k.weight" in asd:
        gguf_t5_format = {  # city96
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        asd_new = {}
        for k, v in asd.items():
            for s, d in gguf_t5_format.items():
                k = k.replace(s, d)
            asd_new[k] = v
        for k in ("shared.weight",):
            asd_new[k] = asd_new[k].dequantize_as_pytorch_parameter()
        asd.clear()
        asd = asd_new

    if "blk.0.attn_norm.weight" in asd:
        gguf_llm_format = {  # city96
            "blk.": "model.layers.",
            "attn_norm": "input_layernorm",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_v_norm.": "self_attn.v_norm.",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "token_embd": "model.embed_tokens",
            "output_norm": "model.norm",
            "output.weight": "lm_head.weight",
        }
        asd_new = {}
        for k, v in asd.items():
            for s, d in gguf_llm_format.items():
                k = k.replace(s, d)
            asd_new[k] = v
        for k in ("model.embed_tokens.weight",):
            asd_new[k] = asd_new[k].dequantize_as_pytorch_parameter()
        asd.clear()
        asd = asd_new

    #   sd / sdxl / wan                  # wan
    if "decoder.conv_in.weight" in asd or "decoder.middle.0.residual.0.gamma" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v

    ##  identify model type
    flux_test_key = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
    svdq_test_key = "model.diffusion_model.single_transformer_blocks.0.mlp_fc1.qweight"
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"

    model_type = "-"
    if legacy_test_key in sd:
        match sd[legacy_test_key].shape[1]:
            case 768:
                model_type = "sd1"
            case 1280:
                model_type = "xlrf"  # sdxl refiner model
            case 2048:
                model_type = "sdxl"
    elif flux_test_key in sd or svdq_test_key in sd:
        model_type = "flux"

    ##  prefixes used by various model types for CLIP-L
    prefix_L = {
        "-": None,
        "sd1": "cond_stage_model.transformer.",
        "xlrf": None,
        "sdxl": "conditioner.embedders.0.transformer.",
        "flux": "text_encoders.clip_l.transformer.",
    }
    ##  prefixes used by various model types for CLIP-G
    prefix_G = {
        "-": None,
        "sd1": None,
        "xlrf": "conditioner.embedders.0.model.transformer.",
        "sdxl": "conditioner.embedders.1.model.transformer.",
        "flux": None,
    }

    ##  VAE format 0 (extracted from model, could be sd1/sdxl)
    if "first_stage_model.decoder.conv_in.weight" in asd:
        if model_type in ("sd1", "xlrf", "sdxl"):
            assert asd["first_stage_model.decoder.conv_in.weight"].shape[1] == 4
            for k, v in asd.items():
                sd[k] = v

    ##  CLIP-G
    CLIP_G = {"conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": "conditioner.embedders.1.model.transformer.", "text_encoders.clip_g.transformer.text_model.encoder.layers.0.layer_norm1.bias": "text_encoders.clip_g.transformer.", "text_model.encoder.layers.0.layer_norm1.bias": "", "transformer.resblocks.0.ln_1.bias": "transformer."}  #   key to identify source model                                                old_prefix
    for CLIP_key in CLIP_G.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 1280:
            new_prefix = prefix_G[model_type]
            old_prefix = CLIP_G[CLIP_key]

            if new_prefix is not None:
                if "resblocks" not in CLIP_key:  # need to convert

                    def convert_transformers(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "{}text_model.embeddings.position_embedding.weight": "{}positional_embedding",
                            "{}text_model.embeddings.token_embedding.weight": "{}token_embedding.weight",
                            "{}text_model.final_layer_norm.weight": "{}ln_final.weight",
                            "{}text_model.final_layer_norm.bias": "{}ln_final.bias",
                            "text_projection.weight": "{}text_projection",
                        }
                        resblock_to_replace = {
                            "layer_norm1": "ln_1",
                            "layer_norm2": "ln_2",
                            "mlp.fc1": "mlp.c_fc",
                            "mlp.fc2": "mlp.c_proj",
                            "self_attn.out_proj": "attn.out_proj",
                        }

                        for x in keys_to_replace:  #   remove trailing 'transformer.' from new prefix
                            k = x.format(prefix_from)
                            statedict[keys_to_replace[x].format(prefix_to[:-12])] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}resblocks.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.q_proj", y)
                                weightsQ = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.k_proj", y)
                                weightsK = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.v_proj", y)
                                weightsV = statedict.pop(k_from)

                                k_to = "{}resblocks.{}.attn.in_proj_{}".format(prefix_to, resblock, y)

                                statedict[k_to] = torch.cat((weightsQ, weightsK, weightsV))
                        return statedict

                    asd = convert_transformers(asd, old_prefix, new_prefix, 32)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    ##  CLIP-L
    CLIP_L = {"cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": "cond_stage_model.transformer.", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": "conditioner.embedders.0.transformer.", "text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias": "text_encoders.clip_l.transformer.", "text_model.encoder.layers.0.layer_norm1.bias": "", "transformer.resblocks.0.ln_1.bias": "transformer."}  #   key to identify source model                                                    old_prefix

    for CLIP_key in CLIP_L.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 768:
            new_prefix = prefix_L[model_type]
            old_prefix = CLIP_L[CLIP_key]

            if new_prefix is not None:
                if "resblocks" in CLIP_key:  # need to convert

                    def transformers_convert(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "positional_embedding": "{}text_model.embeddings.position_embedding.weight",
                            "token_embedding.weight": "{}text_model.embeddings.token_embedding.weight",
                            "ln_final.weight": "{}text_model.final_layer_norm.weight",
                            "ln_final.bias": "{}text_model.final_layer_norm.bias",
                            "text_projection": "text_projection.weight",
                        }
                        resblock_to_replace = {
                            "ln_1": "layer_norm1",
                            "ln_2": "layer_norm2",
                            "mlp.c_fc": "mlp.fc1",
                            "mlp.c_proj": "mlp.fc2",
                            "attn.out_proj": "self_attn.out_proj",
                        }

                        for k in keys_to_replace:
                            statedict[keys_to_replace[k].format(prefix_to)] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
                                weights = statedict.pop(k_from)
                                shape_from = weights.shape[0] // 3
                                for x in range(3):
                                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                                    statedict[k_to] = weights[shape_from * x : shape_from * (x + 1)]
                        return statedict

                    asd = transformers_convert(asd, old_prefix, new_prefix, 12)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    if "encoder.block.0.layer.0.SelfAttention.k.weight" in asd:
        _key = "umt5xxl" if asd["shared.weight"].size(0) == 256384 else "t5xxl"
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}{_key}.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            if k == "spiece_model":
                continue
            sd[f"{text_encoder_key_prefix}{_key}.transformer.{k}"] = v

    elif "encoder.block.0.layer.0.SelfAttention.k.qweight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}t5xxl.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}t5xxl.transformer.{k}"] = True
        sd[f"{text_encoder_key_prefix}t5xxl.transformer.filename"] = str(path)

    if "model.layers.0.post_feedforward_layernorm.weight" in asd:
        assert "model.layers.0.self_attn.q_norm.weight" not in asd
        for k, v in asd.items():
            if k == "spiece_model":
                continue
            sd[f"{text_encoder_key_prefix}gemma2_2b.{k}"] = v

    elif "model.layers.0.self_attn.k_proj.bias" in asd:
        weight = asd["model.layers.0.self_attn.k_proj.bias"]
        assert weight.shape[0] == 512
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen25_7b.{k}"] = v

    elif "model.layers.0.post_attention_layernorm.weight" in asd:
        assert "model.layers.0.self_attn.q_norm.weight" in asd
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen3_4b.transformer.{k}"] = v

    return sd


def preprocess_state_dict(sd):
    if not any(k.startswith("model.diffusion_model") for k in sd.keys()):
        sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}

    return sd


def split_state_dict(sd, additional_state_dicts: list = None):
    sd, metadata = load_torch_file(sd, return_metadata=True)
    sd = preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    if getattr(guess, "nunchaku", False) and ("Z-Image" in guess.huggingface_repo or "Qwen" in guess.huggingface_repo):
        import json

        from nunchaku.utils import get_precision_from_quantization_config

        quantization_config = json.loads(metadata["quantization_config"])
        guess.unet_config.update(
            {
                "precision": get_precision_from_quantization_config(quantization_config),
                "rank": quantization_config.get("rank", 32),
            }
        )

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            _asd = load_torch_file(asd)
            sd = replace_state_dict(sd, _asd, guess, asd)
            del _asd

    guess.clip_target = guess.clip_target(sd)
    guess.model_type = guess.model_type(sd)
    guess.ztsnr = "ztsnr" in sd

    sd = guess.process_vae_state_dict(sd)

    state_dict = {guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix), guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)}

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [k + "."])

    state_dict["ignore"] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f"StateDict Keys: {print_dict}")

    del state_dict["ignore"]

    return state_dict, guess


@torch.inference_mode()
def forge_loader(sd: os.PathLike, additional_state_dicts: list[os.PathLike] = None):
    try:
        state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts=additional_state_dicts)
    except Exception as e:
        from modules.errors import display

        display(e, "forge_loader")
        raise ValueError("Failed to recognize model type!")

    repo_name = estimated_config.huggingface_repo
    backend.args.dynamic_args["kontext"] = "kontext" in str(sd).lower()
    backend.args.dynamic_args["edit"] = "qwen" in str(sd).lower() and "edit" in str(sd).lower()
    backend.args.dynamic_args["nunchaku"] = getattr(estimated_config, "nunchaku", False)

    if getattr(estimated_config, "nunchaku", False):
        estimated_config.unet_config["filename"] = str(sd)

    local_path = os.path.join(dir_path, "huggingface", repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)
    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.pop(component_name, None)
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del component_sd
            if component is not None:
                huggingface_components[component_name] = component

    del state_dicts

    yaml_config = None
    yaml_config_prediction_type = None

    try:
        from pathlib import Path

        import yaml

        config_filename = os.path.splitext(sd)[0] + ".yaml"
        if Path(config_filename).is_file():
            with open(config_filename, "r") as stream:
                yaml_config = yaml.safe_load(stream)
    except ImportError:
        pass

    prediction_types = {
        "EPS": "epsilon",
        "V_PREDICTION": "v_prediction",
        "FLUX": "const",
        "FLOW": "const",
    }

    has_prediction_type = "scheduler" in huggingface_components and hasattr(huggingface_components["scheduler"], "config") and "prediction_type" in huggingface_components["scheduler"].config

    if yaml_config is not None:
        yaml_config_prediction_type: str = yaml_config.get("model", {}).get("params", {}).get("parameterization", "") or yaml_config.get("model", {}).get("params", {}).get("denoiser_config", {}).get("params", {}).get("scaling_config", {}).get("target", "")
        if yaml_config_prediction_type == "v" or yaml_config_prediction_type.endswith(".VScaling"):
            yaml_config_prediction_type = "v_prediction"
        else:
            # Use estimated prediction config if no suitable prediction type found
            yaml_config_prediction_type = ""

    if has_prediction_type:
        if yaml_config_prediction_type:
            huggingface_components["scheduler"].config.prediction_type = yaml_config_prediction_type
        else:
            huggingface_components["scheduler"].config.prediction_type = prediction_types.get(estimated_config.model_type.name, huggingface_components["scheduler"].config.prediction_type)

    for M in possible_models:
        if any(type(estimated_config) is x for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print("Failed to recognize model type!")
    return None
