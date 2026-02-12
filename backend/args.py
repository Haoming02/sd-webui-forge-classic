# https://github.com/comfyanonymous/ComfyUI/blob/v0.7.0/comfy/cli_args.py

import argparse
import enum


class EnumAction(argparse.Action):
    """Argparse `action` for handling Enum"""

    def __init__(self, **kwargs):
        enum_type = kwargs.pop("type", None)
        assert issubclass(enum_type, enum.Enum)

        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(choices)}]")

        super(EnumAction, self).__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser(add_help=False)

parser.add_argument("--loglevel", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID", help="Set the id of device to use (all other devices will not be visible)")
parser.add_argument("--disable-gpu-warning", action="store_true", help="Disable the low VRAM warnings")

parser.add_argument("--text-enc-device", type=str, default=None, metavar="DEVICE", help='Set the device to load text encoder (e.g. "cuda:1")')
parser.add_argument("--vae-device", type=str, default=None, metavar="DEVICE", help='Set the device to load VAE (e.g. "cuda:1")')

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Store the diffusion model in fp32")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Store the diffusion model in bf16")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Store the diffusion model in fp16")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store the diffusion model in fp8_e4m3fn")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store the diffusion model in fp8_e5m2")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store the diffusion model in fp8_e8m0fnu")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16")
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16 (might cause black images)")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store the text encoder in fp32")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store the text encoder in bf16")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store the text encoder in fp16")
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store the text encoder in fp8_e4m3fn")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store the text encoder in fp8_e5m2")

parser.add_argument("--cpu-text-enc", action="store_true", help="Run the text encoder on the CPU")

parser.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use the PyTorch cross attention (override sageattention/flash_attn/xformers)")
parser.add_argument("--force-xformers-vae", action="store_true", help="Force VAE to use xformers attention (meant to use with PyTorch cross attention)")
parser.add_argument("--force-upcast-attention", action="store_true", help="Always upcast to fp32 during attention")

parser.add_argument("--sage", action="store_true", help="install sageattention")
parser.add_argument("--flash", action="store_true", help="install flash_attn")
parser.add_argument("--xformers", action="store_true", help="install xformers")
parser.add_argument("--nunchaku", action="store_true", help="install nunchaku for SVDQ inference")
parser.add_argument("--bnb", action="store_true", help="install bitsandbytes for 4-bit inference")
parser.add_argument("--onnxruntime-gpu", action="store_true", help="install nightly onnxruntime-gpu with cu130 support")

parser.add_argument("--disable-sage", action="store_true", help="disable sageattention")
parser.add_argument("--disable-flash", action="store_true", help="disable flash_attn")
parser.add_argument("--disable-xformers", action="store_true", help="disable xformers")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml")
parser.add_argument("--disable-ipex-optimize", action="store_true", help="Disable ipex.optimize default when loading models with Intel's Extension for PyTorch")
parser.add_argument("--deterministic", action="store_true", help="Use slower deterministic algorithms when possible")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything on the GPU")
vram_group.add_argument("--highvram", action="store_true", help="Keeps models in VRAM after usage")
vram_group.add_argument("--normalvram", action="store_true", help="Force NORMAL_VRAM in case LOW_VRAM gets automatically enabled")
vram_group.add_argument("--lowvram", action="store_true", help="Split the diffusion model in parts to use less VRAM")
vram_group.add_argument("--novram", action="store_true", help="When even LOW_VRAM is still not enough")
vram_group.add_argument("--cpu", action="store_true", help="Use the CPU for everything (slow)")

parser.add_argument("--reserve-vram", type=float, default=None, metavar="GB", help="Set the amount of VRAM you want to reserve for other software (by default some amount is reserved)")
parser.add_argument("--disable-smart-memory", action="store_true", help="Aggressively offload to RAM instead of keeping models in VRAM when possible")
parser.add_argument("--force-non-blocking", action="store_true", help="Use non-blocking operations for all applicable tensors")

parser.add_argument("--cuda-malloc", action="store_true", help="improve memory allocation")
parser.add_argument("--cuda-stream", type=int, nargs="?", metavar="NUM_STREAMS", const=2, help="improve offloading")
parser.add_argument("--pin-shared-memory", action="store_true", help="improve RAM utilization")

parser.add_argument("--fast-fp8", action="store_true", help="torch._scaled_mm")
parser.add_argument("--fast-fp16", action="store_true", help="torch.backends.cuda.matmul.allow_fp16_accumulation")
parser.add_argument("--autotune", action="store_true", help="torch.backends.cudnn.benchmark")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap when loading ckpt/pt files")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap when loading safetensors")


class SageAttentionFuncs(enum.Enum):
    auto = "auto"
    fp16_triton = "fp16_triton"
    fp16_cuda = "fp16_cuda"
    fp8_cuda = "fp8_cuda"


class Sage_quantization_backend(enum.Enum):
    cuda = "cuda"
    triton = "triton"


class Sage_qk_quant_gran(enum.Enum):
    per_warp = "per_warp"
    per_thread = "per_thread"


class Sage_pv_accum_dtype(enum.Enum):
    fp16 = "fp16"
    fp32 = "fp32"
    fp16fp32 = "fp16+fp32"
    fp32fp32 = "fp32+fp32"


sage2 = parser.add_argument_group(description="SageAttention 2")
sage2.add_argument("--sage2-function", type=SageAttentionFuncs, default=SageAttentionFuncs.auto, action=EnumAction)
sage2.add_argument("--sage-quantization-backend", type=Sage_quantization_backend, default=Sage_quantization_backend.triton, action=EnumAction)
sage2.add_argument("--sage-quant-gran", type=Sage_qk_quant_gran, default=Sage_qk_quant_gran.per_thread, action=EnumAction)
sage2.add_argument("--sage-accum-dtype", type=Sage_pv_accum_dtype, default=Sage_pv_accum_dtype.fp32, action=EnumAction)


args, _ = parser.parse_known_args()

# TODO: Stop using this to hack every problem...
dynamic_args = dict(
    embedding_dir=None,
    forge_unet_storage_dtype=None,
    online_lora=False,
    kontext=False,
    edit=False,
    nunchaku=False,
    klein=False,
    wan=False,
    ref_latents=[],
    concat_latent=None,
    is_referencing=False,
    ops=None,
)
"""
Some parameters that are used throughout the Webui
- embedding_dir: `str` - set in modules/sd_models/forge_model_reload
- forge_unet_storage_dtype: `torch.dtype` - set in modules/sd_models/forge_model_reload
- online_lora: `bool` - patch LoRAs on-the-fly
- kontext: `bool` - Flux Kontext
- edit: `bool` - Qwen-Image-Edit
- nunchaku: `bool` - Nunchaku (SVDQ) Models
- klein: `bool` - Flux.2 Klein
- wan: `bool` - Wan 2.2
- ref_latents: `list[torch.Tensor]` - Reference Latent(s) for Flux Kontext & Qwen-Image-Edit
- concat_latent: `torch.Tensor` - Input Latent for Wan 2.2 I2V
- is_referencing: `bool` - Appending Reference Latent(s) (by. ImageStitch)
- ops: `str` - Operations for the Diffusion Model
"""
