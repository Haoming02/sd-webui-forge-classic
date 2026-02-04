import os
import sys

from modules.timer import startup_timer

INITIALIZED = False


def initialize_forge():
    global INITIALIZED
    if INITIALIZED:
        return

    INITIALIZED = True

    # region Comfy
    # https://github.com/Comfy-Org/ComfyUI/blob/v0.10.0/main.py

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"

    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    # endregion

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules_forge", "packages"))

    from backend.args import args

    if args.gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)
        print("Set device to:", args.gpu_device_id)

    from modules_forge.cuda_malloc import get_torch_version, try_cuda_malloc

    if "rocm" in get_torch_version():
        # https://github.com/Comfy-Org/ComfyUI/blob/v0.10.0/main.py
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        os.environ["OCL_SET_SVM_SIZE"] = "262144"

    if args.cuda_malloc:
        try_cuda_malloc()
        startup_timer.record("cuda_malloc")

    from backend import memory_management

    startup_timer.record("memory_management")

    import torch
    import torchvision  # noqa: F401

    startup_timer.record("import torch")

    device = memory_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    memory_management.soft_empty_cache()

    startup_timer.record("warmup")

    from modules_forge.shared import diffusers_dir

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = diffusers_dir

    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = diffusers_dir

    if "HUGGINGFACE_HUB_CACHE" not in os.environ:
        os.environ["HUGGINGFACE_HUB_CACHE"] = diffusers_dir

    if "HUGGINGFACE_ASSETS_CACHE" not in os.environ:
        os.environ["HUGGINGFACE_ASSETS_CACHE"] = diffusers_dir

    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = diffusers_dir

    startup_timer.record("diffusers_dir")

    from modules_forge import patch_basic

    patch_basic.patch_all_basics()

    startup_timer.record("patch basics")

    from backend.huggingface import process

    process()

    startup_timer.record("decompress tokenizers")
