import os
import sys

INITIALIZED = False
MONITOR_MODEL_MOVING = False


def monitor_module_moving():
    if not MONITOR_MODEL_MOVING:
        return

    import torch
    import traceback

    old_to = torch.nn.Module.to

    def new_to(*args, **kwargs):
        traceback.print_stack()
        print("Model Movement")

        return old_to(*args, **kwargs)

    torch.nn.Module.to = new_to
    return


def fix_logging():
    import logging

    logging.getLogger("nunchaku.caching.teacache").addFilter(lambda record: "deprecated" not in record.getMessage().lower())
    logging.getLogger("nunchaku.models.pulid.pulid_forward").addFilter(lambda record: "deprecated" not in record.getMessage().lower())
    logging.getLogger("nunchaku.models.transformers.transformer_flux").addFilter(lambda record: "deprecated" not in record.getMessage().lower())

    _original = logging.basicConfig

    logging.basicConfig = lambda *args, **kwargs: None

    try:
        import nunchaku
    except ImportError:
        pass

    logging.basicConfig = _original


def initialize_forge(startup_timer):
    global INITIALIZED

    if INITIALIZED:
        return

    INITIALIZED = True

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules_forge", "packages"))

    from backend.args import args

    if args.gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)
        print("Set device to:", args.gpu_device_id)

    if args.cuda_malloc:
        from modules_forge.cuda_malloc import try_cuda_malloc

        try_cuda_malloc()

    from backend import memory_management
    import torch
    import torchvision  # noqa: F401
    import pytorch_lightning  # noqa: F401

    startup_timer.record("import torch")

    monitor_module_moving()

    device = memory_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    memory_management.soft_empty_cache()

    from backend import stream
    print("CUDA Using Stream:", stream.should_use_stream())

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

    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()

    fix_logging()

    from backend.huggingface import process
    process()

    startup_timer.record("forge init")
