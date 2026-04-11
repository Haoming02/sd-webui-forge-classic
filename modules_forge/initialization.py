import os


def initialize_forge():
    from ldm_patched.modules import args_parser

    args_parser.args, _ = args_parser.parser.parse_known_args()

    if args_parser.args.gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args_parser.args.gpu_device_id)
        print("Set device to:", args_parser.args.gpu_device_id)

    if args_parser.args.cuda_malloc:
        from modules_forge.cuda_malloc import try_cuda_malloc
        try_cuda_malloc()

    import ldm_patched.modules.model_management as model_management
    import torch

    device = model_management.get_torch_device()
    torch.zeros((1, 1)).to(device, torch.float32)
    model_management.soft_empty_cache()

    import modules_forge.patch_basic
    modules_forge.patch_basic.patch_all_basics()

    from modules_forge import stream
    print("CUDA Stream Activated: ", stream.using_stream)

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
