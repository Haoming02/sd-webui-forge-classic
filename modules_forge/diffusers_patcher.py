import torch
from transformers.modeling_utils import no_init_weights

from backend import memory_management
from backend.operations import using_forge_operations
from backend.patcher.base import ModelPatcher


class DiffusersModelPatcher:
    def __init__(self, pipeline_class, dtype=torch.float16, *args, **kwargs):
        load_device = memory_management.get_torch_device()
        offload_device = torch.device("cpu")

        if not memory_management.should_use_fp16(device=load_device):
            dtype = torch.float32

        self.dtype = dtype

        with no_init_weights():
            with using_forge_operations():
                self.pipeline = pipeline_class.from_pretrained(*args, **kwargs)

        if hasattr(self.pipeline, "unet"):
            if hasattr(self.pipeline.unet, "set_attn_processor"):
                from diffusers.models.attention_processor import AttnProcessor2_0

                self.pipeline.unet.set_attn_processor(AttnProcessor2_0())

        self.pipeline = self.pipeline.to(device=offload_device)

        if self.dtype == torch.float16:
            self.pipeline = self.pipeline.half()

        self.pipeline.eval()

        self.patcher = ModelPatcher(
            model=self.pipeline,
            load_device=load_device,
            offload_device=offload_device,
        )

    def prepare_memory_before_sampling(self, batchsize, latent_width, latent_height):
        area = 2 * batchsize * latent_width * latent_height
        inference_memory = (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)
        memory_management.load_models_gpu(
            models=[self.patcher],
            memory_required=inference_memory,
        )

    def move_tensor_to_current_device(self, x):
        return x.to(device=self.patcher.current_device, dtype=self.dtype)
