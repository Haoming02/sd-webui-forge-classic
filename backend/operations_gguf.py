import torch
from gguf import GGMLQuantizationType

from modules_forge.packages.gguf import quants

QUANTS_MAPPING: dict[GGMLQuantizationType, type[quants.TorchQuant]] = {
    GGMLQuantizationType.Q2_K: quants.Q2_K,
    GGMLQuantizationType.Q3_K: quants.Q3_K,
    GGMLQuantizationType.Q4_0: quants.Q4_0,
    GGMLQuantizationType.Q4_1: quants.Q4_1,
    GGMLQuantizationType.Q4_K: quants.Q4_K,
    GGMLQuantizationType.Q5_0: quants.Q5_0,
    GGMLQuantizationType.Q5_1: quants.Q5_1,
    GGMLQuantizationType.Q5_K: quants.Q5_K,
    GGMLQuantizationType.Q6_K: quants.Q6_K,
    GGMLQuantizationType.Q8_0: quants.Q8_0,
    GGMLQuantizationType.BF16: quants.BF16,
    GGMLQuantizationType.IQ1_S: quants.IQ1_S,
    GGMLQuantizationType.IQ1_M: quants.IQ1_M,
    GGMLQuantizationType.IQ2_XXS: quants.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS: quants.IQ2_XS,
    GGMLQuantizationType.IQ2_S: quants.IQ2_S,
    GGMLQuantizationType.IQ3_XXS: quants.IQ3_XXS,
    GGMLQuantizationType.IQ3_S: quants.IQ3_S,
    GGMLQuantizationType.IQ4_NL: quants.IQ4_NL,
    GGMLQuantizationType.IQ4_XS: quants.IQ4_XS,
}


class ParameterGGUF(torch.nn.Parameter):
    def __init__(self, torch_tensor, *, tensor_type=None, tensor_shape=None, no_init=False):
        super().__init__()
        if no_init:
            return

        self.gguf_cls = QUANTS_MAPPING.get(tensor_type)
        self.real_shape: torch.Size = tensor_shape
        self.computation_dtype = torch.float16
        self.baked = False

    @property
    def shape(self):
        return self.real_shape

    def __new__(cls, torch_tensor, *, tensor_type=None, tensor_shape=None, no_init=False):
        return super().__new__(cls, torch_tensor, requires_grad=False)

    def dequantize_as_pytorch_parameter(self):
        if self.gguf_cls is not None:
            self.gguf_cls.bake(self)
        return torch.nn.Parameter(dequantize_tensor(self), requires_grad=False)

    def copy_with_data(self, data):
        new = ParameterGGUF(data, no_init=True)
        new.gguf_cls = self.gguf_cls
        new.real_shape = self.real_shape
        new.computation_dtype = self.computation_dtype
        new.baked = self.baked
        return new

    def to(self, *args, **kwargs):
        return self.copy_with_data(self.data.to(*args, **kwargs))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))


def dequantize_tensor(tensor: "ParameterGGUF", *, input: torch.Tensor = None) -> torch.Tensor:
    if tensor is None:
        return None

    if not hasattr(tensor, "gguf_cls"):
        return tensor

    if (gguf_cls := tensor.gguf_cls) is not None:
        if input is not None and input.is_cuda and gguf_cls in (quants.Q2_K, quants.Q3_K, quants.Q4_K, quants.Q5_K, quants.Q6_K):
            from modules_forge.packages.gguf.kquant_triton import try_dequantize

            if (weight := try_dequantize(tensor, input)) is not None:
                return weight
        return gguf_cls.dequantize_pytorch(tensor)

    return tensor
