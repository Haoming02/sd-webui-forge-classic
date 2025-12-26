# https://github.com/comfyanonymous/ComfyUI/tree/v0.3.77/comfy/weight_adapter

from typing import Final

from .base import WeightAdapterBase
from .boft import BOFTAdapter
from .glora import GLoRAAdapter
from .loha import LoHaAdapter
from .lokr import LoKrAdapter
from .lora import LoRAAdapter
from .oft import OFTAdapter
from .oftv2 import OFTv2Adapter

adapters: Final[list[type[WeightAdapterBase]]] = [
    BOFTAdapter,
    GLoRAAdapter,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    OFTAdapter,
    OFTv2Adapter,
]

adapter_maps: Final[dict[str, type[WeightAdapterBase]]] = {
    "LoRA": LoRAAdapter,
    "LoHa": LoHaAdapter,
    "LoKr": LoKrAdapter,
    "OFT": OFTAdapter,
    "OFTv2": OFTv2Adapter,
    # "GLoRA": GLoRAAdapter,
    # "BOFT": BOFTAdapter,
}


__all__ = [
    "WeightAdapterBase",
    "adapters",
    "adapter_maps",
] + [a.__name__ for a in adapters]
