# https://github.com/Comfy-Org/ComfyUI/tree/master/comfy/weight_adapter

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
