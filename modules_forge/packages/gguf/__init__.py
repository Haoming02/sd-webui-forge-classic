from .constants import *
from .gguf_reader import *
from .gguf_writer import *
from .lazy import *
from .metadata import *
from .quants import *
from .tensor_mapping import *
from .utility import *


def get_orig_shape(reader: "GGUFReader", tensor_name: str) -> torch.Size | None:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) == 2 and field.types[0] == gguf.GGUFValueType.ARRAY and field.types[1] == gguf.GGUFValueType.INT32:
        return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))
    return None
