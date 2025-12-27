import pickle

import torch

load = pickle.load
unsafe_torch_load = torch.load


class Empty:
    pass


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module: str, name: str):
        if module.startswith("pytorch_lightning"):
            return Empty

        if module.startswith(("collections", "torch", "numpy", "__builtin__")):
            return super().find_class(module, name)

        raise NotImplementedError(f'"{module}.{name}" is forbidden')


class Extra:
    global_extra_handler = None

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        assert Extra.global_extra_handler is None, "already inside an Extra() block"
        Extra.global_extra_handler = self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        Extra.global_extra_handler = None


Unpickler = RestrictedUnpickler
