import pickle
import torch


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
    """
    A class for temporarily setting the global handler for when you can't explicitly call load_with_extra
    (because it's not your code making the torch.load call). The intended use is like this:

    ```
    import torch
    from modules import safe

    def handler(module, name):
        if module == "torch" and name in ["float64", "float16"]:
            return getattr(torch, name)

        return None

    with safe.Extra(handler):
        x = torch.load("model.pt")
    ```
    """

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        global global_extra_handler

        assert global_extra_handler is None, "already inside an Extra() block"
        global_extra_handler = self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        global global_extra_handler

        global_extra_handler = None


unsafe_torch_load = torch.load
global_extra_handler = None

Unpickler = RestrictedUnpickler
