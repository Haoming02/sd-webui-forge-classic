import pickle

load = pickle.load


class Empty:
    pass


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module: str, name: str):
        if module.startswith("pytorch_lightning"):
            return Empty

        if module.startswith(("collections", "torch", "numpy", "__builtin__")):
            return super().find_class(module, name)

        raise NotImplementedError(f'"{module}.{name}" is forbidden')


Unpickler = RestrictedUnpickler
