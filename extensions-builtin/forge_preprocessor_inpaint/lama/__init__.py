from backend.utils import load_torch_file
from lama.modules import make_generator


def load_checkpoint(train_config, path: str, map_location="cpu", strict=False):
    model = make_generator(train_config, **train_config.generator)
    state = load_torch_file(path, device=map_location)
    model.load_state_dict(state, strict=strict)
    return model
