import os
from abc import abstractmethod

from PIL import Image

from modules import devices, modelloader, shared
from modules.images import LANCZOS, NEAREST
from modules.shared import cmd_opts, models_path, opts

# hardcode
UPSCALE_ITERATIONS = 4


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers: list["UpscalerData"] = []
    tile = True

    def __init__(self, create_dirs=False):
        self.scale: int = 1
        self.tile_size: int = opts.ESRGAN_tile
        self.tile_pad: int = opts.ESRGAN_tile_overlap
        self.device = devices.device_esrgan
        self.model_download_path: str = None
        self.img = None
        self.output = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(models_path, self.name)
        if self.model_path:
            if create_dirs and not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

    @abstractmethod
    def do_upscale(self, img: Image.Image, selected_model: str):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path: str):
        raise NotImplementedError

    def upscale(self, img: Image.Image, scale: int, selected_model: str = None):
        self.scale = scale
        dest_w: int = (img.width * scale) // 8 * 8
        dest_h: int = (img.height * scale) // 8 * 8

        for _ in range(UPSCALE_ITERATIONS):
            if shared.state.interrupted:
                break
            img = self.do_upscale(img, selected_model)
            if ((img.width >= dest_w) and (img.height >= dest_h)) or (int(scale) == 1):
                break

        if (img.width != dest_w) or (img.height != dest_h):
            img = img.resize((int(dest_w), int(dest_h)), LANCZOS)

        return img

    def find_models(self, ext_filter=None) -> list[str]:
        return modelloader.load_models(
            model_path=self.model_path,
            model_url=self.model_url,
            command_path=self.user_path,
            ext_filter=ext_filter,
        )


class UpscalerData:
    name: str
    data_path: str
    scaler: Upscaler
    scale: int
    model: None

    def __init__(
        self,
        name: str,
        path: str,
        upscaler: Upscaler = None,
        scale: int = 4,
        model=None,
    ):
        self.name = name
        self.data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model

    def __repr__(self):
        return f"<UpscalerData name={self.name} path={self.data_path} scale={self.scale}>"


class UpscalerNone(Upscaler):
    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "None"
        self.scalers = [UpscalerData("None", None, self)]

    def load_model(self, _):
        return

    def do_upscale(self, img: Image.Image, *args, **kwargs):
        return img


class UpscalerLanczos(Upscaler):
    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]

    def load_model(self, _):
        return

    def do_upscale(self, img: Image.Image, *args, **kwargs):
        return img.resize(
            size=(int(img.width * self.scale), int(img.height * self.scale)),
            resample=LANCZOS,
        )


class UpscalerNearest(Upscaler):
    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]

    def load_model(self, _):
        return

    def do_upscale(self, img: Image.Image, *args, **kwargs):
        return img.resize(
            size=(int(img.width * self.scale), int(img.height * self.scale)),
            resample=NEAREST,
        )
