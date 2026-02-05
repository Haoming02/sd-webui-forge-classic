import re
from functools import lru_cache

from PIL import Image

from backend.memory_management import free_memory, module_size, soft_empty_cache
from modules import devices, errors, modelloader
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model

PREFER_HALF = opts.prefer_fp16_upscalers
if PREFER_HALF:
    print("[Upscalers] Prefer Half-Precision:", PREFER_HALF)

MEM_RATIO = {"DRCT": 0.75, "DAT": 0.25}


class UpscalerESRGAN(Upscaler):
    def __init__(self, dirname: str):
        self.user_path = dirname
        self.model_path = dirname
        super().__init__(True)

        self.name = "ESRGAN"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
        self.model_name = "ESRGAN"
        self.scalers = []

        model_paths = self.find_models(ext_filter=[".pt", ".pth", ".safetensors"])
        if len(model_paths) == 0:
            scaler_data = UpscalerData(self.model_name, self.model_url, self, 4)
            self.scalers.append(scaler_data)

        for file in model_paths:
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)

            if match := re.search(r"(\d)[xX]|[xX](\d)", name):
                scale = int(match.group(1) or match.group(2))
            else:
                scale = 4

            scaler_data = UpscalerData(name, file, self, scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img: Image.Image, selected_model: str):
        soft_empty_cache()

        try:
            model = self.load_model(selected_model)
        except Exception:
            errors.report(f"Unable to load {selected_model}", exc_info=True)
            return img

        free_memory(
            #       (W * H)       * C *          dtype            *    scale    *                  ratio                       *  MB  *                      GPU
            (opts.ESRGAN_tile**2) * 3 * (2 if PREFER_HALF else 4) * model.scale * MEM_RATIO.get(model.architecture.name, 0.05) * 1024 * (1.1 if opts.composite_tiles_on_gpu else 1.0) + module_size(model.model),
            device=devices.device_esrgan,
        )

        return upscale_with_model(
            model=model,
            img=img,
            tile_size=opts.ESRGAN_tile,
            tile_overlap=opts.ESRGAN_tile_overlap,
        )

    @lru_cache(maxsize=4, typed=False)
    def load_model(self, path: str):
        if not path.startswith("http"):
            filename = path
        else:
            filename = modelloader.load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=path.rsplit("/", 1)[-1],
            )

        model = modelloader.load_spandrel_model(filename, device=devices.cpu, prefer_half=PREFER_HALF)
        model.to(devices.device_esrgan)
        return model
