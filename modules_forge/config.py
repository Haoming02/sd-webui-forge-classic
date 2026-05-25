from typing import Final

always_disabled_extensions: Final[list[str]] = [
    "sd-webui-controlnet",
    "multidiffusion-upscaler-for-automatic1111",
]

prefer_official_extensions: Final[dict[str, str]] = {
    "ADetailer": "https://github.com/Haoming02/ADetailer-Neo",
}
