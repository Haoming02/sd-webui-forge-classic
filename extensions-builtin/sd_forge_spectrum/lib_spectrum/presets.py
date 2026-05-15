import os.path
from json import dump, load
from typing import Final

import gradio as gr

from lib_spectrum import logger

PRESET_FILE: Final[os.PathLike] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets.json")
PARAMS: Final[list[type]] = [float, int, float, int, float, int, float]


class PresetManager:
    presets: dict[str, list[float]] = None

    @classmethod
    def load_presets(cls):
        if cls.presets is not None:
            return

        if not os.path.isfile(PRESET_FILE):
            with open(PRESET_FILE, "w+", encoding="utf-8") as json_file:
                dump({}, json_file)

            logger.debug("Creating new empty Presets...")
            cls.presets = {}
            return

        try:
            with open(PRESET_FILE, "r", encoding="utf-8") as json_file:
                cls.presets = load(json_file)
        except Exception:
            logger.error("Failed to load Presets...")
            cls.presets = {}
        else:
            logger.debug("Loaded Presets...")

    @classmethod
    def list_preset(cls) -> list[str]:
        return list(cls.presets.keys())

    @classmethod
    def get_preset(cls, preset_name: str) -> list[float]:
        if (preset := cls.presets.get(preset_name, None)) is None:
            logger.error(f'Preset "{preset_name}" was not found...')
            return [gr.skip()] * len(PARAMS)

        return [gr.update(value=obj(val)) for obj, val in zip(PARAMS, preset)]

    @classmethod
    def save_preset(cls, preset_name: str, *args: float) -> list[str]:
        if preset_name is None or not preset_name.strip():
            logger.error("Invalid Preset Name...")
            return gr.skip()

        cls.presets.update({preset_name: [*args]})

        with open(PRESET_FILE, "w", encoding="utf-8") as json_file:
            dump(cls.presets, json_file)

        logger.info(f'Preset "{preset_name}" Saved!')
        return gr.update(choices=cls.list_preset())

    @classmethod
    def delete_preset(cls, preset_name: str) -> list[str]:
        if preset_name not in cls.presets:
            logger.error(f'Preset "{preset_name}" was not found...')
            return gr.skip()

        del cls.presets[preset_name]

        with open(PRESET_FILE, "w", encoding="utf-8") as json_file:
            dump(cls.presets, json_file)

        logger.info(f'Preset "{preset_name}" Deleted!')
        return gr.update(value=None, choices=cls.list_preset())
