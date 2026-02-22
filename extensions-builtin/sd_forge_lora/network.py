import os
from typing import Final

from modules import cache, errors, hashes, sd_models, shared
from modules_forge.presets import PresetArch

SD_VERSION: Final[list[str]] = ["Unknown"] + PresetArch.choices()


class NetworkOnDisk:
    def __init__(self, name, filename):
        self.name: str = name
        self.filename: os.PathLike = filename
        self.metadata: dict[str, str] = {}
        self.is_safetensors: bool = filename.lower().endswith(".safetensors")

        def read_metadata():
            metadata = sd_models.read_metadata_from_safetensors(filename)
            return metadata

        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file("safetensors-metadata", "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f'reading metadata of "{filename}"')

        self.alias: str = self.metadata.get("ss_output_name", self.name)

        self.hash: bytes = None
        self.shorthash: bytes = None
        self.set_hash(self.metadata.get("sshs_model_hash") or hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or "")

    def set_hash(self, h):
        self.hash = h
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            import networks

            networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or "")

    def get_alias(self) -> str:
        import networks

        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in networks.forbidden_network_aliases:
            return self.name
        else:
            return self.alias


class Network:
    def __init__(self, name, network_on_disk):
        self.name: str = name
        self.network_on_disk: "NetworkOnDisk" = network_on_disk
        self.te_multiplier: float = 1.0
        self.unet_multiplier: float = 1.0

        self.mtime: float = None
        self.mentioned_name: str = None
