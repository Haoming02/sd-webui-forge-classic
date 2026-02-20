from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import networks

from modules import extra_networks, shared


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    remove_symbols = str.maketrans("", "", ":,")

    def __init__(self):
        super().__init__("lora")

        self.errors = {}
        """mapping of network names to the number of errors the network had during operation"""

    def activate(self, p: "StableDiffusionProcessing", params_list: list["extra_networks.ExtraNetworkParams"]):
        self.errors.clear()

        additional: str = shared.opts.sd_lora
        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names: list[str] = []
        te_multipliers: list[float] = []
        unet_multipliers: list[float] = []

        for params in params_list:
            assert params.items

            names.append(params.positional[0])

            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))

            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))

            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)

        networks.load_networks(names, te_multipliers, unet_multipliers)

        if shared.opts.lora_add_hashes_to_infotext:
            if not getattr(p, "is_hr_pass", False) or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}

            for item in networks.loaded_networks:
                if item.network_on_disk.shorthash and item.mentioned_name:
                    p.lora_hashes[item.mentioned_name.translate(self.remove_symbols)] = item.network_on_disk.shorthash

            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(f"{k}: {v}" for k, v in p.lora_hashes.items())

    def deactivate(self, p: "StableDiffusionProcessing"):
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            self.errors.clear()
