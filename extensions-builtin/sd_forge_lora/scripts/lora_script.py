import re

import extra_networks_lora
import gradio as gr
import lora  # noqa
import network
import networks
import ui_extra_networks_lora

from modules import extra_networks, script_callbacks, shared, ui_extra_networks

shared.options_templates.update(
    shared.options_section(
        ("extra_networks", "Extra Networks"),
        {
            "sd_lora": shared.OptionInfo("None", "Add network to prompt", gr.Dropdown, lambda: {"choices": ["None", *networks.available_networks]}, refresh=networks.list_available_networks),
            "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", gr.Radio, {"choices": ["Alias from file", "Filename"]}),
            "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
            "lora_preset_filter": shared.OptionInfo(False, "Filter Lora based on selected Preset"),
        },
    )
)


if shared.cmd_opts.api:
    from fastapi import FastAPI

    def create_lora_json(obj: network.NetworkOnDisk):
        return {
            "name": obj.name,
            "alias": obj.alias,
            "path": obj.filename,
            "metadata": obj.metadata,
        }

    def api_networks(_: gr.Blocks, app: FastAPI):
        @app.get("/sdapi/v1/loras")
        async def get_loras():
            return [create_lora_json(obj) for obj in networks.available_networks.values()]

        @app.post("/sdapi/v1/refresh-loras")
        async def refresh_loras():
            return networks.list_available_networks()

    script_callbacks.on_app_started(api_networks)


re_lora = re.compile("<lora:([^:]+):")


def before_ui():
    ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora())
    networks.extra_network_lora = extra_networks_lora.ExtraNetworkLora()
    extra_networks.register_extra_network(networks.extra_network_lora)


def infotext_pasted(infotext, d: dict):
    hashes = d.get("Lora hashes")
    if not hashes:
        return

    hashes = [x.strip().split(":", 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes}

    def network_replacement(m: re.Match) -> str:
        alias = m.group(1)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)

        network_on_disk = networks.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)

        return f"<lora:{network_on_disk.get_alias()}:"

    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])


script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)
script_callbacks.on_infotext_pasted(infotext_pasted)
