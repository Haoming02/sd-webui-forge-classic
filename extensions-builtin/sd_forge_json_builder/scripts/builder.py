from modules import shared
from modules.script_callbacks import on_app_started, on_before_ui
from modules.ui_extra_networks import ExtraNetworksPage, register_page


class JsonBuilder(ExtraNetworksPage):

    def __init__(self):
        super().__init__("JSON Builder")

    def create_item(self, *args, **kwargs):
        return

    def list_items(self):
        return []


def json_builder_api(demo, app):

    @app.get("/internal/json_builder_prompt_styles")
    async def get_prompt_styles():
        return [{"name": name, "prompt": style.prompt} for name, style in shared.prompt_styles.styles.items()]


on_app_started(json_builder_api)
on_before_ui(lambda: register_page(JsonBuilder()))
