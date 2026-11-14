from modules.script_callbacks import on_before_ui
from modules.ui_extra_networks import ExtraNetworksPage, register_page


class JsonBuilder(ExtraNetworksPage):

    def __init__(self):
        super().__init__("JSON Builder")

    def create_item(self, *args, **kwargs):
        return

    def list_items(self):
        return []


on_before_ui(lambda: register_page(JsonBuilder()))
