from modules import script_callbacks, shared

unet_options: list["SdUnetOption"] = []
current_unet_option = None
current_unet = None


def list_unets():
    unet_options.clear()
    unet_options.extend(script_callbacks.list_unets_callback())


def get_unet_option(option=None) -> "SdUnetOption":
    option = option or shared.opts.sd_unet

    if option == "None":
        return None

    if option == "Automatic":
        name = shared.sd_model.sd_checkpoint_info.model_name

        options = [x for x in unet_options if x.model_name == name]

        option = options[0].label if options else "None"

    return next(iter([x for x in unet_options if x.label == option]), None)


def apply_unet(option=None):
    global current_unet_option
    global current_unet

    new_option = get_unet_option(option)
    if new_option == current_unet_option:
        return

    if current_unet is not None:
        print(f"Dectivating unet: {current_unet.option.label}")
        current_unet.deactivate()

    current_unet_option = new_option
    if current_unet_option is None:
        current_unet = None
        return

    current_unet = current_unet_option.create_unet()
    current_unet.option = current_unet_option
    print(f"Activating unet: {current_unet.option.label}")
    current_unet.activate()


class SdUnetOption:
    model_name = None
    """name of related checkpoint - this option will be selected automatically for unet if the name of checkpoint matches this"""

    label = None
    """name of the unet in UI"""

    def create_unet(self):
        """returns SdUnet object to be used as a Unet instead of built-in unet when making pictures"""
        raise NotImplementedError()
