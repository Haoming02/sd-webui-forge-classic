import torch
import cv2

from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules import model_management
import ldm_patched.modules.clip_vision

from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules_forge.forge_util import resize_image_with_pad
from modules_forge.forge_util import numpy_to_pytorch
from modules.modelloader import load_file_from_url


class PreprocessorParameter:

    def __init__(self, **kwargs):
        self.gradio_update_kwargs = dict(
            minimum=kwargs.pop("minimum", 0.0),
            maximum=kwargs.pop("maximum", 1.0),
            step=kwargs.pop("step", 0.1),
            label=kwargs.pop("label", "n.a."),
            value=kwargs.pop("value", 0.5),
            interactive=kwargs.pop("visible", False),
            # NOTE: now uses `interactive` instead of `visible`
            **kwargs
        )


class Preprocessor:
    def __init__(self):
        self.name: str = "Base"
        self.tags: list[str] = []
        self.model_filename_filters: list[str] = []

        self.slider_resolution = PreprocessorParameter(
            label="Preprocessor Resolution",
            minimum=128,
            maximum=1024,
            value=512,
            step=64,
            visible=True,
        )
        self.slider_1 = PreprocessorParameter()
        self.slider_2 = PreprocessorParameter()

        self.model_patcher: ModelPatcher = None
        self.show_control_mode: bool = True
        self.do_not_need_model: bool = False
        self.sorting_priority: int = 0
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab: bool = True
        self.fill_mask_with_one_when_resize_and_fill: bool = False
        self.use_soft_projection_in_hr_fix: bool = False
        self.expand_mask_when_resize_and_fill: bool = False

    def setup_model_patcher(
        self,
        model,
        load_device=None,
        offload_device=None,
        dtype=torch.float16,
        **kwargs
    ):
        if load_device is None:
            load_device = model_management.get_torch_device()

        if offload_device is None:
            offload_device = torch.device("cpu")

        if not model_management.should_use_fp16(load_device):
            dtype = torch.float32

        model.eval()
        model = model.to(device=offload_device, dtype=dtype)

        self.model_patcher = ModelPatcher(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            **kwargs
        )

        self.model_patcher.dtype = dtype
        return self.model_patcher

    def move_all_model_patchers_to_gpu(self):
        model_management.load_models_gpu([self.model_patcher])

    def send_tensor_to_model_device(self, x):
        return x.to(
            device=self.model_patcher.current_device,
            dtype=self.model_patcher.dtype,
        )

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        pass

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        return cond, mask

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        pass

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        input_mask=None,
        **kwargs
    ):
        return input_image


class PreprocessorNone(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = "None"
        self.slider_resolution = PreprocessorParameter(visible=False)


class PreprocessorCanny(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = "canny"
        self.tags = ["Canny"]
        self.model_filename_filters = ["canny"]
        self.slider_1 = PreprocessorParameter(
            minimum=0,
            maximum=256,
            step=1,
            value=100,
            label="Low Threshold",
            visible=True,
        )
        self.slider_2 = PreprocessorParameter(
            minimum=0,
            maximum=256,
            step=1,
            value=200,
            label="High Threshold",
            visible=True,
        )
        self.sorting_priority = 100
        self.use_soft_projection_in_hr_fix = True

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)
        canny_image = cv2.cvtColor(
            cv2.Canny(input_image, int(slider_1), int(slider_2)),
            cv2.COLOR_GRAY2RGB,
        )
        return remove_pad(canny_image)


add_supported_preprocessor(PreprocessorNone())
add_supported_preprocessor(PreprocessorCanny())


class PreprocessorClipVision(Preprocessor):
    global_cache = {}

    def __init__(self, name, url, filename):
        super().__init__()
        self.name = name
        self.url = url
        self.filename = filename
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.show_control_mode = False
        self.sorting_priority = 1
        self.clipvision = None

    def load_clipvision(self):
        if self.clipvision is not None:
            return self.clipvision

        ckpt_path = load_file_from_url(
            url=self.url,
            model_dir=preprocessor_dir,
            file_name=self.filename,
        )

        if ckpt_path in PreprocessorClipVision.global_cache:
            self.clipvision = PreprocessorClipVision.global_cache[ckpt_path]
        else:
            self.clipvision = ldm_patched.modules.clip_vision.load(ckpt_path)
            PreprocessorClipVision.global_cache[ckpt_path] = self.clipvision

        return self.clipvision

    @torch.no_grad()
    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, **kwargs):
        clipvision = self.load_clipvision()
        return clipvision.encode_image(numpy_to_pytorch(input_image))
