from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad
from modules.modelloader import load_file_from_url

import types
import torch
import numpy as np

from einops import rearrange
from annotator.normalbae.models.NNET import NNET
from annotator.normalbae import load_checkpoint
from torchvision import transforms


class PreprocessorNormalBae(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'normal_dsine'
        self.tags = ['NormalMap']
        self.model_filename_filters = ['normal']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=512, step=16, visible=True)
        self.slider_1 = PreprocessorParameter(
            minimum=0,
            maximum=360,
            step=0.1,
            value=60,
            label="Fov",
            visible=True,
        )
        self.slider_2 = PreprocessorParameter(
            minimum=1,
            maximum=20,
            step=1,
            value=5,
            label="Iterations",
            visible=True,
        )
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.model = None
        self.sorting_priority = 100  # higher goes to top in the list

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
    
        if self.model is None:
            from annotator.normaldsine import NormalDsineDetector

            self.model = NormalDsineDetector()

        result = self.model(
            input_image,
            new_fov=float(slider_1),
            iterations=int(slider_2),
            resulotion=resolution,
        )
        return result


add_supported_preprocessor(PreprocessorNormalBae())
