from __future__ import annotations

import os

import torch

from modules import (
    devices,
    errors,
    face_restoration,
    face_restoration_utils,
    modelloader,
    shared,
)

gfpgan_face_restorer: face_restoration.FaceRestoration | None = None


class FaceRestorerGFPGAN(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "GFPGAN"

    def load_net(self) -> torch.nn.Module:
        os.makedirs(self.model_path, exist_ok=True)
        model_path = modelloader.load_file_from_url(
            url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            model_dir=self.model_path,
            file_name="GFPGANv1.4.pth",
        )
        return modelloader.load_spandrel_model(model_path, device=devices.device_gfpgan).model

    def restore(self, np_image):
        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, return_rgb=False)[0]

        return self.restore_with_helper(np_image, restore_face)


def gfpgan_fix_faces(np_image):
    if gfpgan_face_restorer:
        return gfpgan_face_restorer.restore(np_image)
    print("WARNING: GFPGAN face restorer was not set up")
    return np_image


def setup_model(dirname: str) -> None:
    global gfpgan_face_restorer

    try:
        face_restoration_utils.patch_facexlib(dirname)
        gfpgan_face_restorer = FaceRestorerGFPGAN(model_path=dirname)
        shared.face_restorers.append(gfpgan_face_restorer)
    except Exception:
        errors.report("Error setting up GFPGAN", exc_info=True)
