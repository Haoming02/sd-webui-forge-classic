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

codeformer: face_restoration.FaceRestoration | None = None


class FaceRestorerCodeFormer(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "CodeFormer"

    def load_net(self) -> torch.nn.Module:
        os.makedirs(self.model_path, exist_ok=True)
        model_path = modelloader.load_file_from_url(
            url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            model_dir=self.model_path,
            file_name="codeformer-v0.1.0.pth",
        )
        return modelloader.load_spandrel_model(model_path, device=devices.device_codeformer).model

    def restore(self, np_image, w: float | None = None):
        if w is None:
            w = getattr(shared.opts, "code_former_weight", 0.5)

        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, w=w, adain=True)[0]

        return self.restore_with_helper(np_image, restore_face)


def setup_model(dirname: str) -> None:
    global codeformer
    try:
        codeformer = FaceRestorerCodeFormer(dirname)
        shared.face_restorers.append(codeformer)
    except Exception:
        errors.report("Error setting up CodeFormer", exc_info=True)
