from modules import shared


class FaceRestoration:
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image


def restore_faces(np_image):
    for _restorer in shared.face_restorers:
        if _restorer.name() == shared.opts.face_restoration_model:
            return _restorer.restore(np_image)
    return np_image
