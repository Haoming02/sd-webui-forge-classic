# reference: https://github.com/Comfy-Org/ComfyUI/blob/v0.9.0/comfy/latent_formats.py

import torch


class LatentFormat:
    scale_factor: float = 1.0
    latent_channels: int = 4
    latent_rgb_factors: list[list[float]] = None
    latent_rgb_factors_bias: list[list[float]] = None
    taesd_decoder_name: str = None

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent / self.scale_factor


class SD15(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.18215
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3512,  0.2297,  0.3227],
            [ 0.3250,  0.4974,  0.2350],
            [-0.2829,  0.1762,  0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.13025
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3651,  0.4232,  0.4341],
            [-0.2533, -0.0042,  0.1068],
            [ 0.1076,  0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188],
        ]
        self.latent_rgb_factors_bias = [0.1084, -0.0175, -0.0011]
        self.taesd_decoder_name = "taesdxl_decoder"


class Flux(LatentFormat):
    def __init__(self):
        self.latent_channels = 16
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0346,  0.0244,  0.0681],
            [ 0.0034,  0.0210,  0.0687],
            [ 0.0275, -0.0668, -0.0433],
            [-0.0174,  0.0160,  0.0617],
            [ 0.0859,  0.0721,  0.0329],
            [ 0.0004,  0.0383,  0.0115],
            [ 0.0405,  0.0861,  0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245,  0.0250,  0.1180],
            [ 0.1008,  0.0755, -0.0421],
            [-0.0515,  0.0201,  0.0011],
            [ 0.0428, -0.0012, -0.0036],
            [ 0.0817,  0.0765,  0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778],
        ]
        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class Wan21(LatentFormat):
    def __init__(self):
        self.latent_channels = 16
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.1299, -0.1692,  0.2932],
            [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747],
            [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328],
            [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925],
            [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364],
            [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128],
            [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699],
            [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950],
            [ 0.1984,  0.0913,  0.1861],
        ]
        self.latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]
        self.taesd_decoder_name = "taew2_1"

        self.latents_mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class Flux2(LatentFormat):
    def __init__(self):
        self.latent_channels = 128
        self.latent_rgb_factors = [
            [ 0.0058,  0.0113,  0.0073],
            [ 0.0495,  0.0443,  0.0836],
            [-0.0099,  0.0096,  0.0644],
            [ 0.2144,  0.3009,  0.3652],
            [ 0.0166, -0.0039, -0.0054],
            [ 0.0157,  0.0103, -0.0160],
            [-0.0398,  0.0902, -0.0235],
            [-0.0052,  0.0095,  0.0109],
            [-0.3527, -0.2712, -0.1666],
            [-0.0301, -0.0356, -0.0180],
            [-0.0107,  0.0078,  0.0013],
            [ 0.0746,  0.0090, -0.0941],
            [ 0.0156,  0.0169,  0.0070],
            [-0.0034, -0.0040, -0.0114],
            [ 0.0032,  0.0181,  0.0080],
            [-0.0939, -0.0008,  0.0186],
            [ 0.0018,  0.0043,  0.0104],
            [ 0.0284,  0.0056, -0.0127],
            [-0.0024, -0.0022, -0.0030],
            [ 0.1207, -0.0026,  0.0065],
            [ 0.0128,  0.0101,  0.0142],
            [ 0.0137, -0.0072, -0.0007],
            [ 0.0095,  0.0092, -0.0059],
            [ 0.0000, -0.0077, -0.0049],
            [-0.0465, -0.0204, -0.0312],
            [ 0.0095,  0.0012, -0.0066],
            [ 0.0290, -0.0034,  0.0025],
            [ 0.0220,  0.0169, -0.0048],
            [-0.0332, -0.0457, -0.0468],
            [-0.0085,  0.0389,  0.0609],
            [-0.0076,  0.0003, -0.0043],
            [-0.0111, -0.0460, -0.0614],
        ]
        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.latent_rgb_factors_reshape = lambda t: t.reshape(t.shape[ 0], 32, 2, 2, t.shape[-2], t.shape[-1]).permute(0, 1, 4, 2, 5, 3).reshape(t.shape[ 0], 32, t.shape[-2] * 2, t.shape[-1] * 2)
        self.taesd_decoder_name = "taef2_decoder"

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent
