# reference: https://github.com/Comfy-Org/ComfyUI/blob/v0.38.0/comfy/latent_formats.py

import torch


class LatentFormat:
    scale_factor: float = 1.0
    latent_channels: int = 4
    spacial_downscale_ratio = 8
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


class QwenImage21(LatentFormat):
    def __init__(self):
        self.latent_channels = 64
        self.spacial_downscale_ratio = 16
        self.latent_rgb_factors = [
            [-0.0158, -0.0115, -0.0174],
            [ 0.0030,  0.0120,  0.0027],
            [ 0.0637,  0.0470, -0.0127],
            [ 0.0360,  0.0661, -0.0030],
            [ 0.0159,  0.0181,  0.0082],
            [ 0.0132,  0.0326,  0.0169],
            [ 0.0191,  0.0261,  0.0136],
            [-0.0146, -0.0276, -0.0361],
            [ 0.0187, -0.0024, -0.0072],
            [-0.1059, -0.0090,  0.0350],
            [-0.0195, -0.0226, -0.0138],
            [-0.0295,  0.0024, -0.0215],
            [ 0.0191, -0.0393, -0.0001],
            [-0.0144, -0.0166, -0.0272],
            [ 0.0389,  0.0430,  0.0445],
            [-0.0153, -0.0336,  0.0031],
            [ 0.0339,  0.0122,  0.0220],
            [-0.0136, -0.0078, -0.0120],
            [-0.0340, -0.0282, -0.0245],
            [-0.0133, -0.0176, -0.0133],
            [ 0.0109, -0.0087,  0.0096],
            [-0.0010,  0.0044,  0.0016],
            [ 0.0301,  0.0053,  0.0361],
            [-0.0281, -0.0205, -0.0032],
            [-0.0725,  0.0002,  0.0160],
            [-0.0036,  0.0158,  0.0807],
            [ 0.0087,  0.0040, -0.0053],
            [-0.0260,  0.0183, -0.0077],
            [-0.0039, -0.0035, -0.0107],
            [-0.0026,  0.0172,  0.0237],
            [ 0.0088,  0.0078,  0.0078],
            [-0.0087, -0.0310, -0.0122],
            [-0.0027,  0.0018,  0.0094],
            [-0.0064,  0.0292, -0.0256],
            [ 0.0594,  0.1049,  0.1180],
            [ 0.0103, -0.0103, -0.0026],
            [-0.0091,  0.0025, -0.0015],
            [ 0.0178,  0.0243,  0.0292],
            [-0.0063, -0.0012,  0.0202],
            [ 0.0452,  0.0246,  0.0143],
            [ 0.0149,  0.0270,  0.0052],
            [ 0.1484,  0.0801,  0.0804],
            [-0.0120,  0.0040,  0.0010],
            [ 0.0181,  0.0051, -0.0021],
            [ 0.0132,  0.0050,  0.0019],
            [ 0.0291,  0.0020,  0.0092],
            [ 0.0066, -0.0410, -0.1314],
            [-0.1153, -0.0629, -0.0802],
            [ 0.0258,  0.0378,  0.0298],
            [ 0.0375,  0.1139,  0.0468],
            [-0.0142, -0.0126, -0.0276],
            [ 0.0339,  0.0153,  0.0138],
            [ 0.0346,  0.0211,  0.0267],
            [ 0.0369, -0.0431, -0.0993],
            [-0.0052, -0.0092,  0.0056],
            [-0.0279,  0.0410, -0.0357],
            [ 0.0036,  0.0017, -0.0083],
            [-0.0441, -0.0367, -0.0454],
            [-0.0001, -0.0092, -0.0001],
            [-0.0222, -0.0183, -0.0051],
            [ 0.0039,  0.0053, -0.0184],
            [-0.0094, -0.0075, -0.0143],
            [-0.0066, -0.0088, -0.0063],
            [ 0.0220,  0.0074,  0.0100],
        ]
        self.latent_rgb_factors_bias = [-0.1228, -0.1869, -0.3083]
        self.taesd_decoder_name = "taeqi2_1_decoder"

        self.latents_mean = torch.tensor([0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304]).view(1, self.latent_channels, 1, 1)
        self.latents_std = torch.tensor([3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161]).view(1, self.latent_channels, 1, 1)

    def process_in(self, latent):
        return (latent - self.latents_mean.to(latent.device, latent.dtype)) / self.latents_std.to(latent.device, latent.dtype)

    def process_out(self, latent):
        return latent * self.latents_std.to(latent.device, latent.dtype) + self.latents_mean.to(latent.device, latent.dtype)

class Flux2(LatentFormat):
    def __init__(self):
        self.latent_channels = 128
        self.spacial_downscale_ratio = 16
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
        self.latent_rgb_factors_reshape = lambda t: t.reshape(t.shape[0], 32, 2, 2, t.shape[-2], t.shape[-1]).permute(0, 1, 4, 2, 5, 3).reshape(t.shape[0], 32, t.shape[-2] * 2, t.shape[-1] * 2)
        self.taesd_decoder_name = "taef2_decoder"

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class SDXL_Flux2(Flux2):
    def __init__(self):
        super().__init__()
        self.latent_rgb_factors_reshape = None
        self.latent_channels = 32


class RGB(LatentFormat):
    def __init__(self):
        self.latent_channels = 3
        self.latent_rgb_factors = [
            #  R    G    B
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent
