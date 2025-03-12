from torch import ones_like


class EpsScaling:
    parameterization = "eps"

    def __call__(self, sigma):
        c_skip = ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScaling:
    parameterization = "v"

    def __call__(self, sigma):
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise
