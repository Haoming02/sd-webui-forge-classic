import torch

from modules_forge.packages.huggingface_guess.latent import LatentFormat


class ProcessLatent:
    latent_format: LatentFormat = None

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent_format.process_in(latent)

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent_format.process_out(latent)
