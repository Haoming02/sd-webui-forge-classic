# Adapted from OpenMOSS/MOSS-Image (Apache-2.0), MOSSImageScheduler.

import torch


def moss_sigmas(steps: int, shift: float = 5.0) -> torch.Tensor:
    """Shift the inference grid, then snap to the nearest training timestep."""
    if steps < 1 or shift <= 0:
        raise ValueError("MOSS steps and shift must be positive.")
    total = 1000
    values = torch.linspace(1, shift / (total + shift - 1), steps)
    sigmas = shift * values / (1 + (shift - 1) * values)
    training = torch.linspace(1, 1 / total, total).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat((sigmas, sigmas.new_zeros(1)))
