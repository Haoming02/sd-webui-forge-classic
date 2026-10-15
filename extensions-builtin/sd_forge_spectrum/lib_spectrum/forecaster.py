# https://github.com/ruwwww/ComfyUI-Spectrum-sdxl/blob/main/src/forecaster.py
# Reference: https://github.com/hanjq17/Spectrum/blob/main/src/utils/basis_utils.py

from typing import Optional

import torch
import torch.nn as nn

from backend.memory_management import get_torch_device, should_use_bf16

DTYPE = torch.bfloat16 if should_use_bf16(get_torch_device()) else torch.float16


def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return (x.reshape(1, -1) if x.ndim == 1 else x.reshape(1, -1)), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class ChebyshevForecaster(nn.Module):
    def __init__(self, M: int, K: int, lam: float, t_max: float):
        super().__init__()
        assert K >= M + 2

        self.M = M
        self.K = K
        self.lam = lam
        self.t_max_val = t_max
        self.register_buffer("t_buf", torch.empty(0))
        self._H_buf: Optional[torch.Tensor] = None
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None
        self._XtX_fac: Optional[torch.Tensor] = None
        self._tau_cache: Optional[torch.Tensor] = None
        self._X_cache: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        assert self.t_buf.numel() >= 1
        t_min = torch.zeros(1, device=t.device, dtype=t.dtype)
        t_max = torch.ones(1, device=t.device, dtype=t.dtype) * self.t_max_val
        if torch.isclose(t_max, t_min):
            return torch.zeros_like(t)

        mid = 0.5 * (t_min + t_max)
        rng = t_max - t_min
        return (t - mid) * 2.0 / rng

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        K = taus.shape[0]
        T0 = torch.ones((K, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return T0

        T1 = taus
        cols = [T0, T1]
        for m in range(2, self.M + 1):
            Tm = 2 * taus * cols[-1] - cols[-2]
            cols.append(Tm)

        return torch.cat(cols[: self.M + 1], dim=1)

    @property
    def P(self) -> int:
        return self.M + 1

    def update(self, t: float | torch.Tensor, h: torch.Tensor) -> None:
        device = h.device

        t = torch.as_tensor(t, dtype=DTYPE, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(device)

        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape

        if self.t_buf.numel() == 0:
            self.t_buf = t[None]
            self._H_buf = h_flat
        else:
            delta = h_flat - self._H_buf[-1]
            self._last_delta_norm = delta.norm(p=2)
            self.t_buf = torch.cat([self.t_buf, t[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K :]
                self._H_buf = self._H_buf[-self.K :]

        self._coef = None
        self._XtX_fac = None
        self._tau_cache = None
        self._X_cache = None

    def last_delta(self) -> torch.Tensor:
        if self._last_delta_norm is None:
            return torch.tensor(1e-6, device=self.t_buf.device if self.t_buf.numel() else "cpu")
        return self._last_delta_norm

    def ready(self) -> bool:
        return self.t_buf.numel() >= min(self.K, self.M + 2)

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return

        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)
        H = self._H_buf.to(torch.float32)
        _, P = X.shape
        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.transpose(0, 1)
        XtX = Xt @ X + lamI

        try:
            L = torch.linalg.cholesky(XtX)
        except:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(P, device=X.device))

        XtH = Xt @ H
        C = torch.cholesky_solve(XtH, L).to(DTYPE)
        self._coef = C
        self._XtX_fac = L
        self._tau_cache = taus
        self._X_cache = X.to(DTYPE)

    @torch.inference_mode()
    def predict(self, t_star: float | torch.Tensor) -> torch.Tensor:
        assert self._shape is not None
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])
        h_flat = x_star @ self._coef
        return _unflatten(h_flat, self._shape)


class Spectrum(nn.Module):
    def __init__(self, cheb: "ChebyshevForecaster", w: float, taylor_order: int = 1, enable_blend: bool = True, prefer: str = "auto", alpha: float = 6.0, ema_beta: float = 0.9):
        super().__init__()

        self.cheb: "ChebyshevForecaster" = cheb
        self.w: float = w
        self.taylor_order: int = taylor_order
        self.enable_blend: bool = enable_blend
        self.prefer: str = prefer
        self.alpha: float = alpha
        self.ema_beta: float = ema_beta

    @torch.inference_mode()
    def _local_taylor_discrete(self, t_star: torch.Tensor) -> torch.Tensor:
        H = self.cheb._H_buf
        t = self.cheb.t_buf
        h_i = H[-1]
        t_i = t[-1]
        if t.numel() < 2:
            return _unflatten(h_i.clone().reshape(1, -1), self.cheb._shape)

        h_im1 = H[-2]
        t_im1 = t[-2]
        dh1 = h_i - h_im1
        dt_last = (t_i - t_im1).clamp_min(1e-8)
        k = ((t_star - t_i) / dt_last).to(h_i.dtype)
        out = h_i + k * dh1

        if self.taylor_order >= 2 and t.numel() >= 3:
            h_im2 = H[-3]
            d2 = h_i - 2 * h_im1 + h_im2
            out = out + 0.5 * k * (k - 1.0) * d2
        if self.taylor_order >= 3 and t.numel() >= 4:
            h_im3 = H[-4]
            d3 = h_i - 3 * h_im1 + 3 * h_im2 - h_im3
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3

        return _unflatten(out.reshape(1, -1), self.cheb._shape)

    @torch.inference_mode()
    def predict(self, t_star: float | torch.Tensor, return_weight: bool = False):
        device = self.cheb.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=DTYPE, device=device)
        h_taylor = self._local_taylor_discrete(t_star)

        if not self.cheb.ready():
            return (h_taylor, 0.0) if return_weight else h_taylor

        h_cheb = self.cheb.predict(t_star)
        assert (w := self.w) is not None
        h_mix = (1 - w) * h_taylor + w * h_cheb
        return (h_mix, float(w)) if return_weight else h_mix

    def update(self, t: float | torch.Tensor, h: torch.Tensor):
        return self.cheb.update(t, h)

    def last_delta(self) -> torch.Tensor:
        return self.cheb.last_delta()

    def ready(self) -> bool:
        return self.cheb.ready()
