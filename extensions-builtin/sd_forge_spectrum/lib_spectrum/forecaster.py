# https://github.com/ruwwww/comfyui-spectrum-sdxl

import math

import torch


class FastChebyshevForecaster:
    def __init__(self, m: int, lam: float, steps: int):
        self.M = m
        self.K = max(m + 2, 8)
        self.lam = lam
        self.H_buf = []
        self.T_buf = []
        self.time_buf = []
        self.shape = None
        self.dtype = None
        self.t_max = float(steps)

    def _taus(self, t: float) -> float:
        return (t / self.t_max) * 2.0 - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1)
        T = [torch.ones((taus.shape[0], 1), device=taus.device, dtype=torch.float32)]
        if self.M > 0:
            T.append(taus)
            for _ in range(2, self.M + 1):
                T.append(2 * taus * T[-1] - T[-2])
        return torch.cat(T[: self.M + 1], dim=1)

    def update(self, cnt: int, h: torch.Tensor):
        if self.shape and h.shape != self.shape:
            self.reset_buffers()

        self.shape = h.shape
        self.dtype = h.dtype

        self.H_buf.append(h.view(-1))
        self.T_buf.append(self._taus(cnt))
        self.time_buf.append(cnt)
        if len(self.H_buf) > self.K:
            self.H_buf.pop(0)
            self.T_buf.pop(0)
            self.time_buf.pop(0)

    def predict(self, cnt: int, w: float) -> torch.Tensor:
        device = self.H_buf[-1].device

        H = torch.stack(self.H_buf, dim=0).to(torch.float32)
        T = torch.tensor(self.T_buf, dtype=torch.float32, device=device)

        X = self._build_design(T)
        lamI = self.lam * torch.eye(self.M + 1, device=device)
        XtX = X.T @ X + lamI

        try:
            L = torch.linalg.cholesky(XtX)
        except RuntimeError:
            jitter = 1e-5 * XtX.diag().mean()
            L = torch.linalg.cholesky(XtX + jitter * torch.eye(self.M + 1, device=device))

        XtH = X.T @ H
        coef = torch.cholesky_solve(XtH, L)

        tau_star = torch.tensor([self._taus(cnt)], device=device)
        x_star = self._build_design(tau_star)

        pred_cheb = (x_star @ coef).squeeze(0)

        if len(self.H_buf) >= 2:
            h_i = self.H_buf[-1].to(torch.float32)
            h_im1 = self.H_buf[-2].to(torch.float32)

            t_i = self.time_buf[-1]
            t_im1 = self.time_buf[-2]

            dt_last = t_i - t_im1
            k = (cnt - t_i) / dt_last if dt_last > 1e-8 else 1.0

            h_taylor = h_i + k * (h_i - h_im1)
        else:
            h_taylor = self.H_buf[-1].to(torch.float32)

        res = (1 - w) * h_taylor + w * pred_cheb
        return res.to(self.dtype).view(self.shape)

    def reset_buffers(self):
        self.H_buf.clear()
        self.T_buf.clear()
        self.time_buf.clear()


class SpectrumNode:

    @staticmethod
    def patch(model, steps: int, w: float, m: int, lam: float, window_size: int, flex_window: float, warmup_steps: int, stop_caching_step: float):
        state = {"forecaster": None, "cnt": 0, "num_cached": 0, "curr_ws": float(window_size), "last_t": -1, "total_runs": 0, "estimated_total_steps": steps}

        def spectrum_unet_wrapper(model_function, kwargs):
            x, timestep, c = kwargs["input"], kwargs["timestep"], kwargs["c"]
            t_scalar = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)

            if t_scalar > state["last_t"]:
                if state["forecaster"]:
                    state["forecaster"].reset_buffers()
                state["cnt"] = 0
                state["num_cached"] = 0
                state["curr_ws"] = float(window_size)
                state["forecaster"] = None
                state["total_runs"] += 1

            state["last_t"] = t_scalar

            is_micro_final = False
            auto_stop = int(state["estimated_total_steps"] * stop_caching_step)
            if state["cnt"] >= auto_stop:
                is_micro_final = True

            do_actual = True
            if state["cnt"] >= warmup_steps and not is_micro_final:
                do_actual = (state["num_cached"] + 1) % math.floor(state["curr_ws"]) == 0

            if do_actual:
                out = model_function(x, timestep, **c)
                if state["forecaster"] is None:
                    state["forecaster"] = FastChebyshevForecaster(m=m, lam=lam, steps=steps)

                state["forecaster"].update(state["cnt"], out)
                if state["cnt"] >= warmup_steps:
                    state["curr_ws"] += flex_window
                state["num_cached"] = 0
            else:
                out = state["forecaster"].predict(state["cnt"], w=w).to(x.dtype)
                state["num_cached"] += 1

            state["cnt"] += 1
            return out

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(spectrum_unet_wrapper)

        return new_model
