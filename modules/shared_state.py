import datetime
import logging
import threading
import time
from contextlib import nullcontext
from typing import Optional

import torch
from PIL import Image

from backend import stream
from modules import devices, errors, shared

log = logging.getLogger(__name__)


class State:
    skipped = False
    interrupted = False
    stopping_generation = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = "0"
    preview_step: int = 0
    sampling_step: int = 0
    sampling_steps: int = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    time_start = None
    server_start = None
    _server_command_signal = threading.Event()
    _server_command: Optional[str] = None

    def __init__(self):
        self.server_start = time.time()
        self.vae_stream = None
        if stream.should_use_stream():
            self.vae_stream = torch.cuda.Stream(device=devices.device, priority=1)

    @property
    def need_restart(self) -> bool:
        # Compatibility getter for need_restart.
        return self.server_command == "restart"

    @need_restart.setter
    def need_restart(self, value: bool) -> None:
        # Compatibility setter for need_restart.
        if value:
            self.server_command = "restart"

    @property
    def server_command(self):
        return self._server_command

    @server_command.setter
    def server_command(self, value: Optional[str]) -> None:
        """
        Set the server command to `value` and signal that it's been set.
        """
        self._server_command = value
        self._server_command_signal.set()

    def wait_for_server_command(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for server command to get set; return and clear the value and signal.
        """
        if self._server_command_signal.wait(timeout):
            self._server_command_signal.clear()
            req = self._server_command
            self._server_command = None
            return req
        return None

    def request_restart(self) -> None:
        self.interrupt()
        self.server_command = "restart"
        log.info("Received restart request")

    def skip(self):
        self.skipped = True
        log.info("Received skip request")

    def interrupt(self):
        self.interrupted = True
        log.info("Received interrupt request")

    def stop_generating(self):
        self.stopping_generation = True
        log.info("Received stop generating request")

    def nextjob(self):
        if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.preview_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "stopping_generation": self.stopping_generation,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self, job: str = "(unknown)"):
        self.sampling_step = 0
        self.preview_step = 0
        self.time_start = time.time()
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent: torch.Tensor = None
        self.current_image: Image.Image = None
        self.current_image_sampling_step = 0
        self.id_live_preview = 0
        self.skipped = False
        self.interrupted = False
        self.stopping_generation = False
        self.textinfo = None
        self.job = job
        devices.torch_gc()
        log.info("Starting job %s", job)

    def end(self):
        duration = time.time() - self.time_start
        log.info("Ending job %s (%.2f seconds)", self.job, duration)
        self.job = ""
        self.job_count = 0

        devices.torch_gc()

    @torch.inference_mode()
    def set_current_image(self):
        if not shared.opts.live_previews_enable or shared.opts.show_progress_every_n_steps == -1:
            return
        if self.preview_step >= self.sampling_steps:
            return
        if self.preview_step - self.current_image_sampling_step >= shared.opts.show_progress_every_n_steps:
            self.do_set_current_image()

    @torch.inference_mode()
    def do_set_current_image(self):
        if self.current_latent is None:
            return

        import modules.sd_samplers

        try:
            _video: bool = self.current_latent.ndim == 5 and self.current_latent.size(2) > 1

            vae_context = nullcontext()
            if self.vae_stream is not None:
                self.vae_stream.wait_stream(stream.current_stream)
                vae_context = stream.stream_context()(self.vae_stream)

            with vae_context:
                if _video:
                    self.assign_current_image(modules.sd_samplers.sample_to_video(self.current_latent))
                elif shared.opts.show_progress_grid:
                    self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
                else:
                    self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

            self.current_image_sampling_step = self.sampling_step

        except Exception:
            errors.record_exception()

    @torch.inference_mode()
    def assign_current_image(self, image: Image.Image):
        if shared.opts.live_previews_image_format == "jpeg" and not getattr(image, "is_animated", False) and image.mode != "RGB":
            image = image.convert("RGB")
        self.current_image = image
        self.id_live_preview += 1
