from __future__ import annotations

import base64
import io
import random
import string
import time
from collections import OrderedDict
from typing import List

import gradio as gr
from pydantic import BaseModel, Field

import modules.shared as shared
from modules.shared import opts
from fastapi.responses import StreamingResponse, Response

current_task = None
pending_tasks = OrderedDict()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2


def start_task(id_task):
    global current_task

    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)


def create_task_id(task_type):
    N = 7
    res = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    return f"task({task_type}-{res})"


def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()


class PendingTasksResponse(BaseModel):
    size: int = Field(title="Pending task size")
    tasks: List[str] = Field(title="Pending task ids")


class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float | None = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float | None = Field(default=None, title="ETA in secs")
    live_preview: str | None = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int | None = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: str | None = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)
    # endpoint that returns the live preview image as binary data (useful to fetch as blob on frontend)
    return app.add_api_route("/internal/live-preview/{id_live}", live_preview_image, methods=["GET"])


def get_pending_tasks():
    pending_tasks_ids = list(pending_tasks)
    pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def progressapi(req: ProgressRequest):
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks

    if not active:
        textinfo = "Waiting..."
        if queued:
            sorted_queued = sorted(pending_tasks.keys(), key=lambda x: pending_tasks[x])
            queue_index = sorted_queued.index(req.id_task)
            textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    progress = 0

    job_count, job_no = shared.state.job_count, shared.state.job_no
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    elapsed_since_start = time.time() - shared.state.time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    live_preview = None
    id_live_preview = req.id_live_preview

    if opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                # Instead of embedding the image as a data URI (which consumes more memory on the
                # frontend), expose a small URL that returns the raw image bytes. The frontend will
                # fetch that URL and create a blob URL for display.
                id_live_preview = shared.state.id_live_preview
                # include format in the returned URL implicitly handled by the image endpoint
                live_preview = f"./internal/live-preview/{id_live_preview}"

    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)


def live_preview_image(id_live: int):
    """Return the current live preview image as raw bytes.

    The frontend requests this via fetch and converts the response to a Blob, then to a blob URL
    with URL.createObjectURL to avoid keeping large base64 strings in memory.
    """
    # Ensure current_image is up-to-date
    shared.state.set_current_image()
    image = shared.state.current_image
    if image is None:
        return Response(status_code=404)

    buffered = io.BytesIO()

    # Same saving heuristics as before
    if opts.live_previews_image_format == "png":
        if max(*image.size) <= 256:
            save_kwargs = {"optimize": True}
        else:
            save_kwargs = {"optimize": False, "compress_level": 1}
    else:
        image = image.convert("RGB")
        save_kwargs = {}

    image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
    buffered.seek(0)

    media_type = f"image/{opts.live_previews_image_format}"
    # webp uses image/webp
    if opts.live_previews_image_format == "jpg":
        media_type = "image/jpeg"
    if opts.live_previews_image_format == "jpeg":
        media_type = "image/jpeg"

    return StreamingResponse(buffered, media_type=media_type)


def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
