# syntax=docker/dockerfile:1
# Forge Classic – neo branch | NVIDIA GPU (RTX 3060, CUDA 12.4)

ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

# Match the CUDA wheels PyTorch was compiled against.
# Available variants: cu118, cu121, cu124, cu126 …
# See: https://download.pytorch.org/whl/
ARG TORCH_INDEX=cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Allow uv to fetch the Python 3.12 standalone runtime if not on the system.
    UV_PYTHON_DOWNLOADS=automatic

# ── uv binary (from the official image — no installer script, no PATH guessing)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg               – required by the project (av / media processing)
# libtcmalloc-minimal4 – reduces memory fragmentation for large model workloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ffmpeg \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (UID 99 / GID 100 = Unraid's nobody:users) ─────────────────
# GID 100 may already exist in the base image; the || true swallows the error.
RUN groupadd -g 100 users 2>/dev/null || true \
    && useradd -u 99 -g 100 -d /home/forge -m -s /bin/bash forge

WORKDIR /home/forge/sd-webui

# ── Source code ───────────────────────────────────────────────────────────────
# Cloned at build time; rebuild the image to pick up upstream changes.
RUN git clone --branch neo --depth 1 \
        https://github.com/Haoming02/sd-webui-forge-classic.git . \
    && chown -R 99:100 /home/forge

USER forge

# ── Python 3.12 virtual environment ──────────────────────────────────────────
# --seed pre-installs pip/setuptools/wheel for packages that expect them.
# uv fetches the Python 3.12 standalone build automatically if needed.
RUN uv venv venv --python 3.12 --seed

ENV VIRTUAL_ENV=/home/forge/sd-webui/venv \
    PATH="/home/forge/sd-webui/venv/bin:$PATH" \
    # Override the Unraid template's PYTHON default to use the venv runtime.
    PYTHON=/home/forge/sd-webui/venv/bin/python3.12

# Install PyTorch with CUDA support before requirements.txt so the pinned
# CUDA-enabled wheel is not overwritten by a CPU-only fallback.
RUN uv pip install \
        torch \
        torchvision \
        xformers \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# Install remaining project requirements.
# The bare `torch` entry in requirements.txt will already be satisfied above.
RUN uv pip install -r requirements.txt imageio-ffmpeg

# Pre-install gradio and extension/preprocessor dependencies so that
# prepare_environment() on first start becomes a fast verification pass
# rather than a ~65s download+install run.
RUN uv pip install \
        python-dotenv \
        pillow-avif-plugin \
        lxml \
        filetype \
        "hnswlib>=0.0.0,<1.0.0" \
        send2trash \
        ZipUnicode \
        beautifulsoup4 \
        pysocks \
        addict \
        ftfy \
        fvcore \
        mediapipe \
        onnx \
        onnxruntime \
        timm \
        yapf \
        insightface

# ── Persistent data directories ───────────────────────────────────────────────
# These are bind-mount targets; actual data lives on the Unraid host.
# `outputs` is symlinked to `output` because Forge writes to the former
# while the Unraid template mounts the latter.
RUN mkdir -p \
        models/Stable-diffusion \
        models/VAE \
        models/Lora \
        models/ControlNet \
        extensions \
        output \
    && ln -s /home/forge/sd-webui/output /home/forge/sd-webui/outputs

# ── Runtime ───────────────────────────────────────────────────────────────────
COPY --chown=99:100 entrypoint.sh /home/forge/sd-webui/entrypoint.sh
RUN chmod +x /home/forge/sd-webui/entrypoint.sh

EXPOSE 7860

# Allow extra time for initial model loading before the health probe fires.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=15 \
    CMD curl -f http://localhost:7860/ || exit 1

ENTRYPOINT ["/home/forge/sd-webui/entrypoint.sh"]
