# syntax=docker/dockerfile:1
# Forge Classic – neo branch | NVIDIA GPU

ARG CUDA_VERSION=12.6.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

# Match the CUDA wheels PyTorch was compiled against.
# Available variants: cu118, cu121, cu124, cu126 …
# See: https://download.pytorch.org/whl/
ARG TORCH_INDEX=cu126

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Allow uv to fetch the Python 3.13 standalone runtime if not on the system.
    UV_PYTHON_DOWNLOADS=automatic \
    # Tells prepare_environment() which index to use for optional packages (e.g. xformers).
    TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_INDEX}

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

# ── Python 3.13 virtual environment ──────────────────────────────────────────
# --seed pre-installs pip/setuptools/wheel for packages that expect them.
RUN uv venv venv --python 3.13 --seed

ENV VIRTUAL_ENV=/home/forge/sd-webui/venv \
    PATH="/home/forge/sd-webui/venv/bin:$PATH" \
    # Override the Unraid template's PYTHON default to use the venv runtime.
    PYTHON=/home/forge/sd-webui/venv/bin/python3.13

# ── PyTorch pre-install ───────────────────────────────────────────────────────
# Baked in at build time so the ~2 GB download does not happen on first start.
# prepare_environment() checks is_installed("torch") and skips this step when
# torch is already present, so the runtime install path is never hit.
RUN uv pip install \
        torch \
        torchvision \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

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

# Allow extra time for prepare_environment() to run on first start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=15 \
    CMD curl -f http://localhost:7860/ || exit 1

ENTRYPOINT ["/home/forge/sd-webui/entrypoint.sh"]
