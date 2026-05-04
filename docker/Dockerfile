# syntax=docker/dockerfile:1
# Forge Neo | NVIDIA GPU

ARG CUDA_VERSION=12.6.3

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# See: https://download.pytorch.org/whl/
ARG TORCH_INDEX=cu126

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_NO_CACHE=1 \
    TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_INDEX}

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        google-perftools \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Non-Root User (UID 99 / GID 100 = nobody:users)
RUN groupadd -g 100 users 2>/dev/null || true \
    && useradd -u 99 -g 100 -d /home/forge -m -s /bin/bash forge

WORKDIR /home/forge/sd-webui

# Clone at build time
RUN git clone --branch neo --depth 1 --filter blob:none \
        https://github.com/Haoming02/sd-webui-forge-classic.git . \
    && chown -R 99:100 /home/forge

USER forge

# Python 3.13 Virtual ENVironment
RUN uv venv venv --python 3.13 --seed

ENV VIRTUAL_ENV="/home/forge/sd-webui/venv" \
    PATH="/home/forge/sd-webui/venv/bin:$PATH" \
    PYTHON="/home/forge/sd-webui/venv/bin/python3.13"

# PyTorch Pre-Install
RUN uv pip install torch torchvision \
    --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# Persistent Data Directories
RUN mkdir -p \
        models/Stable-diffusion \
        models/VAE \
        models/Lora \
        models/ControlNet \
        extensions \
        output

# Runtime
COPY --chown=99:100 entrypoint.sh /home/forge/sd-webui/entrypoint.sh
RUN chmod +x /home/forge/sd-webui/entrypoint.sh

EXPOSE 7860

# extra time for prepare_environment() to run on first start
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=15 \
    CMD curl -f http://localhost:7860/ || exit 1

ENTRYPOINT ["/home/forge/sd-webui/entrypoint.sh"]
