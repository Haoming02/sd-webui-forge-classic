# SD-WebUI Forge Neo (Docker)

Docker image for [sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) by Haoming02.

**Docker Hub:** `oromis995/sd-forge-neo`

> ⚠ Requires an NVIDIA GPU. The application cannot run without one.
> ⚠ Ensure NVIDIA drivers are up to date (560+ required for CUDA 12.6).

---

## Unraid deployment

| Container path | Purpose |
|---|---|
| `/home/forge/sd-webui/models` | Checkpoints, VAEs, LoRAs, ControlNet weights |
| `/home/forge/sd-webui/output` | Generated images |
| `/home/forge/sd-webui/extensions` | User-installed extensions |

The container runs as UID 99 / GID 100 (`nobody:users`) to match Unraid's default share permissions.

---

## Manual docker run

```bash
docker run -d \
  --gpus all \
  -p 7860:7860 \
  -v /path/to/models:/home/forge/sd-webui/models \
  -v /path/to/outputs:/home/forge/sd-webui/output \
  -v /path/to/extensions:/home/forge/sd-webui/extensions \
  oromis995/sd-forge-neo:latest
```

Pass extra flags via `-e COMMANDLINE_ARGS="..."` if needed (e.g. `--api`, `--xformers`, `--cuda-malloc`).

Access the WebUI at `http://<host-ip>:7860`.

---

## Building locally

```bash
git clone --branch neo https://github.com/Haoming02/sd-webui-forge-classic
cd sd-webui-forge-classic/docker
docker build -t forge-neo-local .
```

To target a different CUDA variant:
```bash
docker build --build-arg TORCH_INDEX=cu124 -t forge-neo-local .
```

---

## Image details

| | |
|---|---|
| Base | `nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04` |
| Python | 3.13 via uv |
| PyTorch | Latest stable from `download.pytorch.org/whl/cu126` |
| User | `forge` (UID 99 / GID 100) |
| Port | 7860 |

**First-start note:** On the first run `prepare_environment()` installs gradio, requirements, and any other dependencies. This may take a few minutes. Packages persist in the container's writable layer across restarts; recreating the container (e.g. on image update) will repeat this step once.
