# SD-WebUI Forge Neo (Docker)

Docker image for [sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) by Haoming02.

**Docker Hub:** `oromis95/sd-forge-neo`
**GitHub:** https://github.com/oromis995/Forge-Neo-Docker

> ⚠ Requires an NVIDIA GPU. The application cannot run without one.
> ⚠ Ensure NVIDIA drivers are up to date (550+ recommended for CUDA 12.4).

---

## Unraid deployment

Import `sd-forge-neo.xml` from this repo into Unraid's Docker template manager.

| Container path | Purpose |
|---|---|
| `/home/forge/sd-webui/models` | Checkpoints, VAEs, LoRAs, ControlNet weights |
| `/home/forge/sd-webui/output` | Generated images |
| `/home/forge/sd-webui/extensions` | User-installed extensions |

The container runs as UID 99 / GID 100 (`nobody:users`) to match Unraid's default share permissions.

Default `COMMANDLINE_ARGS`:
```
--api --listen --cuda-malloc --xformers --skip-torch-cuda-test --enable-insecure-extension-access
```

---

## Manual docker run

```bash
docker run -d \
  --gpus all \
  -p 7860:7860 \
  -v /path/to/models:/home/forge/sd-webui/models \
  -v /path/to/outputs:/home/forge/sd-webui/output \
  -v /path/to/extensions:/home/forge/sd-webui/extensions \
  -e COMMANDLINE_ARGS="--api --listen --cuda-malloc --xformers --skip-torch-cuda-test --enable-insecure-extension-access" \
  oromis995/sd-forge-neo:latest
```

Access the WebUI at `http://<host-ip>:7860`.

---

## Building locally

```bash
git clone https://github.com/oromis995/Forge-Neo-Docker
cd Forge-Neo-Docker
docker build -t forge-neo-local .
```

To target a different CUDA variant:
```bash
docker build --build-arg TORCH_INDEX=cu126 -t forge-neo-local .
```

---

## Image details

| | |
|---|---|
| Base | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| Python | 3.12 via uv — 3.13 has no xformers wheels |
| PyTorch | Latest stable from `download.pytorch.org/whl/cu124` |
| xformers | Co-installed from the PyTorch index for ABI compatibility |
| User | `forge` (UID 99 / GID 100) |
| Port | 7860 |

**First-start note:** On the first run `prepare_environment()` installs gradio (version-pinned internally by Forge) and a small number of extension packages. These persist in the container's writable layer across restarts. Recreating the container (e.g. on image update) will repeat this step once.
