# MOSS-Image

Forge Neo supports the dense MOSS-Image-Base and MOSS-Image-SFT architecture,
with its Qwen3.5-9B text encoder and FLUX.2 VAE. No Transformers/Diffusers upgrade
or external ComfyUI installation is required.

## Model files

Place consolidated component weights in these folders:

| Component | Folder |
| --- | --- |
| MOSS-Image DiT | `models/Stable-diffusion` |
| Qwen3.5-9B text encoder | `models/text_encoder` |
| FLUX.2 VAE | `models/VAE` |

For a local Hugging Face bundle containing `transformer/`, `text_encoder/` and
`vae/`, run from the Forge directory, using Forge's Python environment:

```powershell
python tools/convert_moss.py "X:\path\to\MOSS-Image-Base"
```

The converter combines shards and preserves the original weights and dtype.
It writes the three components into the folders above and refuses to overwrite
existing files. Use `--name moss-image-sft` for a second model. Distributed
training checkpoints must first be exported to the Hugging Face format.

## UI

Restart Forge after installing this integration. Select **moss** under
**UI Preset**, refresh the model list, select the MOSS DiT checkpoint, then select
the Qwen3.5 encoder and FLUX.2 VAE in **VAE / Text Encoder**.

Defaults: **Euler**, **MOSS** scheduler, **50** steps, **CFG 3**, **Shift 5**.
Start with 1024 × 1024. The MOSS scheduler uses the reference implementation's
shifted grid snapped to training timesteps. The negative CFG branch skips main
DiT block 22, as in the reference defaults.

Captions and layout JSON both go in the positive prompt. Text is encoded with
the Qwen thinking prefix, a 4096-token limit, and the final normalized hidden
state. Each prompt is encoded at its own length; padding is masked in the DiT.
Prompt emphasis weights are disabled. Negative prompts default to empty.

The model uses Forge's existing offload and weight-storage controls. BF16 and
FP32 computation are supported; the text encoder avoids FP16 storage by default.
FP8 DiT storage uses Forge's normal manual casting. The integration targets
text-to-image and ordinary img2img, without reference-image editing.

## Validation

```powershell
python -m unittest discover -s tests -p test_moss.py -v
```

The tests use small random models and local tokenizer assets; model downloads
are not required. Full-size image generation still requires the model weights.

Reference: [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), including
its native ComfyUI model and exported `pipeline_moss_image.py` implementation.
