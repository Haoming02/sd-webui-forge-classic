<h1 align="center">Stable Diffusion WebUI Forge - Neo</h1>

<p align="center"><sup>
[ <a href="https://github.com/Haoming02/sd-webui-forge-classic/tree/classic#stable-diffusion-webui-forge---classic">Classic</a> | Neo ]
<br>
<a href="https://ko-fi.com/Haoming"><img src="https://img.shields.io/badge/Kofi-0D1117.svg?logo=ko-fi&logoColor=white"></a>
</sup></p>

<p align="center"><img src="html\ui.webp" width=512 alt="UI"></p>

<blockquote><i>
<b>Stable Diffusion WebUI Forge</b> is a platform on top of the original <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Stable Diffusion WebUI</a> by <ins>AUTOMATIC1111</ins>, to make development easier, optimize resource management, speed up inference, and study experimental features.<br>
The name "Forge" is inspired by "Minecraft Forge". This project aims to become the Forge of Stable Diffusion WebUI.<br>
<p align="right">- <b>lllyasviel</b><br>
<sup>(paraphrased)</sup></p>
</i></blockquote>

<br>

"**Neo**" mainly serves as an continuation for the "`latest`" version of Forge, which was built on [Gradio](https://github.com/gradio-app/gradio) `4.40.0` before lllyasviel became too busy... Additionally, this fork is focused on optimization and usability, with the main goal of being able to run the latest popular models, on an easy-to-use GUI without any bloatwares.

> [!Tip]
> [How to Install](#installation)

<br>

## Features [Feb.]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] Support [Anima](https://huggingface.co/circlestone-labs/Anima)
- [X] Support [Flux.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
    - `4B`, `9B`
    - `txt2img`, `img2img`, `inpaint`
- [X] Support [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)
    - `z-image`, `z-image-turbo`
- [X] Support [Wan 2.2](https://github.com/Wan-Video/Wan2.2)
    - `txt2img`, `img2img`, `txt2vid`, `img2vid`
    - use `Refiner` to achieve **High Noise** / **Low Noise** switching
        - enable `Refiner` in **Settings/Refiner**

> [!Important]
> To export a video, you need to have **[FFmpeg](https://ffmpeg.org/)** installed

- [X] Support [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) / [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
    - `txt2img` / `img2img`, `inpaint`

> [!Note]
> Since the layers between **Qwen-Image** and **Qwen-Image-Edit** are exactly the same, to be properly detected as an **Edit** model, the model needs to include "`qwen`" and "`edit`" in its path, either the file name or folder name.

- [X] Support [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
    - `img2img`, `inpaint`

> [!Note]
> Since the layers between **Flux-Dev**, **Flux-Krea**, and **Flux-Kontext** are exactly the same, to be properly detected as a **Kontext** model, the model needs to include "`kontext`" in its path, either the file name or folder name.

- [X] Support Multi-Image Inputs for **Qwen-Image-Edit** and **Flux-Kontext**
- [X] Support [Nunchaku](https://github.com/nunchaku-tech/nunchaku) (`SVDQ`) Models
    - `flux-dev`, `flux-krea`, `flux-kontext`, `qwen-image`, `qwen-image-edit`, `z-image-turbo`
    - support LoRA for `Flux` and `Qwen`
    - see [Commandline](#by-neo)
- [X] Support [Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
    - `Neta-Lumina`, `NetaYume-Lumina`
- [X] Support [Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD)

> [!Tip]
> Check out [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) for where to get each model and the accompanying modules

> [!Tip]
> Check out [Inference References](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References) for how to use each model and the recommended parameters

<br>

- [X] Rewrite Preset System
    - now remembers the checkpoint/module selection and parameters for each preset
- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - requires **manually** installing [uv](https://github.com/astral-sh/uv/releases)
    - drastically speed up installation
    - see [Commandline](#by-neo)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention), [FlashAttention](https://github.com/Dao-AILab/flash-attention), `fp16_accumulation`, `torch._scaled_mm`
    - see [Commandline](#by-neo)
- [X] Implement Triton Kernel for `matmul` in `torch.int8`
    - speed up `bf16` models
    - enable by selecting `int8` in the `Diffusion in Low Bits`
- [X] Implement [Radial Attention](https://github.com/mit-han-lab/radial-attention)
    - speed up `Wan 2.2`
    - requires **manually** installing [SpargeAttn](https://github.com/thu-ml/SpargeAttn)
- [X] Implement fast `state_dict` switching for Refiner
    - enable in **Settings/Refiner**
- [X] Implement Seed Variance Enhancer
    - improve seed-to-seed variance for distilled models
- [X] Implement RescaleCFG
    - reduce burnt colors; mainly for `v-pred` checkpoints
    - enable in **Settings/UI Alternatives**
- [X] Implement MaHiRo
    - alternative CFG calculation; improve prompt adherence
    - enable in **Settings/UI Alternatives**
- [X] Implement [Epsilon Scaling](https://github.com/comfyanonymous/ComfyUI/pull/10132)
    - enable in **Settings/Stable Diffusion**
- [X] Implement Torch.Compile
- [X] Support loading upscalers in `half` precision
    - speed up; reduce quality
    - enable in **Settings/Upscaling**
- [X] Support running tile composition on GPU
    - enable in **Settings/Upscaling**
- [X] Update `spandrel`
    - support new Upscaler architectures
- [X] Add support for `.avif`, `.heif`, and `.jxl` image formats
- [X] Automatically determine the optimal row count for `X/Y/Z Plot`

#### Removed Features

- [X] SD2
- [X] SD3
- [X] Forge Spaces
- [X] Hypernetworks
- [X] CLIP Interrogator
- [X] Deepbooru Interrogator
- [X] Textual Inversion Training
- [X] Most built-in Extensions
- [X] Some built-in Scripts
- [X] Some Samplers
- [X] Sampler in RadioGroup
- [X] Unix `.sh` launch scripts
    - You can still use this WebUI by simply copying a launch script from other working WebUI

#### Optimizations

- [X] **[Comfy]** Rewrite the Backend *(`memory_management.py`, `ModelPatcher`, `attention.py`, etc.)*
- [X] No longer `git` `clone` any repository on fresh install
- [X] Fix memory leak when switching checkpoints
- [X] Speed up launch time
- [X] Improve timer logs
- [X] Remove unused `cmd_args`
- [X] Remove unused `args_parser`
- [X] Remove unused `shared_options`
- [X] Remove legacy codes
- [X] Fix some typos
- [X] Fix automatic `Tiled VAE` fallback
- [X] Pad conditioning for SDXL
- [X] Remove redundant upscaler codes
    - put every upscaler inside the `ESRGAN` folder
- [X] Improve `ForgeCanvas`
    - brush adjustments
    - customization
    - deobfuscate
    - eraser
    - hotkeys
- [X] Optimize upscaler logics
- [X] Optimize certain operations in `Spandrel`
- [X] Speed up model loading
- [X] Improve memory management
- [X] Improve color correction
- [X] Update the implementation for `X/Y/Z Plot`
- [X] Update the implementation for `Soft Inpainting`
- [X] Update the implementation for `MultiDiffusion`
- [X] Update the implementation for `uni_pc` and `LCM` samplers
- [X] Update the implementation of LoRAs
- [X] Revamp settings
    - improve formatting
    - update descriptions
- [X] Check for Extension updates in parallel
- [X] Move `embeddings` folder into `models` folder
- [X] ControlNet Rewrite
    - change Units to `gr.Tab`
    - remove multi-inputs, as they are "[misleading](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)"
- [X] Disable Refiner by default
    - enable again in **Settings/Refiner**
- [X] No longer install `bitsandbytes` by default
    - see [Commandline](#by-neo)
- [X] Lint & Format
- [X] Update `Pillow`
    - faster image processing
- [X] Update `protobuf`
    - faster `insightface` loading
- [X] Update to latest PyTorch
    - `torch==2.10.0+cu130`

> [!Note]
> If your GPU does not support the latest PyTorch, manually [install](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations#older-pytorch) older version of PyTorch

- [X] No longer install `open-clip` twice
- [X] Update some packages to newer versions
- [X] Update recommended Python to `3.13.12`
- [X] many more... :tm:

<br>

## Commandline
> These flags can be added after the `set COMMANDLINE_ARGS=` line in the `webui-user.bat` *(separate each flag with space)*

> [!Tip]
> Use `python launch.py --help` to see all available flags

- `--xformers`: Install the `xformers` package to speed up generation
- `--port`: Specify a server port to use
    - defaults to `7860`
- `--api`: Enable [API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) access

#### by. Neo

- Add the following flags to slightly improve the model loading; in certain situations, they may cause `OutOfMemory` errors instead...
    - `--cuda-malloc`
    - `--cuda-stream`
    - `--pin-shared-memory`

- `--uv`: Replace the `python -m pip` calls with `uv pip` to massively speed up package installation
    - requires **uv** to be installed first *(see [Installation](#installation))*
- `--uv-symlink`: Same as above; but additionally pass `--link-mode symlink` to the commands
    - significantly reduces installation size (`~7 GB` to `~100 MB`)

> [!Important]
> Using `symlink` means it will directly access the packages from the cache folders; refrain from clearing the cache when setting this option

- `--model-ref`: Points to a central `models` folder that contains all your models
    - said folder should contain subfolders like `Stable-diffusion`, `Lora`, `VAE`, `ESRGAN`, etc.

> [!Important]
> This simply **replaces** the `models` folder, rather than adding on top of it

- `--forge-ref-a1111-home`: Point to an Automatic1111 installation to load its `models` folders
    - **i.e.** `Stable-diffusion`, `text_encoder`, etc.

- `--forge-ref-comfy-home`: Point to a ComfyUI installation to load its `models` folders
    - **i.e.** `diffusion_models`, `clip`, etc.
- `--forge-ref-comfy-yaml`: Point to the ComfyUI `extra_model_paths.yaml` to load its configurations
    - **i.e.** `base_path`, `checkpoints`, etc.

<br>

- `--sage`: Install the `sageattention` package to speed up generation
    - will also attempt to install `triton` automatically
- `--flash`: Install the `flash_attn` package to speed up generation
- `--nunchaku`: Install the `nunchaku` package to inference SVDQ models
- `--bnb`: Install the `bitsandbytes` package to do low-bits (`nf4`) inference
- `--onnxruntime-gpu`: Install the `onnxruntime` with the latest GPU support

<br>

- `--fast-fp8`: Use the `torch._scaled_mm` function when the model type is `float8_e4m3fn`
- `--fast-fp16`: Enable the `allow_fp16_accumulation` option
- `--autotune`: Enable the `torch.backends.cudnn.benchmark` option
    - this is slower in my experience...

<br>

## Installation

0. Install **[git](https://git-scm.com/downloads)**
1. Clone the Repo
    ```bash
    git clone https://github.com/Haoming02/sd-webui-forge-classic sd-webui-forge-neo --branch neo
    ```

2. Setup Python

<br>

<details>
<summary>Recommended Method</summary>

- Install **[uv](https://github.com/astral-sh/uv#installation)**
- Set up **venv**
    ```bash
    cd sd-webui-forge-neo
    uv venv venv --python 3.13 --seed
    ```
- Add the `--uv` flag to `webui-user.bat`

</details>

<br>

<details>
<summary>Deprecated Method</summary>

- Install **[Python 3.13.12](https://www.python.org/downloads/release/python-31312/)**
    - Remember to enable `Add Python to PATH`

</details>

<br>

3. **(Optional)** Configure [Commandline](#commandline)
4. Launch the WebUI via `webui-user.bat`
5. During the first launch, it will automatically install all the requirements
6. Once the installation is finished, the WebUI will start in a browser automatically

<br>

> [!Tip]
> Check out [Extra Installations](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations) for how to install `git`, `uv`, and `FFmpeg`

<br>

## Attention Functions

> [!Important]
> The `--xformers`, `--flash`, and `--sage` args are only responsible for installing the packages, **not** whether its respective attention is used *(this also means you can remove them once the packages are successfully installed)*

> [!Caution]
> Do **not** just blindly install all of them <br>
> Nowadays the native PyTorch `scaled_dot_product_attention` is usually as fast, and also more stable

**Forge Neo** tries to import the packages and automatically choose the first available attention function in the following order:

1. `SageAttention`
2. `FlashAttention`
3. `xformers`
4. `PyTorch`
5. `Basic`

> [!Note]
> To skip a specific attention, add the respective disable arg such as `--disable-sage`

<br>

## Issues & Requests

- **Issues** about removed features will simply be ignored
- **Issues** regarding installation will be ignored if it's obviously user-error
- **Issues** caused by [StabilityMatrix](https://github.com/LykosAI/StabilityMatrix) will also be ignored
    - only open an Issue if you can reproduce it on a clean install following the official [Installation](#installation) guide
- Linux, macOS, AMD, Intel will not be officially supported, as I cannot verify nor maintain them...

<br>

> [!Tip]
> Check out the [Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)~

<hr>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <b>city96</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>

<br>

<p align="right">
<sub><i>Buy me a <a href="https://ko-fi.com/Haoming">Coffee</a>~ ☕
</i></sub>
</p>
