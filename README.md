<h1 align="center">Stable Diffusion WebUI Forge - Neo</h1>

<p align="center"><sup>
[ <a href="https://github.com/Haoming02/sd-webui-forge-classic/tree/classic#stable-diffusion-webui-forge---classic">Classic</a> | Neo ]
</sup></p>

<p align="center"><img src="html\ui.webp" width=512 alt="UI"></p>

<blockquote><i>
<b>Stable Diffusion WebUI Forge</b> is a platform on top of the original <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Stable Diffusion WebUI</a> by <ins>AUTOMATIC1111</ins>, to make development easier, optimize resource management, speed up inference, and study experimental features.<br>
The name "Forge" is inspired by "Minecraft Forge". This project aims to become the Forge of Stable Diffusion WebUI.<br>
<p align="right">- <b>lllyasviel</b><br>
<sup>(paraphrased)</sup></p>
</i></blockquote>

<br>

"**Neo**" mainly serves as an continuation for the "`latest`" version of Forge, which was built on [Gradio](https://github.com/gradio-app/gradio) `4.40.0` before lllyasviel became too busy... Additionally, this fork is focused on optimization and usability, with the main goal of being the lightest WebUI without any bloatwares.

> [!Tip]
> [How to Install](#installation)

<br>

## Features [Dec.]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] Support [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [X] Support [Wan 2.2](https://github.com/Wan-Video/Wan2.2)
    - `txt2img`, `img2img`, `txt2vid`, `img2vid`
    - use `Refiner` to achieve **High Noise** / **Low Noise** switching
        - enable `Refiner` in **Settings/Refiner**

> [!Important]
> To export a video, you need to have **[FFmpeg](https://ffmpeg.org/)** installed

- [X] Support [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- [X] Support [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
    - `img2img`, `inpaint`

> [!Note]
> Since the layers between **Qwen-Image** and **Qwen-Image-Edit** are exactly the same, to be properly detected as an **Edit** model, the model needs to include "`qwen`" and "`edit`" in its path, either the file name or folder name.

- [X] Support [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
    - `img2img`, `inpaint`

> [!Note]
> Since the layers between **Flux-Dev**, **Flux-Krea**, and **Flux-Kontext** are exactly the same, to be properly detected as a **Kontext** model, the model needs to include "`kontext`" in its path, either the file name or folder name.

- [X] Support Multi-Image Inputs for **Qwen-Image-Edit** and **Flux-Kontext**
- [X] Support [Nunchaku](https://github.com/nunchaku-tech/nunchaku) (`SVDQ`) Models
    - `flux-dev`, `flux-krea`, `flux-kontext`, `qwen-image`, `qwen-image-edit`, `t5`, `z-image-turbo`
    - support LoRAs
    - see [Commandline](#by-neo)
- [X] Support [Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
    - `Neta-Lumina`, `NetaYume-Lumina`
- [X] Support [Chroma](https://huggingface.co/lodestones/Chroma)
    - special thanks: [@croquelois](https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2925)

> [!Tip]
> Check out [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) for where to get each model and the accompanying modules

> [!Tip]
> Check out [Inference References](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References) for how to use each model and the recommended parameters

<br>

- [X] Rewrite Preset System
    - now actually remembers the checkpoint/module selection and parameters for each preset
- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - requires **manually** installing [uv](https://github.com/astral-sh/uv/releases)
    - drastically speed up installation
    - see [Commandline](#by-neo)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention), [FlashAttention](https://github.com/Dao-AILab/flash-attention), and fast `fp16_accumulation`
    - see [Commandline](#by-neo)
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
- [X] Support loading upscalers in `half` precision
    - speed up; reduce quality
    - enable in **Settings/Upscaling**
- [X] Support running tile composition on GPU
    - enable in **Settings/Upscaling**
- [X] Update `spandrel`
    - support new Upscaler architectures
- [X] Add support for `.avif`, `.heif`, and `.jxl` image formats

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

- [X] No longer `git` `clone` any repository on fresh install
- [X] Fix memory leak when switching checkpoints
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
- [X] Improve memory management
- [X] Improve color correction
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
    - `torch==2.9.1+cu130`
    - `xformers==0.0.33`

> [!Note]
> If your GPU does not support the latest PyTorch, manually [install](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations#older-pytorch) older version of PyTorch

- [X] No longer install `open-clip` twice
- [X] Update some packages to newer versions
- [X] Update recommended Python to `3.11.9`
- [X] many more... :tm:

<br>

## Commandline
> These flags can be added after the `set COMMANDLINE_ARGS=` line in the `webui-user.bat` *(separate each flag with space)*

#### A1111 built-in

- `--xformers`: Install the `xformers` package to speed up generation
- `--port`: Specify a server port to use
    - defaults to `7860`
- `--api`: Enable [API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) access

<br>

- Once you have successfully launched the WebUI, you can add the following flags to bypass some validation steps in order to improve the Startup time
    - `--skip-prepare-environment`
    - `--skip-install`
    - `--skip-python-version-check`
    - `--skip-torch-cuda-test`
    - `--skip-version-check`

> [!Important]
> Remove them if you are installing an Extension, as those also block Extension from installing requirements

#### by. Forge

- For RTX **30** and above, you can add the following flags to slightly increase the performance; but in rare occurrences, they may cause `OutOfMemory` errors or even crash the WebUI; and in certain configurations, they may even lower the speed instead
    - `--cuda-malloc`
    - `--cuda-stream`
    - `--pin-shared-memory`

- `--forge-ref-a1111-home`: Point to an Automatic1111 installation to load its `models` folders
    - **i.e.** `Stable-diffusion`, `text_encoder`

#### by. Neo

- `--uv`: Replace the `python -m pip` calls with `uv pip` to massively speed up package installation
    - requires **uv** to be installed first *(see [Installation](#installation))*
- `--uv-symlink`: Same as above; but additionally pass `--link-mode symlink` to the commands
    - significantly reduces installation size (`~7 GB` to `~100 MB`)

> [!Important]
> Using `symlink` means it will directly access the packages from the cache folders; refrain from clearing the cache when setting this option

- `--forge-ref-comfy-home`: Point to a ComfyUI installation to load its `models` folders
    - **i.e.** `diffusion_models`, `clip`

- `--model-ref`: Points to a central `models` folder that contains all your models
    - said folder should contain subfolders like `Stable-diffusion`, `Lora`, `VAE`, `ESRGAN`, etc.

> [!Important]
> This simply **replaces** the `models` folder, rather than adding on top of it

- `--sage`: Install the `sageattention` package to speed up generation
    - will also attempt to install `triton` automatically
- `--flash`: Install the `flash_attn` package to speed up generation
- `--nunchaku`: Install the `nunchaku` package to inference SVDQ models
- `--bnb`: Install the `bitsandbytes` package to do low-bits (`nf4`) inference
- `--onnxruntime-gpu`: Install the `onnxruntime` with the latest GPU support
- `--fast-fp16`: Enable the `allow_fp16_accumulation` option

<details>
<summary>with SageAttention 2</summary>

- `--sage2-function`: Select the function used by **SageAttention 2**
    - **options:**
        - `auto` (default)
        - `fp16_triton`
        - `fp16_cuda`
        - `fp8_cuda`

> If you are getting `NaN` errors, try play around with them

</details>

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
    uv venv venv --python 3.11 --seed
    ```
- Add the `--uv` flag to `webui-user.bat`

</details>

<br>

<details>
<summary>Deprecated Method</summary>

- Install **[Python 3.11.9](https://www.python.org/downloads/release/python-3119/)**
    - Remember to enable `Add Python to PATH`

</details>

<br>

3. **(Optional)** Configure [Commandline](#commandline)
4. Launch the WebUI via `webui-user.bat`
5. During the first launch, it will automatically install all the requirements
6. Once the installation is finished, the WebUI will start in a browser automatically

<br>

## Attention Functions

> [!Important]
> The `--xformers`, `--flash`, and `--sage` args are only responsible for installing the packages, **not** whether its respective attention is used *(this also means you can remove them once the packages are successfully installed)*

**Forge Neo** tries to import the packages and automatically choose the first available attention function in the following order:

1. `SageAttention`
2. `FlashAttention`
3. `xformers`
4. `PyTorch`
5. `Basic`

> [!Tip]
> To skip a specific attention, add the respective disable arg such as `--disable-sage`

> [!Note]
> The **VAE** only checks for `xformers`, so `--xformers` is still recommended even if you already have `--sage`

In my experience, the speed of each attention function for SDXL is ranked in the following order:

- `SageAttention` ≥ `FlashAttention` > `xformers` > `PyTorch` >> `Basic`

<br>

> [!Tip]
> Check out the [Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)~

<br>

## Issues & Requests

- **Issues** about removed features will simply be ignored
- **Issues** regarding installation will be ignored if it's obviously user-error
- Non-Windows platforms will not be officially supported, as I cannot verify nor maintain them

</details>

<hr>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <b>city96</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>
