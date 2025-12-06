<h1 align="center">Stable Diffusion WebUI Forge - Classic</h1>

<p align="center"><sup>
[ Classic | <a href="https://github.com/Haoming02/sd-webui-forge-classic/tree/neo#stable-diffusion-webui-forge---neo">Neo</a> ]
</sup></p>

<p align="center"><img src="html\ui.webp" width=512 alt="UI"></p>

<blockquote><i>
<b>Stable Diffusion WebUI Forge</b> is a platform on top of the original <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Stable Diffusion WebUI</a> by <ins>AUTOMATIC1111</ins>, to make development easier, optimize resource management, speed up inference, and study experimental features.<br>
The name "Forge" is inspired by "Minecraft Forge". This project aims to become the Forge of Stable Diffusion WebUI.<br>
<p align="right">- <b>lllyasviel</b><br>
<sup>(paraphrased)</sup></p>
</i></blockquote>

<br>

"**Classic**" mainly serves as an archive for the "`previous`" version of Forge, which was built on [Gradio](https://github.com/gradio-app/gradio) `3.41.2` before the major changes *(see the original [announcement](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/801))* were introduced. Additionally, this fork is focused exclusively on **SD1** and **SDXL** checkpoints, having various optimizations implemented, with the main goal of being the lightest WebUI without any bloatwares.

> [!Tip]
> [How to Install](#installation)

<br>

## Features [Dec.]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - requires **manually** installing [uv](https://github.com/astral-sh/uv/releases)
    - drastically speed up installation
    - see [Commandline](#by-classic)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention), [FlashAttention](https://github.com/Dao-AILab/flash-attention), and fast `fp16_accumulation`
    - see [Commandline](#by-classic)
- [X] Support fast `fp8` operation *(`torch._scaled_mm`)*
    - requires RTX **40** +
    - requires **UNet Weights in fp8** option
    - ~10% speed up; reduce quality
    - enable in **Settings/Optimizations**

> [!Note]
> The `fp16_accumulation` requires `fp16` precision, thus is not compatible with the fast `fp8` operation

<br>

- [X] Persistent LoRA Patching
    - speed up LoRA loading in subsequent generations
    - see [Commandline](#by-classic)
- [X] Patch LoRA in-place
    - reduce VRAM usage when loading LoRA
    - enable in **Settings/Extra Networks**
- [X] Implement new Samplers
    - *(ported from reForge Webui)*
- [X] Implement Scheduler dropdown
    - *(backported from Automatic1111 Webui upstream)*
    - enable in **Settings/UI Alternatives**
- [X] Add `CFG` slider to the `Hires. fix` section
- [X] Implement RescaleCFG
    - reduce burnt colors; mainly for `v-pred` checkpoints
    - enable in **Settings/UI Alternatives**
- [X] Implement MaHiRo
    - alternative CFG calculation; improve prompt adherence
    - enable in **Settings/UI Alternatives**
- [X] Implement [Epsilon Scaling](https://github.com/comfyanonymous/ComfyUI/pull/10132)
    - enable in **Settings/Stable Diffusion**
- [X] Implement full precision calculation for `Mask blur` blending
    - enable in **Settings/img2img**
- [X] Support loading upscalers in `half` precision
    - speed up; reduce quality
    - enable in **Settings/Upscaling**
- [X] Support running tile composition on GPU
    - enable in **Settings/Upscaling**
- [X] Allow `newline` in LoRA metadata
    - *(backported from Automatic1111 Webui upstream)*
- [X] Implement sending parameters from generation result rather than from UI
    - **e.g.** send the prompts instead of `Wildcard` syntax
    - enable in **Settings/Infotext**
- [X] Implement tiling optimization for VAE
    - reduce memory usage; reduce speed
    - enable in **Settings/VAE**
- [X] Implement `diskcache` for hashes
    - *(backported from Automatic1111 Webui upstream)*
- [X] Implement `skip_early_cond`
    - *(backported from Automatic1111 Webui upstream)*
    - enable in **Settings/Optimizations**
- [X] Allow inserting the upscaled image to the Gallery instead of overriding the input image
    - *(backported from upstream [PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16405))*
- [X] Support `v-pred` **SDXL** checkpoints *(**e.g.** [NoobAI](https://civitai.com/models/833294?modelVersionId=1190596))*
- [X] Support new LoRA architectures
- [X] Update `spandrel`
    - support new Upscaler architectures
- [X] Add support for `.avif`, `.heif`, and `.jxl` image formats
- [X] Automatically determine the optimal row count for `X/Y/Z Plot`
- [X] Support new LoRA architectures
- [X] `DepthAnything v2` Preprocessor
- [X] Support [NoobAI Inpaint](https://civitai.com/models/1376234/noobai-inpainting-controlnet) ControlNet
- [X] Support [Union](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) / [ProMax](https://huggingface.co/brad-twinkl/controlnet-union-sdxl-1.0-promax) ControlNet
    - they simply always show up in the dropdown

#### Removed Features

- [X] SD2
- [X] Alt-Diffusion
- [X] Instruct-Pix2Pix
- [X] Hypernetworks
- [X] SVD
- [X] Z123
- [X] CLIP Interrogator
- [X] Deepbooru Interrogator
- [X] Textual Inversion Training
- [X] Checkpoint Merging
- [X] LDSR
- [X] Most built-in Extensions
- [X] Some built-in Scripts
- [X] Some Samplers
- [X] Sampler in RadioGroup
- [X] `test` scripts
- [X] Some Preprocessors *(ControlNet)*
- [X] `Photopea` and `openpose_editor` *(ControlNet)*
- [X] Unix `.sh` launch scripts
    - You can still use this WebUI by simply copying a launch script from other working WebUI

#### Optimizations

- [X] **[Freedom]** Natively integrate the `SD1` and `SDXL` logics
    - no longer `git` `clone` any repository on fresh install
    - no more random hacks and monkey patches
- [X] Fix `canvas-zoom-and-pan` built-in extension
    - no more infinite-resizing bug when using `Send to` buttons
- [X] Fix RAM and VRAM leak when switching checkpoints
- [X] Clean up the `ldm_patched` *(**i.e.** `comfy`)* folder
- [X] Remove unused `cmd_args`
- [X] Remove unused `args_parser`
- [X] Remove unused `shared_options`
- [X] Remove legacy codes
- [X] Fix some typos
- [X] Remove redundant upscaler codes
    - put every upscaler inside the `ESRGAN` folder
- [X] Optimize upscaler logics
- [X] Optimize certain operations in `Spandrel`
- [X] Optimize the creation of Extra Networks pages
    - *(backported from Automatic1111 Webui upstream)*
- [X] Improve color correction
- [X] Improve hash caching
- [X] Improve error logs
    - no longer print `TypeError: 'NoneType' object is not iterable`
- [X] Improve implementation for `Comments`
- [X] Update the implementation for `uni_pc` sampler
- [X] Revamp settings
    - improve formatting
    - update descriptions
- [X] Check for Extension updates in parallel
- [X] Move `embeddings` folder into `models` folder
- [X] ControlNet Rewrite
    - change Units to `gr.Tab`
    - remove multi-inputs, as they are "[misleading](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)"
    - change `visible` toggle to `interactive` toggle; now the UI will no longer jump around
    - improve `Presets` application
    - fix `Inpaint not masked` mode
- [X] Disable Refiner by default
    - enable again in **Settings/UI Alternatives**
- [X] Disable Tree View by default
    - enable again in **Settings/Extra Networks**
- [X] Hide Sampler Parameters by default
    - enable again by adding **--adv-samplers** flag
- [X] Hide some X/Y/Z Plot options by default
    - enable again by adding **--adv-xyz** flag
- [X] Run `text encoder` on CPU by default
- [X] Fix `pydantic` Errors
- [X] Fix `Soft Inpainting`
- [X] Fix `Controllllite`
- [X] Fix `MultiDiffusion`
- [X] Fix `SD Upscale`
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

- `--no-download-sd-model`: Do not download a default checkpoint
    - can be removed after you download some checkpoints of your choice
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

#### by. Classic

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

- `--persistent-patches`: Enable the persistent LoRA patching
    - no longer apply LoRA every single generation, if the weight is unchanged
    - save around 1 second per generation when using LoRA

- `--sage`: Install the `sageattention` package to speed up generation
    - will also attempt to install `triton` automatically
- `--flash`: Install the `flash_attn` package to speed up generation
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
    git clone https://github.com/Haoming02/sd-webui-forge-classic
    ```

2. Setup Python

<br>

<details>
<summary>Recommended Method</summary>

- Install **[uv](https://github.com/astral-sh/uv#installation)**
- Set up **venv**
    ```bash
    cd sd-webui-forge-classic
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

**Forge Classic** tries to import the packages and automatically choose the first available attention function in the following order:

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

> [!Important]
> When using `SageAttention 2`, both positive prompts and negative prompts are required; omitting negative prompts may cause `NaN` issues

<br>

> [!Tip]
> Check out the [Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)~

<br>

## Issues & Requests

- **Issues** about removed features will simply be ignored
- **Issues** regarding installation will be ignored if it's obviously user-error
- **Feature Request** not related to performance or optimization will simply be ignored
    - For cutting edge features, check out the [Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) branch instead
    - Non-Windows platforms will not be officially supported, as I cannot verify nor maintain them

</details>

<hr>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>
