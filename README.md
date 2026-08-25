<h1 align="center">Stable Diffusion WebUI Forge - Neo</h1>

<p align="center"><sup>
[ <b>Neo</b> | <a href="https://github.com/Haoming02/sd-webui-forge-classic/tree/classic#stable-diffusion-webui-forge---classic">Classic</a> ]
</sup></p>

<p align="center"><img src="html\ui.webp" width=512 alt="UI"></p>

<blockquote><i>
<b>Stable Diffusion WebUI Forge</b> is a platform on top of the original <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Stable Diffusion WebUI</a> by <ins>AUTOMATIC1111</ins>, to make development easier, optimize resource management, speed up inference, and study experimental features.<br>
The name "Forge" is inspired by "Minecraft Forge". This project aims to become the Forge of Stable Diffusion WebUI.<br>
<p align="right">- <b>lllyasviel</b><br>
<sup>(paraphrased)</sup></p>
</i></blockquote>

<br>

"**Neo**" mainly serves as an continuation for the "`latest`" version of Forge, which was built on [Gradio](https://github.com/gradio-app/gradio) `4.40.0` before lllyasviel became too busy... Additionally, this fork is focused on optimization and usability, with the main goal of being able to run the latest popular models via an easy-to-use GUI.

> [!Tip]
> [How to Install](#installation)

<br>

## Features [Aug.]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] Support [Krea 2](https://huggingface.co/krea/Krea-2-Turbo)
    - `Turbo` / `Raw`
- [X] Support **Krea 2 Identity Edit**
    - require specific [LoRA](https://civitai.com/models/2761113/krea-2-identity-edit)
    - enable in **Settings/Stable Diffusion**
- [X] Support [Anima](https://huggingface.co/circlestone-labs/Anima)
- [X] Support **Anima Edit**
    - require specific [LoRA](https://civitai.com/models/2650553/anima-edit)
    - enable in **Settings/Stable Diffusion**
- [X] Support [Flux.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
    - `4B` / `9B` (**not** `FLUX.2-Dev`)

> [!Important]
> To use `Flux.2-Klein` for regular `img2img`, toggle the functionality in **Settings/Stable Diffusion**

- [X] Support [Ernie-Image](https://huggingface.co/baidu/ERNIE-Image)
    - `ernie-image` / `ernie-image-turbo`
- [X] Support [PiD 1.5](https://huggingface.co/nvidia/PiD)
    - `sdxl` / `qwen` / `flux1` / `flux2` (**not** `PixelDiT`)
    - use `PiD Integrated` to automatically upscale after generation
- [X] Support [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)
    - `z-image` / `z-image-turbo`
- [X] Support [Wan 2.2](https://github.com/Wan-Video/Wan2.2)
    - `14B` (**not** `5B`)
    - use `Refiner` to achieve **High Noise** / **Low Noise** switching

> [!Important]
> To export a video, you need to have **[FFmpeg](https://ffmpeg.org/)** installed

- [X] Support [Mugen](https://huggingface.co/CabalResearch/Mugen)
    - display the `Shift` slider for `xl` preset in **Settings/Presets/XL**
- [X] Support advanced **SDXL** models

> [!Note]
> - **v-prediction:** `state_dict` must include "`v_pred`"
> - **Zero Terminal SNR:** `state_dict` must include "`ztsnr`"
> - **Rectified Flow:** the model must include "`rectified`" in its path *(**e.g.** file name or folder name)*

- [X] Support [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) / [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

> [!Note]
> To be detected as an **Edit** model, the model must include "`qwen`" and "`edit`" in its path *(**e.g.** file name or folder name)*

- [X] Support [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

> [!Note]
> To be detected as a **Kontext** model, the model must include "`kontext`" in its path *(**e.g.** file name or folder name)*

- Implement `ImageStitch Integrated`
    - [X] support multi-image inputs for **Edit** models
    - [X] support FirstLastFrameToVideo for `Wan 2.2`
- [X] Support [Nunchaku](https://github.com/nunchaku-tech/nunchaku) (`SVDQ`) Models (**Deprecated**)
    - `flux-dev`, `flux-krea`, `flux-kontext`, `qwen-image`, `qwen-image-edit`, `z-image-turbo`
    - only `Flux` and `Qwen` support LoRA currently
    - see [Commandline](#by-neo)
- [X] Support [Lumina-Image-2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
    - `Neta-Lumina` / `NetaYume-Lumina`
- [X] Support [Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD)
- [X] Support **MixedPrecision** Models
    - `fp4mixed` / `fp8mixed` / `mxfp8` / `nvfp4` / `fp8_scaled` / `int8_convrot` / `convrot_w4a4`
- [X] Support [Flux.2-Small-Decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/blob/main/full_encoder_small_decoder.safetensors) & [Qwen2D VAE](https://huggingface.co/Anzhc/Qwen2D-VAE/blob/main/Qwen2D_VAE.safetensors)

<br>

> [!Tip]
> Check out [Download Models](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) for where to get each model and the accompanying modules

> [!Tip]
> Check out [Inference References](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References) for how to use each model and the recommended parameters

<br>

- [X] Rewrite Preset System
    - now save the checkpoint/module selection and parameters per each Preset

> [!Note]
> This overrides the `UI Defaults` for the controlled parameters

<br>

- [X] Enforce Resolution Steps
    - dimensions must be multiples of `64` by default
    - adjust in **Settings/System**
- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - drastically speed up installation
    - require **manually** installing [uv](https://github.com/astral-sh/uv/releases)
    - see [Commandline](#by-neo)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention), [FlashAttention](https://github.com/Dao-AILab/flash-attention), `fp16_accumulation`, `torch._scaled_mm`
    - see [Commandline](#by-neo)
- [X] Implement [Radial Attention](https://github.com/mit-han-lab/radial-attention)
    - speed up `Wan 2.2`
    - require **manually** installing [SpargeAttn](https://github.com/thu-ml/SpargeAttn)
- [X] Implement fast `state_dict` switching for Refiner
    - enable in **Settings/Refiner**
- [X] Implement RescaleCFG
    - reduce burnt colors; mainly for `v-pred` checkpoints
    - enable in **Settings/UI Alternatives**
- [X] Implement MaHiRo
    - alternative CFG calculation; improve prompt adherence
    - enable in **Settings/UI Alternatives**
- [X] Implement [Spectrum](https://github.com/hanjq17/Spectrum)
    - training-free acceleration for all models
- [X] Implement [Epsilon Scaling](https://github.com/comfyanonymous/ComfyUI/pull/10132)
    - enable in **Settings/Stable Diffusion**
- [X] Implement `torch.compile`
    - speed up inference after compilation
- [X] Implement alternative Prompt Box layouts
- [X] Implement tiled `Conv2d` for VAE
    - reduce memory usage; reduce speed
    - see [Commandline](#by-neo)
- [X] Implement full precision calculation for `Mask blur` blending
    - enable in **Settings/img2img**
- [X] Support TAESD / TAEHV live preview for all models
- [X] Support video previews for ExtraNetworks
- [X] Support loading upscalers in `half` precision
    - speed up; reduce quality
    - enable in **Settings/Upscaling**
- [X] Support running tile composition on GPU
    - enable in **Settings/Upscaling**
- [X] Support (short) videos in **Extras** tab
- [X] Add support for `.avif`, `.heif`, and `.jxl` image formats
- [X] Automatically determine the optimal row count for `X/Y/Z Plot`
- [X] Update **LLLite** Controlnet
    - [SDXL](https://huggingface.co/kohya-ss/controlnet-lllite/tree/main) / [Anima](https://huggingface.co/kohya-ss/Anima-LLLite/tree/main)
- [X] Support **Union** Controlnet
    - [SDXL](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) / [Chenkin](https://civitai.com/models/2527960/chenkin-unicontrol-xl)
- [X] Support **Region** Controlnet
    - [Anima](https://huggingface.co/Sen-sou/Anima-LLLite-Regional-Controlnet)
    - use the **Region** mode to create color masks

#### Removed Features

- [X] SD2
- [X] SD3
- [X] Forge Spaces
- [X] Hypernetworks
- [X] CLIP Interrogator
- [X] Deepbooru Interrogator
- [X] Textual Inversion Training
- [X] Some built-in Extensions
- [X] Some built-in Scripts
- [X] Some Samplers & Schedulers
- [X] Some Compatibility Settings
- [X] Stealth Infotext
- [X] `bitsandbytes` Support

#### Optimizations

- [X] **[Comfy]** Rewrite the Backend *(`memory_management.py`, `ModelPatcher`, `attention.py`, etc.)*
- [X] No longer `git` `clone` any repository on fresh install
- [X] No longer install `open-clip`
- [X] Fix memory leak when switching checkpoints
- [X] Restore the ability to drag-and-drop images onto `gr.Image` that already contains image
- [X] Speed up launch time
- [X] Improve timer logs
- [X] Remove unused `cmd_args`
- [X] Remove unused `args_parser`
- [X] Remove unused `shared_options`
- [X] Remove legacy codes
- [X] Fix some typos
- [X] Fix automatic `Tiled VAE` fallback
- [X] Fix `Tiling` for SD1 and SDXL
- [X] Pad conditioning for SDXL
- [X] Remove duplicated upscaler codes
- [X] Update [spandrel](https://github.com/chaiNNer-org/spandrel)
    - support new upscaler architectures

> [!Important]
> Put every upscaler (`.pth` / `.safetensors`) inside the `ESRGAN` folder

> [!Tip]
> Check out [OpenModelDB](https://openmodeldb.info/) for where to get upscalers

- [X] Improve `ForgeCanvas`
    - brush adjustments
    - customization
    - deobfuscate
    - eraser
    - hotkeys
    - mobile friendly
- [X] Optimize upscaler logics
- [X] Optimize certain operations in `Spandrel`
- [X] Optimize certain operations for `VAE`
- [X] Speed up model loading
- [X] Improve color correction
- [X] Improve memory management
- [X] Unload all models on `OutOfMemory`
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
- [X] Infotext Rewrite
    - allow switching Models and Modules
    - save `emphasis` properly
    - correct default values
- [X] ControlNet Rewrite
    - change Units to `gr.Tab`
    - improve `masks` & `buttons`
    - remove multi-inputs, as they are "[misleading](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)"
- [X] Checkpoint Merger Rewrite
    - update ui
    - support modern models
- [X] Refiner Rewrite
    - update ui
    - improve tooltips
    - implement `Refiner CFG Scale`
- [X] Improve inference speed
- [X] Improve non-Nvidia support
- [X] Lint & Format
- [X] Update `Pillow`
    - faster image processing
- [X] Update `protobuf`
    - faster `insightface` loading
- [X] Update to latest PyTorch
    - `torch==2.13.0+cu130` on NVIDIA
    - `torch==2.9.1+rocm7.13` on AMD *(see [AMD.md](AMD.md))*

> [!Note]
> If your GPU does not support the latest PyTorch, manually [install](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations#older-pytorch) older version of PyTorch

- [X] Update some packages to newer versions
- [X] Update recommended Python to `3.13.12`
- [X] many more... :tm:

<br>

## Commandline
> These flags can be added after the `set COMMANDLINE_ARGS=` line in the `webui-user.bat` *(in the same line ; separate each flag with space)*

> [!Tip]
> Use `python launch.py --help` to see all available flags

- `--xformers`: Install the `xformers` package to speed up generation

> [!Warning]
> `xformers` does **not** support `RTX 50s`, and is **not** available on AMD

- `--gpu-backend`: Pin the compute backend instead of auto-detecting it
    - one of `cuda`, `rocm`, `zluda`, `directml`, `cpu`
    - see **[AMD.md](AMD.md)** for the AMD ones

- `--port`: Specify a server port to use
    - defaults to `7860`
- `--api`: Enable [API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) access

#### by. Neo

- `--cuda-malloc`: Improve memory allocation
- `--cuda-stream`: Enable async weight offloading
- `--pin-shared-memory`: Improve RAM utilization
- `--expandable-segments`: Enable experimental PyTorch allocator *(may prevent `OutOfMemory` errors on certain platforms)*

<br>

- `--uv`: Replace the `python -m pip` calls with `uv pip` to massively speed up package installation
    - requires **uv** to be installed first *(see [Extra Installations](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations))*
- `--uv-symlink`: Same as above; but additionally pass `--link-mode symlink` to the commands
    - significantly reduces installation size (`~7 GB` to `~100 MB`)
- `--uv-local-cache`: Same as above; but additionally set `UV_CACHE_DIR` to a `.uv-cache` folder within WebUI directory
    - speed up installation on non-default drive *(**i.e.** not `C:` on Windows)*
    - allow clean uninstallation by simply deleting the WebUI directory

> [!Important]
> `symlink` means it will directly access the packages from the cache folder instead of copying the packages over ; refrain from clearing the cache when using this option

<br>

- `--model-ref`: Points to a central `models` folder that contains all your models
    - said folder should contain subfolders like `Stable-diffusion`, `Lora`, `VAE`, `ESRGAN`, etc.

> [!Important]
> This simply **replaces** the `models` folder rather than adding on top of it

<br>

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
- `--onnxruntime-gpu`: Install the `onnxruntime` with the latest GPU support

<br>

- `--fast-fp8`: Use the `torch._scaled_mm` function when the model type is `float8_e4m3fn`
- `--fast-fp16`: Enable the `allow_fp16_accumulation` option
- `--autotune`: Enable the `torch.backends.cudnn.benchmark` option
    - this is slower in my experience...
- `--tiled-conv2d`: Replace `Conv2d` ops with tiled variants
    - has greater reduction for **SD1** and **SDXL** VAE; less for **Wan** VAE
    - `64` / `128` / `256` / `512`

<br>

- `--enable-triton-backend`: Enable the use of Triton backend in `comfy-kitchen`
    - *might* speed up inference

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
4. Launch the WebUI via `webui-user.bat` *(**AMD Radeon**: use `webui-user-amd.bat` instead)*
5. During the first launch, it will automatically install all the requirements
6. Once the installation is finished, the WebUI will start in a browser automatically

<br>

> [!Tip]
> - For **AMD**, see **[AMD.md](AMD.md)** — the correct PyTorch build (ROCm, or
>   ZLUDA as a fallback) is selected and installed automatically; `webui-user-amd.bat`
>   is a ready-made launcher. Third-party walkthroughs such as
>   <a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#amd-forge-neo-with-rocm"><ins>CS1o</ins> 's Guide</a>
>   remain useful background reading.
> - For **Linux** and **macOS**, refer to [Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix)
> - For **Docker** (`Nvidia`), refer to [Docker](docker/)

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
- **Issues** that is obviously user-error will simply be ignored
- **Issues** regarding **AMD** GPU will simply be ignored
- **Issues** running non-official models will simply be ignored
    - do not just randomly download every single finetune/quant you find
- **Issues** about 3rd-party Extensions will simply be ignored
    - extension should support the UI, not the other way around
- **Issues** caused by [StabilityMatrix](https://github.com/LykosAI/StabilityMatrix) will simply be ignored
    - only open an Issue if you can reproduce it on a clean install following the official [Installation](#installation) instruction

> [!Caution]
> - If you post **NSFW** images/videos, you will immediately be banned
>     - the sole discretion is on me ; if you are unsure, just generate `cats` and `dogs`...

<hr>

> [!Tip]
> Check out the [Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki) & [FAQ](https://github.com/Haoming02/sd-webui-forge-classic/issues/414)

<br>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <b>city96</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>

<br>

<p align="right">
<sub><i>
Buy me a <a href="https://ko-fi.com/Haoming">Coffee</a> ☕~
</i></sub>
<br>
<sub><i>
<a href="https://paypal.me/hmgamingdonation">PayPal</a> me 💳~
</i></sub>
</p>
