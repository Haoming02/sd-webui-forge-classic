@echo off

:: ===========================================================================
::  Launcher for AMD Radeon GPUs on Windows.
::
::  Tested target: Radeon RX 6800 (gfx1030, 16 GB) / Ryzen 5 3600 / 64 GB RAM
::                 / Windows 10 x64
::
::  Requirements:
::    * Python 3.13 (or 3.12) 64-bit, "Add python.exe to PATH" ticked
::    * Git for Windows
::    * A current AMD Adrenalin driver
::
::  The first launch downloads a ROCm build of PyTorch (a few GB) and creates
::  the "venv" folder. Read AMD.md if anything goes wrong.
:: ===========================================================================

:: Leave blank to use whatever "python" resolves to, or point at a specific
:: interpreter, e.g. set PYTHON="C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"
:: set PYTHON=
:: set GIT=
:: set VENV_DIR=

:: --gpu-backend rocm     native ROCm PyTorch (recommended for RX 6800)
:: --pin-shared-memory    use the 64 GB of system RAM to hold offloaded weights
set COMMANDLINE_ARGS=--gpu-backend rocm --pin-shared-memory

:: ---------------------------------------------------------------------------
::  Other options worth knowing about
:: ---------------------------------------------------------------------------
:: --gpu-backend zluda      fall back to ZLUDA (needs the AMD HIP SDK 6.2/6.4);
::                          use this if the ROCm wheels do not cover your card
:: --zluda-nightly          ZLUDA nightly: adds hipBLASLt and MIOpen support
:: --cuda-stream 2          overlap weight transfers with compute
:: --lowvram / --novram     more aggressive offloading for very large models
:: --reserve-vram 1.0       leave 1 GB of VRAM for the desktop compositor
:: --use-pytorch-cross-attention   SDPA instead of the sliced fallback: faster
::                          at low resolutions, more VRAM-hungry at high ones
:: --fp16-unet              force fp16 weights (RDNA 2 has no fast bf16)
:: --disable-gpu-warning    silence the low-VRAM warnings
:: --reinstall-torch        rebuild the PyTorch install from scratch
::
:: NOT usable on AMD (they are CUDA-only and will be ignored with a message):
::   --xformers --sage --flash --nunchaku --onnxruntime-gpu --cuda-malloc

call webui.bat
