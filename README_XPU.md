# Intel XPU (Windows) Setup

This guide covers installing and running Forge Neo on Intel XPU (iGPU/Arc) on
Windows using the PyTorch XPU wheels. It also includes basic troubleshooting.

## Requirements

- Windows 10/11 (64-bit)
- Intel GPU driver 32.0.101.8xxx or newer
- Python 3.11.x (64-bit)
- Git
- Microsoft Visual C++ 2015-2022 Redistributable (x64)
  - https://aka.ms/vs/17/release/vc_redist.x64.exe

> Note: Do not run `oneapi-vars.bat` in the same shell where you launch the
> Web UI. It can override DLLs bundled with the PyTorch XPU wheel and cause
> load errors. Use a clean Command Prompt for launching.

## Install (fresh venv)

```bat
git clone https://github.com/Skimcult/sd-webui-forge-neo.git
cd sd-webui-forge-neo

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install PyTorch XPU

```bat
pip install torch==2.10.0+xpu torchvision==0.25.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

### Verify XPU is available

```bat
python -c "import torch; print(torch.__version__); print('xpu', torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'no xpu')"
```

If `xpu True` appears, continue.

### Install the remaining dependencies

```bat
pip install -r requirements.txt
```

## Launch

Recommended launch flags (faster startup and full XPU residency):

```bat
python launch.py --gpu-only --skip-install --skip-prepare-environment --no-hashing
```

You can also set these once in `webui-user.bat`:

```bat
set COMMANDLINE_ARGS=--gpu-only --skip-install --skip-prepare-environment --no-hashing
```

## Troubleshooting

### WinError 127 / torch_python.dll fails to load
This is almost always a missing runtime or DLL override.

1) Install the VC++ redistributable (link above) and reboot.
2) Ensure you are launching from a clean Command Prompt without oneAPI
   environment scripts.

### XPU is False

- Confirm the XPU wheel is installed (`torch==2.10.0+xpu`).
- Verify the Intel GPU driver is up to date.
- Make sure you are not running a CUDA wheel by mistake.

### launch.py tries to install CUDA

Run from your venv and keep the XPU torch installed:

```bat
venv\Scripts\activate
python -c "import torch; print(torch.__version__)"
```

If it is not `+xpu`, reinstall the XPU wheel as shown above.

