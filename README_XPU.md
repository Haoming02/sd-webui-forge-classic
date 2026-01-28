# Intel XPU Setup (Windows / Linux / WSL2)

This guide covers installing and running Forge Neo on Intel XPU (iGPU/Arc) on
Windows, Linux, and WSL2 using the PyTorch XPU wheels. It also includes basic
troubleshooting.

## Windows Requirements

- Windows 10/11 (64-bit)
- Intel GPU driver 32.0.101.8xxx or newer
- Python 3.11.x (64-bit)
- Git
- Microsoft Visual C++ 2015-2022 Redistributable (x64)
  - https://aka.ms/vs/17/release/vc_redist.x64.exe

> Note: Do not run `oneapi-vars.bat` in the same shell where you launch the
> Web UI. It can override DLLs bundled with the PyTorch XPU wheel and cause
> load errors. Use a clean Command Prompt for launching.

## Windows Install (fresh venv)

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

## Windows Launch

Recommended launch flags (faster startup and full XPU residency):

```bat
python launch.py --gpu-only --skip-install --skip-prepare-environment --no-hashing
```

You can also set these once in `webui-user.bat`:

```bat
set COMMANDLINE_ARGS=--gpu-only --skip-install --skip-prepare-environment --no-hashing
```

## Linux / WSL2 Setup

### Requirements

- Ubuntu 22.04/24.04 (native or WSL2)
- Intel GPU driver with Level Zero (native) or Windows Intel driver with WSL2 GPU support
- Python 3.11, Git

### Install system GPU runtime (Ubuntu)

```bash
sudo apt update
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero
```

### Create venv and install PyTorch XPU

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

pip install torch==2.10.0+xpu torchvision==0.25.0+xpu --index-url https://download.pytorch.org/whl/xpu
```

Optional (Linux only, if wheels are available for your distro):

```bash
pip install --find-links https://pytorch-extension.intel.com/release-whl/stable/xpu intel-extension-for-pytorch==2.10.0
```

### Verify XPU is available

```bash
python -c "import torch; print(torch.__version__); print('xpu', torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'no xpu')"
```

### Launch (Linux / WSL2)

```bash
python launch.py --gpu-only --skip-install --skip-prepare-environment --no-hashing
```

WSL2 notes:

- Update Windows Intel GPU driver to the latest version.
- Update WSL: `wsl --update`
- Use `http://localhost:7860` from Windows to access the UI.
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

### Linux / WSL2: XPU is False

- Ensure `intel-level-zero-gpu` and `intel-opencl-icd` are installed.
- On WSL2, update Windows Intel driver and run `wsl --update`.

### launch.py tries to install CUDA

Run from your venv and keep the XPU torch installed:

```bat
venv\Scripts\activate
python -c "import torch; print(torch.__version__)"
```

If it is not `+xpu`, reinstall the XPU wheel as shown above.

