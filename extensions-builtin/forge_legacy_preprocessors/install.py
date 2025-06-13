import os
from pathlib import Path
from modules.errors import display

import launch

repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def install_requirements(req_file):
    with open(req_file, "r") as file:
        for package in file.readlines():
            try:
                package = package.strip()
                if not launch.is_installed(package):
                    launch.run_pip(
                        f"install {package}",
                        f"Legacy Preprocessor Requirement: {package}",
                    )
            except Exception as e:
                display(e, "cnet req")
                print(f"Failed to install {package}, some Preprocessors may not work...")


install_requirements(main_req_file)


def try_install_from_wheel(pkg_name: str, wheel_url: str):
    if launch.is_installed(pkg_name):
        return

    try:
        launch.run_pip(
            f"install {wheel_url}",
            f"Legacy Preprocessor Requirement: {pkg_name}",
        )
    except Exception as e:
        display(e, "cnet req")
        print(f"Failed to install {pkg_name}, some Preprocessors may not work...")


try_install_from_wheel(
    "handrefinerportable",
    wheel_url=os.environ.get(
        "HANDREFINER_WHEEL",
        "https://github.com/huchenlei/HandRefinerPortable/releases/download/v1.0.1/handrefinerportable-2024.2.12.0-py2.py3-none-any.whl",
    ),
)

try_install_from_wheel(
    "depth_anything",
    wheel_url=os.environ.get(
        "DEPTH_ANYTHING_WHEEL",
        "https://github.com/huchenlei/Depth-Anything/releases/download/v1.0.0/depth_anything-2024.1.22.0-py2.py3-none-any.whl",
    ),
)

try_install_from_wheel(
    "depth_anything_v2",
    wheel_url=os.environ.get(
        "DEPTH_ANYTHING_V2_WHEEL",
        "https://github.com/MackinationsAi/UDAV2-ControlNet/releases/download/v1.0.0/depth_anything_v2-2024.7.1.0-py2.py3-none-any.whl",
    ),
)
