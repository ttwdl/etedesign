"""检查当前 Python 环境是否能跑 AR-EMT 项目。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 00_check_env.py

这个脚本不训练，只检查：
  - Python 路径；
  - PyTorch；
  - CUDA / GPU；
  - numpy、scipy、matplotlib、PIL、tmm、tensorboard。
"""

from __future__ import annotations

import importlib
import sys


def check_module(name: str):
    """检查一个 Python 包是否能 import。"""

    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
        return module
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    print("Python 环境检查")
    print(f"  sys.executable = {sys.executable}")
    print(f"  python version = {sys.version.split()[0]}")
    print()

    numpy = check_module("numpy")
    scipy = check_module("scipy")
    matplotlib = check_module("matplotlib")
    pil = check_module("PIL")
    tmm = check_module("tmm")
    tensorboard = check_module("tensorboard")
    torch = check_module("torch")
    print()

    if torch is not None:
        print("PyTorch CUDA 检查")
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU = {torch.cuda.get_device_name(0)}")
            print(f"  CUDA runtime = {torch.version.cuda}")
        else:
            print("  没检测到 CUDA，训练会用 CPU，速度会慢很多。")
        print()

    required = [numpy, scipy, matplotlib, pil, tmm, tensorboard, torch]
    if all(x is not None for x in required):
        print("环境检查通过，可以继续运行 01_debug_tmm_emt.py。")
    else:
        print("环境检查没有通过，请先补齐上面 [FAIL] 的包。")


if __name__ == "__main__":
    main()
