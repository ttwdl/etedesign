"""检查当前 Python 环境能不能跑 AR-EMT 项目（只检查，不训练）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 00_check_env.py

它会依次确认：
  - 用的是哪个 Python；
  - PyTorch 装没装、能不能用 GPU(CUDA)；
  - numpy / scipy / matplotlib / PIL / tmm / tensorboard 有没有齐。
全绿了再去跑后面的脚本。
"""

from __future__ import annotations

import importlib
import sys


def check_module(name: str):
    """尝试 import 一个包；成功打印版本，失败打印原因。返回模块或 None。"""

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
    print(f"  sys.executable = {sys.executable}")          # 当前用的 python 解释器路径
    print(f"  python version = {sys.version.split()[0]}")
    print()

    # 逐个检查项目依赖（tmm 只在这里检查一下是否安装；本项目实际用的是自带的可微 TMM）
    numpy = check_module("numpy")
    scipy = check_module("scipy")
    matplotlib = check_module("matplotlib")
    pil = check_module("PIL")
    tmm = check_module("tmm")
    tensorboard = check_module("tensorboard")
    torch = check_module("torch")
    print()

    # 单独把 CUDA(GPU) 情况打印清楚，因为它最影响训练速度
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
        print("环境检查没通过，请先补齐上面 [FAIL] 的包。")


if __name__ == "__main__":
    main()
