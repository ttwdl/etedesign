"""启动 TensorBoard 实时看板（在浏览器里看训练曲线/结构变化）。

用法：
  1. 先跑 03_train_ar_emt.py，它会把日志写到 runs/ar_emt_36ch_t06_tor15_50；
  2. 再跑本脚本:
       & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 06_start_tensorboard.py
  3. 浏览器打开下面提示的地址（默认 http://localhost:6010）。

端口被占用就把 USER_SETTINGS["port"] 换成别的（如 6010）。
logdir 要和 03 里的 tensorboard_dir 一致。
"""

from __future__ import annotations

import subprocess
import sys


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    "logdir": "runs/ar_emt_36ch_t06_tor15_50",   # 要和 03_train_ar_emt.py 的 tensorboard_dir 一致
    "port": 6010,
}


def main() -> None:
    settings = USER_SETTINGS
    # 用当前 python 去调用 tensorboard 模块，避免 PATH 里找不到 tensorboard 命令
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", settings["logdir"],
        "--port", str(settings["port"]),
        "--host", "localhost",
    ]

    print("启动 TensorBoard:")
    print("  " + " ".join(cmd))
    print(f"浏览器地址: http://localhost:{settings['port']}")
    print("端口被占用就把 USER_SETTINGS['port'] 改成 6010 等其它端口。")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
