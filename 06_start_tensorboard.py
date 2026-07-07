"""启动 TensorBoard 实时看板。

先运行训练脚本 03_train_ar_emt.py，它会把日志写到 runs/ar_emt_live。
然后运行本脚本：
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 06_start_tensorboard.py

浏览器打开：
  http://localhost:6007
"""

from __future__ import annotations

import subprocess
import sys


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    "logdir": "runs/ar_emt_live",
    "port": 6009,
}


def main() -> None:
    settings = USER_SETTINGS
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        settings["logdir"],
        "--port",
        str(settings["port"]),
        "--host",
        "localhost",
    ]

    print("启动 TensorBoard:")
    print("  " + " ".join(cmd))
    print(f"浏览器地址: http://localhost:{settings['port']}")
    print("如果端口被占用，把 USER_SETTINGS['port'] 改成 6008 或其他端口。")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
