"""训练 AR-EMT 光学编码器 + MLP 解码器（主训练脚本）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 03_train_ar_emt.py

平时只改下面的 USER_SETTINGS，不用在命令行加参数。

一句话流程：
  一条 151 维光谱 S(λ)
    → 经过 16 个 AR-EMT 滤光片，得到 16 个测量值（可选：给测量值加噪声）
    → MLP 解码器还原出 151 维光谱 Ŝ(λ)
    → 让 Ŝ(λ) 尽量接近 S(λ)。

损失(loss)由 4 项组成，后 3 项都能单独开关（把对应权重设 0 即可）：
  loss = MSE(还原误差, 主目标)
       + lambda_sam  · 光谱角      (轻微保住谱形，防止把峰抹平)
       + lambda_trans· 吞吐量惩罚  (别让滤光片整体太暗)
       + lambda_coh  · 通道去相关  (让 16 个滤光片形状尽量互补, 重建更好还原)

约定：
  - best checkpoint 只看 val_mse；test 集完全不碰，留给 04_eval_report.py 做最终汇报。
  - 训练时用带噪声 + 随机入射角的测量；验证/评估用干净、0 度的测量。
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from pathlib import Path

# Windows + conda 里，PyTorch / NumPy / SciPy / Matplotlib 有时会重复加载 Intel OpenMP。
# 如果不提前设置，可能出现 “OMP: Error #15: Initializing libiomp5md.dll”。
# 这行只影响当前脚本进程；以后如果你重装环境彻底解决冲突，可以删掉它。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接存图（服务器/后台跑也不会报错）
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    add_measurement_noise,
    emt_condition,
    evaluate_fixed_angle,
    geometry_report,
    measurement_matrix_coherence,
    sam_loss,
    structure_rows,
    tor_percent,
)


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    # ---- 路径 ----
    # absolute 数据缓存目录。先运行 02_prepare_data.py 生成。
    "data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
    "tensorboard_dir": "runs/ar_emt_live",

    # ---- 设备 / 复现 ----
    "device": "cuda",      # 有 NVIDIA GPU 用 cuda，没有就改 cpu
    "seed": 2026,

    # ---- 训练规模 ----
    "epochs": 150,         # 想快速试跑改成 2~3
    "batch_size": 512,
    "eval_batch_size": 4096,

    # ---- 入射角 ----
    # 训练时给每条光谱一个随机入射角，让模型对小角度更稳。
    #   fixed0     : 只用 0 度（最快，但只学 0 度）
    #   per_batch  : 每个 batch 随机一个角度
    #   per_sample : 每条光谱各自随机一个角度（最贴近真实，但最慢）
    "angle_mode": "per_sample",
    "angle_max_deg": 5.0,

    # ---- 测量噪声（重要）----
    # 真实探测器一定有噪声；干净训练会让重建“纸面好看、上机就崩”，所以默认开一点。
    #   noise_rel : 相对(光度)噪声, 正比信号本身, 与数据尺度无关, 最稳妥。0.01 = 1%。
    #   noise_abs : 绝对(读出)噪声, 固定大小; 需要时按你测量值尺度设(如 0.02)。默认关。
    "noise_rel": 0.01,
    "noise_abs": 0.0,

    # ---- loss 各项权重 ----
    "t_target": 0.75,       # 吞吐量下限目标：希望 16 条透过谱的平均透过率别低于它
    "lambda_trans": 0.05,   # 吞吐量惩罚权重
    "lambda_coh": 0.02,     # 通道去相关权重(新增)：越大越逼 16 个滤光片形状互补
    "lambda_sam": 0.05,     # 光谱角权重(新增)：轻微保住谱形；不想要就设 0

    # ---- 优化器 ----
    # 物理结构参数和解码器分两组：结构参数学习率更小、不加 weight decay。
    "decoder_lr": 1e-3,
    "physics_lr": 2e-4,
    "decoder_weight_decay": 1e-4,
    "grad_clip_norm": 1.0,

    # ---- 学习率自动衰减 ----
    # val_mse 连续 patience 次不下降，就把学习率乘以 factor。
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    # ---- 显示 / 保存 ----
    "progress_every_batches": 20,   # 每多少个 batch 刷新一次进度行
    "eval_every_epochs": 1,         # 每多少个 epoch 评估+保存一次
    "save_live_plots": True,        # 是否边训练边存曲线图
    "use_tensorboard": True,        # 是否写 TensorBoard(用 06_start_tensorboard.py 打开)

    # resume=True 且存在 last checkpoint 时，接着上次继续训练。
    # 想从头重训就改成 False（并清空/换掉旧 checkpoint 目录）。
    "resume": False,

    # ---- 几何约束 ----
    "period_nm": 180.0,
    "g_min_nm": 40.0,
    "d_min_nm": 60.0,
    "enforce_d_min": True,
}


# =============================================================================
# 一些小工具
# =============================================================================


def set_seed(seed: int) -> None:
    """固定随机种子，让每次训练尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """读取 02_prepare_data.py 生成的 train/val/test/wl 缓存。"""

    train_path = data_dir / "train_spectra.npy"
    val_path = data_dir / "val_spectra.npy"
    test_path = data_dir / "test_spectra.npy"
    wl_path = data_dir / "wl_nm.npy"
    missing = [p for p in [train_path, val_path, test_path, wl_path] if not p.exists()]
    if missing:
        names = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"数据缓存不完整，请先运行 02_prepare_data.py。缺少:\n{names}")

    train = torch.from_numpy(np.load(train_path).astype(np.float32))
    val = torch.from_numpy(np.load(val_path).astype(np.float32))
    test = torch.from_numpy(np.load(test_path).astype(np.float32))
    wl_nm = torch.from_numpy(np.load(wl_path).astype(np.float32))

    print(f"读取数据缓存: {data_dir}")
    print(f"  train: {tuple(train.shape)}, min={train.min():.4f}, max={train.max():.4f}, mean={train.mean():.4f}")
    print(f"  val  : {tuple(val.shape)}, min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")
    print(f"  test : {tuple(test.shape)}, min={test.min():.4f}, max={test.max():.4f}, mean={test.mean():.4f}")
    print(f"  wl_nm: {tuple(wl_nm.shape)}")
    return train, val, test, wl_nm


def make_alpha(batch_size: int, settings: dict, device: torch.device) -> torch.Tensor:
    """按 angle_mode 生成本次训练用的入射角(度)。"""

    mode = settings["angle_mode"]
    max_deg = float(settings["angle_max_deg"])
    if mode == "fixed0":
        return torch.zeros(1, device=device)
    if mode == "per_batch":
        return torch.rand(1, device=device) * max_deg
    if mode == "per_sample":
        return torch.rand(batch_size, device=device) * max_deg
    raise ValueError(f"未知 angle_mode: {mode}")


def make_optimizer(model: AREMTModel, settings: dict) -> torch.optim.Optimizer:
    """创建 AdamW，分两组参数：
    1) 物理结构参数(rho/h_c/t_r/AR)：学习率小，不加 weight decay；
    2) 解码器参数：学习率大，加一点 weight decay 抑制过拟合。
    """

    physics_params = [model.rho, model.raw_h_c, model.raw_t_r, model.raw_ar]
    return torch.optim.AdamW(
        [
            {"params": physics_params, "lr": settings["physics_lr"], "weight_decay": 0.0, "name": "physics"},
            {"params": model.decoder.parameters(), "lr": settings["decoder_lr"],
             "weight_decay": settings["decoder_weight_decay"], "name": "decoder"},
        ]
    )


def make_scheduler(optimizer: torch.optim.Optimizer, settings: dict):
    """val_mse 长时间不降就自动降低学习率。"""

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings["scheduler_factor"],
        patience=settings["scheduler_patience"],
        min_lr=settings["scheduler_min_lr"],
    )


def save_csv(rows: list[dict], path: Path) -> None:
    """把一组字典存成 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---- 训练日志：只记“训练/验证曲线 + 关键光学量”，结构参数另存一份 CSV ----
TRAIN_LOG_HEADER = [
    "epoch", "lr_physics", "lr_decoder",
    "train_loss", "train_mse", "val_mse", "val_psnr", "val_sam",
    "T0_mean", "T0_min", "tor_percent", "coherence0", "grad_norm",
]


def init_train_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRAIN_LOG_HEADER).writeheader()


def append_train_log(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRAIN_LOG_HEADER).writerow(row)


def save_structure_csv(model: AREMTModel, path: Path) -> None:
    """保存每个通道的 D/P、D、gap、填充因子、n_eff。"""

    save_csv(structure_rows(model), path)


# =============================================================================
# 画图 / TensorBoard
# =============================================================================


def make_spectra_figure(model: AREMTModel) -> plt.Figure:
    """画当前 16 个通道的 0 度透过谱。"""

    device = next(model.parameters()).device
    with torch.no_grad():
        t0 = model.transmission(torch.tensor([0.0], device=device))[0].detach().cpu()
    wl = model.wl_nm.detach().cpu()

    fig = plt.figure(figsize=(9, 5))
    for idx in range(t0.shape[0]):
        plt.plot(wl, t0[idx], lw=1.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("Current 16-channel spectra, alpha=0 deg")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    return fig


def save_progress_plot(log_path: Path, out_path: Path) -> None:
    """把 train_log.csv 画成 4 张训练曲线：MSE / 平均透过率 / 区分度 / 学习率。"""

    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    epoch = [int(r["epoch"]) for r in rows]
    train_mse = [float(r["train_mse"]) for r in rows]
    val_mse = [float(r["val_mse"]) for r in rows]
    t_mean = [float(r["T0_mean"]) for r in rows]
    tor = [float(r["tor_percent"]) for r in rows]
    coh = [float(r["coherence0"]) for r in rows]
    lr_decoder = [float(r["lr_decoder"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epoch, train_mse, marker="o", label="train MSE")
    axes[0, 0].plot(epoch, val_mse, marker="o", label="val MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("MSE"); axes[0, 0].legend()

    axes[0, 1].plot(epoch, t_mean, marker="o", color="tab:green")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("T0_mean"); axes[0, 1].set_ylim(0.0, 1.05)

    # 左下同时画 tor(区分度, 越大越好) 和 coherence(相关性, 越小越好)
    axes[1, 0].plot(epoch, tor, marker="o", color="tab:orange", label="tor % (↑好)")
    ax_coh = axes[1, 0].twinx()
    ax_coh.plot(epoch, coh, marker="s", color="tab:red", label="coherence (↓好)")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("tor (%)"); ax_coh.set_ylabel("coherence")
    axes[1, 0].legend(loc="upper left"); ax_coh.legend(loc="upper right")

    axes[1, 1].plot(epoch, lr_decoder, marker="o", color="tab:purple")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("decoder lr")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_tensorboard(writer: SummaryWriter, epoch: int, row: dict, model: AREMTModel) -> None:
    """把这一轮的关键数字写进 TensorBoard。"""

    writer.add_scalar("loss/train_loss", float(row["train_loss"]), epoch)
    writer.add_scalar("loss/train_mse", float(row["train_mse"]), epoch)
    writer.add_scalar("loss/val_mse", float(row["val_mse"]), epoch)
    writer.add_scalar("quality/val_psnr", float(row["val_psnr"]), epoch)
    writer.add_scalar("quality/val_sam", float(row["val_sam"]), epoch)
    writer.add_scalar("optics/T0_mean", float(row["T0_mean"]), epoch)
    writer.add_scalar("optics/T0_min", float(row["T0_min"]), epoch)
    writer.add_scalar("optics/tor_percent", float(row["tor_percent"]), epoch)
    writer.add_scalar("optics/coherence0", float(row["coherence0"]), epoch)
    writer.add_scalar("train/grad_norm", float(row["grad_norm"]), epoch)
    writer.add_scalar("train/lr_physics", float(row["lr_physics"]), epoch)
    writer.add_scalar("train/lr_decoder", float(row["lr_decoder"]), epoch)

    fig = make_spectra_figure(model)
    writer.add_figure("spectra/current_0deg", fig, epoch)
    plt.close(fig)
    writer.flush()


# =============================================================================
# checkpoint 存 / 取
# 注意：checkpoint 的字段(keys)保持不变，04/05/07 等脚本都按这些字段读取。
# =============================================================================


def checkpoint_dict(model, optimizer, scheduler, config, wl_nm, settings, epoch, best_val_mse) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.__dict__,
        "wl_nm": wl_nm.detach().cpu(),
        "settings": settings,
        "epoch": epoch,
        "best_val_mse": best_val_mse,
    }


def load_resume_if_needed(model, optimizer, scheduler, last_path: Path, settings: dict, device) -> tuple[int, float]:
    """如果开了 resume 且存在 last checkpoint，就从上次断点继续训练。"""

    if not settings["resume"] or not last_path.exists():
        return 1, math.inf

    ckpt = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_val_mse = float(ckpt.get("best_val_mse", math.inf))
    print(f"从 last checkpoint 继续训练: {last_path}")
    print(f"  start_epoch={start_epoch}, best_val_mse={best_val_mse:.6e}")
    return start_epoch, best_val_mse


def print_structure_brief(model: AREMTModel) -> None:
    """终端里简短打印当前结构参数。"""

    params = model.physical_parameters()
    ratio = params["ratio"].detach().cpu()
    ar = params["ar_nm"].detach().cpu()
    h_c = float(params["h_c_nm"].detach().cpu())
    t_r = float(params["t_r_nm"].detach().cpu())
    period = model.config.period_nm
    emt = emt_condition(model.config, ratio.max())
    status = "OK" if emt["ok"] else "FAIL"
    print(f"  h_c={h_c:.2f} nm, t_r={t_r:.2f} nm, AR=[{ar[0]:.2f}, {ar[1]:.2f}, {ar[2]:.2f}, {ar[3]:.2f}] nm")
    print(f"  D/P=[{float(ratio.min()):.4f}, {float(ratio.max()):.4f}], "
          f"D=[{float(ratio.min()) * period:.2f}, {float(ratio.max()) * period:.2f}] nm, "
          f"EMT={status}, margin={emt['margin_nm']:.2f} nm")


# =============================================================================
# 训练一个 epoch —— 这里是整套训练的“心脏”，看懂这段就看懂了全部
# =============================================================================


def run_one_epoch(model, train_cpu, optimizer, mse_fn, settings, device, epoch, n_epochs, writer) -> dict:
    """训练一个 epoch，返回本轮的平均 loss / mse / 透过率 / 区分度等。"""

    model.train()
    batch_size = int(settings["batch_size"])
    n_train = train_cpu.shape[0]
    n_batches = math.ceil(n_train / batch_size)
    perm = torch.randperm(n_train)  # 每个 epoch 打乱一次样本顺序

    # 累加器，用来算这一轮的平均值
    loss_sum = mse_sum = t_mean_sum = coh_sum = 0.0
    grad_norm_last = 0.0
    n_seen = 0
    epoch_start = time.time()

    for batch_index, start in enumerate(range(0, n_train, batch_size), start=1):
        idx = perm[start:start + batch_size]
        batch = train_cpu[idx].to(device, non_blocking=True)          # 真实光谱 S(λ), [B,151]
        alpha = make_alpha(batch.shape[0], settings, device)          # 入射角(度)

        # ---------------- 前向：把物理和网络串起来 ----------------
        # 这几步故意写开，方便你看清数据怎么一步步变过去：
        t = model.transmission(alpha)                                 # 16 条透过谱 [A,16,151]
        t_use = t[0] if t.shape[0] == 1 else t                        # 供 measure 用的形状
        meas = model.measure(batch, t_use)                            # 压成 16 个测量值 [B,16]
        meas = add_measurement_noise(meas, settings["noise_rel"], settings["noise_abs"])  # 训练时加噪
        pred = model.decoder(meas)                                    # 还原回 151 维 [B,151]

        # ---------------- loss：主目标 + 三个约束 ----------------
        loss_mse = mse_fn(pred, batch)                                # 主目标：还原误差
        loss = loss_mse

        if settings["lambda_sam"] > 0:                               # 谱形约束(可选)
            loss = loss + settings["lambda_sam"] * sam_loss(pred, batch)

        t_mean = t.mean()                                            # 平均透过率
        if settings["lambda_trans"] > 0:                            # 吞吐量约束：别让滤光片太暗
            loss_trans = torch.relu(torch.tensor(settings["t_target"], device=device) - t_mean).square()
            loss = loss + settings["lambda_trans"] * loss_trans

        coh = measurement_matrix_coherence(t_use)                    # 通道相关性(越小越好)
        if settings["lambda_coh"] > 0:                              # 去相关约束：逼 16 个滤光片互补
            loss = loss + settings["lambda_coh"] * coh

        # ---------------- 反向 + 更新 ----------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip_norm"])  # 梯度裁剪防爆
        optimizer.step()

        # ---------------- 累加 + 打印进度 ----------------
        bs = batch.shape[0]
        loss_sum += float(loss.detach().cpu()) * bs
        mse_sum += float(loss_mse.detach().cpu()) * bs
        t_mean_sum += float(t_mean.detach().cpu()) * bs
        coh_sum += float(coh.detach().cpu()) * bs
        grad_norm_last = float(grad_norm.detach().cpu())
        n_seen += bs

        if (batch_index == 1 or batch_index == n_batches
                or (settings["progress_every_batches"] > 0 and batch_index % settings["progress_every_batches"] == 0)):
            if writer is not None:
                gb = (epoch - 1) * n_batches + batch_index
                writer.add_scalar("batch/loss", float(loss.detach().cpu()), gb)
                writer.add_scalar("batch/mse", float(loss_mse.detach().cpu()), gb)
            elapsed = time.time() - epoch_start
            pct = batch_index / n_batches * 100.0
            print(f"\repoch {epoch:04d}/{n_epochs} batch {batch_index:04d}/{n_batches:04d} ({pct:5.1f}%) | "
                  f"loss={float(loss.detach().cpu()):.4e} | mse={float(loss_mse.detach().cpu()):.4e} | "
                  f"Tmean={float(t_mean.detach().cpu()):.4f} | coh={float(coh.detach().cpu()):.4f} | "
                  f"grad={grad_norm_last:.3e} | {elapsed:6.1f}s", end="", flush=True)
    print()

    return {
        "train_loss": loss_sum / n_seen,
        "train_mse": mse_sum / n_seen,
        "train_Tmean": t_mean_sum / n_seen,
        "train_coherence": coh_sum / n_seen,
        "grad_norm": grad_norm_last,
    }


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    settings = USER_SETTINGS
    set_seed(settings["seed"])

    device = torch.device(settings["device"] if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(settings["checkpoint_dir"])
    results_dir = Path(settings["results_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_cpu, val_cpu, _test_cpu, wl_nm = load_cache(Path(settings["data_dir"]))
    if device.type == "cuda":
        train_cpu = train_cpu.pin_memory()  # 锁页内存，CPU→GPU 拷贝更快

    config = GeometryConfig(
        period_nm=settings["period_nm"],
        g_min_nm=settings["g_min_nm"],
        d_min_nm=settings["d_min_nm"],
        enforce_d_min=settings["enforce_d_min"],
    )
    print()
    print(geometry_report(config))
    print()

    model = AREMTModel(wl_nm, config).to(device)
    optimizer = make_optimizer(model, settings)
    scheduler = make_scheduler(optimizer, settings)
    mse_fn = nn.MSELoss()

    best_path = checkpoint_dir / "ar_emt_best.pt"
    last_path = checkpoint_dir / "ar_emt_last.pt"
    train_log_path = results_dir / "train_log.csv"
    progress_plot_path = results_dir / "train_progress.png"
    spectra_plot_path = results_dir / "train_current_spectra_0deg.png"
    current_structure_path = results_dir / "ar_emt_current_structure.csv"

    start_epoch, best_val_mse = load_resume_if_needed(model, optimizer, scheduler, last_path, settings, device)
    if start_epoch == 1:
        init_train_log(train_log_path)

    writer = SummaryWriter(log_dir=settings["tensorboard_dir"]) if settings["use_tensorboard"] else None

    print(f"device={device}")
    print(f"epochs={settings['epochs']}, batch_size={settings['batch_size']}, angle_mode={settings['angle_mode']}")
    print(f"noise: rel={settings['noise_rel']}, abs={settings['noise_abs']} | "
          f"lambda: trans={settings['lambda_trans']}, coh={settings['lambda_coh']}, sam={settings['lambda_sam']}")
    print(f"训练日志: {train_log_path}")
    print(f"训练曲线: {progress_plot_path}")
    print(f"看 TensorBoard: 运行 06_start_tensorboard.py")
    print()

    total_start = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        train_info = run_one_epoch(
            model, train_cpu, optimizer, mse_fn, settings, device,
            epoch, int(settings["epochs"]), writer,
        )

        # 不到评估轮次就跳过后面的评估/保存(最后一轮一定评估)
        if epoch % settings["eval_every_epochs"] != 0 and epoch != int(settings["epochs"]):
            continue

        # ---- 验证：干净、0 度 ----
        val_metrics = evaluate_fixed_angle(model, val_cpu, angle_deg=0.0, batch_size=settings["eval_batch_size"])
        scheduler.step(val_metrics["mse"])

        # ---- 记录当前 0 度滤光片的几个观察量 ----
        with torch.no_grad():
            t0 = model.transmission(torch.tensor([0.0], device=device))[0]
            tor = tor_percent(t0)
            coherence0 = float(measurement_matrix_coherence(t0).detach().cpu())
            t_mean_eval = float(t0.mean().detach().cpu())
            t_min_eval = float(t0.min().detach().cpu())

        row = {
            "epoch": epoch,
            "lr_physics": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "train_loss": train_info["train_loss"],
            "train_mse": train_info["train_mse"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
            "val_sam": val_metrics["sam"],
            "T0_mean": t_mean_eval,
            "T0_min": t_min_eval,
            "tor_percent": tor,
            "coherence0": coherence0,
            "grad_norm": train_info["grad_norm"],
        }

        append_train_log(train_log_path, row)
        save_structure_csv(model, current_structure_path)  # 结构参数单独存一份
        if settings["save_live_plots"]:
            save_progress_plot(train_log_path, progress_plot_path)
            fig = make_spectra_figure(model)
            fig.savefig(spectra_plot_path, dpi=160)
            plt.close(fig)
        if writer is not None:
            write_tensorboard(writer, epoch, row, model)

        # 每个评估轮都存一份 last checkpoint（断点续训用）
        torch.save(checkpoint_dict(model, optimizer, scheduler, config, wl_nm, settings, epoch, best_val_mse), last_path)

        print(f"epoch {epoch:04d}/{settings['epochs']} | train_mse={row['train_mse']:.6e} | "
              f"val_mse={row['val_mse']:.6e} | psnr={row['val_psnr']:.2f} | sam={row['val_sam']:.4f} | "
              f"T0_mean={row['T0_mean']:.4f} T0_min={row['T0_min']:.4f} | "
              f"tor={row['tor_percent']:.3f}% coh={row['coherence0']:.4f} | grad={row['grad_norm']:.3e}")
        print_structure_brief(model)

        # 只有 val_mse 创新低时，才更新 best checkpoint
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            torch.save(checkpoint_dict(model, optimizer, scheduler, config, wl_nm, settings, epoch, best_val_mse), best_path)
            save_structure_csv(model, results_dir / "ar_emt_best_structure.csv")
            print(f"  ✔ 新的 best，已保存: {best_path}")
        print()

    if writer is not None:
        writer.close()

    print(f"训练完成。last checkpoint: {last_path}")
    print(f"最佳 val_mse: {best_val_mse:.6e}")
    print(f"总耗时: {(time.time() - total_start) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
