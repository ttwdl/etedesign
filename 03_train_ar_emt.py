"""训练 AR-EMT 光学编码器 + MLP 解码器。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 03_train_ar_emt.py

平时只改下面 USER_SETTINGS，不需要在命令行后面加参数。

当前训练目标：
  输入一条 151 维光谱 S(λ)
  -> 经过 16 个 AR-EMT 滤光片得到 16 个测量值
  -> MLP 解码回 151 维光谱 Ŝ(λ)
  -> 用 MSE 让 Ŝ(λ) 接近 S(λ)

注意：
  tor 只记录，不进 loss。
  best checkpoint 只看 val_mse，test 集留到 04_eval_report.py 最终评估。
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    emt_condition,
    evaluate_fixed_angle,
    geometry_report,
    structure_rows,
    tor_percent,
)


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    # 新 absolute 数据缓存。先运行 02_prepare_data.py 生成。
    "data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",

    # 输出目录。重构后重新训练，旧目录已经清空。
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
    "tensorboard_dir": "runs/ar_emt_live",

    # 设备。电脑有 NVIDIA GPU 时用 cuda，没有就改成 cpu。
    "device": "cuda",
    "seed": 2026,

    # 训练规模。想快速试跑可以把 epochs 改成 2 或 3。
    "epochs": 150,
    "batch_size": 512,
    "eval_batch_size": 4096,

    # 训练入射角。per_sample 表示每条光谱随机一个 0-5 度角。
    # fixed0 更快，但只学 0 度。
    "angle_mode": "per_sample",  # fixed0 / per_batch / per_sample
    "angle_max_deg": 5.0,

    # 测量噪声。这里是加在 16 通道测量值上的很小噪声。
    # absolute 数据保持积分和，测量值可能比 0-1 大，所以这个值先设很小。
    "noise_max": 0.0,

    # loss = MSE + lambda_trans * max(0, T_target - T_mean)^2
    # 透过率约束只是防止滤光片太暗，主目标仍是重建 MSE。
    "t_target": 0.75,
    "lambda_trans": 0.05,

    # 优化器：decoder 用 weight decay，物理结构参数不加 weight decay。
    "decoder_lr": 1e-3,
    "physics_lr": 2e-4,
    "decoder_weight_decay": 1e-4,
    "grad_clip_norm": 1.0,

    # 学习率调度：val_mse 连续几次不降，就把学习率乘 factor。
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    # 进度显示和保存频率。
    "progress_every_batches": 20,
    "eval_every_epochs": 1,
    "save_live_plots": True,
    "use_tensorboard": True,

    # resume=True 时，如果 checkpoints/ar_emt_last.pt 存在，就接着训练。
    # 如果你想从头重新训练，把它改成 False。
    "resume": True,

    # 几何约束。
    "period_nm": 180.0,
    "g_min_nm": 40.0,
    "d_min_nm": 60.0,
    "enforce_d_min": True,
}


def set_seed(seed: int) -> None:
    """固定随机种子，方便复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cache(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """读取 02_prepare_data.py 生成的缓存。"""

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
    """生成训练时使用的入射角。"""

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
    """创建 AdamW 优化器。

    分两组参数：
    1. 物理结构参数：rho、h_c、t_r、AR 厚度，不加 weight decay；
    2. decoder 参数：MLP 权重和偏置，加很小 weight decay，减少过拟合。
    """

    physics_params = [model.rho, model.raw_h_c, model.raw_t_r, model.raw_ar]
    return torch.optim.AdamW(
        [
            {
                "params": physics_params,
                "lr": settings["physics_lr"],
                "weight_decay": 0.0,
                "name": "physics",
            },
            {
                "params": model.decoder.parameters(),
                "lr": settings["decoder_lr"],
                "weight_decay": settings["decoder_weight_decay"],
                "name": "decoder",
            },
        ]
    )


def make_scheduler(optimizer: torch.optim.Optimizer, settings: dict):
    """创建学习率调度器。

    val_mse 如果长时间不下降，就自动降低学习率。
    """

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=settings["scheduler_factor"],
        patience=settings["scheduler_patience"],
        min_lr=settings["scheduler_min_lr"],
    )


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """保存 CSV，小表格统一用这个函数。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def init_train_log(path: Path) -> None:
    """初始化训练日志 CSV。"""

    header = [
        "epoch",
        "lr_physics",
        "lr_decoder",
        "train_loss",
        "train_mse",
        "train_Tmean",
        "val_mse",
        "val_psnr",
        "val_sam",
        "T0_mean",
        "T0_min",
        "tor_percent",
        "grad_norm",
        "h_c_nm",
        "t_r_nm",
        "top_L_nm",
        "top_H_nm",
        "bottom_H_nm",
        "bottom_L_nm",
        "ratio_min",
        "ratio_max",
        "D_min_nm",
        "D_max_nm",
        "emt_margin_nm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()


def append_train_log(path: Path, row: dict[str, float | int]) -> None:
    """向训练日志追加一行。"""

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_structure_csv(model: AREMTModel, path: Path) -> None:
    """保存每个通道的 D/P、D、gap、f、n_eff。"""

    save_csv(structure_rows(model), path)


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
    """把 train_log.csv 画成训练曲线图。"""

    if not log_path.exists():
        return

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        return

    epoch = [int(r["epoch"]) for r in rows]
    train_mse = [float(r["train_mse"]) for r in rows]
    val_mse = [float(r["val_mse"]) for r in rows]
    t_mean = [float(r["T0_mean"]) for r in rows]
    tor = [float(r["tor_percent"]) for r in rows]
    lr_decoder = [float(r["lr_decoder"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epoch, train_mse, marker="o", label="train MSE")
    axes[0, 0].plot(epoch, val_mse, marker="o", label="val MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()

    axes[0, 1].plot(epoch, t_mean, marker="o", color="tab:green")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("T0_mean")
    axes[0, 1].set_ylim(0.0, 1.05)

    axes[1, 0].plot(epoch, tor, marker="o", color="tab:orange")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("tor (%)")

    axes[1, 1].plot(epoch, lr_decoder, marker="o", color="tab:purple")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("decoder lr")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    row: dict[str, float | int],
    model: AREMTModel,
) -> None:
    """把训练状态写入 TensorBoard。"""

    writer.add_scalar("loss/train_loss", float(row["train_loss"]), epoch)
    writer.add_scalar("loss/train_mse", float(row["train_mse"]), epoch)
    writer.add_scalar("loss/val_mse", float(row["val_mse"]), epoch)
    writer.add_scalar("quality/val_psnr", float(row["val_psnr"]), epoch)
    writer.add_scalar("quality/val_sam", float(row["val_sam"]), epoch)

    writer.add_scalar("optics/T_train_mean", float(row["train_Tmean"]), epoch)
    writer.add_scalar("optics/T0_mean", float(row["T0_mean"]), epoch)
    writer.add_scalar("optics/T0_min", float(row["T0_min"]), epoch)
    writer.add_scalar("optics/tor_percent", float(row["tor_percent"]), epoch)
    writer.add_scalar("optics/emt_margin_nm", float(row["emt_margin_nm"]), epoch)

    writer.add_scalar("train/grad_norm", float(row["grad_norm"]), epoch)
    writer.add_scalar("train/lr_physics", float(row["lr_physics"]), epoch)
    writer.add_scalar("train/lr_decoder", float(row["lr_decoder"]), epoch)

    writer.add_scalar("structure/h_c_nm", float(row["h_c_nm"]), epoch)
    writer.add_scalar("structure/t_r_nm", float(row["t_r_nm"]), epoch)
    writer.add_scalar("structure/top_L_nm", float(row["top_L_nm"]), epoch)
    writer.add_scalar("structure/top_H_nm", float(row["top_H_nm"]), epoch)
    writer.add_scalar("structure/bottom_H_nm", float(row["bottom_H_nm"]), epoch)
    writer.add_scalar("structure/bottom_L_nm", float(row["bottom_L_nm"]), epoch)
    writer.add_scalar("structure/D_min_nm", float(row["D_min_nm"]), epoch)
    writer.add_scalar("structure/D_max_nm", float(row["D_max_nm"]), epoch)

    fig = make_spectra_figure(model)
    writer.add_figure("spectra/current_0deg", fig, epoch)
    plt.close(fig)
    writer.flush()


def checkpoint_dict(
    model: AREMTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: GeometryConfig,
    wl_nm: torch.Tensor,
    settings: dict,
    epoch: int,
    best_val_mse: float,
) -> dict:
    """整理 checkpoint 内容。"""

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


def load_resume_if_needed(
    model: AREMTModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    last_path: Path,
    settings: dict,
    device: torch.device,
) -> tuple[int, float]:
    """如果打开 resume，就从 last checkpoint 接着训。"""

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
    """在终端简短打印当前结构参数。"""

    params = model.physical_parameters()
    ratio = params["ratio"].detach().cpu()
    ar = params["ar_nm"].detach().cpu()
    h_c = float(params["h_c_nm"].detach().cpu())
    t_r = float(params["t_r_nm"].detach().cpu())
    period = model.config.period_nm
    emt = emt_condition(model.config, ratio.max())
    status = "OK" if emt["ok"] else "FAIL"
    print(
        f"  h_c={h_c:.2f} nm, t_r={t_r:.2f} nm, "
        f"AR=[{ar[0]:.2f}, {ar[1]:.2f}, {ar[2]:.2f}, {ar[3]:.2f}] nm"
    )
    print(
        f"  D/P=[{float(ratio.min()):.4f}, {float(ratio.max()):.4f}], "
        f"D=[{float(ratio.min()) * period:.2f}, {float(ratio.max()) * period:.2f}] nm, "
        f"EMT={status}, margin={emt['margin_nm']:.2f} nm"
    )


def run_one_epoch(
    model: AREMTModel,
    train_cpu: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mse_fn: nn.Module,
    settings: dict,
    device: torch.device,
    epoch: int,
    n_epochs: int,
    writer: SummaryWriter | None,
) -> dict[str, float]:
    """训练一个 epoch。"""

    model.train()
    batch_size = int(settings["batch_size"])
    n_train = train_cpu.shape[0]
    n_batches = math.ceil(n_train / batch_size)
    perm = torch.randperm(n_train)

    loss_sum = 0.0
    mse_sum = 0.0
    t_mean_sum = 0.0
    grad_norm_last = 0.0
    n_seen = 0
    epoch_start = time.time()

    for batch_index, start in enumerate(range(0, n_train, batch_size), start=1):
        idx = perm[start:start + batch_size]
        batch = train_cpu[idx].to(device, non_blocking=True)
        alpha = make_alpha(batch.shape[0], settings, device)

        pred, t_matrix = model(batch, alpha, noise_max=settings["noise_max"])
        loss_mse = mse_fn(pred, batch)
        t_mean = t_matrix.mean()
        loss_trans = torch.relu(torch.tensor(settings["t_target"], device=device) - t_mean).square()
        loss = loss_mse + settings["lambda_trans"] * loss_trans

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip_norm"])
        optimizer.step()

        bs = batch.shape[0]
        loss_sum += float(loss.detach().cpu()) * bs
        mse_sum += float(loss_mse.detach().cpu()) * bs
        t_mean_sum += float(t_mean.detach().cpu()) * bs
        grad_norm_last = float(grad_norm.detach().cpu())
        n_seen += bs

        if (
            batch_index == 1
            or batch_index == n_batches
            or (settings["progress_every_batches"] > 0 and batch_index % settings["progress_every_batches"] == 0)
        ):
            global_batch = (epoch - 1) * n_batches + batch_index
            if writer is not None:
                writer.add_scalar("batch/loss", float(loss.detach().cpu()), global_batch)
                writer.add_scalar("batch/mse", float(loss_mse.detach().cpu()), global_batch)
                writer.add_scalar("batch/Tmean", float(t_mean.detach().cpu()), global_batch)
                writer.add_scalar("batch/grad_norm", grad_norm_last, global_batch)

            elapsed = time.time() - epoch_start
            pct = batch_index / n_batches * 100.0
            print(
                "\r"
                f"epoch {epoch:04d}/{n_epochs} "
                f"batch {batch_index:04d}/{n_batches:04d} ({pct:5.1f}%) | "
                f"loss={float(loss.detach().cpu()):.4e} | "
                f"mse={float(loss_mse.detach().cpu()):.4e} | "
                f"Tmean={float(t_mean.detach().cpu()):.4f} | "
                f"grad={grad_norm_last:.3e} | "
                f"elapsed={elapsed:6.1f}s",
                end="",
                flush=True,
            )
    print()

    return {
        "train_loss": loss_sum / n_seen,
        "train_mse": mse_sum / n_seen,
        "train_Tmean": t_mean_sum / n_seen,
        "grad_norm": grad_norm_last,
    }


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
        train_cpu = train_cpu.pin_memory()

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
    print(f"训练日志: {train_log_path}")
    print(f"训练曲线: {progress_plot_path}")
    print(f"TensorBoard: http://localhost:6007")
    print()

    total_start = time.time()
    for epoch in range(start_epoch, int(settings["epochs"]) + 1):
        train_info = run_one_epoch(
            model=model,
            train_cpu=train_cpu,
            optimizer=optimizer,
            mse_fn=mse_fn,
            settings=settings,
            device=device,
            epoch=epoch,
            n_epochs=int(settings["epochs"]),
            writer=writer,
        )

        if epoch % settings["eval_every_epochs"] != 0 and epoch != int(settings["epochs"]):
            continue

        val_metrics = evaluate_fixed_angle(
            model,
            val_cpu,
            angle_deg=0.0,
            batch_size=settings["eval_batch_size"],
        )
        scheduler.step(val_metrics["mse"])

        with torch.no_grad():
            t0 = model.transmission(torch.tensor([0.0], device=device))[0]
            tor = tor_percent(t0)
            t_mean_eval = float(t0.mean().detach().cpu())
            t_min_eval = float(t0.min().detach().cpu())

        params = model.physical_parameters()
        ratio_cpu = params["ratio"].detach().cpu()
        ar_cpu = params["ar_nm"].detach().cpu()
        h_c_cpu = float(params["h_c_nm"].detach().cpu())
        t_r_cpu = float(params["t_r_nm"].detach().cpu())
        emt = emt_condition(model.config, ratio_cpu.max())

        row = {
            "epoch": epoch,
            "lr_physics": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "train_loss": train_info["train_loss"],
            "train_mse": train_info["train_mse"],
            "train_Tmean": train_info["train_Tmean"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
            "val_sam": val_metrics["sam"],
            "T0_mean": t_mean_eval,
            "T0_min": t_min_eval,
            "tor_percent": tor,
            "grad_norm": train_info["grad_norm"],
            "h_c_nm": h_c_cpu,
            "t_r_nm": t_r_cpu,
            "top_L_nm": float(ar_cpu[0]),
            "top_H_nm": float(ar_cpu[1]),
            "bottom_H_nm": float(ar_cpu[2]),
            "bottom_L_nm": float(ar_cpu[3]),
            "ratio_min": float(ratio_cpu.min()),
            "ratio_max": float(ratio_cpu.max()),
            "D_min_nm": float(ratio_cpu.min()) * model.config.period_nm,
            "D_max_nm": float(ratio_cpu.max()) * model.config.period_nm,
            "emt_margin_nm": float(emt["margin_nm"]),
        }

        append_train_log(train_log_path, row)
        save_structure_csv(model, current_structure_path)
        if settings["save_live_plots"]:
            save_progress_plot(train_log_path, progress_plot_path)
            fig = make_spectra_figure(model)
            fig.savefig(spectra_plot_path, dpi=160)
            plt.close(fig)
        if writer is not None:
            write_tensorboard(writer, epoch, row, model)

        torch.save(
            checkpoint_dict(model, optimizer, scheduler, config, wl_nm, settings, epoch, best_val_mse),
            last_path,
        )

        print(
            f"epoch {epoch:04d}/{settings['epochs']} | "
            f"train_mse={row['train_mse']:.6e} | "
            f"val_mse={row['val_mse']:.6e} | "
            f"psnr={row['val_psnr']:.2f} | sam={row['val_sam']:.4f} | "
            f"T0_mean={row['T0_mean']:.4f} T0_min={row['T0_min']:.4f} | "
            f"tor={row['tor_percent']:.3f}% | grad={row['grad_norm']:.3e}"
        )
        print_structure_brief(model)
        print(f"  已更新: {train_log_path}, {current_structure_path}")

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            torch.save(
                checkpoint_dict(model, optimizer, scheduler, config, wl_nm, settings, epoch, best_val_mse),
                best_path,
            )
            save_structure_csv(model, results_dir / "ar_emt_best_structure.csv")
            print(f"  保存 best checkpoint: {best_path}")
        print()

    if writer is not None:
        writer.close()

    print(f"训练完成。last checkpoint: {last_path}")
    print(f"最佳 val_mse: {best_val_mse:.6e}")
    print(f"总耗时: {(time.time() - total_start) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
