"""评估训练好的 AR-EMT checkpoint（最终结果汇报用）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 04_eval_report.py

它会读取 checkpoints_16ch_t06_tor20_150/ar_emt_best.pt 和 test_spectra.npy，然后输出：
  - 不同入射角下的重建精度表（eval_angles.csv）；
  - 结构参数表（eval_structure.csv）；
  - 制造误差 Monte Carlo（eval_fabrication_mc.csv）；
  - 测量噪声鲁棒性（eval_noise_robustness.csv）；
  - 透过谱图。

为什么用 test 集：
  训练时选 best 看的是 val_mse；test 集没参与训练、也没参与选 best，
  所以用它当“最终、诚实”的成绩最合适。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

# Windows + conda 里，PyTorch / NumPy / SciPy / Matplotlib 有时会重复加载 Intel OpenMP。
# 如果不提前设置，可能出现 “OMP: Error #15: Initializing libiomp5md.dll”。
# 这行只影响当前脚本进程；以后如果你重装环境彻底解决冲突，可以删掉它。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    add_measurement_noise,
    evaluate_fixed_angle,
    geometry_report,
    metric_mse_psnr_sam,
    model_kwargs_from_settings,
    structure_rows,
    tor_percent,
)


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    "checkpoint": "checkpoints_16ch_t06_tor20_150/ar_emt_best.pt",
    "data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",
    "output_dir": "results_16ch_t06_tor20_150",
    "device": "cuda",

    "test_size": 0,   # 0 表示用 test_spectra.npy 的全部数据

    # 要评估的入射角(度)
    "angles_deg": [0.0, 0.5, 2.0, 5.0, 8.0, 10.0],

    # 制造误差 Monte Carlo 次数。想快点改 5，正式汇报改 50。
    "mc": 20,

    # 测量噪声鲁棒性：分别在这些“相对噪声”水平下评估重建(0=无噪声做基准)。
    "noise_eval_levels": [0.0, 0.01, 0.02, 0.05],

    "seed": 2026,
    "batch_size": 4096,
}


def save_csv(rows: list[dict], path: Path) -> None:
    """保存 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_test_data(data_dir: Path, test_size: int) -> torch.Tensor:
    """读取 test 集。"""

    path = data_dir / "test_spectra.npy"
    if not path.exists():
        raise FileNotFoundError(f"找不到测试集: {path}，请先运行 02_prepare_data.py")
    data = torch.from_numpy(np.load(path).astype(np.float32))
    if test_size > 0:
        data = data[:test_size]
    print(f"读取 test 数据: {path}, shape={tuple(data.shape)}")
    print(f"  min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    return data


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[AREMTModel, dict]:
    """读取 checkpoint 并恢复模型（结构参数 + 解码器权重）。"""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GeometryConfig(**ckpt["config"])
    wl_nm = ckpt["wl_nm"].to(dtype=torch.float32)
    model = AREMTModel(wl_nm, config, **model_kwargs_from_settings(ckpt.get("settings", {}))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(geometry_report(config))
    print(f"读取 checkpoint: {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch')}, best_val_mse={ckpt.get('best_val_mse')}, "
          f"best_val_score={ckpt.get('best_val_score')}")
    print()
    return model, ckpt


def evaluate_with_overrides(
    model: AREMTModel,
    spectra: torch.Tensor,
    angle_deg: float,
    ratio: torch.Tensor,
    h_c: torch.Tensor,
    t_r: torch.Tensor,
    ar: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    """用“临时指定的结构参数”评估（供制造误差 MC 用）。

    注意：这里只扰动物理结构参数（模拟加工误差），解码器权重不变。
    """

    device = next(model.parameters()).device
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        t = model.transmission(
            torch.tensor([angle_deg], device=device),
            ratio_override=ratio, h_c_override=h_c, t_r_override=t_r, ar_override=ar,
        )[0]
        for start in range(0, spectra.shape[0], batch_size):
            batch = spectra[start:start + batch_size].to(device)
            meas = model.measure(batch, t)
            pred = model.decoder(meas)
            preds.append(pred.cpu())
            targets.append(batch.cpu())
    return metric_mse_psnr_sam(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def run_mc_fabrication(model: AREMTModel, spectra: torch.Tensor, n_mc: int, seed: int, batch_size: int) -> dict[str, float]:
    """制造误差 Monte Carlo：随机扰动结构参数很多次，看重建平均掉多少。

    当前扰动幅度（第一版工艺误差假设，以后按真实工艺能力改）：
    - D    : ±3 nm
    - H_total: ±5 nm（全局）
    - h_c  : ±2 nm，随后令 t_r = H_total - h_c，保持总厚度平整，并重新检查深宽比
    - AR 4层: ±2 nm
    """

    if n_mc <= 0:
        return {"mse": float("nan"), "l1": float("nan"), "diff_l1": float("nan"),
                "psnr": float("nan"), "sam": float("nan")}

    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    params = model.physical_parameters()
    base_ratio = params["ratio"].detach()
    base_hc = params["h_c_nm"].detach()
    base_core = params["core_total_nm"].detach()
    base_ar = params["ar_nm"].detach()
    period = model.config.period_nm

    rows = []
    for _ in range(n_mc):
        # D 的 ±3nm 扰动换算成 D/P 的扰动(除以周期)
        d_delta = (torch.rand(base_ratio.shape, generator=gen, device=device) * 6.0 - 3.0) / period
        ratio = torch.clamp(base_ratio + d_delta, model.r_min, model.r_max)
        core_noise = torch.rand((), generator=gen, device=device) * 10.0 - 5.0
        core_total = torch.clamp(base_core + core_noise, *model.core_total_range)
        h_c_noise = torch.rand(base_hc.shape, generator=gen, device=device) * 4.0 - 2.0
        h_low, h_high = model.h_c_bounds(ratio, core_total)
        h_c = torch.minimum(torch.maximum(base_hc + h_c_noise, h_low), h_high)
        t_r = core_total - h_c
        ar = torch.clamp(base_ar + (torch.rand(base_ar.shape, generator=gen, device=device) * 4.0 - 2.0), *model.ar_range)
        rows.append(evaluate_with_overrides(model, spectra, 0.0, ratio, h_c, t_r, ar, batch_size))

    return {
        "mse": float(np.mean([r["mse"] for r in rows])),
        "l1": float(np.mean([r["l1"] for r in rows])),
        "diff_l1": float(np.mean([r["diff_l1"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "sam": float(np.mean([r["sam"] for r in rows])),
    }


def run_noise_robustness(model: AREMTModel, spectra: torch.Tensor, levels: list[float],
                         seed: int, batch_size: int) -> list[dict[str, float]]:
    """测量噪声鲁棒性：在几个“相对噪声”水平下评估重建(0=无噪声基准)。

    因为训练时加了噪声，这里应看到即使噪声升高，重建也不会立刻崩，
    说明模型学到的是“抗噪的还原”，而不是死记硬背干净测量。
    """

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    rows = []
    with torch.no_grad():
        t = model.transmission(torch.tensor([0.0], device=device))[0]
        for rel in levels:
            preds, targets = [], []
            for start in range(0, spectra.shape[0], batch_size):
                batch = spectra[start:start + batch_size].to(device)
                meas = model.measure(batch, t)
                meas = add_measurement_noise(meas, rel_sigma=rel, abs_sigma=0.0)  # 测试时加噪
                pred = model.decoder(meas)
                preds.append(pred.cpu())
                targets.append(batch.cpu())
            m = metric_mse_psnr_sam(torch.cat(preds, 0), torch.cat(targets, 0))
            rows.append({"noise_rel": rel, **m})
            print(f"  noise_rel={rel:5.3f} | mse={m['mse']:.6e} | psnr={m['psnr']:.2f} | sam={m['sam']:.4f}")
    return rows


def plot_spectra(model: AREMTModel, output_dir: Path) -> None:
    """存两张透过谱图：0 度单独一张；0 度实线 + 5 度虚线叠一张。"""

    device = next(model.parameters()).device
    wl = model.wl_nm.detach().cpu().numpy()
    with torch.no_grad():
        t0 = model.transmission(torch.tensor([0.0], device=device))[0].detach().cpu()
        t5 = model.transmission(torch.tensor([5.0], device=device))[0].detach().cpu()

    plt.figure(figsize=(9, 5))
    for idx in range(t0.shape[0]):
        plt.plot(wl, t0[idx], lw=1.0)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Transmission")
    plt.title("AR-EMT spectra, alpha=0 deg"); plt.ylim(0.0, 1.05)
    plt.tight_layout(); plt.savefig(output_dir / "eval_spectra_0deg.png", dpi=180); plt.close()

    plt.figure(figsize=(9, 5))
    for idx in range(t0.shape[0]):
        plt.plot(wl, t0[idx], lw=1.0, alpha=0.9)
        plt.plot(wl, t5[idx], lw=0.8, alpha=0.45, ls="--")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Transmission")
    plt.title("AR-EMT spectra, solid=0 deg, dashed=5 deg"); plt.ylim(0.0, 1.05)
    plt.tight_layout(); plt.savefig(output_dir / "eval_spectra_0deg_5deg.png", dpi=180); plt.close()


def main() -> None:
    settings = USER_SETTINGS
    device = torch.device(settings["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ckpt = load_model(Path(settings["checkpoint"]), device)
    test = load_test_data(Path(settings["data_dir"]), settings["test_size"])

    # ---- 角度表：不同入射角下的重建精度 + 透过率 ----
    angle_rows = []
    for angle in settings["angles_deg"]:
        metrics = evaluate_fixed_angle(model, test, angle_deg=angle, batch_size=settings["batch_size"])
        with torch.no_grad():
            t = model.transmission(torch.tensor([angle], device=device))[0]
            row = {
                "angle_deg": angle,
                "mse": metrics["mse"], "l1": metrics["l1"], "diff_l1": metrics["diff_l1"],
                "psnr": metrics["psnr"], "sam": metrics["sam"],
                "T_mean": float(t.mean().cpu()), "T_min": float(t.min().cpu()),
                "T_peak_median": float(torch.median(t.max(dim=1).values).cpu()),
                "tor_percent": tor_percent(t),
            }
        angle_rows.append(row)
        print(f"angle={angle:4.1f} deg | mse={row['mse']:.6e} | l1={row['l1']:.6e} | "
              f"diff={row['diff_l1']:.6e} | psnr={row['psnr']:.2f} | "
              f"sam={row['sam']:.4f} | T_mean={row['T_mean']:.4f} | tor={row['tor_percent']:.3f}%")

    save_csv(angle_rows, output_dir / "eval_angles.csv")
    save_csv(structure_rows(model), output_dir / "eval_structure.csv")
    plot_spectra(model, output_dir)

    # ---- 制造误差 MC ----
    print()
    print(f"制造误差 Monte Carlo (mc={settings['mc']})：")
    mc_metrics = run_mc_fabrication(model, test, n_mc=settings["mc"], seed=settings["seed"] + 100,
                                    batch_size=settings["batch_size"])
    save_csv([{"mc_count": settings["mc"], **mc_metrics}], output_dir / "eval_fabrication_mc.csv")
    print(f"  平均 | mse={mc_metrics['mse']:.6e} | l1={mc_metrics['l1']:.6e} | "
          f"diff={mc_metrics['diff_l1']:.6e} | psnr={mc_metrics['psnr']:.2f} | sam={mc_metrics['sam']:.4f}")

    # ---- 测量噪声鲁棒性 ----
    print()
    print("测量噪声鲁棒性：")
    noise_rows = run_noise_robustness(model, test, settings["noise_eval_levels"],
                                      seed=settings["seed"] + 200, batch_size=settings["batch_size"])
    save_csv(noise_rows, output_dir / "eval_noise_robustness.csv")

    print()
    print(f"评估结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
