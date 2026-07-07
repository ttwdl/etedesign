"""评估训练好的 AR-EMT checkpoint。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 04_eval_report.py

这个脚本只做最终评估：
  - 读取 checkpoints/ar_emt_best.pt；
  - 读取 test_spectra.npy；
  - 输出角度表、制造误差、结构参数和透过谱图。

注意：
  训练时保存 best 用的是 val_mse。
  这里的 test 集没有参与训练和选 best，更适合当最终结果汇报。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    evaluate_fixed_angle,
    geometry_report,
    metric_mse_psnr_sam,
    structure_rows,
    tor_percent,
)


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    "checkpoint": "checkpoints/ar_emt_best.pt",
    "data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",
    "output_dir": "results",
    "device": "cuda",

    # 0 表示用 test_spectra.npy 全部数据。
    "test_size": 0,

    # 要评估的入射角，单位是度。
    "angles_deg": [0.0, 0.5, 2.0, 5.0, 8.0, 10.0],

    # 制造误差 Monte Carlo 次数。想快一点可以改成 5，正式汇报可以改成 50。
    "mc": 20,
    "seed": 2026,
    "batch_size": 4096,
}


def save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """保存 CSV 表格。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_test_data(data_dir: Path, test_size: int) -> torch.Tensor:
    """读取测试集。"""

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
    """读取 checkpoint 并恢复模型。"""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GeometryConfig(**ckpt["config"])
    wl_nm = ckpt["wl_nm"].to(dtype=torch.float32)
    model = AREMTModel(wl_nm, config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(geometry_report(config))
    print(f"读取 checkpoint: {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch')}, best_val_mse={ckpt.get('best_val_mse')}")
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
    """用指定结构参数评估。

    制造误差 MC 会扰动结构参数，但 decoder 权重不变。
    """

    device = next(model.parameters()).device
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        t = model.transmission(
            torch.tensor([angle_deg], device=device),
            ratio_override=ratio,
            h_c_override=h_c,
            t_r_override=t_r,
            ar_override=ar,
        )[0]
        for start in range(0, spectra.shape[0], batch_size):
            batch = spectra[start:start + batch_size].to(device)
            meas = model.measure(batch, t)
            pred = model.decoder(meas)
            preds.append(pred.cpu())
            targets.append(batch.cpu())
    return metric_mse_psnr_sam(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def run_mc_fabrication(model: AREMTModel, spectra: torch.Tensor, n_mc: int, seed: int, batch_size: int) -> dict[str, float]:
    """制造误差 Monte Carlo。

    当前扰动设置：
    - D: ±3 nm；
    - h_c: ±2 nm；
    - t_r: ±5 nm；
    - AR 四层: ±2 nm。

    这些数值只是第一版工艺误差假设，以后可以按真实工艺能力修改。
    """

    if n_mc <= 0:
        return {"mse": float("nan"), "psnr": float("nan"), "sam": float("nan")}

    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    params = model.physical_parameters()
    base_ratio = params["ratio"].detach()
    base_hc = params["h_c_nm"].detach()
    base_tr = params["t_r_nm"].detach()
    base_ar = params["ar_nm"].detach()
    period = model.config.period_nm

    rows = []
    for _ in range(n_mc):
        d_delta = (torch.rand(base_ratio.shape, generator=gen, device=device) * 6.0 - 3.0) / period
        ratio = torch.clamp(base_ratio + d_delta, model.r_min, model.r_max)
        h_c = torch.clamp(base_hc + (torch.rand((), generator=gen, device=device) * 4.0 - 2.0), *model.h_c_range)
        t_r = torch.clamp(base_tr + (torch.rand((), generator=gen, device=device) * 10.0 - 5.0), *model.t_r_range)
        ar = torch.clamp(base_ar + (torch.rand(base_ar.shape, generator=gen, device=device) * 4.0 - 2.0), *model.ar_range)
        rows.append(evaluate_with_overrides(model, spectra, 0.0, ratio, h_c, t_r, ar, batch_size))

    return {
        "mse": float(np.mean([r["mse"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "sam": float(np.mean([r["sam"] for r in rows])),
    }


def plot_spectra(model: AREMTModel, output_dir: Path) -> None:
    """保存透过谱图。"""

    device = next(model.parameters()).device
    wl = model.wl_nm.detach().cpu().numpy()
    with torch.no_grad():
        t0 = model.transmission(torch.tensor([0.0], device=device))[0].detach().cpu()
        t5 = model.transmission(torch.tensor([5.0], device=device))[0].detach().cpu()

    plt.figure(figsize=(9, 5))
    for idx in range(t0.shape[0]):
        plt.plot(wl, t0[idx], lw=1.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("AR-EMT spectra, alpha=0 deg")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "eval_spectra_0deg.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for idx in range(t0.shape[0]):
        plt.plot(wl, t0[idx], lw=1.0, alpha=0.9)
        plt.plot(wl, t5[idx], lw=0.8, alpha=0.45, ls="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("AR-EMT spectra, solid=0 deg, dashed=5 deg")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "eval_spectra_0deg_5deg.png", dpi=180)
    plt.close()


def main() -> None:
    settings = USER_SETTINGS
    device = torch.device(settings["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _ckpt = load_model(Path(settings["checkpoint"]), device)
    test = load_test_data(Path(settings["data_dir"]), settings["test_size"])

    angle_rows = []
    for angle in settings["angles_deg"]:
        metrics = evaluate_fixed_angle(model, test, angle_deg=angle, batch_size=settings["batch_size"])
        with torch.no_grad():
            t = model.transmission(torch.tensor([angle], device=device))[0]
            row = {
                "angle_deg": angle,
                "mse": metrics["mse"],
                "psnr": metrics["psnr"],
                "sam": metrics["sam"],
                "T_mean": float(t.mean().cpu()),
                "T_min": float(t.min().cpu()),
                "T_peak_median": float(torch.median(t.max(dim=1).values).cpu()),
                "tor_percent": tor_percent(t),
            }
        angle_rows.append(row)
        print(
            f"angle={angle:4.1f} deg | mse={row['mse']:.6e} | psnr={row['psnr']:.2f} | "
            f"sam={row['sam']:.4f} | T_mean={row['T_mean']:.4f} | tor={row['tor_percent']:.3f}%"
        )

    save_csv(angle_rows, output_dir / "eval_angles.csv")
    save_csv(structure_rows(model), output_dir / "eval_structure.csv")
    plot_spectra(model, output_dir)

    mc_metrics = run_mc_fabrication(
        model,
        test,
        n_mc=settings["mc"],
        seed=settings["seed"] + 100,
        batch_size=settings["batch_size"],
    )
    save_csv([{"mc_count": settings["mc"], **mc_metrics}], output_dir / "eval_fabrication_mc.csv")

    print()
    print(
        f"制造扰动 MC({settings['mc']}) | mse={mc_metrics['mse']:.6e} | "
        f"psnr={mc_metrics['psnr']:.2f} | sam={mc_metrics['sam']:.4f}"
    )
    print(f"评估结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
