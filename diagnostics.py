"""Phase 0 诊断：判断瓶颈在滤光片编码器 Φ，还是在 MLP 解码器。

直接运行：
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' diagnostics.py

这个脚本只做“诊断”，不训练新模型、不改 loss、不覆盖已有 checkpoint/results。

它会比较 16 / 25 / 36 通道三个已经训练好的模型：
  1. 训练集 PCA：看看 CAVE 光谱本身到底有多少自由度；
  2. Φ 有效秩：看看滤光片矩阵实际提供了多少互补测量；
  3. 线性 pinv 重建：不用 MLP，只用最小二乘从测量值反推光谱；
  4. MLP vs pinv：如果 MLP 只比 pinv 好一点，说明主要瓶颈在编码器；
  5. subspace 残差：看 Φ 的行空间能覆盖多少真实光谱变化方向。

输出目录：
  diagnostics_phase0_16_25_36/
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

# Windows + conda 里，PyTorch / NumPy / SciPy / Matplotlib 有时会重复加载 Intel OpenMP。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    measurement_matrix_coherence,
    metric_mse_psnr_sam,
    model_kwargs_from_settings,
    tor_percent,
)


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    # 02_prepare_data.py 生成的场景级 train/val/test 缓存。
    "data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",

    # 本脚本输出目录。这里是诊断结果，不是训练结果。
    "output_dir": "diagnostics_phase0_16_25_36",

    # 每次 pinv / subspace 诊断最多用多少条 test 光谱。
    # 0 表示全部使用。当前 test=10000，很小，可以全用。
    "test_size": 0,

    # PCA 只用 train。当前 train=100000，151 维，SVD 可以直接跑。
    "pca_top_k": [3, 5, 8, 10, 15, 20, 25, 31],
    "pca_thresholds": [0.99, 0.995, 0.999],

    # subspace 投影里 G=Phi Phi^T 可能病态，所以加一个很小的对角正则。
    "subspace_eps": 1e-4,

    # 三个正式模型。注意：这里不纳入刚才中止的 coh020 半程实验。
    "models": [
        {
            "name": "16ch_t06_tor20_150",
            "channels": 16,
            "checkpoint": "checkpoints_16ch_t06_tor20_150/ar_emt_best.pt",
            "results_dir": "results_16ch_t06_tor20_150",
        },
        {
            "name": "25ch_t06_tor20_150",
            "channels": 25,
            "checkpoint": "checkpoints_25ch_t06_tor20_150/ar_emt_best.pt",
            "results_dir": "results_25ch_t06_tor20_150",
        },
        {
            "name": "36ch_t06_tor15_150",
            "channels": 36,
            "checkpoint": "checkpoints_36ch_t06_tor15_150/ar_emt_best.pt",
            "results_dir": "results_36ch_t06_tor15_150",
        },
    ],
}


# =============================================================================
# 基础工具：读写 CSV / 数据 / checkpoint
# =============================================================================


def save_csv(rows: list[dict], path: Path) -> None:
    """保存 CSV。rows 是一行一个 dict。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    """读取 CSV；如果文件不存在，返回空列表。"""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_spectra_cache(data_dir: Path, test_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 train/test/wl。

    train 只用于 PCA 和 subspace 诊断；
    test 用于 pinv 线性重建诊断；
    wl 是 400-700 nm 的 151 个波长点。
    """

    train_path = data_dir / "train_spectra.npy"
    test_path = data_dir / "test_spectra.npy"
    wl_path = data_dir / "wl_nm.npy"
    missing = [p for p in [train_path, test_path, wl_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("数据缓存不完整，缺少：\n" + "\n".join(str(p) for p in missing))

    train = np.load(train_path).astype(np.float32)
    test = np.load(test_path).astype(np.float32)
    wl_nm = np.load(wl_path).astype(np.float32)
    if test_size > 0:
        test = test[:test_size]

    print(f"读取数据缓存: {data_dir}")
    print(f"  train: {train.shape}, min={train.min():.4f}, max={train.max():.4f}, mean={train.mean():.4f}")
    print(f"  test : {test.shape}, min={test.min():.4f}, max={test.max():.4f}, mean={test.mean():.4f}")
    print(f"  wl_nm: {wl_nm.shape}")
    return train, test, wl_nm


def load_model_and_phi(checkpoint_path: Path) -> tuple[AREMTModel, torch.Tensor, dict]:
    """读取 checkpoint，并提取 0 度下的滤光片矩阵 Phi。

    Phi 的 shape 是 [C, 151]：
      C 是滤光片通道数；
      151 是 400-700 nm 每 2 nm 一个采样点。
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GeometryConfig(**ckpt["config"])
    wl_nm = ckpt["wl_nm"].to(dtype=torch.float32)
    model = AREMTModel(wl_nm, config, **model_kwargs_from_settings(ckpt.get("settings", {}))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        phi = model.transmission(torch.tensor([0.0], dtype=torch.float32))[0].detach().cpu()
    return model, phi, ckpt


def read_mlp_angle0(results_dir: Path) -> dict:
    """读取 04_eval_report.py 已经算好的 0 度 MLP test 指标。"""

    rows = read_csv_rows(results_dir / "eval_angles.csv")
    for row in rows:
        if abs(float(row["angle_deg"]) - 0.0) < 1e-9:
            return row
    raise RuntimeError(f"{results_dir / 'eval_angles.csv'} 中没有 angle_deg=0 的行。")


def to_float(row: dict, key: str) -> float:
    """CSV 里所有值都是字符串，这里转成 float。"""

    return float(row[key])


# =============================================================================
# 诊断 1：训练光谱 PCA
# =============================================================================


def run_pca(train: np.ndarray, top_k: list[int], thresholds: list[float]) -> tuple[list[dict], np.ndarray, dict]:
    """对训练光谱做 PCA 维度诊断。

    这里不需要显式求 PCA 主成分，只要看奇异值能解释多少方差。
    train shape: [N, 151]
    """

    print()
    print("PCA 诊断：正在对训练集做 SVD ...")
    centered = train - train.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    var = sv ** 2
    ratio = np.cumsum(var) / np.sum(var)

    rows = []
    for k in top_k:
        idx = min(k, ratio.shape[0]) - 1
        rows.append({"top_k": k, "explained_variance": float(ratio[idx])})
        print(f"  top-{k:2d} PCA 解释方差 = {ratio[idx] * 100:.3f}%")

    threshold_info = {}
    for th in thresholds:
        need = int(np.searchsorted(ratio, th) + 1)
        threshold_info[f"k_for_{th:.3f}"] = need
        print(f"  达到 {th * 100:.1f}% 方差需要 k = {need}")

    return rows, ratio, threshold_info


def plot_pca_curve(ratio: np.ndarray, output_path: Path) -> None:
    """画 PCA 累计解释方差曲线。"""

    x = np.arange(1, ratio.shape[0] + 1)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(x, ratio * 100.0, marker="o", ms=3, lw=1.5)
    for y in [99.0, 99.5, 99.9]:
        plt.axhline(y, ls="--", lw=0.9, alpha=0.6)
        plt.text(x[-1], y, f" {y:.1f}%", va="center", fontsize=9)
    plt.xlabel("PCA 成分数 k")
    plt.ylabel("累计解释方差 (%)")
    plt.title("训练光谱 PCA 维度诊断")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# =============================================================================
# 诊断 2：Phi 有效秩 / pinv 线性重建 / subspace 残差
# =============================================================================


def effective_rank(singular_values: np.ndarray) -> float:
    """有效秩：把奇异值当成概率分布，用熵估计“等效有几个维度”。"""

    s = singular_values.astype(np.float64)
    p = s / (np.sum(s) + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def numerical_rank(singular_values: np.ndarray, shape: tuple[int, int]) -> int:
    """数值秩：比有效秩更硬的 rank 判断。"""

    if singular_values.size == 0:
        return 0
    tol = singular_values.max() * max(shape) * np.finfo(np.float64).eps
    return int(np.sum(singular_values > tol))


def subspace_residual_np(spectra: np.ndarray, phi: np.ndarray, eps: float) -> float:
    """把光谱投影到 row(Phi)，计算平均 MSE 残差。

    如果这个值很大，说明滤光片行空间没有覆盖真实光谱变化方向；
    后续的 subspace_residual_loss 就是在训练时直接压这个量。
    """

    phi64 = phi.astype(np.float64)
    s64 = spectra.astype(np.float64)
    g = phi64 @ phi64.T + eps * np.eye(phi64.shape[0], dtype=np.float64)
    basis = np.linalg.solve(g, phi64)          # [C,151]
    proj = (s64 @ phi64.T) @ basis             # [N,151]
    return float(np.mean((s64 - proj) ** 2))


def pinv_reconstruction_metrics(test: np.ndarray, phi: np.ndarray) -> dict[str, float]:
    """不用 MLP，只用 Phi 的 Moore-Penrose 伪逆做线性重建。

    y = S @ Phi.T
    S_pinv = y @ pinv(Phi).T
    """

    y = test @ phi.T
    recon = y @ np.linalg.pinv(phi).T
    recon = recon.astype(np.float32)
    return metric_mse_psnr_sam(torch.from_numpy(recon), torch.from_numpy(test.astype(np.float32)))


def diagnose_one_model(item: dict, train: np.ndarray, test: np.ndarray, eps: float) -> tuple[dict, np.ndarray]:
    """诊断一个 checkpoint，返回汇总行和奇异值。"""

    print()
    print(f"诊断模型: {item['name']}")
    model, phi_torch, ckpt = load_model_and_phi(Path(item["checkpoint"]))
    phi = phi_torch.numpy().astype(np.float64)
    c, n_wl = phi.shape

    singular_values = np.linalg.svd(phi, compute_uv=False)
    eff_rank = effective_rank(singular_values)
    num_rank = numerical_rank(singular_values, phi.shape)
    cond = float(singular_values.max() / max(singular_values.min(), 1e-12))
    coherence = float(measurement_matrix_coherence(phi_torch).detach().cpu())
    tor = tor_percent(phi_torch)

    pinv_metrics = pinv_reconstruction_metrics(test, phi.astype(np.float32))
    mlp = read_mlp_angle0(Path(item["results_dir"]))

    train_subspace = subspace_residual_np(train, phi, eps)
    test_subspace = subspace_residual_np(test, phi, eps)

    mlp_mse = to_float(mlp, "mse")
    pinv_mse = pinv_metrics["mse"]
    improvement = (pinv_mse - mlp_mse) / max(pinv_mse, 1e-12)

    row = {
        "name": item["name"],
        "channels": c,
        "checkpoint_epoch": ckpt.get("epoch", ""),
        "phi_shape": f"{c}x{n_wl}",
        "phi_effective_rank": eff_rank,
        "phi_numerical_rank": num_rank,
        "phi_condition_number": cond,
        "phi_coherence": coherence,
        "phi_tor_percent": tor,
        "phi_singular_min": float(singular_values.min()),
        "phi_singular_max": float(singular_values.max()),
        "train_subspace_residual": train_subspace,
        "test_subspace_residual": test_subspace,
        "mlp_test_mse": mlp_mse,
        "mlp_test_l1": to_float(mlp, "l1"),
        "mlp_test_diff_l1": to_float(mlp, "diff_l1"),
        "mlp_test_psnr": to_float(mlp, "psnr"),
        "mlp_test_sam": to_float(mlp, "sam"),
        "mlp_T_mean": to_float(mlp, "T_mean"),
        "mlp_tor_percent": to_float(mlp, "tor_percent"),
        "pinv_test_mse": pinv_metrics["mse"],
        "pinv_test_l1": pinv_metrics["l1"],
        "pinv_test_diff_l1": pinv_metrics["diff_l1"],
        "pinv_test_psnr": pinv_metrics["psnr"],
        "pinv_test_sam": pinv_metrics["sam"],
        "mlp_mse_improvement_over_pinv": improvement,
    }

    print(f"  Phi shape = {row['phi_shape']}")
    print(f"  effective rank = {eff_rank:.3f} / {c}, cond = {cond:.3e}")
    print(f"  coherence = {coherence:.4f}, tor = {tor:.3f}%")
    print(f"  MLP mse = {mlp_mse:.6e}, pinv mse = {pinv_mse:.6e}, MLP 相对提升 = {improvement * 100:.2f}%")
    print(f"  test subspace residual = {test_subspace:.6e}")
    return row, singular_values


def plot_singular_values(sv_by_name: dict[str, np.ndarray], output_path: Path) -> None:
    """把三组模型的 Phi 奇异值画在一张图里。"""

    plt.figure(figsize=(7.6, 4.8))
    for name, sv in sv_by_name.items():
        x = np.arange(1, len(sv) + 1)
        plt.semilogy(x, sv, marker="o", ms=3, lw=1.4, label=name)
    plt.xlabel("奇异值序号")
    plt.ylabel("奇异值（对数坐标）")
    plt.title("滤光片矩阵 Φ 的奇异值谱")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# =============================================================================
# Markdown 汇报
# =============================================================================


def conclusion_text(model_rows: list[dict], threshold_info: dict) -> tuple[str, str]:
    """根据诊断数字生成简短结论和 Phase 1 建议。"""

    best_mlp = min(model_rows, key=lambda r: r["mlp_test_mse"])
    best_pinv = min(model_rows, key=lambda r: r["pinv_test_mse"])
    row25 = next(r for r in model_rows if r["channels"] == 25)
    row36 = next(r for r in model_rows if r["channels"] == 36)
    k99 = threshold_info.get("k_for_0.990", "")

    parts = []
    parts.append(f"PCA 结果显示，训练光谱达到 99% 解释方差需要约 {k99} 个主成分。")
    parts.append(f"MLP 正式 test MSE 最好的是 {best_mlp['name']}，MSE={best_mlp['mlp_test_mse']:.6e}。")
    parts.append(f"pinv 线性重建最好的是 {best_pinv['name']}，MSE={best_pinv['pinv_test_mse']:.6e}。")

    if row36["phi_effective_rank"] <= row25["phi_effective_rank"] * 1.10:
        parts.append("36 通道的有效秩没有随通道数明显提高，说明更多通道没有带来等比例的新信息。")
    else:
        parts.append("36 通道有效秩有提高，但正式 MLP 重建仍不如 25 通道，说明通道数量不是唯一因素。")
    if row36["phi_condition_number"] > row25["phi_condition_number"] * 10.0:
        parts.append("36 通道条件数远高于 25 通道，说明它虽然通道更多，但反演更病态、更不稳。")

    if row25["mlp_mse_improvement_over_pinv"] < 0.0:
        advice = (
            "建议 Phase 1 仍优先做 subspace_residual，但结论要更准确："
            "25ch 的 clean 0° pinv 已经低于 MLP，说明 Φ 行空间本身有潜力，"
            "当前瓶颈不只是“编码器不够”，也包括 MLP/训练目标/带噪随机角训练没有吃满线性上限。"
        )
    elif row25["mlp_mse_improvement_over_pinv"] < 0.20:
        advice = "建议 Phase 1 优先做 subspace_residual：MLP 相对 pinv 提升有限，主要瓶颈更像在编码器 Φ。"
    else:
        advice = "建议 Phase 1 仍先做 subspace_residual，但要注意 MLP 先验确实有贡献；暂不进入 Phase 3。"
    return " ".join(parts), advice


def write_report(
    output_dir: Path,
    pca_rows: list[dict],
    threshold_info: dict,
    model_rows: list[dict],
) -> None:
    """写 diag_report.md。"""

    summary, advice = conclusion_text(model_rows, threshold_info)
    lines = [
        "# Phase 0 诊断报告：16/25/36 通道编码器瓶颈分析",
        "",
        "本报告只做诊断：没有改 loss，没有训练新模型，没有用 test 集选择 checkpoint。",
        "",
        "## 总结",
        "",
        summary,
        "",
        advice,
        "",
        "## PCA：训练光谱内在维度",
        "",
        "| top-k | 累计解释方差 |",
        "| ---: | ---: |",
    ]
    for row in pca_rows:
        lines.append(f"| {row['top_k']} | {row['explained_variance'] * 100:.3f}% |")

    lines.extend([
        "",
        "| 阈值 | 需要 PCA 成分数 |",
        "| ---: | ---: |",
    ])
    for key, value in threshold_info.items():
        th = key.replace("k_for_", "")
        lines.append(f"| {float(th) * 100:.1f}% | {value} |")

    lines.extend([
        "",
        "## 模型诊断总表",
        "",
        "| 模型 | C | MLP MSE | pinv MSE | MLP相对pinv提升 | MLP SAM | pinv SAM | T_mean | Φ有效秩 | Φ条件数 | coherence | tor% | test subspace残差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in model_rows:
        lines.append(
            f"| {row['name']} | {row['channels']} | "
            f"{row['mlp_test_mse']:.6e} | {row['pinv_test_mse']:.6e} | "
            f"{row['mlp_mse_improvement_over_pinv'] * 100:.2f}% | "
            f"{row['mlp_test_sam']:.5f} | {row['pinv_test_sam']:.5f} | "
            f"{row['mlp_T_mean']:.5f} | {row['phi_effective_rank']:.2f} | "
            f"{row['phi_condition_number']:.2e} | {row['phi_coherence']:.5f} | "
            f"{row['phi_tor_percent']:.3f} | {row['test_subspace_residual']:.6e} |"
        )

    lines.extend([
        "",
        "表中 `MLP相对pinv提升` 的含义：正数表示 MLP 比 pinv 更好；负数表示 clean 0° pinv 反而更好。",
        "",
        "关键解读：36ch 的 `test subspace残差` 很低，但 `Φ条件数` 极大、coherence 也最高，说明它的行空间看似覆盖了数据方向，实际测量反演非常病态。这能解释为什么 36 通道没有带来更好的 MLP test MSE。",
        "",
        "## 决策门",
        "",
        "- 如果 MLP 只比 pinv 好一点：主要瓶颈在编码器 Φ，Phase 1 应优先做 `subspace_residual`。",
        "- 如果 pinv 反而比 MLP 好：说明 Φ 的线性信息量有潜力，但当前 MLP/训练目标/鲁棒训练没有吃满这个上限。",
        "- 如果 36 通道有效秩没有明显增加：说明多通道发生冗余，盲目加通道不是主方向。",
        "- Phase 3 空间网络不进入本轮修改，必须等 Phase 0/1/2 汇报后再单独确认。",
        "",
        "## 输出文件",
        "",
        "- `pca_explained_variance.csv`：PCA top-k 方差解释率。",
        "- `model_diagnostics.csv`：每个模型的 MLP、pinv、Φ 有效秩和 subspace 残差。",
        "- `pca_curve.png`：PCA 累计解释方差曲线。",
        "- `phi_singular_values.png`：三组 Φ 的奇异值谱。",
    ])

    (output_dir / "diag_report.md").write_text("\n".join(lines), encoding="utf-8")


def append_experiment_log(model_rows: list[dict]) -> None:
    """把 Phase 0 摘要追加到 experiment_log.md。

    这里的“外部 MSE”没有做，因为 Phase 0 只诊断 train/test cache 和已有 eval。
    """

    path = Path("experiment_log.md")
    section_title = "## Phase 0 诊断：16/25/36 通道编码器瓶颈分析"
    lines = [
        "",
        section_title,
        "",
        "| 实验名 | 改了什么(单变量) | test MSE | test L1 | test SAM | PSNR | 外部 MSE | T_mean | Φ 有效秩 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in model_rows:
        conclusion = (
            f"pinv MSE={row['pinv_test_mse']:.2e}; "
            f"coh={row['phi_coherence']:.3f}; "
            f"subspace={row['test_subspace_residual']:.2e}"
        )
        lines.append(
            f"| {row['name']} | Phase 0 诊断，不改训练 | "
            f"{row['mlp_test_mse']:.6e} | {row['mlp_test_l1']:.6e} | "
            f"{row['mlp_test_sam']:.5f} | {row['mlp_test_psnr']:.2f} | N/A | "
            f"{row['mlp_T_mean']:.5f} | {row['phi_effective_rank']:.2f} | {conclusion} |"
        )
    lines.append("")
    lines.append("Phase 0 诊断输出目录：`diagnostics_phase0_16_25_36/`。")
    lines.append("")

    new_section = "\n".join(lines)
    if path.exists():
        old_text = path.read_text(encoding="utf-8")
        start = old_text.find(section_title)
        if start >= 0:
            next_start = old_text.find("\n## ", start + len(section_title))
            if next_start >= 0:
                old_text = old_text[:start].rstrip() + "\n" + old_text[next_start:].lstrip()
            else:
                old_text = old_text[:start].rstrip()
        text = old_text.rstrip() + "\n" + new_section
    else:
        text = new_section.lstrip()
    path.write_text(text, encoding="utf-8")


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    settings = USER_SETTINGS
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, _wl_nm = load_spectra_cache(Path(settings["data_dir"]), int(settings["test_size"]))

    pca_rows, pca_curve, threshold_info = run_pca(
        train,
        top_k=list(settings["pca_top_k"]),
        thresholds=list(settings["pca_thresholds"]),
    )
    save_csv(pca_rows, output_dir / "pca_explained_variance.csv")
    plot_pca_curve(pca_curve, output_dir / "pca_curve.png")

    model_rows = []
    sv_by_name = {}
    for item in settings["models"]:
        row, sv = diagnose_one_model(item, train, test, eps=float(settings["subspace_eps"]))
        model_rows.append(row)
        sv_by_name[item["name"]] = sv

    save_csv(model_rows, output_dir / "model_diagnostics.csv")
    plot_singular_values(sv_by_name, output_dir / "phi_singular_values.png")
    write_report(output_dir, pca_rows, threshold_info, model_rows)
    append_experiment_log(model_rows)

    print()
    print("Phase 0 诊断完成。重点看：")
    print(f"  {output_dir / 'diag_report.md'}")
    print(f"  {output_dir / 'model_diagnostics.csv'}")
    print(f"  {output_dir / 'pca_curve.png'}")
    print(f"  {output_dir / 'phi_singular_values.png'}")


if __name__ == "__main__":
    main()
