"""Phase 1：先修解码目标，再引入 subspace 约束的批量实验脚本。

直接运行：
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 11_run_phase1_loss_experiments.py

这个脚本只负责 Phase 1，不进入 Phase 2/3：
  1. 先跑 3 组 25 通道、50 epoch 筛选实验；
  2. 只看 train/val 日志，不碰 test 集选择模型；
  3. 输出 phase1_screen50_summary.csv / .md；
  4. 如果你把 USER_SETTINGS["run_formal_150"] 改成 True，
     它会把筛选出的推荐配置从头跑 150 epoch，并最后用 04_eval_report.py 做 test 汇报。

为什么要单独写 runner？
  平时你仍然可以只打开 03_train_ar_emt.py 直接训练；
  这个文件只是帮你连续跑一组“有对照关系”的实验，避免手动改来改去出错。
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import math
import os
from pathlib import Path

# Windows + conda 里可能重复加载 OpenMP。这里沿用其它脚本的处理方式。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    measurement_matrix_coherence,
    metric_mse_psnr_sam,
    model_kwargs_from_settings,
    phi_effective_rank,
    tor_percent,
)


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    # False 表示如果某个 Phase 1 输出目录已经完整存在，就跳过，避免覆盖。
    # 如果你明确想重跑同名 Phase 1 实验，再改成 True。
    "force_rerun": False,

    # 数据仍然用 absolute 100k cache：train 用来训练，val 用来选 best，test 只给 formal 最终评估。
    "data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",

    # 先跑 50 轮筛选。跑完以后看 phase1_screen50_summary.md 再决定。
    "screen_epochs": 50,

    # 默认不自动跑 150 轮，避免你还没看筛选结果就占用很久 GPU。
    # 想让它筛选完自动正式训练，把这里改成 True。
    "run_formal_150": False,
    "formal_epochs": 150,

    # Formal 150 跑完以后才会调用 04_eval_report.py 使用 test 集。
    "formal_eval_angles_deg": [0.0, 0.5, 2.0, 5.0, 8.0, 10.0],
    "formal_eval_mc": 20,

    # 筛选通过门槛：平均透过率太低的配置不进入 formal 候选。
    "min_T0_mean_for_formal": 0.60,
}


# 三个 Phase 1 筛选实验。
# 它们是递进关系：实验 2 在实验 1 基础上加 subspace；实验 3 在实验 2 基础上换主 loss。
SCREEN_EXPERIMENTS = [
    {
        "name": "phase1_sam15_select_50",
        "explain": "只修 best 选择分数，并把 lambda_sam 从 0.05 提高到 0.15；主损仍用 MSE。",
        "overrides": {
            "lambda_sam": 0.15,
            "selection_score_mode": "l1_diff_sam",
            "sel_lambda_sam": 0.3,
        },
    },
    {
        "name": "phase1_subspace010_50",
        "explain": "在实验 1 基础上加入 subspace_residual=0.1，并关闭旧 coherence 约束。",
        "overrides": {
            "lambda_sam": 0.15,
            "selection_score_mode": "l1_diff_sam",
            "sel_lambda_sam": 0.3,
            "lambda_subspace": 0.10,
            "lambda_coh": 0.0,
        },
    },
    {
        "name": "phase1_charb_50",
        "explain": "在实验 2 基础上把主损改成 Charbonnier，并加入二阶差分曲率约束。",
        "overrides": {
            "lambda_sam": 0.15,
            "selection_score_mode": "l1_diff_sam",
            "sel_lambda_sam": 0.3,
            "lambda_subspace": 0.10,
            "lambda_coh": 0.0,
            "loss_mode": "charbonnier",
            "lambda_l1": 0.0,
            "lambda_diff2": 0.05,
        },
    },
]


def load_module(path: Path, module_name: str):
    """按文件路径导入脚本模块。

    训练/评估脚本文件名以数字开头，不能直接 import，所以这里按路径加载。
    """

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def common_train_settings(name: str, epochs: int) -> dict:
    """生成 Phase 1 的共同训练设置。

    这里固定 25 通道，因为 Phase 0 已经显示 25ch 是当前最好的主基线。
    """

    return {
        "data_dir": USER_SETTINGS["data_dir"],
        "checkpoint_dir": f"checkpoints_{name}",
        "results_dir": f"results_{name}",
        "tensorboard_dir": f"runs/ar_emt_{name}",
        "device": "cuda",
        "seed": 2026,
        "epochs": int(epochs),
        "batch_size": 512,
        "eval_batch_size": 4096,
        "angle_mode": "per_sample",
        "angle_max_deg": 5.0,
        "n_channels": 25,
        "hidden_dims": (768, 384),
        "h_c_range": (250.0, 1500.0),
        "t_r_range": (0.0, 1500.0),
        "core_total_nm": 1000.0,
        "core_total_range": (800.0, 1800.0),
        "aspect_ratio_max": 10.0,
        "noise_rel": 0.01,
        "noise_abs": 0.0,
        "t_target": 0.60,
        "lambda_trans": 0.03,
        "lambda_coh": 0.005,
        "lambda_sam": 0.05,
        "lambda_l1": 0.10,
        "lambda_diff": 0.20,
        "lambda_diff2": 0.0,
        "lambda_subspace": 0.0,
        "subspace_eps": 1e-4,
        "loss_mode": "mse",
        "charbonnier_eps": 1e-3,
        "selection_score_mode": "old",
        "sel_lambda_sam": 0.3,
        "tor_target_percent": 2.0,
        "lambda_tor": 0.01,
        "decoder_lr": 1e-3,
        "physics_lr": 2e-4,
        "decoder_weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-6,
        "progress_every_batches": 20,
        "eval_every_epochs": 1,
        "save_live_plots": True,
        "use_tensorboard": True,
        "resume": False,
        "period_nm": 180.0,
        "g_min_nm": 40.0,
        "d_min_nm": 60.0,
        "enforce_d_min": True,
    }


def screen_train_settings(exp: dict) -> dict:
    """把共同设置和某个实验的改动合并。"""

    settings = common_train_settings(exp["name"], int(USER_SETTINGS["screen_epochs"]))
    settings.update(exp["overrides"])
    return settings


def formal_train_settings(winner: dict) -> dict:
    """把筛选胜出的配置改成 150 epoch 正式训练目录。

    注意：正式版是从头训练，不复用 50 轮筛选 checkpoint。
    这样最终结果更干净，也不会把“筛选过程的中途状态”混进正式结果。
    """

    name = "phase1_formal_selected_150"
    settings = common_train_settings(name, int(USER_SETTINGS["formal_epochs"]))
    settings.update(winner["overrides"])
    settings["use_tensorboard"] = True
    settings["progress_every_batches"] = 20
    return settings


def eval_settings_for(train_settings: dict) -> dict:
    """生成 formal 最终评估设置。只有 formal 阶段才会用 test 集。"""

    return {
        "checkpoint": str(Path(train_settings["checkpoint_dir"]) / "ar_emt_best.pt"),
        "data_dir": train_settings["data_dir"],
        "output_dir": train_settings["results_dir"],
        "device": train_settings["device"],
        "test_size": 0,
        "angles_deg": list(USER_SETTINGS["formal_eval_angles_deg"]),
        "mc": int(USER_SETTINGS["formal_eval_mc"]),
        "noise_eval_levels": [0.0, 0.01, 0.02, 0.05],
        "seed": train_settings["seed"],
        "batch_size": train_settings["eval_batch_size"],
    }


def read_csv_rows(path: Path) -> list[dict]:
    """读取 CSV；如果不存在就返回空列表。"""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path) -> None:
    """保存一组 dict 到 CSV。"""

    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def float_or_nan(value) -> float:
    """把 CSV 字符串转成 float；空值转成 NaN，方便排序和判断。"""

    if value is None or value == "":
        return float("nan")
    return float(value)


def best_val_row(results_dir: Path) -> dict | None:
    """从 train_log.csv 里找 val_recon_loss 最低的一行。

    50 轮筛选只看验证集，不看 test。
    """

    rows = read_csv_rows(results_dir / "train_log.csv")
    if not rows:
        return None
    return min(rows, key=lambda r: float_or_nan(r.get("val_recon_loss")))


def experiment_is_complete(settings: dict) -> bool:
    """判断某个训练实验是否已经完整跑完。"""

    best_path = Path(settings["checkpoint_dir"]) / "ar_emt_best.pt"
    rows = read_csv_rows(Path(settings["results_dir"]) / "train_log.csv")
    if not best_path.exists() or not rows:
        return False
    max_epoch = max(int(float(row["epoch"])) for row in rows if row.get("epoch", "") != "")
    return max_epoch >= int(settings["epochs"])


def run_training(settings: dict, module_name: str) -> None:
    """调用 03_train_ar_emt.py 训练一次。"""

    train_mod = load_module(Path("03_train_ar_emt.py"), module_name)
    train_mod.USER_SETTINGS = settings
    train_mod.main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_screen_experiment(exp: dict) -> None:
    """运行一组 50 轮筛选实验；已有完整结果时默认跳过。"""

    settings = screen_train_settings(exp)
    if experiment_is_complete(settings) and not USER_SETTINGS["force_rerun"]:
        print(f"\n跳过 {exp['name']}：已有完整 50 轮结果。")
        return

    if (Path(settings["results_dir"]).exists() or Path(settings["checkpoint_dir"]).exists()) and not USER_SETTINGS["force_rerun"]:
        print(f"\n跳过 {exp['name']}：目录已存在但结果不完整。")
        print("  为避免覆盖旧文件，请确认后把 USER_SETTINGS['force_rerun'] 改成 True 再跑。")
        return

    print("\n" + "=" * 80)
    print(f"开始 50 轮筛选实验: {exp['name']}")
    print(exp["explain"])
    print(f"结果目录: {settings['results_dir']}")
    print("=" * 80)
    run_training(settings, f"train_{exp['name']}")


def collect_screen_summary() -> list[dict]:
    """收集三组 50 轮筛选结果，生成对比表。"""

    rows = []
    min_t = float(USER_SETTINGS["min_T0_mean_for_formal"])
    for exp in SCREEN_EXPERIMENTS:
        settings = screen_train_settings(exp)
        results_dir = Path(settings["results_dir"])
        row = best_val_row(results_dir)
        if row is None:
            rows.append({
                "experiment": exp["name"],
                "status": "missing",
                "explain": exp["explain"],
                "best_epoch": "",
                "val_recon_loss": "",
                "val_mse": "",
                "val_l1": "",
                "val_diff_l1": "",
                "val_sam": "",
                "T0_mean": "",
                "tor_percent": "",
                "coherence0": "",
                "phi_effective_rank": "",
                "train_subspace": "",
                "eligible_for_150": "no",
                "reason": "没有 train_log.csv 或训练未完成",
            })
            continue

        t_mean = float_or_nan(row.get("T0_mean"))
        val_score = float_or_nan(row.get("val_recon_loss"))
        eligible = math.isfinite(val_score) and math.isfinite(t_mean) and t_mean >= min_t
        reason = "通过筛选门槛" if eligible else f"T0_mean<{min_t:.2f} 或 val 分数异常"
        rows.append({
            "experiment": exp["name"],
            "status": "ok",
            "explain": exp["explain"],
            "best_epoch": row.get("epoch", ""),
            "val_recon_loss": row.get("val_recon_loss", ""),
            "val_mse": row.get("val_mse", ""),
            "val_l1": row.get("val_l1", ""),
            "val_diff_l1": row.get("val_diff_l1", ""),
            "val_sam": row.get("val_sam", ""),
            "T0_mean": row.get("T0_mean", ""),
            "tor_percent": row.get("tor_percent", ""),
            "coherence0": row.get("coherence0", ""),
            "phi_effective_rank": row.get("phi_effective_rank", ""),
            "train_subspace": row.get("train_subspace", ""),
            "eligible_for_150": "yes" if eligible else "no",
            "reason": reason,
        })
    return rows


def choose_recommended(rows: list[dict]) -> dict | None:
    """在通过 T0_mean 门槛的配置里，选 val_recon_loss 最低的一组。"""

    eligible = [
        r for r in rows
        if r["eligible_for_150"] == "yes" and math.isfinite(float_or_nan(r["val_recon_loss"]))
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda r: float_or_nan(r["val_recon_loss"]))


def write_screen_markdown(rows: list[dict], recommended: dict | None, path: Path) -> None:
    """写 50 轮筛选汇报。"""

    lines = [
        "# Phase 1 50 轮筛选汇报",
        "",
        "本汇报只使用训练集和验证集日志，没有使用 test 集选择模型。",
        "",
        "| 实验 | best epoch | val_recon_loss | val MSE | val L1 | val SAM | T0_mean | tor% | coherence | Phi有效秩 | subspace | 是否进150轮 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['best_epoch']} | {row['val_recon_loss']} | "
            f"{row['val_mse']} | {row['val_l1']} | {row['val_sam']} | {row['T0_mean']} | "
            f"{row['tor_percent']} | {row['coherence0']} | {row['phi_effective_rank']} | "
            f"{row['train_subspace']} | {row['eligible_for_150']} |"
        )

    lines.extend([
        "",
        "## 选择规则",
        "",
        f"- 先剔除 `T0_mean < {float(USER_SETTINGS['min_T0_mean_for_formal']):.2f}` 或 val 分数异常的配置。",
        "- 剩余配置里，选 `val_recon_loss` 最低的一组进入 150 轮正式训练。",
        "- `val_recon_loss` 是验证集重建选择分数，不包含 test，也不直接包含 tor/coherence/T0。",
        "",
    ])
    if recommended is None:
        lines.append("当前没有配置通过筛选门槛，暂不建议进入 150 轮。")
    else:
        lines.append(
            f"推荐进入 150 轮的配置：`{recommended['experiment']}`，"
            f"best epoch={recommended['best_epoch']}，"
            f"val_recon_loss={recommended['val_recon_loss']}。"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_model_and_phi(checkpoint_path: Path) -> tuple[AREMTModel, torch.Tensor, dict]:
    """读取 checkpoint，并提取 0 度 Phi，用于 formal 后的 pinv/秩诊断。"""

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


def pinv_metrics(test: np.ndarray, phi: torch.Tensor) -> dict[str, float]:
    """formal 诊断用：不用 MLP，只用 Phi 伪逆做 clean 0 度线性重建。"""

    phi_np = phi.numpy().astype(np.float32)
    y = test.astype(np.float32) @ phi_np.T
    recon = y @ np.linalg.pinv(phi_np).T
    return metric_mse_psnr_sam(torch.from_numpy(recon.astype(np.float32)), torch.from_numpy(test.astype(np.float32)))


def read_angle0(results_dir: Path) -> dict | None:
    """读取 04_eval_report.py 生成的 0 度 test 指标。"""

    rows = read_csv_rows(results_dir / "eval_angles.csv")
    return next((r for r in rows if abs(float(r["angle_deg"]) - 0.0) < 1e-9), None)


def collect_formal_summary(train_settings: dict, selected_name: str) -> list[dict]:
    """formal 150 完成后汇总 test、pinv、Phi 诊断。"""

    results_dir = Path(train_settings["results_dir"])
    checkpoint_path = Path(train_settings["checkpoint_dir"]) / "ar_emt_best.pt"
    angle0 = read_angle0(results_dir)
    if angle0 is None:
        raise RuntimeError(f"找不到 formal eval 0 度结果: {results_dir / 'eval_angles.csv'}")

    _model, phi, ckpt = load_model_and_phi(checkpoint_path)
    test = np.load(Path(train_settings["data_dir"]) / "test_spectra.npy").astype(np.float32)
    pinv = pinv_metrics(test, phi)
    singular_values = torch.linalg.svdvals(phi).numpy()

    row = {
        "formal_experiment": train_settings["results_dir"],
        "selected_from": selected_name,
        "checkpoint_epoch": ckpt.get("epoch", ""),
        "mlp_test_mse": angle0["mse"],
        "mlp_test_l1": angle0["l1"],
        "mlp_test_diff_l1": angle0["diff_l1"],
        "mlp_test_sam": angle0["sam"],
        "mlp_test_psnr": angle0["psnr"],
        "pinv_clean0_mse": pinv["mse"],
        "pinv_clean0_l1": pinv["l1"],
        "pinv_clean0_diff_l1": pinv["diff_l1"],
        "pinv_clean0_sam": pinv["sam"],
        "pinv_clean0_psnr": pinv["psnr"],
        "T_mean": angle0["T_mean"],
        "tor_percent": angle0["tor_percent"],
        "phi_effective_rank": float(phi_effective_rank(phi)),
        "phi_condition_number": float(singular_values.max() / max(singular_values.min(), 1e-12)),
        "phi_coherence": float(measurement_matrix_coherence(phi)),
    }
    return [row]


def write_formal_markdown(rows: list[dict], path: Path) -> None:
    """写 formal 150 汇报。"""

    row = rows[0]
    lines = [
        "# Phase 1 150 轮正式结果",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
        f"| 选择来源 | {row['selected_from']} |",
        f"| checkpoint epoch | {row['checkpoint_epoch']} |",
        f"| MLP test MSE | {float(row['mlp_test_mse']):.6e} |",
        f"| MLP test L1 | {float(row['mlp_test_l1']):.6e} |",
        f"| MLP test SAM | {float(row['mlp_test_sam']):.5f} |",
        f"| MLP test PSNR | {float(row['mlp_test_psnr']):.2f} |",
        f"| pinv clean0 MSE | {float(row['pinv_clean0_mse']):.6e} |",
        f"| pinv clean0 SAM | {float(row['pinv_clean0_sam']):.5f} |",
        f"| T_mean | {float(row['T_mean']):.5f} |",
        f"| tor_percent | {float(row['tor_percent']):.3f}% |",
        f"| Phi 有效秩 | {float(row['phi_effective_rank']):.2f} |",
        f"| Phi 条件数 | {float(row['phi_condition_number']):.3e} |",
        f"| Phi coherence | {float(row['phi_coherence']):.5f} |",
        "",
        "基线目标：对标 `25ch_t06_tor20_150`，其 0 度 test MSE 约为 `1.53e-4`，SAM 约为 `0.0655`。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_formal_to_experiment_log(rows: list[dict]) -> None:
    """把 formal 结果追加到 experiment_log.md。"""

    row = rows[0]
    path = Path("experiment_log.md")
    section_title = "## Phase 1 formal：loss/selection/subspace 修改实验"
    lines = [
        "",
        section_title,
        "",
        "| 实验名 | 改了什么 | test MSE | test L1 | test SAM | PSNR | pinv MSE | T_mean | Phi有效秩 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {row['formal_experiment']} | 由 `{row['selected_from']}` 筛选进入 150 轮 | "
            f"{float(row['mlp_test_mse']):.6e} | {float(row['mlp_test_l1']):.6e} | "
            f"{float(row['mlp_test_sam']):.5f} | {float(row['mlp_test_psnr']):.2f} | "
            f"{float(row['pinv_clean0_mse']):.6e} | {float(row['T_mean']):.5f} | "
            f"{float(row['phi_effective_rank']):.2f} | "
            f"tor={float(row['tor_percent']):.3f}%, coh={float(row['phi_coherence']):.4f} |"
        ),
        "",
    ]

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


def run_formal_if_enabled(recommended: dict | None) -> None:
    """如果开关允许，就把推荐配置跑 150 轮并做 test 汇报。"""

    if not USER_SETTINGS["run_formal_150"]:
        return
    if recommended is None:
        print("没有通过筛选的配置，跳过 150 轮 formal。")
        return

    winner = next(exp for exp in SCREEN_EXPERIMENTS if exp["name"] == recommended["experiment"])
    train_settings = formal_train_settings(winner)
    if experiment_is_complete(train_settings) and not USER_SETTINGS["force_rerun"]:
        print(f"\n跳过 formal：已有完整结果 {train_settings['results_dir']}")
    elif (Path(train_settings["results_dir"]).exists() or Path(train_settings["checkpoint_dir"]).exists()) and not USER_SETTINGS["force_rerun"]:
        print(f"\nformal 目录已存在但不完整: {train_settings['results_dir']}")
        print("为避免覆盖旧文件，请确认后把 USER_SETTINGS['force_rerun'] 改成 True 再跑。")
        return
    else:
        print("\n" + "=" * 80)
        print(f"开始 150 轮 formal: {winner['name']} -> {train_settings['results_dir']}")
        print("=" * 80)
        run_training(train_settings, "train_phase1_formal_selected_150")

    eval_done = (Path(train_settings["results_dir"]) / "eval_angles.csv").exists()
    if eval_done and not USER_SETTINGS["force_rerun"]:
        print("\n跳过 formal test 评估：eval_angles.csv 已存在。")
    else:
        print("\n开始 formal test 评估")
        eval_mod = load_module(Path("04_eval_report.py"), "eval_phase1_formal_selected_150")
        eval_mod.USER_SETTINGS = eval_settings_for(train_settings)
        eval_mod.main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    formal_rows = collect_formal_summary(train_settings, selected_name=winner["name"])
    save_csv(formal_rows, Path("phase1_formal_summary.csv"))
    write_formal_markdown(formal_rows, Path("phase1_formal_summary.md"))
    append_formal_to_experiment_log(formal_rows)
    print("formal 汇总已保存: phase1_formal_summary.csv / phase1_formal_summary.md")


def main() -> None:
    print("Phase 1 loss / selection / subspace 筛选实验")
    print("本脚本默认只跑 50 轮筛选，不用 test 选模型。")
    print()
    for exp in SCREEN_EXPERIMENTS:
        print(f"- {exp['name']}: {exp['explain']}")

    for exp in SCREEN_EXPERIMENTS:
        run_screen_experiment(exp)

    rows = collect_screen_summary()
    recommended = choose_recommended(rows)
    save_csv(rows, Path("phase1_screen50_summary.csv"))
    write_screen_markdown(rows, recommended, Path("phase1_screen50_summary.md"))

    print()
    print("50 轮筛选汇总已保存:")
    print("  phase1_screen50_summary.csv")
    print("  phase1_screen50_summary.md")
    if recommended is None:
        print("当前没有配置通过筛选门槛。")
    else:
        print(f"推荐配置: {recommended['experiment']} | val_recon_loss={recommended['val_recon_loss']}")

    run_formal_if_enabled(recommended)


if __name__ == "__main__":
    main()
