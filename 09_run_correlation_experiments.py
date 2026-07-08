"""批量跑“通道相关性 / tor 区分度”消融实验。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 09_run_correlation_experiments.py

这个脚本专门回答一个问题：
  如果把滤光片通道去相关约束(lambda_coh)或 tor 区分度约束调大，
  重建精度会变好还是变差？

为了先快速看趋势，下面所有实验都保持：
  - 25 个滤光片通道；
  - T_target = 0.60；
  - 从 CAVE absolute 数据里切出来的同一份小缓存；
  - 同一个随机种子 seed=2026；
  - 先跑 30 轮筛选，不直接跑 150 轮。

输出:
  - checkpoints_25ch_corrquick_xxx_30/
  - results_25ch_corrquick_xxx_30/
  - correlation_experiment_summary.csv

注意：
  这个脚本不会删除旧实验。如果某个输出目录已经存在，默认会跳过，避免覆盖。
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


# Windows + conda 里可能重复加载 OpenMP。这里沿用其它脚本的处理方式。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    # False 表示如果结果目录已存在就跳过，避免覆盖旧实验。
    # 如果你明确想重跑，把它改成 True，但会覆盖同名输出。
    "force_rerun": False,

    # 从完整 100k cache 里切小数据，快速看趋势。
    "source_data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",
    "quick_data_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_corrquick_20k",
    "quick_train_size": 20000,
    "quick_val_size": 5000,
    "quick_test_size": 5000,

    # 先跑 30 轮筛选。筛出来好的，再单独跑 150 轮正式版。
    "epochs": 30,

    # 评估 Monte Carlo 次数。快速筛选先用 3，正式汇报可以用 20 或 50。
    "eval_mc": 3,

    # 快速筛选先只看 0 度，正式版再看多角度。
    "eval_angles_deg": [0.0],
}


# 这里是本次实验矩阵。
# name 会进入目录名：results_25ch_corrquick_{name}_{epochs}。
EXPERIMENTS = [
    {
        "name": "base",
        "explain": "快速筛选 baseline：保持当前 25ch 主线约束",
        "lambda_coh": 0.005,
        "lambda_tor": 0.010,
        "tor_target_percent": 2.0,
    },
    {
        "name": "coh010",
        "explain": "只把通道去相关权重从 0.005 加到 0.010",
        "lambda_coh": 0.010,
        "lambda_tor": 0.010,
        "tor_target_percent": 2.0,
    },
    {
        "name": "coh020",
        "explain": "只把通道去相关权重从 0.005 加到 0.020",
        "lambda_coh": 0.020,
        "lambda_tor": 0.010,
        "tor_target_percent": 2.0,
    },
    {
        "name": "tor30",
        "explain": "主要加强 tor 下限：目标从 2.0% 加到 3.0%，权重加到 0.020",
        "lambda_coh": 0.005,
        "lambda_tor": 0.020,
        "tor_target_percent": 3.0,
    },
    {
        "name": "coh010_tor25",
        "explain": "温和组合：lambda_coh=0.010，同时 tor 目标提高到 2.5%",
        "lambda_coh": 0.010,
        "lambda_tor": 0.020,
        "tor_target_percent": 2.5,
    },
]


def load_module(path: Path, module_name: str):
    """按文件路径导入脚本模块。

    训练脚本文件名以数字开头，不能直接 import，所以用 importlib 按路径加载。
    """

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def train_settings_for(exp: dict) -> dict:
    """生成某个实验的训练参数。

    这些参数基本复制当前 25ch 主线，只改 lambda_coh / lambda_tor / tor_target_percent。
    """

    name = exp["name"]
    epochs = int(USER_SETTINGS["epochs"])
    return {
        "data_dir": USER_SETTINGS["quick_data_dir"],
        "checkpoint_dir": f"checkpoints_25ch_corrquick_{name}_{epochs}",
        "results_dir": f"results_25ch_corrquick_{name}_{epochs}",
        "tensorboard_dir": f"runs/ar_emt_25ch_corrquick_{name}_{epochs}",
        "device": "cuda",
        "seed": 2026,
        "epochs": epochs,
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
        "lambda_coh": float(exp["lambda_coh"]),
        "lambda_sam": 0.05,
        "lambda_l1": 0.10,
        "lambda_diff": 0.20,
        "tor_target_percent": float(exp["tor_target_percent"]),
        "lambda_tor": float(exp["lambda_tor"]),
        "decoder_lr": 1e-3,
        "physics_lr": 2e-4,
        "decoder_weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-6,
        "progress_every_batches": 0,
        "eval_every_epochs": 1,
        "save_live_plots": True,
        "use_tensorboard": False,
        "resume": False,
        "period_nm": 180.0,
        "g_min_nm": 40.0,
        "d_min_nm": 60.0,
        "enforce_d_min": True,
    }


def eval_settings_for(train_settings: dict) -> dict:
    """生成某个实验的评估参数。"""

    return {
        "checkpoint": str(Path(train_settings["checkpoint_dir"]) / "ar_emt_best.pt"),
        "data_dir": train_settings["data_dir"],
        "output_dir": train_settings["results_dir"],
        "device": train_settings["device"],
        "test_size": 0,
        "angles_deg": list(USER_SETTINGS["eval_angles_deg"]),
        "mc": int(USER_SETTINGS["eval_mc"]),
        "noise_eval_levels": [0.0, 0.01, 0.02, 0.05],
        "seed": train_settings["seed"],
        "batch_size": train_settings["eval_batch_size"],
    }


def ensure_quick_cache() -> None:
    """从完整 absolute cache 切出一份小 cache，加快消融实验。

    这里只是复制前 N 条光谱，不做任何归一化或重新处理。
    原始完整数据不会被覆盖。
    """

    source = Path(USER_SETTINGS["source_data_dir"])
    target = Path(USER_SETTINGS["quick_data_dir"])
    target.mkdir(parents=True, exist_ok=True)

    expected = [
        target / "train_spectra.npy",
        target / "val_spectra.npy",
        target / "test_spectra.npy",
        target / "wl_nm.npy",
    ]
    if all(p.exists() for p in expected):
        print(f"快速数据缓存已存在: {target}")
        return

    print(f"生成快速数据缓存: {target}")
    train = np.load(source / "train_spectra.npy", mmap_mode="r")[: int(USER_SETTINGS["quick_train_size"])]
    val = np.load(source / "val_spectra.npy", mmap_mode="r")[: int(USER_SETTINGS["quick_val_size"])]
    test = np.load(source / "test_spectra.npy", mmap_mode="r")[: int(USER_SETTINGS["quick_test_size"])]
    wl = np.load(source / "wl_nm.npy")

    np.save(target / "train_spectra.npy", np.asarray(train, dtype=np.float32))
    np.save(target / "val_spectra.npy", np.asarray(val, dtype=np.float32))
    np.save(target / "test_spectra.npy", np.asarray(test, dtype=np.float32))
    np.save(target / "wl_nm.npy", np.asarray(wl, dtype=np.float32))


def read_csv_first(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def collect_one_result(results_dir: Path) -> dict:
    """从一个 results 目录提取对比用的关键指标。"""

    eval_rows = read_csv_rows(results_dir / "eval_angles.csv")
    angle0 = next((r for r in eval_rows if abs(float(r["angle_deg"]) - 0.0) < 1e-9), None)
    train_rows = read_csv_rows(results_dir / "train_log.csv")
    best_train = None
    if train_rows and "val_recon_loss" in train_rows[0]:
        best_train = min(train_rows, key=lambda r: float(r["val_recon_loss"]))
    elif train_rows:
        best_train = train_rows[-1]
    fab = read_csv_first(results_dir / "eval_fabrication_mc.csv")

    return {
        "results_dir": results_dir.name,
        "epochs": int(float(best_train["epoch"])) if best_train and "epoch" in best_train else None,
        "best_val_mse": num(best_train.get("val_mse")) if best_train else None,
        "best_val_l1": num(best_train.get("val_l1")) if best_train else None,
        "best_val_sam": num(best_train.get("val_sam")) if best_train else None,
        "coherence0": num(best_train.get("coherence0")) if best_train else None,
        "test_mse": num(angle0.get("mse")) if angle0 else None,
        "test_l1": num(angle0.get("l1")) if angle0 else None,
        "test_diff_l1": num(angle0.get("diff_l1")) if angle0 else None,
        "test_sam": num(angle0.get("sam")) if angle0 else None,
        "test_psnr": num(angle0.get("psnr")) if angle0 else None,
        "T_mean": num(angle0.get("T_mean")) if angle0 else None,
        "tor_percent": num(angle0.get("tor_percent")) if angle0 else None,
        "fab_mse": num(fab.get("mse")) if fab else None,
        "fab_sam": num(fab.get("sam")) if fab else None,
    }


def write_summary(rows: list[dict], path: Path) -> None:
    """保存总对比表。"""

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(exp: dict) -> None:
    """训练 + 评估单个实验。"""

    train_settings = train_settings_for(exp)
    results_dir = Path(train_settings["results_dir"])
    checkpoint_dir = Path(train_settings["checkpoint_dir"])
    best_path = checkpoint_dir / "ar_emt_best.pt"

    already_done = best_path.exists() and (results_dir / "eval_angles.csv").exists()
    if already_done and not USER_SETTINGS["force_rerun"]:
        print(f"\n跳过 {exp['name']}：已有 checkpoint 和 eval_angles.csv")
        return

    print("\n" + "=" * 80)
    print(f"开始实验: {exp['name']}")
    print(exp["explain"])
    print(f"输出: {results_dir}")
    print("=" * 80)

    train_mod = load_module(Path("03_train_ar_emt.py"), f"train_{exp['name']}")
    train_mod.USER_SETTINGS = train_settings
    train_mod.main()

    # 释放一下显存，避免连续跑多个实验时显存碎片堆积。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n开始最终 test 评估")
    eval_mod = load_module(Path("04_eval_report.py"), f"eval_{exp['name']}")
    eval_mod.USER_SETTINGS = eval_settings_for(train_settings)
    eval_mod.main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    print("通道相关性 / tor 消融实验")
    print("本轮先用 20k/5k/5k 小数据跑 30 epoch 快速筛选，不覆盖已有实验。")
    print()
    ensure_quick_cache()
    print()
    for exp in EXPERIMENTS:
        print(f"- {exp['name']}: {exp['explain']}")

    for exp in EXPERIMENTS:
        run_experiment(exp)

    # 汇总本轮快速筛选实验。
    rows = []
    epochs = int(USER_SETTINGS["epochs"])
    for exp in EXPERIMENTS:
        path = Path(f"results_25ch_corrquick_{exp['name']}_{epochs}")
        if path.exists():
            row = collect_one_result(path)
            row["lambda_coh"] = exp["lambda_coh"]
            row["lambda_tor"] = exp["lambda_tor"]
            row["tor_target_percent"] = exp["tor_target_percent"]
            rows.append(row)

    summary_path = Path("correlation_experiment_summary.csv")
    write_summary(rows, summary_path)

    print()
    print(f"汇总表已保存: {summary_path}")
    ranked = [r for r in rows if r.get("test_mse") is not None]
    ranked.sort(key=lambda r: float(r["test_mse"]))
    print("按 test_mse 排名：")
    for r in ranked:
        print(
            f"  {r['results_dir']:<32} "
            f"test_mse={r['test_mse']:.6e} "
            f"sam={r['test_sam']:.4f} "
            f"tor={r['tor_percent']:.3f}% "
            f"coh={r['coherence0']:.4f} "
            f"T={r['T_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
