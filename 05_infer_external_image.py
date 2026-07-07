"""用训练好的 AR-EMT 模型，重建一张“没参与训练”的外部高光谱图片/场景。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 05_infer_external_image.py

它解决的问题：
  训练完之后，你想把一张真实高光谱 cube 输入进来，模拟它经过 16 个滤光片后的
  测量值，再用保存好的解码器把光谱重建出来，并画图/存结果看效果。

支持的输入：
1. CAVE 场景目录：里面有 31 张 *_ms_*.png 波段图；
2. npy 文件：shape 可为 [H,W,31]、[H,W,151]、[31,H,W]、[151,H,W] 或 [N,151]；
3. mat 文件：自动找第一个“最后一维是 31 或 151”的数组。

不支持普通 RGB 图片：RGB 只有 3 个通道，信息不够；这里需要高光谱 cube 作仿真输入。
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
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.io import loadmat

from ar_emt_common import AREMTModel, GeometryConfig, metric_mse_psnr_sam, model_kwargs_from_settings


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    # 用 best checkpoint 推理。训练没结束时也可临时改成 checkpoints/ar_emt_last.pt。
    "checkpoint": "checkpoints_longtor_50/ar_emt_best.pt",

    # 默认拿一个 CAVE 场景做例子；可改成别的 CAVE 目录或 npy/mat 文件。
    "input_path": r"E:\hyperspectral_datasets\CAVE\extracted\balloons_ms\balloons_ms",

    # 推理结果单独放这里，别和训练结果混。
    "output_dir": "results_infer_longtor_50",

    "device": "cuda",
    "angle_deg": 0.0,
    "batch_size": 4096,

    # 画几条像素光谱做检查，格式 (y, x)。
    # 若输入是 [N,151] 这种没有图像宽高的数据，就按样本编号取前几个。
    "plot_pixels": [(80, 80), (180, 260), (320, 320)],
}


# 波长网格：模型用 151 点，CAVE 原始 31 点。
WL_151 = np.linspace(400.0, 700.0, 151).astype(np.float32)
WL_31 = np.linspace(400.0, 700.0, 31).astype(np.float32)


# =============================================================================
# 读图 / 清理 / 插值（和 02_prepare_data.py 保持一致：绝对强度，不逐条归一化）
# =============================================================================


def image_to_float01(path: Path) -> np.ndarray:
    """读一张 PNG 按位深缩放到 0~1（8-bit/255，16-bit/65535），不做逐像素归一化。"""

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    if np.issubdtype(arr.dtype, np.uint8):
        scale = 255.0
    elif np.issubdtype(arr.dtype, np.uint16):
        scale = 65535.0
    else:
        max_val = float(np.nanmax(arr))
        scale = 1.0 if max_val <= 1.0 else max_val

    out = arr.astype(np.float32) / scale
    return np.clip(out, 0.0, 1.0)


def clean_cube(cube: np.ndarray) -> np.ndarray:
    """清理高光谱 cube：NaN/inf→0，裁到 0~1，不逐条归一化。"""

    cube = np.asarray(cube, dtype=np.float32)
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    cube = np.clip(cube, 0.0, 1.0)
    return cube.astype(np.float32)


def interpolate_cube_to_151(cube: np.ndarray) -> np.ndarray:
    """把最后一维为 31 的 cube 三次样条插值成 151。"""

    cube = clean_cube(cube)
    if cube.shape[-1] == 151:
        return cube
    if cube.shape[-1] != 31:
        raise ValueError(f"光谱维必须是 31 或 151，实际 shape={cube.shape}")

    original_shape = cube.shape[:-1]
    flat = cube.reshape(-1, 31)
    cs = CubicSpline(WL_31, flat, axis=1)
    out = cs(WL_151).reshape(*original_shape, 151)
    return clean_cube(out)


def move_spectral_axis_to_last(arr: np.ndarray) -> np.ndarray:
    """把光谱维挪到最后。有些 npy 是 [31,H,W]/[151,H,W]，我们要 [H,W,31]/[H,W,151]。"""

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in {31, 151}:
        return arr
    if arr.ndim == 3 and arr.shape[0] in {31, 151}:
        return np.moveaxis(arr, 0, -1)
    if arr.ndim == 2 and arr.shape[-1] == 151:
        return arr
    raise ValueError(f"不认识的高光谱数据 shape={arr.shape}")


# =============================================================================
# 按输入类型读取
# =============================================================================


def load_cave_scene(scene_dir: Path) -> tuple[np.ndarray, tuple[int, int] | None, str]:
    """读 CAVE 场景目录，返回 [H,W,151]。"""

    pngs = sorted(scene_dir.glob("*_ms_*.png"))[:31]
    if len(pngs) != 31:
        raise ValueError(f"{scene_dir} 中没有找到 31 张 *_ms_*.png 波段图。")

    bands = [image_to_float01(path) for path in pngs]
    cube31 = np.stack(bands, axis=-1)
    cube151 = interpolate_cube_to_151(cube31)
    h, w = cube151.shape[:2]
    return cube151, (h, w), f"CAVE scene: {scene_dir}"


def load_npy(path: Path) -> tuple[np.ndarray, tuple[int, int] | None, str]:
    arr = move_spectral_axis_to_last(np.load(path))
    cube = interpolate_cube_to_151(arr)
    image_shape = tuple(cube.shape[:2]) if cube.ndim == 3 else None
    return cube, image_shape, f"npy: {path}"


def load_mat(path: Path) -> tuple[np.ndarray, tuple[int, int] | None, str]:
    data = loadmat(path)
    for key, arr in data.items():
        if key.startswith("__") or not isinstance(arr, np.ndarray):
            continue
        try:
            moved = move_spectral_axis_to_last(arr)
            cube = interpolate_cube_to_151(moved)
            image_shape = tuple(cube.shape[:2]) if cube.ndim == 3 else None
            return cube, image_shape, f"mat: {path}, variable={key}"
        except Exception:
            continue
    raise ValueError(f"{path} 中没有找到光谱维为 31 或 151 的数组。")


def load_input_cube(input_path: Path) -> tuple[np.ndarray, tuple[int, int] | None, str]:
    """根据路径类型分派到对应的读取函数。"""

    if input_path.is_dir():
        return load_cave_scene(input_path)
    if input_path.suffix.lower() == ".npy":
        return load_npy(input_path)
    if input_path.suffix.lower() == ".mat":
        return load_mat(input_path)
    raise ValueError("需要 CAVE 场景目录、npy 或 mat 高光谱数据；普通 RGB 图片不能用于此仿真。")


def load_model(checkpoint_path: Path, device: torch.device) -> AREMTModel:
    """读取训练保存的完整模型；推理只需要“结构参数 + 解码器权重”。"""

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GeometryConfig(**ckpt["config"])
    wl_nm = ckpt["wl_nm"].to(dtype=torch.float32)
    model = AREMTModel(wl_nm, config, **model_kwargs_from_settings(ckpt.get("settings", {}))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"读取 checkpoint: {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch')}, best_val_mse={ckpt.get('best_val_mse')}, "
          f"best_val_score={ckpt.get('best_val_score')}")
    return model


def run_inference(model: AREMTModel, spectra: np.ndarray, angle_deg: float,
                  batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """批量推理。输入 [N,151]；输出 重建 [N,151] 和 16 通道测量 [N,16]。

    这里不加噪声（模拟“理想读数下”的重建效果）；想看抗噪表现请用 04 的噪声鲁棒性评估。
    """

    spectra_tensor = torch.from_numpy(spectra.astype(np.float32))
    pred_chunks, meas_chunks = [], []
    with torch.no_grad():
        t = model.transmission(torch.tensor([angle_deg], device=device))[0]
        for start in range(0, spectra_tensor.shape[0], batch_size):
            batch = spectra_tensor[start:start + batch_size].to(device)
            meas = model.measure(batch, t)
            pred = model.decoder(meas)
            pred_chunks.append(pred.cpu().numpy().astype(np.float32))
            meas_chunks.append(meas.cpu().numpy().astype(np.float32))
    return np.concatenate(pred_chunks, axis=0), np.concatenate(meas_chunks, axis=0)


def save_summary_csv(summary: dict, path: Path) -> None:
    """保存一行推理摘要。"""

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


# =============================================================================
# 画图
# =============================================================================


def plot_selected_spectra(gt_flat, pred_flat, image_shape, plot_pixels, out_path: Path) -> None:
    """画几个像素位置的真实光谱(实线)和重建光谱(虚线)对比。"""

    plt.figure(figsize=(9, 5))
    if image_shape is None:
        for idx in range(min(3, gt_flat.shape[0])):
            plt.plot(WL_151, gt_flat[idx], lw=1.8, label=f"gt sample {idx}")
            plt.plot(WL_151, pred_flat[idx], lw=1.2, ls="--", label=f"pred sample {idx}")
    else:
        h, w = image_shape
        for y, x in plot_pixels:
            yy = int(np.clip(y, 0, h - 1))
            xx = int(np.clip(x, 0, w - 1))
            idx = yy * w + xx
            plt.plot(WL_151, gt_flat[idx], lw=1.8, label=f"gt ({yy},{xx})")
            plt.plot(WL_151, pred_flat[idx], lw=1.2, ls="--", label=f"pred ({yy},{xx})")

    plt.xlabel("Wavelength (nm)"); plt.ylabel("Intensity")
    plt.title("Selected pixel spectra"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def plot_error_map(gt_flat, pred_flat, image_shape, out_path: Path) -> None:
    """画每个像素的光谱重建 MSE 误差图（越亮误差越大）。"""

    h, w = image_shape
    mse = np.mean((pred_flat - gt_flat) ** 2, axis=1).reshape(h, w)
    plt.figure(figsize=(6, 5))
    plt.imshow(mse, cmap="magma"); plt.colorbar(label="MSE")
    plt.title("Per-pixel reconstruction MSE"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def plot_measurement_preview(meas_flat, image_shape, out_path: Path) -> None:
    """画 16 个滤光片各自的测量图（相当于 16 张“伪彩通道图”）。"""

    h, w = image_shape
    meas = meas_flat.reshape(h, w, 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ch, ax in enumerate(axes.ravel()):
        im = ax.imshow(meas[:, :, ch], cmap="viridis")
        ax.set_title(f"ch{ch}", fontsize=9); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Simulated 16-channel measurements", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def main() -> None:
    settings = USER_SETTINGS
    device = torch.device(settings["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(settings["checkpoint"]), device)
    cube, image_shape, source_desc = load_input_cube(Path(settings["input_path"]))

    # 把 cube 拍平成 [N,151] 送进模型
    if cube.ndim == 3:
        h, w, n_wl = cube.shape
        spectra_flat = cube.reshape(h * w, n_wl)
    elif cube.ndim == 2:
        spectra_flat = cube
    else:
        raise ValueError(f"输入 cube 维度不对: {cube.shape}")

    print(f"输入数据: {source_desc}")
    print(f"  cube shape={cube.shape}")
    print(f"  flat spectra shape={spectra_flat.shape}")
    print(f"  min={spectra_flat.min():.6f}, max={spectra_flat.max():.6f}, mean={spectra_flat.mean():.6f}")

    pred_flat, meas_flat = run_inference(
        model=model, spectra=spectra_flat, angle_deg=float(settings["angle_deg"]),
        batch_size=int(settings["batch_size"]), device=device,
    )

    # 有真值(输入本身)就算个整体指标
    metrics = metric_mse_psnr_sam(torch.from_numpy(pred_flat), torch.from_numpy(spectra_flat))
    summary = {
        "source": source_desc, "checkpoint": settings["checkpoint"], "angle_deg": settings["angle_deg"],
        "n_spectra": spectra_flat.shape[0],
        "mse": metrics["mse"], "l1": metrics["l1"], "diff_l1": metrics["diff_l1"],
        "psnr": metrics["psnr"], "sam": metrics["sam"],
        "input_min": float(spectra_flat.min()), "input_max": float(spectra_flat.max()), "input_mean": float(spectra_flat.mean()),
        "pred_min": float(pred_flat.min()), "pred_max": float(pred_flat.max()), "pred_mean": float(pred_flat.mean()),
    }
    save_summary_csv(summary, output_dir / "inference_summary.csv")

    # 存原始数组，方便你后续自己分析
    np.save(output_dir / "input_spectra_151.npy", spectra_flat.astype(np.float32))
    np.save(output_dir / "measurement_16ch.npy", meas_flat.astype(np.float32))
    np.save(output_dir / "reconstructed_spectra_151.npy", pred_flat.astype(np.float32))

    # 有图像宽高时，额外存成 cube 并画误差图/测量图
    if image_shape is not None:
        h, w = image_shape
        np.save(output_dir / "input_cube_151.npy", spectra_flat.reshape(h, w, 151).astype(np.float32))
        np.save(output_dir / "reconstructed_cube_151.npy", pred_flat.reshape(h, w, 151).astype(np.float32))
        np.save(output_dir / "measurement_16ch_image.npy", meas_flat.reshape(h, w, 16).astype(np.float32))
        plot_error_map(spectra_flat, pred_flat, image_shape, output_dir / "reconstruction_error_map.png")
        plot_measurement_preview(meas_flat, image_shape, output_dir / "measurement_channels_preview.png")

    plot_selected_spectra(spectra_flat, pred_flat, image_shape, settings["plot_pixels"],
                          output_dir / "selected_pixel_spectra.png")

    print()
    print("推理完成")
    print(f"  mse={metrics['mse']:.6e}, l1={metrics['l1']:.6e}, "
          f"diff={metrics['diff_l1']:.6e}, psnr={metrics['psnr']:.2f}, sam={metrics['sam']:.4f}")
    print(f"  结果已保存到: {output_dir}")
    print("  重点看:")
    print(f"    {output_dir / 'selected_pixel_spectra.png'}")
    if image_shape is not None:
        print(f"    {output_dir / 'reconstruction_error_map.png'}")
        print(f"    {output_dir / 'measurement_channels_preview.png'}")


if __name__ == "__main__":
    main()
