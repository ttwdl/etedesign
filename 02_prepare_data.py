"""准备训练数据缓存（把 CAVE 高光谱图片变成 train/val/test 三份 npy）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 02_prepare_data.py

默认读取已解压的 CAVE PNG 数据：
  E:\\hyperspectral_datasets\\CAVE\\extracted
输出缓存：
  E:\\hyperspectral_datasets\\CAVE\\data_cache_absolute_100k

==================== 两个关键设计点，务必理解 ====================

【1】保留“绝对强度”，不做逐条归一化
  旧做法会把每条光谱除以它自己的最大值，只剩下“形状”，丢掉了像素间明暗关系。
  这里只按图像位深统一缩放：8-bit PNG 除以 255，16-bit PNG 除以 65535。
  这样更接近真实探测器读到的强度，像素之间“谁亮谁暗”被保留下来。

【2】按“场景”划分 train/val/test，而不是按“像素”
  CAVE 一共才三十多个场景，同一张图里相邻像素几乎一模一样。
  如果把所有场景的像素混在一起再随机切三份，训练集和验证集就会出现
  “来自同一张图的像素” —— 相当于验证时提前看过答案，val/test 分数会虚高。
  正确做法：先把“整张场景”分给 train / val / test，再各自从自己的场景里采样像素。
  这样三份数据来自完全不同的场景，验证/测试分数才诚实可信。
  （注意：这条只对 CAVE PNG 生效。npy/npz/mat/synthetic 这些回退路径，
    因为拿不到“场景”概念，仍走“先合并再随机切”，代码里有标注。）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.io import loadmat


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    # 数据来源模式：
    #   "dir"       : 从目录读 CAVE PNG（推荐），或 npy/npz/mat
    #   "npy"       : 只读一个 npy
    #   "synthetic" : 生成假光谱，只用来检查代码能不能跑通
    "mode": "dir",

    # CAVE 解压目录（不要指到旧缓存目录）。
    "input_dir": r"E:\hyperspectral_datasets\CAVE\extracted",
    "input_npy": "",

    # 新缓存目录，和旧缓存分开，避免覆盖。
    "output_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",

    # 每份要多少条光谱。三份互不重叠。
    "train_size": 100000,
    "val_size": 10000,
    "test_size": 10000,

    # 按场景划分时，验证/测试各占多少比例的“场景数”（剩下的都给训练）。
    # 例：0.15 + 0.15 表示 15% 场景做验证、15% 做测试、70% 做训练。
    "val_scene_frac": 0.15,
    "test_scene_frac": 0.15,

    # 随机种子固定后，场景划分和像素采样顺序都可复现。
    "seed": 2026,
}


# 模型用 400-700 nm、每 2 nm 一点，共 151 点；CAVE 原始是每 10 nm 一点，共 31 点。
WL_151 = np.linspace(400.0, 700.0, 151).astype(np.float32)
WL_31 = np.linspace(400.0, 700.0, 31).astype(np.float32)


# =============================================================================
# 光谱清理 / 插值 / 读图（这几个函数是“绝对强度”处理的核心，逻辑没变）
# =============================================================================


def clean_spectra(spectra: np.ndarray) -> np.ndarray:
    """整理光谱数组，但不做逐条归一化。

    输入可以是 [N, 波长] 或 [H, W, 波长]，输出统一成 [N, 波长]。
    只做三件事：
    1. NaN / inf → 0；
    2. 负值裁成 0（光强不应为负）；
    3. 大于 1 的值裁到 1（PNG 已按位深缩放到 0~1）。
    """

    spectra = np.asarray(spectra, dtype=np.float32)
    spectra = spectra.reshape(-1, spectra.shape[-1])
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
    spectra = np.clip(spectra, 0.0, 1.0)
    return spectra.astype(np.float32)


def interpolate_to_151(spectra: np.ndarray) -> np.ndarray:
    """把 31 通道光谱三次样条插值到 151 通道；插值后仍只裁剪、不逐条归一化。"""

    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.shape[-1] == 151:
        return clean_spectra(spectra)
    if spectra.shape[-1] != 31:
        raise ValueError(f"只支持最后一维为 31 或 151 的光谱，实际 shape={spectra.shape}")

    flat = spectra.reshape(-1, 31)
    cs = CubicSpline(WL_31, flat, axis=1)
    out = cs(WL_151)
    return clean_spectra(out)


def image_to_float01(path: Path) -> np.ndarray:
    """读一张 PNG，按位深缩放到 0~1（保留绝对强度的关键）。

    - 8-bit  → 除以 255；
    - 16-bit → 除以 65535；
    - 不做逐像素/逐光谱最大值归一化。
    """

    img = Image.open(path)
    arr = np.asarray(img)

    if arr.ndim == 3:
        # CAVE 每个波段本应是灰度图；万一遇到 RGB 就取平均，保证能跑。
        arr = arr.mean(axis=2)

    if np.issubdtype(arr.dtype, np.uint8):
        scale = 255.0
    elif np.issubdtype(arr.dtype, np.uint16):
        scale = 65535.0
    else:
        # 少数情况 PIL 返回 int32/float：若最大值<=1 认为已是 0~1，否则按当前最大值估一个缩放。
        max_val = float(np.nanmax(arr))
        scale = 1.0 if max_val <= 1.0 else max_val

    out = arr.astype(np.float32) / scale
    return np.clip(out, 0.0, 1.0)


# =============================================================================
# 找 CAVE 场景 / 读一个场景 / 从若干场景里采样像素
# =============================================================================


def find_cave_scenes(input_dir: Path) -> list[Path]:
    """找 CAVE 解压后的场景目录。

    每个场景通常含 31 张 PNG，文件名类似 balloons_ms_01.png ... balloons_ms_31.png。
    """

    scenes = []
    for path in input_dir.rglob("*"):
        if not path.is_dir():
            continue
        pngs = sorted(path.glob("*_ms_*.png"))
        if len(pngs) >= 31:
            scenes.append(path)
    return sorted(set(scenes))


def load_cave_png_scene(scene_dir: Path) -> np.ndarray:
    """读一个 CAVE 场景，返回 [H*W, 31]。每个波段先按位深缩放到 0~1，再堆成光谱 cube。"""

    pngs = sorted(scene_dir.glob("*_ms_*.png"))[:31]
    if len(pngs) != 31:
        raise ValueError(f"CAVE 场景 {scene_dir} 不是 31 个波段，实际 {len(pngs)}")

    bands = [image_to_float01(path) for path in pngs]
    cube = np.stack(bands, axis=-1)
    return clean_spectra(cube.reshape(-1, 31))


def sample_pixels_from_scenes(scenes: list[Path], n_samples: int, seed: int, tag: str) -> np.ndarray:
    """从给定的一批场景里，均匀采样 n_samples 条像素光谱，返回 [n_samples, 31]。

    做法：先给每个场景平摊一个采样配额，各场景采样后拼起来；
    若总数超过目标，再随机抽到目标数量。保证每个场景都有代表。
    """

    if not scenes:
        raise RuntimeError(f"[{tag}] 没有可用场景，无法采样。")

    rng = np.random.default_rng(seed)
    per_scene = int(np.ceil(n_samples / len(scenes)))
    chunks = []
    print(f"[{tag}] 使用 {len(scenes)} 个场景，每个场景最多采样 {per_scene} 条：")

    for scene in scenes:
        spectra31 = load_cave_png_scene(scene)
        take = min(per_scene, spectra31.shape[0])
        idx = rng.choice(spectra31.shape[0], size=take, replace=False)
        chunks.append(spectra31[idx])
        print(f"    {scene.name}: {spectra31.shape[0]} pixels -> sample {take}")

    spectra31 = np.concatenate(chunks, axis=0)
    if spectra31.shape[0] > n_samples:
        idx = rng.choice(spectra31.shape[0], size=n_samples, replace=False)
        spectra31 = spectra31[idx]
    elif spectra31.shape[0] < n_samples:
        raise ValueError(f"[{tag}] 采到的像素不足：需要 {n_samples}，实际 {spectra31.shape[0]}，"
                         f"请减小该份 size 或增加场景数量。")
    return spectra31


def split_cave_scenes(
    scenes: list[Path],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """把“整张场景”打乱后分成 train / val / test 三组（纯函数，方便理解和测试）。

    关键：划分的是“场景”，不是像素。这样三份数据来自完全不同的图，杜绝串味。
    """

    n = len(scenes)
    if n < 3:
        raise RuntimeError(f"CAVE 场景太少（{n} 个），无法按场景划分 train/val/test。")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    if n_val + n_test >= n:
        raise RuntimeError(f"验证+测试占用的场景数({n_val}+{n_test})不能 >= 总场景数({n})，请调小比例。")

    val_idx = order[:n_val]
    test_idx = order[n_val:n_val + n_test]
    train_idx = order[n_val + n_test:]

    pick = lambda ids: [scenes[i] for i in ids]
    return pick(train_idx), pick(val_idx), pick(test_idx)


# =============================================================================
# 回退路径：npy / npz / mat / synthetic（拿不到“场景”，仍用合并后随机切）
# =============================================================================


def make_synthetic_spectra(n_samples: int, seed: int) -> np.ndarray:
    """生成假光谱，只用于冒烟测试（确认代码能跑通）。

    每条光谱是若干高斯峰叠加，再乘一个随机亮度系数来模拟“绝对强度”，
    不做逐条归一化。这些不是实验数据。
    """

    rng = np.random.default_rng(seed)
    spectra = np.zeros((n_samples, WL_151.size), dtype=np.float32)

    for i in range(n_samples):
        y = np.zeros(WL_151.size, dtype=np.float32)
        n_peaks = rng.integers(1, 5)
        for _ in range(n_peaks):
            center = rng.uniform(410.0, 690.0)
            width = rng.uniform(12.0, 75.0)
            amp = rng.uniform(0.15, 1.0)
            y += amp * np.exp(-0.5 * ((WL_151 - center) / width) ** 2)
        y += rng.uniform(0.01, 0.08)
        y *= rng.uniform(0.15, 1.0)
        spectra[i] = y

    return clean_spectra(spectra)


def load_spectra_from_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    return interpolate_to_151(arr)


def load_spectra_from_npz(path: Path) -> list[np.ndarray]:
    out = []
    data = np.load(path)
    for key in data.files:
        arr = data[key]
        if arr.ndim >= 2 and arr.shape[-1] in {31, 151}:
            out.append(interpolate_to_151(arr))
    return out


def load_spectra_from_mat(path: Path) -> list[np.ndarray]:
    out = []
    data = loadmat(path)
    for key, arr in data.items():
        if key.startswith("__"):
            continue
        if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[-1] in {31, 151}:
            out.append(interpolate_to_151(arr))
    return out


def load_pooled_from_dir(input_dir: Path, max_samples: int, seed: int) -> np.ndarray:
    """从目录读 npy/npz/mat，合并成一个大池子（仅在没有 CAVE 场景时走这里）。"""

    chunks: list[np.ndarray] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".npy":
                chunks.append(load_spectra_from_npy(path))
            elif path.suffix.lower() == ".npz":
                chunks.extend(load_spectra_from_npz(path))
            elif path.suffix.lower() == ".mat":
                chunks.extend(load_spectra_from_mat(path))
        except Exception as exc:
            print(f"跳过 {path}: {type(exc).__name__}: {exc}")

    if not chunks:
        raise RuntimeError(f"没有在 {input_dir} 找到最后一维为 31 或 151 的光谱数组。")

    spectra = clean_spectra(np.concatenate(chunks, axis=0))
    if spectra.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(spectra.shape[0], size=max_samples, replace=False)
        spectra = spectra[idx]
    return spectra


# =============================================================================
# 保存
# =============================================================================


def save_splits(train: np.ndarray, val: np.ndarray, test: np.ndarray, output_dir: Path, split_note: str) -> None:
    """把已经切好的三份光谱插值到 151、清理并保存成 npy，同时写一份说明文件。"""

    train = interpolate_to_151(train)
    val = interpolate_to_151(val)
    test = interpolate_to_151(test)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_spectra.npy", train.astype(np.float32))
    np.save(output_dir / "val_spectra.npy", val.astype(np.float32))
    np.save(output_dir / "test_spectra.npy", test.astype(np.float32))
    np.save(output_dir / "wl_nm.npy", WL_151.astype(np.float32))

    info = (
        "AR-EMT absolute cache\n"
        "scale: CAVE PNG bit-depth scaling, no per-spectrum normalization\n"
        f"split: {split_note}\n"
        f"train_spectra.npy: {train.shape}, min={train.min():.6f}, max={train.max():.6f}, mean={train.mean():.6f}\n"
        f"val_spectra.npy  : {val.shape}, min={val.min():.6f}, max={val.max():.6f}, mean={val.mean():.6f}\n"
        f"test_spectra.npy : {test.shape}, min={test.min():.6f}, max={test.max():.6f}, mean={test.mean():.6f}\n"
        f"wl_nm.npy        : {WL_151.shape}\n"
    )
    (output_dir / "data_info.txt").write_text(info, encoding="utf-8")
    print(f"保存完成: {output_dir}")
    print(info)


def pooled_random_split(spectra: np.ndarray, train_size: int, val_size: int, test_size: int,
                        seed: int, output_dir: Path) -> None:
    """把一个大池子打乱后随机切三份（仅用于 npy/mat/synthetic 这些独立样本）。"""

    spectra = clean_spectra(spectra)
    need = train_size + val_size + test_size
    if spectra.shape[0] < need:
        raise ValueError(f"光谱数量不足：需要 {need}，实际 {spectra.shape[0]}。")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(spectra.shape[0])[:need]
    train = spectra[idx[:train_size]]
    val = spectra[idx[train_size:train_size + val_size]]
    test = spectra[idx[train_size + val_size:]]
    save_splits(train, val, test, output_dir, split_note="pooled random split (samples assumed independent)")


# =============================================================================
# 主流程
# =============================================================================


def main() -> None:
    s = USER_SETTINGS
    output_dir = Path(s["output_dir"])

    if s["mode"] == "synthetic":
        total = s["train_size"] + s["val_size"] + s["test_size"]
        spectra = make_synthetic_spectra(total, s["seed"])
        print("使用合成光谱生成缓存（仅冒烟测试）。")
        pooled_random_split(spectra, s["train_size"], s["val_size"], s["test_size"], s["seed"], output_dir)
        return

    if s["mode"] == "npy":
        if not s["input_npy"]:
            raise ValueError("mode='npy' 时必须填写 USER_SETTINGS['input_npy']")
        total = s["train_size"] + s["val_size"] + s["test_size"]
        spectra = load_spectra_from_npy(Path(s["input_npy"]))
        print(f"从 npy 读取光谱: {s['input_npy']}")
        pooled_random_split(spectra, s["train_size"], s["val_size"], s["test_size"], s["seed"], output_dir)
        return

    if s["mode"] != "dir":
        raise ValueError(f"未知 mode: {s['mode']}")

    # ---- mode == "dir" ----
    input_dir = Path(s["input_dir"])
    if not s["input_dir"]:
        raise ValueError("mode='dir' 时必须填写 USER_SETTINGS['input_dir']")

    scenes = find_cave_scenes(input_dir)
    if scenes:
        # ★ 推荐路径：CAVE PNG，按场景划分 train/val/test
        print(f"检测到 CAVE PNG 场景 {len(scenes)} 个，按场景划分 train/val/test。")
        train_scenes, val_scenes, test_scenes = split_cave_scenes(
            scenes, s["val_scene_frac"], s["test_scene_frac"], s["seed"]
        )
        print(f"  train 场景 {len(train_scenes)} 个 | val 场景 {len(val_scenes)} 个 | test 场景 {len(test_scenes)} 个")
        # 每份用不同的 seed 偏移，避免三份采样到“相同的场景内位置”这种巧合
        train31 = sample_pixels_from_scenes(train_scenes, s["train_size"], s["seed"] + 1, "train")
        val31 = sample_pixels_from_scenes(val_scenes, s["val_size"], s["seed"] + 2, "val")
        test31 = sample_pixels_from_scenes(test_scenes, s["test_size"], s["seed"] + 3, "test")
        save_splits(train31, val31, test31, output_dir,
                    split_note=f"scene-level split (train/val/test scenes = "
                               f"{len(train_scenes)}/{len(val_scenes)}/{len(test_scenes)})")
    else:
        # 回退路径：目录里是 npy/npz/mat，没有场景概念 → 合并后随机切
        print(f"未找到 CAVE 场景，改为读取 npy/npz/mat 并随机切分: {input_dir}")
        total = s["train_size"] + s["val_size"] + s["test_size"]
        spectra = load_pooled_from_dir(input_dir, max_samples=total, seed=s["seed"])
        pooled_random_split(spectra, s["train_size"], s["val_size"], s["test_size"], s["seed"], output_dir)


if __name__ == "__main__":
    main()
