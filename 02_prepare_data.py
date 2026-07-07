"""准备训练数据缓存。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 02_prepare_data.py

本脚本默认读取已经解压好的 CAVE PNG 数据：
  E:\\hyperspectral_datasets\\CAVE\\extracted

输出新的 absolute 缓存：
  E:\\hyperspectral_datasets\\CAVE\\data_cache_absolute_100k

重要变化：
  旧版本会把每条光谱除以自己的最大值，所以只保留光谱形状。
  新版本不做逐条归一化，只按图像位深缩放：
    8-bit  PNG: 除以 255
    16-bit PNG: 除以 65535
  这样像素之间的明暗关系会被保留下来，更接近真实探测器强度。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.io import loadmat


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    # 数据来源模式：
    #   "dir"       : 从目录里读 CAVE PNG / npy / npz / mat
    #   "npy"       : 只读一个 npy
    #   "synthetic" : 生成假光谱，只用来检查代码能不能跑通
    "mode": "dir",

    # CAVE 解压目录。不要写到 data_cache_100k，那个是旧缓存。
    "input_dir": r"E:\hyperspectral_datasets\CAVE\extracted",
    "input_npy": "",

    # 新缓存目录。这个目录和旧的 data_cache_100k 分开，避免覆盖旧结果。
    "output_dir": r"E:\hyperspectral_datasets\CAVE\data_cache_absolute_100k",

    # 数据量。训练、验证、测试三份互不重叠。
    "train_size": 100000,
    "val_size": 10000,
    "test_size": 10000,

    # 随机种子固定后，每次抽样顺序相同，方便复现。
    "seed": 2026,
}


WL_151 = np.linspace(400.0, 700.0, 151).astype(np.float32)
WL_31 = np.linspace(400.0, 700.0, 31).astype(np.float32)


def clean_spectra(spectra: np.ndarray) -> np.ndarray:
    """整理光谱数组，但不做逐条归一化。

    输入可以是 [N, 波长数]，也可以是 [H, W, 波长数]。
    输出统一变成 [N, 波长数]。

    这里只做三件事：
    1. NaN / inf 变成 0；
    2. 负值裁掉，因为光强不应为负；
    3. 大于 1 的值裁到 1，因为 PNG 已经按位深缩放到 0-1。
    """

    spectra = np.asarray(spectra, dtype=np.float32)
    spectra = spectra.reshape(-1, spectra.shape[-1])
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
    spectra = np.clip(spectra, 0.0, 1.0)
    return spectra.astype(np.float32)


def interpolate_to_151(spectra: np.ndarray) -> np.ndarray:
    """把 31 通道 CAVE 光谱插值到 151 通道。

    CAVE 原始波长通常是 400-700 nm，每 10 nm 一个点，共 31 点。
    训练模型用 400-700 nm，每 2 nm 一个点，共 151 点。

    注意：插值后仍然不做逐条归一化，只裁剪到 0-1。
    """

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
    """读取一张 PNG，并按位深缩放到 0-1。

    这个函数是“保留绝对强度”的关键：
    - 不会让每个像素单独除以最大值；
    - 只按照图像格式的位深做统一缩放。
    """

    img = Image.open(path)
    arr = np.asarray(img)

    if arr.ndim == 3:
        # 理论上 CAVE 每个波段是灰度图。如果遇到 RGB，就取平均，保持代码能跑。
        arr = arr.mean(axis=2)

    if np.issubdtype(arr.dtype, np.uint8):
        scale = 255.0
    elif np.issubdtype(arr.dtype, np.uint16):
        scale = 65535.0
    else:
        # 少数情况下 PIL 会返回 int32/float。这里尽量保守：
        # 如果最大值已经 <=1，就认为它已经是 0-1；
        # 否则用当前 dtype 能表示的最大整数估一个缩放。
        max_val = float(np.nanmax(arr))
        scale = 1.0 if max_val <= 1.0 else max_val

    out = arr.astype(np.float32) / scale
    return np.clip(out, 0.0, 1.0)


def make_synthetic_spectra(n_samples: int, seed: int) -> np.ndarray:
    """生成假光谱，只用于冒烟测试。

    这些光谱不是实验数据，只是让训练脚本能快速跑通。
    为了模拟“绝对强度”，这里会给每条光谱一个随机亮度系数，
    不再把每条光谱强行归一到最大值 1。
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
    """读取一个 npy 文件。

    如果 npy 里面已经是 0-1，就直接用。
    如果里面是 0-255 或 0-65535 这种整数灰度，建议你先自己确认单位。
    这里为了安全，会裁剪到 0-1，不做逐条归一化。
    """

    arr = np.load(path)
    return interpolate_to_151(arr)


def load_spectra_from_npz(path: Path) -> list[np.ndarray]:
    """读取 npz 中所有看起来像光谱的数据。"""

    out = []
    data = np.load(path)
    for key in data.files:
        arr = data[key]
        if arr.ndim >= 2 and arr.shape[-1] in {31, 151}:
            out.append(interpolate_to_151(arr))
    return out


def load_spectra_from_mat(path: Path) -> list[np.ndarray]:
    """读取 mat 中所有看起来像光谱的数据。"""

    out = []
    data = loadmat(path)
    for key, arr in data.items():
        if key.startswith("__"):
            continue
        if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[-1] in {31, 151}:
            out.append(interpolate_to_151(arr))
    return out


def find_cave_scenes(input_dir: Path) -> list[Path]:
    """寻找 CAVE 解压后的场景目录。

    CAVE 每个场景通常包含 31 张 PNG，文件名类似：
        balloons_ms_01.png
        balloons_ms_02.png
        ...
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
    """读取一个 CAVE 场景，返回 [H*W, 31]。

    每个波段先按位深缩放到 0-1，然后堆成一个光谱 cube。
    """

    pngs = sorted(scene_dir.glob("*_ms_*.png"))[:31]
    if len(pngs) != 31:
        raise ValueError(f"CAVE 场景 {scene_dir} 不是 31 个波段，实际 {len(pngs)}")

    bands = [image_to_float01(path) for path in pngs]
    cube = np.stack(bands, axis=-1)
    return clean_spectra(cube.reshape(-1, 31))


def load_cave_png_dir(input_dir: Path, max_samples: int, seed: int) -> np.ndarray:
    """从 CAVE PNG 场景中抽样像素光谱。

    CAVE 全部像素很多，没有必要全部加载进训练。
    这里按场景均匀抽样，保证不同场景都有代表。
    """

    scenes = find_cave_scenes(input_dir)
    if not scenes:
        raise RuntimeError(f"没有在 {input_dir} 找到 CAVE PNG 场景。")

    rng = np.random.default_rng(seed)
    per_scene = int(np.ceil(max_samples / len(scenes)))
    chunks = []
    print(f"检测到 CAVE PNG 场景 {len(scenes)} 个，每个场景最多抽样 {per_scene} 条。")

    for scene in scenes:
        spectra31 = load_cave_png_scene(scene)
        take = min(per_scene, spectra31.shape[0])
        idx = rng.choice(spectra31.shape[0], size=take, replace=False)
        chunks.append(spectra31[idx])
        print(f"  {scene.name}: {spectra31.shape[0]} pixels -> sample {take}")

    spectra31 = np.concatenate(chunks, axis=0)
    if spectra31.shape[0] > max_samples:
        idx = rng.choice(spectra31.shape[0], size=max_samples, replace=False)
        spectra31 = spectra31[idx]

    return interpolate_to_151(spectra31)


def load_spectra_from_dir(input_dir: Path, max_samples: int, seed: int) -> np.ndarray:
    """从目录读取光谱。

    优先识别 CAVE PNG 场景；如果不是 CAVE PNG，就尝试读取 npy/npz/mat。
    """

    cave_scenes = find_cave_scenes(input_dir)
    if cave_scenes:
        return load_cave_png_dir(input_dir, max_samples=max_samples, seed=seed)

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


def split_and_save(
    spectra: np.ndarray,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    output_dir: Path,
) -> None:
    """打乱并保存 train/val/test 三份缓存。"""

    spectra = clean_spectra(spectra)
    need = train_size + val_size + test_size
    if spectra.shape[0] < need:
        raise ValueError(f"光谱数量不足：需要 {need}，实际 {spectra.shape[0]}。")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(spectra.shape[0])[:need]
    train = spectra[idx[:train_size]]
    val = spectra[idx[train_size:train_size + val_size]]
    test = spectra[idx[train_size + val_size:]]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_spectra.npy", train.astype(np.float32))
    np.save(output_dir / "val_spectra.npy", val.astype(np.float32))
    np.save(output_dir / "test_spectra.npy", test.astype(np.float32))
    np.save(output_dir / "wl_nm.npy", WL_151.astype(np.float32))

    info = (
        "AR-EMT absolute cache\n"
        "scale: CAVE PNG bit-depth scaling, no per-spectrum normalization\n"
        f"train_spectra.npy: {train.shape}, min={train.min():.6f}, max={train.max():.6f}, mean={train.mean():.6f}\n"
        f"val_spectra.npy  : {val.shape}, min={val.min():.6f}, max={val.max():.6f}, mean={val.mean():.6f}\n"
        f"test_spectra.npy : {test.shape}, min={test.min():.6f}, max={test.max():.6f}, mean={test.mean():.6f}\n"
        f"wl_nm.npy        : {WL_151.shape}\n"
    )
    (output_dir / "data_info.txt").write_text(info, encoding="utf-8")

    print(f"保存完成: {output_dir}")
    print(info)


def main() -> None:
    settings = USER_SETTINGS
    total = settings["train_size"] + settings["val_size"] + settings["test_size"]

    if settings["mode"] == "synthetic":
        spectra = make_synthetic_spectra(total, settings["seed"])
        print("使用合成光谱生成缓存。")
    elif settings["mode"] == "npy":
        if not settings["input_npy"]:
            raise ValueError("mode='npy' 时必须填写 USER_SETTINGS['input_npy']")
        spectra = load_spectra_from_npy(Path(settings["input_npy"]))
        print(f"从 npy 读取光谱: {settings['input_npy']}")
    elif settings["mode"] == "dir":
        if not settings["input_dir"]:
            raise ValueError("mode='dir' 时必须填写 USER_SETTINGS['input_dir']")
        spectra = load_spectra_from_dir(Path(settings["input_dir"]), max_samples=total, seed=settings["seed"])
        print(f"从目录读取光谱: {settings['input_dir']}")
    else:
        raise ValueError(f"未知 mode: {settings['mode']}")

    split_and_save(
        spectra=spectra,
        train_size=settings["train_size"],
        val_size=settings["val_size"],
        test_size=settings["test_size"],
        seed=settings["seed"],
        output_dir=Path(settings["output_dir"]),
    )


if __name__ == "__main__":
    main()
