"""画当前最优 AR-EMT 滤光片结构示意图（看最终设计长什么样）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 07_plot_design_schematic.py

输出:
  results_25ch_t06_tor20_50/design_schematic.png

这张图分三块：左=纵向层结构，右上=滤光片 D/h_c 分布，下=0 度透过谱。
注意：图不是按真实厚度比例画的（EMT 腔几百 nm，AR 层几十 nm，按真实比例会看不清），
     但图上标注的数值都是真实训练值。
"""

from __future__ import annotations

import os
from pathlib import Path

# Windows + conda 里，PyTorch / NumPy / SciPy / Matplotlib 有时会重复加载 Intel OpenMP。
# 如果不提前设置，可能出现 “OMP: Error #15: Initializing libiomp5md.dll”。
# 这行只影响当前脚本进程；以后如果你重装环境彻底解决冲突，可以删掉它。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from ar_emt_common import AREMTModel, GeometryConfig, model_kwargs_from_settings, structure_rows


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    "checkpoint": "checkpoints_25ch_t06_tor20_50/ar_emt_best.pt",
    "output_png": "results_25ch_t06_tor20_50/design_schematic.png",
}


def setup_font() -> None:
    """设置中文字体，避免中文变方框。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_model(checkpoint_path: Path) -> AREMTModel:
    """从 checkpoint 恢复模型（放 CPU 上画图就够了）。"""

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GeometryConfig(**ckpt["config"])
    wl_nm = ckpt["wl_nm"].to(dtype=torch.float32)
    model = AREMTModel(wl_nm, config, **model_kwargs_from_settings(ckpt.get("settings", {})))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def draw_layer_stack(ax, model: AREMTModel) -> None:
    """画纵向层结构。

    新模型里每个通道的 h_c / t_r 可以不同，但 h_c + t_r 固定。
    这张剖面图取中间通道做代表，并在文字里标出全通道范围。
    """

    params = model.physical_parameters()
    ar = params["ar_nm"].detach().cpu().numpy()
    h_c_all = params["h_c_nm"].detach().cpu().numpy()
    t_r_all = params["t_r_nm"].detach().cpu().numpy()
    core_total = float(params["core_total_nm"].detach().cpu())
    aspect = params["aspect_ratio"].detach().cpu().numpy()
    aspect_max = float(params["aspect_ratio_max"].detach().cpu())
    mid_ch = len(h_c_all) // 2
    h_c = float(h_c_all[mid_ch])
    t_r = float(t_r_all[mid_ch])
    rows = structure_rows(model)
    d_mid = rows[mid_ch]["D_nm"]   # 取中间通道的柱径，画 EMT 腔里的柱子示意
    period = model.config.period_nm

    # 从上到下的层（名字, 厚度nm, 颜色）
    layers = [
        ("top L: SiO2", ar[0], "#9ecae1"),
        ("top H: TiO2", ar[1], "#fdae6b"),
        ("residual SU-8", t_r, "#c7e9c0"),
        ("EMT cavity\nTiO2 pillars + SU-8 fill", h_c, "#fff7bc"),
        ("bottom H: TiO2", ar[2], "#fdae6b"),
        ("bottom L: SiO2", ar[3], "#9ecae1"),
    ]

    def draw_height(thickness_nm: float) -> float:
        # 用开平方压缩厚度差异，让几十 nm 的薄层也能看见（纯为可视化，不代表真实比例）
        return max(0.18, np.sqrt(thickness_nm) / 7.0)

    x0, width = 0.12, 0.78
    y = 0.0
    y_positions = []
    for name, thickness, color in reversed(layers):   # reversed: 从下往上堆
        h = draw_height(float(thickness))
        ax.add_patch(Rectangle((x0, y), width, h, facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(x0 + width + 0.04, y + h / 2, f"{name}: {thickness:.2f} nm", va="center", fontsize=9)
        y_positions.append((name, y, h))
        y += h

    # 底部基底 + 顶部空气
    ax.add_patch(Rectangle((x0, -0.38), width, 0.38, facecolor="#d9d9d9", edgecolor="black", linewidth=0.8))
    ax.text(x0 + width / 2, -0.19, "fused silica substrate", ha="center", va="center", fontsize=9)
    ax.text(x0 + width / 2, y + 0.18, "air", ha="center", va="center", fontsize=10)

    # 在 EMT 腔里画几根 TiO2 柱子示意
    cavity = [v for v in y_positions if v[0].startswith("EMT cavity")][0]
    _, cy, ch = cavity
    pillar_width = width * (d_mid / period) * 0.22
    for px in [x0 + width * 0.28, x0 + width * 0.50, x0 + width * 0.72]:
        ax.add_patch(Rectangle((px - pillar_width / 2, cy), pillar_width, ch,
                               facecolor="#e6550d", edgecolor="#7f2704", linewidth=0.6))
    ax.text(x0 + width / 2, cy + ch / 2, "SU-8 fills gaps\nTiO2 pillars",
            ha="center", va="center", fontsize=9, color="black")
    ax.text(
        x0,
        y + 0.02,
        f"shown: ch{mid_ch}; trained global H_total = h_c + t_r = {core_total:.1f} nm\n"
        f"h_c range {h_c_all.min():.1f}-{h_c_all.max():.1f} nm, "
        f"t_r range {t_r_all.min():.1f}-{t_r_all.max():.1f} nm\n"
        f"aspect h_c/D range {aspect.min():.2f}-{aspect.max():.2f}, limit {aspect_max:.1f}",
        ha="left",
        va="bottom",
        fontsize=8,
    )

    ax.set_xlim(0, 1.85)
    ax.set_ylim(-0.45, y + 0.70)
    ax.set_title("Vertical stack")
    ax.axis("off")


def draw_channel_layout(ax, model: AREMTModel) -> None:
    """画所有滤光片的柱径和 EMT 腔厚分布。

    16 通道会自动画成 4x4，25 通道会自动画成 5x5。
    """

    rows = structure_rows(model)
    n_channels = len(rows)
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    d_values = np.array([r["D_nm"] for r in rows])
    gap_values = np.array([r["gap_nm"] for r in rows])
    hc_values = np.array([r["h_c_nm"] for r in rows])
    tr_values = np.array([r["t_r_nm"] for r in rows])
    period = model.config.period_nm

    ax.set_aspect("equal")
    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_title(f"{n_channels} filters: D and h_c change, shared H_total")

    for ch in range(n_rows * n_cols):
        i = ch // n_cols
        j = ch % n_cols
        x, y = j, i
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#f7f7f7", edgecolor="black", linewidth=0.8))
        if ch >= n_channels:
            continue
        radius = 0.38 * d_values[ch] / d_values.max()   # 圆点大小按柱径归一化
        ax.add_patch(Circle((x + 0.5, y + 0.40), radius, facecolor="#e6550d", edgecolor="#7f2704"))
        ax.text(
            x + 0.5,
            y + 0.78,
            f"ch{ch}\nD={d_values[ch]:.0f} G={gap_values[ch]:.0f}\n"
            f"h={hc_values[ch]:.0f} r={tr_values[ch]:.0f}",
            ha="center",
            va="center",
            fontsize=6.0 if n_channels > 16 else 7.0,
        )

    ax.text(n_cols / 2, n_rows + 0.25, f"Period P = {period:.1f} nm", ha="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def draw_spectra(ax, model: AREMTModel) -> None:
    """画 0 度透过谱。"""

    with torch.no_grad():
        t0 = model.transmission(torch.tensor([0.0]))[0].detach().cpu().numpy()
    wl = model.wl_nm.detach().cpu().numpy()

    for idx in range(t0.shape[0]):
        ax.plot(wl, t0[idx], lw=1.0)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Transmission")
    ax.set_title("0 deg transmission spectra"); ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)


def main() -> None:
    setup_font()
    settings = USER_SETTINGS
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}，请先运行 03_train_ar_emt.py")

    model = load_model(checkpoint)
    out_path = Path(settings["output_png"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.2])
    ax_stack = fig.add_subplot(gs[0, 0])
    ax_layout = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[1, :])

    draw_layer_stack(ax_stack, model)
    draw_channel_layout(ax_layout, model)
    draw_spectra(ax_spec, model)

    fig.suptitle("AR-EMT filter design from best checkpoint", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"结构示意图已保存: {out_path}")


if __name__ == "__main__":
    main()
