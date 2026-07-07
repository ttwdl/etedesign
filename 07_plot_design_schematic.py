"""画当前最优 AR-EMT 滤光片结构示意图。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 07_plot_design_schematic.py

输出:
  results/design_schematic.png

这张图用于看最终结构，不是严格按真实厚度比例绘制。
原因是 EMT 腔可能几百 nm，而 AR 层几十 nm，按真实比例画会很难看清。
图中标注的数值是真实训练值。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from ar_emt_common import AREMTModel, GeometryConfig, structure_rows


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    "checkpoint": "checkpoints/ar_emt_best.pt",
    "output_png": "results/design_schematic.png",
}


def setup_font() -> None:
    """设置中文字体。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_model(checkpoint_path: Path) -> AREMTModel:
    """从 checkpoint 恢复模型。"""

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GeometryConfig(**ckpt["config"])
    wl_nm = ckpt["wl_nm"].to(dtype=torch.float32)
    model = AREMTModel(wl_nm, config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def draw_layer_stack(ax, model: AREMTModel) -> None:
    """画纵向层结构。

    真实结构：
        air / SiO2 / TiO2 / residual SU-8 /
        EMT cavity / TiO2 / SiO2 / fused silica
    """

    params = model.physical_parameters()
    ar = params["ar_nm"].detach().cpu().numpy()
    h_c = float(params["h_c_nm"].detach().cpu())
    t_r = float(params["t_r_nm"].detach().cpu())
    rows = structure_rows(model)
    d_mid = rows[len(rows) // 2]["D_nm"]
    period = model.config.period_nm

    layers = [
        ("top L: SiO2", ar[0], "#9ecae1"),
        ("top H: TiO2", ar[1], "#fdae6b"),
        ("residual SU-8", t_r, "#c7e9c0"),
        ("EMT cavity\nTiO2 pillars + SU-8 fill", h_c, "#fff7bc"),
        ("bottom H: TiO2", ar[2], "#fdae6b"),
        ("bottom L: SiO2", ar[3], "#9ecae1"),
    ]

    def draw_height(thickness_nm: float) -> float:
        # 用 sqrt 压缩厚度差异，让薄层也能看见。
        return max(0.18, np.sqrt(thickness_nm) / 7.0)

    x0, width = 0.12, 0.78
    y = 0.0
    y_positions = []
    for name, thickness, color in reversed(layers):
        h = draw_height(float(thickness))
        ax.add_patch(Rectangle((x0, y), width, h, facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(
            x0 + width + 0.04,
            y + h / 2,
            f"{name}: {thickness:.2f} nm",
            va="center",
            fontsize=9,
        )
        y_positions.append((name, y, h))
        y += h

    ax.add_patch(Rectangle((x0, -0.38), width, 0.38, facecolor="#d9d9d9", edgecolor="black", linewidth=0.8))
    ax.text(x0 + width / 2, -0.19, "fused silica substrate", ha="center", va="center", fontsize=9)
    ax.text(x0 + width / 2, y + 0.18, "air", ha="center", va="center", fontsize=10)

    cavity = [v for v in y_positions if v[0].startswith("EMT cavity")][0]
    _, cy, ch = cavity
    pillar_width = width * (d_mid / period) * 0.22
    for px in [x0 + width * 0.28, x0 + width * 0.50, x0 + width * 0.72]:
        ax.add_patch(
            Rectangle(
                (px - pillar_width / 2, cy),
                pillar_width,
                ch,
                facecolor="#e6550d",
                edgecolor="#7f2704",
                linewidth=0.6,
            )
        )
    ax.text(
        x0 + width / 2,
        cy + ch / 2,
        "SU-8 fills gaps\nTiO2 pillars",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )

    ax.set_xlim(0, 1.85)
    ax.set_ylim(-0.45, y + 0.45)
    ax.set_title("Vertical stack")
    ax.axis("off")


def draw_channel_layout(ax, model: AREMTModel) -> None:
    """画 4x4 滤光片的柱径分布。"""

    rows = structure_rows(model)
    d_values = np.array([r["D_nm"] for r in rows]).reshape(4, 4)
    gap_values = np.array([r["gap_nm"] for r in rows]).reshape(4, 4)
    period = model.config.period_nm

    ax.set_aspect("equal")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()
    ax.set_title("4x4 filters: only D changes")

    for i in range(4):
        for j in range(4):
            x, y = j, i
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#f7f7f7", edgecolor="black", linewidth=0.8))
            radius = 0.38 * d_values[i, j] / d_values.max()
            ax.add_patch(Circle((x + 0.5, y + 0.42), radius, facecolor="#e6550d", edgecolor="#7f2704"))
            ch = i * 4 + j
            ax.text(
                x + 0.5,
                y + 0.78,
                f"ch{ch}\nD={d_values[i,j]:.1f} nm\nG={gap_values[i,j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.text(2, 4.25, f"Period P = {period:.1f} nm", ha="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_spectra(ax, model: AREMTModel) -> None:
    """画 0 度透过谱。"""

    with torch.no_grad():
        t0 = model.transmission(torch.tensor([0.0]))[0].detach().cpu().numpy()
    wl = model.wl_nm.detach().cpu().numpy()

    for idx in range(t0.shape[0]):
        ax.plot(wl, t0[idx], lw=1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission")
    ax.set_title("0 deg transmission spectra")
    ax.set_ylim(0.0, 1.05)
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

    fig = plt.figure(figsize=(13, 9))
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
