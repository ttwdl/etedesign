"""画 AR-EMT 端到端训练网络结构图。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 08_plot_network_architecture.py

输出:
  results/network_architecture_ppt.png
  results/network_architecture_ppt.svg

这张图用于 PPT 展示，重点说明：
1. 输入是一条光谱，不是图像 patch；
2. 光学编码器是可训练物理结构 + 可微 TMM；
3. 16 个滤光片把 151 维光谱压缩成 16 维测量；
4. MLP 解码器把 16 维测量重建回 151 维光谱；
5. loss 只用 MSE 和透过率约束，tor 只记录不参与 loss。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    "output_dir": "results",
    "output_stem": "network_architecture_ppt",
}


def setup_font() -> None:
    """设置中文字体。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def add_box(
    ax,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    body: str,
    facecolor: str,
    title_size: int = 13,
    body_size: int = 10,
) -> None:
    """画一个说明框。"""

    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#2b2b2b",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=title_size, weight="bold")
    ax.text(x + 0.08, y + h - 0.34, body, ha="left", va="top", fontsize=body_size, linespacing=1.35)


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
    curve: float = 0.0,
    color: str = "#333333",
    dashed: bool = False,
) -> None:
    """画流程箭头。"""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.7,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.08,
            text,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )


def draw_decoder_nodes(ax) -> None:
    """在 MLP 框里画 16->500->151 的全连接示意。"""

    x_cols = [9.20, 9.80, 10.40]
    y_nodes = {
        "in": [3.78, 3.62, 3.46, 3.30],
        "hid": [3.84, 3.68, 3.52, 3.36, 3.20],
        "out": [3.78, 3.62, 3.46, 3.30],
    }

    for y0 in y_nodes["in"]:
        ax.scatter(x_cols[0], y0, s=65, color="#7fb3d5", edgecolor="#1b4f72", zorder=5)
    for y0 in y_nodes["hid"]:
        ax.scatter(x_cols[1], y0, s=65, color="#f7dc6f", edgecolor="#7d6608", zorder=5)
    for y0 in y_nodes["out"]:
        ax.scatter(x_cols[2], y0, s=65, color="#82e0aa", edgecolor="#145a32", zorder=5)

    for ya in y_nodes["in"]:
        for yb in y_nodes["hid"]:
            ax.plot([x_cols[0], x_cols[1]], [ya, yb], color="#888888", linewidth=0.45, alpha=0.55)
    for ya in y_nodes["hid"]:
        for yb in y_nodes["out"]:
            ax.plot([x_cols[1], x_cols[2]], [ya, yb], color="#888888", linewidth=0.45, alpha=0.55)

    ax.text(x_cols[0], 3.03, "16", ha="center", fontsize=10)
    ax.text(x_cols[1], 3.03, "500", ha="center", fontsize=10)
    ax.text(x_cols[2], 3.03, "151", ha="center", fontsize=10)


def main() -> None:
    setup_font()
    settings = USER_SETTINGS
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.0, 5.72, "AR-EMT 端到端训练网络结构", ha="center", va="center", fontsize=24, weight="bold")
    ax.text(
        6.0,
        5.42,
        "输入是按位深缩放后的光谱强度；物理编码器给出 16 通道测量；MLP 线性输出重建 151 维光谱",
        ha="center",
        va="center",
        fontsize=13,
        color="#333333",
    )

    add_box(
        ax,
        (0.30, 3.55),
        (1.70, 1.35),
        "输入光谱",
        "S(λ)\nshape: [B,151]\nλ = 400-700 nm\n保留相对明暗\n不做逐条归一化",
        "#d6eaf8",
    )

    add_box(
        ax,
        (2.45, 3.45),
        (2.45, 1.55),
        "可训练光学编码器",
        "16 个滤光片\n每通道只改变 D/P\nρ → r=D/P → D → f → n_eff\n共享厚度: h_c, t_r, AR 层",
        "#fdebd0",
    )

    add_box(
        ax,
        (5.35, 3.45),
        (1.25, 1.55),
        "可微 TMM",
        "由层结构计算\nT_m(λ, α)\n输出透过谱\nshape: [16,151]",
        "#e8daef",
        title_size=12,
        body_size=9,
    )

    add_box(
        ax,
        (7.05, 3.45),
        (1.45, 1.55),
        "光电测量",
        "y_m = Σ S(λ)T_m(λ)\n保持积分和\nshape: [B,16]",
        "#d5f5e3",
        title_size=12,
        body_size=9,
    )

    add_box(
        ax,
        (8.90, 3.00),
        (2.85, 2.00),
        "解码器 MLP",
        "Linear: 16 → 500\nLeakyReLU(0.01)\nLinear: 500 → 151\n最后一层线性输出\n下方圆点只是维度示意",
        "#fcf3cf",
        body_size=9.5,
    )
    draw_decoder_nodes(ax)

    add_box(
        ax,
        (10.05, 1.35),
        (1.65, 1.10),
        "重建光谱",
        "Ŝ(λ)\nshape: [B,151]\n目标是接近输入 S(λ)",
        "#d4efdf",
    )

    add_box(
        ax,
        (3.50, 0.75),
        (3.80, 1.35),
        "训练目标",
        "loss = MSE(Ŝ, S) + λ_trans · max(0, T_target - T_mean)^2\n默认 T_target = 0.75, λ_trans = 0.05\ntor 只记录，不参与 loss",
        "#f9e79f",
    )

    add_box(
        ax,
        (0.55, 0.75),
        (2.25, 1.35),
        "会被更新的参数",
        "编码器: ρ[16], h_c, t_r, AR[4]\n解码器: 两个 Linear 的权重和偏置\n优化器: AdamW + 梯度裁剪",
        "#fadbd8",
        body_size=9.5,
    )

    add_box(
        ax,
        (7.75, 0.55),
        (2.90, 0.90),
        "训练/验证/测试",
        "train 更新参数，val 选择 best。\ntest 只在最终评估脚本里使用。",
        "#ebedef",
        title_size=12,
        body_size=9.5,
    )

    add_arrow(ax, (2.00, 4.25), (2.45, 4.25), "输入")
    add_arrow(ax, (4.90, 4.25), (5.35, 4.25), "结构参数")
    add_arrow(ax, (6.60, 4.25), (7.05, 4.25), "透过谱")
    add_arrow(ax, (8.50, 4.25), (8.90, 4.25), "16维测量")
    add_arrow(ax, (10.90, 3.00), (10.90, 2.45), "输出")
    add_arrow(ax, (10.05, 1.90), (7.30, 1.40), "与真值比较", curve=-0.12)

    add_arrow(ax, (5.05, 1.50), (3.40, 3.45), "反向传播更新光学结构", curve=-0.25, color="#2874a6", dashed=True)
    add_arrow(ax, (7.30, 1.50), (10.15, 3.00), "反向传播更新 MLP", curve=0.25, color="#2874a6", dashed=True)

    ax.text(
        6.0,
        0.20,
        "当前模型不是 CNN，也不学习空间邻域；它学习的是单条光谱经过 16 个物理滤光片后的重建关系。",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    png_path = output_dir / f"{settings['output_stem']}.png"
    svg_path = output_dir / f"{settings['output_stem']}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"网络结构图 PNG 已保存: {png_path}")
    print(f"网络结构图 SVG 已保存: {svg_path}")


if __name__ == "__main__":
    main()
