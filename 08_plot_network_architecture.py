"""画 AR-EMT 端到端训练网络结构图（给 PPT 用）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 08_plot_network_architecture.py

输出:
  results/network_architecture_ppt.png
  results/network_architecture_ppt.svg

这张图想讲清 5 件事：
1. 输入是“一条光谱”，不是图像 patch；
2. 光学编码器 = 可训练物理结构 + 可微 TMM；
3. 16 个滤光片把 151 维光谱压成 16 维测量（训练时还会加噪声）；
4. MLP 解码器（16→512→256→151，输出经 Softplus 保证非负）把测量重建回光谱；
5. loss = MSE + L1 + diff_L1 + 光谱角 + 吞吐量约束 + 通道去相关。

注意：这张图纯手绘示意，改了模型结构记得也来这里同步文字，别让图“说谎”。
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    "output_dir": "results_flat_hc_50",
    "output_stem": "network_architecture_ppt",
}


def setup_font() -> None:
    """设置中文字体，避免中文变成方框。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, xy, size, title, body, facecolor, title_size=13, body_size=10) -> None:
    """画一个圆角说明框：上面粗体标题，下面正文。"""

    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4, edgecolor="#2b2b2b", facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=title_size, weight="bold")
    ax.text(x + 0.08, y + h - 0.34, body, ha="left", va="top", fontsize=body_size, linespacing=1.35)


def add_arrow(ax, start, end, text=None, curve=0.0, color="#333333", dashed=False) -> None:
    """画一支流程箭头，可带文字标注。dashed=True 用于表示“反向传播”。"""

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=18, linewidth=1.7, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.08, text, ha="center", va="bottom", fontsize=10, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def draw_decoder_nodes(ax) -> None:
    """在解码器框里画 16 → 512 → 256 → 151 的四层全连接示意（只是示意，圆点数量不代表真实维度）。"""

    x_cols = [9.15, 9.70, 10.25, 10.80]          # 四列的横坐标
    y_nodes = {
        "in":  [3.78, 3.62, 3.46, 3.30],          # 16
        "h1":  [3.84, 3.68, 3.52, 3.36, 3.20],    # 512
        "h2":  [3.84, 3.68, 3.52, 3.36, 3.20],    # 256
        "out": [3.78, 3.62, 3.46, 3.30],          # 151
    }
    colors = ["#7fb3d5", "#f7dc6f", "#f5b041", "#82e0aa"]
    edges = ["#1b4f72", "#7d6608", "#7e5109", "#145a32"]
    keys = ["in", "h1", "h2", "out"]

    # 画圆点
    for xi, key, col, ec in zip(x_cols, keys, colors, edges):
        for y0 in y_nodes[key]:
            ax.scatter(xi, y0, s=60, color=col, edgecolor=ec, zorder=5)

    # 相邻两列之间连线
    for a in range(3):
        for ya in y_nodes[keys[a]]:
            for yb in y_nodes[keys[a + 1]]:
                ax.plot([x_cols[a], x_cols[a + 1]], [ya, yb], color="#888888", linewidth=0.4, alpha=0.5)

    # 每列下面标维度
    for xi, label in zip(x_cols, ["16", "512", "256", "151"]):
        ax.text(xi, 3.03, label, ha="center", fontsize=10)


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

    # ---- 标题 ----
    ax.text(6.0, 5.72, "AR-EMT 端到端训练网络结构", ha="center", va="center", fontsize=24, weight="bold")
    ax.text(6.0, 5.42,
            "输入是按位深缩放后的光谱强度；物理编码器给出 16 通道测量(训练时加噪)；MLP 输出重建 151 维光谱",
            ha="center", va="center", fontsize=13, color="#333333")

    # ---- 主干各个框 ----
    add_box(ax, (0.30, 3.55), (1.70, 1.35), "输入光谱",
            "S(λ)\nshape: [B,151]\nλ = 400-700 nm\n保留相对明暗\n不做逐条归一化", "#d6eaf8")

    add_box(ax, (2.45, 3.45), (2.45, 1.55), "可训练光学编码器",
            "16 个滤光片\n每通道改变 D/P 和 h_c\nρ → r=D/P → D → f → n_eff\n约束: h_c + t_r = 常数", "#fdebd0")

    add_box(ax, (5.35, 3.45), (1.25, 1.55), "可微 TMM",
            "由层结构计算\nT_m(λ, α)\n输出透过谱\nshape: [16,151]", "#e8daef", title_size=12, body_size=9)

    add_box(ax, (7.05, 3.45), (1.45, 1.55), "光电测量 + 噪声",
            "y_m = Σ S(λ)T_m(λ)\n积分求和\n训练时加噪声\nshape: [B,16]", "#d5f5e3", title_size=12, body_size=9)

    add_box(ax, (8.90, 3.00), (2.85, 2.00), "解码器 MLP",
            "Linear: 16 → 512\nLinear: 512 → 256\nLeakyReLU(0.01)\nLinear: 256 → 151\nSoftplus 输出(≥0)", "#fcf3cf", body_size=9.5)
    draw_decoder_nodes(ax)

    add_box(ax, (10.05, 1.35), (1.65, 1.10), "重建光谱",
            "Ŝ(λ) ≥ 0\nshape: [B,151]\n目标是接近 S(λ)", "#d4efdf")

    # ---- 训练目标 / 参数 / 数据划分 ----
    add_box(ax, (3.30, 0.70), (4.25, 1.45), "训练目标 (loss)",
            "loss = MSE(Ŝ,S) + λ_l1·L1\n"
            "     + λ_diff·一阶差分L1 + λ_sam·光谱角\n"
            "     + λ_trans·吞吐量惩罚\n     + λ_coh·通道去相关", "#f9e79f", body_size=9.5)

    add_box(ax, (0.55, 0.70), (2.55, 1.45), "会被更新的参数",
            "编码器: ρ[16], h_c[16], AR[4]\nt_r 由 H_total-h_c 得到\n解码器: Linear 权重/偏置\nAdamW + 梯度裁剪", "#fadbd8", body_size=9.0)

    add_box(ax, (7.85, 0.55), (2.85, 0.90), "训练/验证/测试",
            "train 更新参数, val 选 best。\ntest 只在 04 最终评估用。\n数据按“场景”划分, 互不串味。", "#ebedef", title_size=12, body_size=9.0)

    # ---- 正向箭头 ----
    add_arrow(ax, (2.00, 4.25), (2.45, 4.25), "输入")
    add_arrow(ax, (4.90, 4.25), (5.35, 4.25), "结构参数")
    add_arrow(ax, (6.60, 4.25), (7.05, 4.25), "透过谱")
    add_arrow(ax, (8.50, 4.25), (8.90, 4.25), "16维测量")
    add_arrow(ax, (10.90, 3.00), (10.90, 2.45), "输出")
    add_arrow(ax, (10.05, 1.90), (7.55, 1.45), "与真值比较", curve=-0.12)

    # ---- 反向传播（虚线，蓝色）----
    add_arrow(ax, (5.05, 1.45), (3.40, 3.45), "反向传播更新光学结构", curve=-0.25, color="#2874a6", dashed=True)
    add_arrow(ax, (7.30, 1.45), (10.15, 3.00), "反向传播更新 MLP", curve=0.25, color="#2874a6", dashed=True)

    ax.text(6.0, 0.18,
            "当前模型不是 CNN, 也不学空间邻域；它学的是“单条光谱经过 16 个物理滤光片后如何重建”。",
            ha="center", va="center", fontsize=11, color="#333333")

    png_path = output_dir / f"{settings['output_stem']}.png"
    svg_path = output_dir / f"{settings['output_stem']}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"网络结构图 PNG 已保存: {png_path}")
    print(f"网络结构图 SVG 已保存: {svg_path}")


if __name__ == "__main__":
    main()
