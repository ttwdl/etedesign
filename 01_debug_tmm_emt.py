"""调试 AR-EMT 的几何约束、EMT 条件和可微 TMM（只检查，不训练）。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 01_debug_tmm_emt.py

它做四件事，帮你在正式训练前确认“物理部分是对的、梯度能传回去”：
1. 打印几何约束(D/P、D、gap)和 EMT 条件；
2. 快速对比几种 AR 层序的透过率高低，看看增透趋势对不对；
3. 检查 TMM 的梯度能不能反向传播到结构参数（能，才谈得上训练）；
4. 存一张初始多通道透过谱图，肉眼看看形状。
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
import torch

from ar_emt_common import (
    AREMTModel,
    GeometryConfig,
    build_ar_emt_transmission,
    emt_neff_from_ratio,
    geometry_limits,
    geometry_report,
    material_n,
    model_kwargs_from_settings,
    tmm_transmission_unpolarized,
    tor_percent,
)


# =============================================================================
# 用户设置区：平时只改这里
# =============================================================================
USER_SETTINGS = {
    "period_nm": 180.0,
    "g_min_nm": 40.0,
    "d_min_nm": 60.0,
    "enforce_d_min": True,
    "output_dir": "results_36ch_t06_tor15_50",
    "device": "cuda",

    # 和 03_train_ar_emt.py 保持一致：这次试 36 通道。
    "n_channels": 36,
    "hidden_dims": (1024, 512),
    "h_c_range": (250.0, 1500.0),
    "t_r_range": (0.0, 1500.0),
    "core_total_nm": 1000.0,
    "core_total_range": (800.0, 1800.0),
    "aspect_ratio_max": 10.0,
}


def qw_thickness_for_order(order: str, lambda0_nm: float = 530.0) -> list[float]:
    """按层序字符串生成“四分之一波长”厚度。

    L 表示 SiO2，H 表示 TiO2。例如 "LH" = 先 SiO2 再 TiO2。
    只用于本调试脚本里试不同层序，正式训练固定用 L-H / H-L。
    """

    n_map = {"L": 1.46, "H": 2.35}
    return [lambda0_nm / (4.0 * n_map[ch]) for ch in order]


def build_order_transmission(
    ratio: torch.Tensor,
    wl_nm: torch.Tensor,
    top_order: str,
    bottom_order: str,
    h_c_nm: float = 600.0,
    t_r_nm: float = 50.0,
    alpha_deg: float = 0.0,
) -> torch.Tensor:
    """只用于“试不同 AR 层序”的 TMM 计算（不是训练用的那条固定层序）。

    训练脚本固定用：空气 / L-H / SU-8 / EMT / H-L / 熔石英。
    这里多试几种顺序，只是为了确认“加了增透层，透过率确实变高”这个趋势。
    """

    n_struct = ratio.numel()

    def const_layer(name: str) -> torch.Tensor:
        return material_n(name, wl_nm)[None, :].expand(n_struct, wl_nm.numel())

    n_layers = [const_layer("air")]
    d_layers = []

    # 顶部 AR 层：按 top_order 一层层加
    for ch, thick in zip(top_order, qw_thickness_for_order(top_order)):
        n_layers.append(const_layer("sio2" if ch == "L" else "tio2"))
        d_layers.append(torch.tensor(thick, device=wl_nm.device))

    # 残余 SU-8 + EMT 腔
    n_layers.append(const_layer("su8"))
    d_layers.append(torch.tensor(t_r_nm, device=wl_nm.device))
    n_layers.append(emt_neff_from_ratio(ratio, wl_nm))
    d_layers.append(torch.tensor(h_c_nm, device=wl_nm.device))

    # 底部 AR 层
    for ch, thick in zip(bottom_order, qw_thickness_for_order(bottom_order)):
        n_layers.append(const_layer("sio2" if ch == "L" else "tio2"))
        d_layers.append(torch.tensor(thick, device=wl_nm.device))

    n_layers.append(const_layer("fused_silica"))
    return tmm_transmission_unpolarized(
        n_layers, d_layers, wl_nm, torch.tensor([alpha_deg], device=wl_nm.device),
    )[0]


def summarize_t(name: str, t_matrix: torch.Tensor) -> None:
    """打印一组透过谱的关键数字（均值/最小/最大/区分度）。"""

    t = t_matrix.detach()
    print(f"{name:16s} T_mean={float(t.mean()):.4f}, T_min={float(t.min()):.4f}, "
          f"T_max={float(t.max()):.4f}, tor={tor_percent(t):.3f}%")


def main() -> None:
    settings = USER_SETTINGS
    device = torch.device(settings["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GeometryConfig(
        period_nm=settings["period_nm"],
        g_min_nm=settings["g_min_nm"],
        d_min_nm=settings["d_min_nm"],
        enforce_d_min=settings["enforce_d_min"],
    )
    print(geometry_report(config))
    print()

    # 波长网格 + n_channels 个通道的 D/P（在可行区间里均匀铺开）
    wl_nm = torch.linspace(400.0, 700.0, 151, device=device)
    limits = geometry_limits(config)
    ratio = torch.linspace(limits["r_min"], limits["r_max"], int(settings["n_channels"]), device=device)

    # ---- (1)(2) 层序快速对比：所有厚度先用四分之一波初值 ----
    print("层序快速对比（所有厚度先用四分之一波初值）：")
    t_hl_lh = build_order_transmission(ratio, wl_nm, top_order="HL", bottom_order="LH")
    t_lh_hl = build_order_transmission(ratio, wl_nm, top_order="LH", bottom_order="HL")
    t_no_ar = build_order_transmission(ratio, wl_nm, top_order="", bottom_order="")
    summarize_t("HL / LH", t_hl_lh)
    summarize_t("LH / HL", t_lh_hl)
    summarize_t("no AR", t_no_ar)   # 没有增透层做对照，应明显更低
    print()

    # ---- (3) 检查 TMM 梯度能不能传回结构参数 ----
    print("检查 TMM 梯度是否能传回结构参数：")
    model = AREMTModel(wl_nm.cpu(), config, **model_kwargs_from_settings(settings)).to(device)
    t0 = model.transmission(torch.tensor([0.0], device=device))[0]
    loss = t0.mean()
    loss.backward()   # 只要能 backward 且各参数 grad 不为 0，就说明结构参数可训练
    print(f"  mean(T) = {float(loss.detach()):.6f}")
    print(f"  rho grad norm       = {float(model.rho.grad.norm()):.6e}")
    print(f"  H_total grad        = {float(model.raw_core_total.grad):.6e}")
    print(f"  h_c grad norm       = {float(model.raw_h_c.grad.norm()):.6e}")
    print("  t_r 不再单独训练：t_r_l = H_total - h_c_l，所以它的梯度等价地走到 h_c_l 上。")
    print(f"  AR thickness grad   = {float(model.raw_ar.grad.norm()):.6e}")
    print()

    # ---- (4) 存初始透过谱图 ----
    params = model.physical_parameters()
    h_c = params["h_c_nm"].detach().cpu()
    t_r = params["t_r_nm"].detach().cpu()
    core_total = float(params["core_total_nm"].detach().cpu())
    aspect = params["aspect_ratio"].detach().cpu()
    aspect_max = float(params["aspect_ratio_max"].detach().cpu())
    print(f"初始平整约束: h_c=[{float(h_c.min()):.2f}, {float(h_c.max()):.2f}] nm, "
          f"t_r=[{float(t_r.min()):.2f}, {float(t_r.max()):.2f}] nm, "
          f"H_total={core_total:.2f} nm")
    print(f"初始深宽比约束: h_c/D=[{float(aspect.min()):.2f}, {float(aspect.max()):.2f}], "
          f"limit={aspect_max:.1f}")
    t_init_0 = build_ar_emt_transmission(
        params["ratio"], params["h_c_nm"], params["t_r_nm"], params["ar_nm"],
        wl_nm, torch.tensor([0.0], device=device),
    )[0]
    t_init_5 = build_ar_emt_transmission(
        params["ratio"], params["h_c_nm"], params["t_r_nm"], params["ar_nm"],
        wl_nm, torch.tensor([5.0], device=device),
    )[0]
    summarize_t("initial 0deg", t_init_0)
    summarize_t("initial 5deg", t_init_5)   # 换个入射角，确认斜入射也算得动

    plt.figure(figsize=(9, 5))
    for idx in range(t_init_0.shape[0]):
        plt.plot(wl_nm.detach().cpu(), t_init_0[idx].detach().cpu(), lw=1.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(f"Initial {t_init_0.shape[0]}-channel AR-EMT spectra, alpha=0 deg")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    out_png = output_dir / "debug_initial_spectra_0deg.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"初始透过谱图已保存: {out_png}")
    print("调试完成。")


if __name__ == "__main__":
    main()
