"""调试 AR-EMT 几何约束、EMT 条件和可微 TMM。

直接运行:
  & 'C:\\Users\\23\\.conda\\envs\\TMM\\python.exe' 01_debug_tmm_emt.py

这个脚本不训练，只做四件事：
1. 打印 D/P、D、gap、EMT 条件；
2. 快速对比不同 AR 层序的透过率；
3. 检查 TMM 是否能反向传播到结构参数；
4. 保存初始 16 通道透过谱图。
"""

from __future__ import annotations

from pathlib import Path

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
    tmm_transmission_unpolarized,
    tor_percent,
)


# =========================
# 用户设置区：平时只改这里
# =========================
USER_SETTINGS = {
    "period_nm": 180.0,
    "g_min_nm": 40.0,
    "d_min_nm": 60.0,
    "enforce_d_min": True,
    "output_dir": "results",
    "device": "cuda",
}


def qw_thickness_for_order(order: str, lambda0_nm: float = 530.0) -> list[float]:
    """按层序字符串生成四分之一波厚度。

    L 表示 SiO2，H 表示 TiO2。
    例如 "LH" 表示先 SiO2 再 TiO2。
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
    """只用于调试层序的 TMM 计算。

    训练脚本固定使用 air / L-H / SU-8 / EMT / H-L / fused silica。
    这里多试几个顺序，是为了确认增透结构趋势。
    """

    n_struct = ratio.numel()

    def const_layer(name: str) -> torch.Tensor:
        return material_n(name, wl_nm)[None, :].expand(n_struct, wl_nm.numel())

    n_layers = [const_layer("air")]
    d_layers = []

    for ch, thick in zip(top_order, qw_thickness_for_order(top_order)):
        n_layers.append(const_layer("sio2" if ch == "L" else "tio2"))
        d_layers.append(torch.tensor(thick, device=wl_nm.device))

    n_layers.append(const_layer("su8"))
    d_layers.append(torch.tensor(t_r_nm, device=wl_nm.device))

    n_layers.append(emt_neff_from_ratio(ratio, wl_nm))
    d_layers.append(torch.tensor(h_c_nm, device=wl_nm.device))

    for ch, thick in zip(bottom_order, qw_thickness_for_order(bottom_order)):
        n_layers.append(const_layer("sio2" if ch == "L" else "tio2"))
        d_layers.append(torch.tensor(thick, device=wl_nm.device))

    n_layers.append(const_layer("fused_silica"))
    return tmm_transmission_unpolarized(
        n_layers,
        d_layers,
        wl_nm,
        torch.tensor([alpha_deg], device=wl_nm.device),
    )[0]


def summarize_t(name: str, t_matrix: torch.Tensor) -> None:
    """打印一组透过谱的关键数字。"""

    t = t_matrix.detach()
    print(
        f"{name:16s} "
        f"T_mean={float(t.mean()):.4f}, "
        f"T_min={float(t.min()):.4f}, "
        f"T_max={float(t.max()):.4f}, "
        f"tor={tor_percent(t):.3f}%"
    )


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

    wl_nm = torch.linspace(400.0, 700.0, 151, device=device)
    limits = geometry_limits(config)
    ratio = torch.linspace(limits["r_min"], limits["r_max"], 16, device=device)

    print("层序快速对比：所有厚度先用四分之一波初值")
    t_hl_lh = build_order_transmission(ratio, wl_nm, top_order="HL", bottom_order="LH")
    t_lh_hl = build_order_transmission(ratio, wl_nm, top_order="LH", bottom_order="HL")
    t_no_ar = build_order_transmission(ratio, wl_nm, top_order="", bottom_order="")
    summarize_t("HL / LH", t_hl_lh)
    summarize_t("LH / HL", t_lh_hl)
    summarize_t("no AR", t_no_ar)
    print()

    print("检查 TMM 梯度是否能传回结构参数")
    model = AREMTModel(wl_nm.cpu(), config).to(device)
    t0 = model.transmission(torch.tensor([0.0], device=device))[0]
    loss = t0.mean()
    loss.backward()
    print(f"  mean(T) = {float(loss.detach()):.6f}")
    print(f"  rho grad norm = {float(model.rho.grad.norm()):.6e}")
    print(f"  h_c grad = {float(model.raw_h_c.grad):.6e}")
    print(f"  t_r grad = {float(model.raw_t_r.grad):.6e}")
    print(f"  AR thickness grad norm = {float(model.raw_ar.grad.norm()):.6e}")
    print()

    params = model.physical_parameters()
    t_init_0 = build_ar_emt_transmission(
        params["ratio"],
        params["h_c_nm"],
        params["t_r_nm"],
        params["ar_nm"],
        wl_nm,
        torch.tensor([0.0], device=device),
    )[0]
    t_init_5 = build_ar_emt_transmission(
        params["ratio"],
        params["h_c_nm"],
        params["t_r_nm"],
        params["ar_nm"],
        wl_nm,
        torch.tensor([5.0], device=device),
    )[0]

    summarize_t("initial 0deg", t_init_0)
    summarize_t("initial 5deg", t_init_5)

    plt.figure(figsize=(9, 5))
    for idx in range(t_init_0.shape[0]):
        plt.plot(wl_nm.detach().cpu(), t_init_0[idx].detach().cpu(), lw=1.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title("Initial 16-channel AR-EMT spectra, alpha=0 deg")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    out_png = output_dir / "debug_initial_spectra_0deg.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"初始透过谱图已保存: {out_png}")
    print("调试完成。")


if __name__ == "__main__":
    main()
