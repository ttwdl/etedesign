"""AR-EMT 项目的共用代码。

这个文件只放多个脚本都会用到的东西：
1. 几何约束：D/P、D、gap、EMT 条件；
2. 材料折射率：TiO2、SiO2、SU-8、空气、熔石英；
3. 可微 TMM：用 PyTorch 写，所以结构参数可以反向传播；
4. AR-EMT 模型：物理编码器 + MLP 解码器；
5. 常用指标：MSE、PSNR、SAM、tor、结构参数导出。

单位约定：
- 长度默认都是 nm；
- 光谱 shape 默认是 [样本数, 波长点数]；
- 透过谱 shape 默认是 [通道数, 波长点数]；
- 当前数据使用按位深缩放后的相对强度，不做逐条光谱最大值归一化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class GeometryConfig:
    """几何约束配置。

    period_nm:
        超表面周期 P。当前默认固定 180 nm，不训练。
    ratio_min_design / ratio_max_design:
        设计希望的 D/P 范围。用户希望是 0.1 到 0.9。
    g_min_nm:
        最小间隙 gap = P - D。它会限制 D/P 的实际上限。
    d_min_nm:
        最小柱径 D。默认 60 nm，更稳妥。
    enforce_d_min:
        是否启用 D >= d_min_nm。如果关掉，下限只由 ratio_min_design 控制。
    lambda_min_nm:
        EMT 检查使用的最短波长。当前光谱从 400 nm 开始。
    """

    period_nm: float = 180.0
    ratio_min_design: float = 0.1
    ratio_max_design: float = 0.9
    g_min_nm: float = 40.0
    d_min_nm: float = 60.0
    enforce_d_min: bool = True
    lambda_min_nm: float = 400.0


def complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    """根据实数 dtype 选择复数 dtype。

    TMM 里面会出现复数相位，所以折射率和矩阵都用复数。
    """

    if real_dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def geometry_limits(config: GeometryConfig) -> dict[str, float]:
    """计算真正可训练的 D/P 范围。

    设计上希望 D/P 在 [0.1, 0.9]，但实际还要满足制造约束：
    - D >= d_min_nm；
    - gap = P - D >= g_min_nm。

    例如 P=180 nm、D_min=60 nm、gap_min=40 nm：
    - r_min=max(0.1, 60/180)=0.3333；
    - r_max=min(0.9, 1-40/180)=0.7778。
    """

    period = float(config.period_nm)
    r_min = float(config.ratio_min_design)
    r_max = float(config.ratio_max_design)

    if config.enforce_d_min:
        r_min = max(r_min, float(config.d_min_nm) / period)

    r_max = min(r_max, 1.0 - float(config.g_min_nm) / period)

    if r_max <= r_min:
        raise ValueError(
            "几何约束没有可行区间："
            f"r_min={r_min:.4f}, r_max={r_max:.4f}。"
            "请减小 d_min_nm / g_min_nm，或者增大 period_nm。"
        )

    return {
        "r_min": r_min,
        "r_max": r_max,
        "d_min_nm": r_min * period,
        "d_max_nm": r_max * period,
        "gap_min_nm": period * (1.0 - r_max),
        "period_nm": period,
    }


def bounded_value(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """把无界训练变量映射到 [low, high]。

    神经网络优化器更擅长优化无界参数。这里用 sigmoid 做重参数化：
        raw 是优化器真正更新的参数；
        bounded_value(raw) 才是有物理意义的厚度或 D/P。
    这样训练时不会跑出制造边界。
    """

    return low + (high - low) * torch.sigmoid(raw)


def raw_from_bounded(value: torch.Tensor | float | Iterable[float], low: float, high: float) -> torch.Tensor:
    """把一个已知的物理初值反解成 raw 初值。

    例子：希望 h_c 初始为 600 nm，但训练参数实际存 raw_h_c。
    这个函数就是把 600 nm 转成对应的 raw_h_c。
    """

    value_tensor = torch.as_tensor(value, dtype=torch.float32)
    x = (value_tensor - low) / (high - low)
    x = torch.clamp(x, 1e-5, 1.0 - 1e-5)
    return torch.logit(x)


def ratio_to_fill_factor(ratio: torch.Tensor) -> torch.Tensor:
    """由 D/P 计算圆柱在正方周期中的填充因子。

    圆柱面积 = pi * (D/2)^2
    单元面积 = P^2
    所以 f = pi * D^2 / (4P^2) = pi * (D/P)^2 / 4。
    """

    return math.pi * ratio.square() / 4.0


def material_n(name: str, wl_nm: torch.Tensor) -> torch.Tensor:
    """返回材料折射率 n(lambda)。

    当前先用简单透明模型，方便复现和调试：
    - TiO2: n=2.35；
    - SiO2 / fused silica: n=1.46；
    - SU-8: 简单 Cauchy 公式；
    - air: n=1。

    如果以后有真实椭偏数据，可以在这里改成 CSV 插值。
    """

    key = name.lower()
    wl = wl_nm.to(dtype=torch.float32)
    cdtype = complex_dtype_from_real(wl.dtype)

    if key in {"air"}:
        n_real = torch.ones_like(wl)
    elif key in {"tio2", "h"}:
        n_real = torch.full_like(wl, 2.35)
    elif key in {"sio2", "silica", "l", "fused_silica", "substrate", "glass"}:
        n_real = torch.full_like(wl, 1.46)
    elif key in {"su8", "resist", "photoresist"}:
        wl_um = wl / 1000.0
        n_real = 1.566 + 0.00796 / wl_um.square() + 0.00014 / wl_um.pow(4)
    else:
        raise ValueError(f"未知材料: {name}")

    return n_real.to(cdtype)


def emt_neff_from_ratio(
    ratio: torch.Tensor,
    wl_nm: torch.Tensor,
    pillar_material: str = "tio2",
    fill_material: str = "su8",
    mode: str = "volume",
) -> torch.Tensor:
    """由 D/P 计算 EMT 腔层的有效折射率。

    ratio:
        [M]，M 是滤光片通道数。当前 M=16。
    返回:
        [M, Nwl]，每个通道、每个波长一个 n_eff。

    当前默认 volume mixing:
        eps_eff = (1-f)*eps_SU8 + f*eps_TiO2

    物理图像：
        EMT 腔不是纯 TiO2，也不是纯 SU-8；
        它是 TiO2 柱子嵌在 SU-8 里，所以折射率由填充因子决定。
    """

    ratio = torch.as_tensor(ratio, dtype=wl_nm.dtype, device=wl_nm.device)
    if ratio.ndim == 0:
        ratio = ratio[None]

    f = ratio_to_fill_factor(ratio)[:, None]
    n_fill = material_n(fill_material, wl_nm)[None, :]
    n_pillar = material_n(pillar_material, wl_nm)[None, :]

    eps_fill = n_fill.square()
    eps_pillar = n_pillar.square()

    if mode == "volume":
        eps_eff = (1.0 - f) * eps_fill + f * eps_pillar
    elif mode == "mg2d":
        eps_eff = eps_fill * ((eps_pillar + eps_fill) + f * (eps_pillar - eps_fill)) / (
            (eps_pillar + eps_fill) - f * (eps_pillar - eps_fill)
        )
    else:
        raise ValueError(f"未知 EMT 模式: {mode}")

    return torch.sqrt(eps_eff)


def emt_condition(config: GeometryConfig, ratio_max: float | torch.Tensor) -> dict[str, float | bool]:
    """检查 EMT 条件 P < lambda_min / n_eff_max。

    这个检查不是训练损失，只是提醒当前周期是否仍适合等效介质近似。
    """

    with torch.no_grad():
        if isinstance(ratio_max, torch.Tensor):
            r_max = float(ratio_max.detach().max().cpu())
        else:
            r_max = float(ratio_max)

        wl_min = torch.tensor([config.lambda_min_nm], dtype=torch.float32)
        neff_max = float(emt_neff_from_ratio(torch.tensor([r_max]), wl_min).real.max())
        limit_nm = float(config.lambda_min_nm) / neff_max
        ok = float(config.period_nm) < limit_nm

    return {
        "period_nm": float(config.period_nm),
        "lambda_min_nm": float(config.lambda_min_nm),
        "ratio_max": r_max,
        "neff_max": neff_max,
        "emt_limit_nm": limit_nm,
        "margin_nm": limit_nm - float(config.period_nm),
        "ok": ok,
    }


def geometry_report(config: GeometryConfig) -> str:
    """生成几何约束和 EMT 检查的文字报告。"""

    limits = geometry_limits(config)
    emt = emt_condition(config, limits["r_max"])
    status = "满足" if emt["ok"] else "不满足"
    return (
        "几何约束:\n"
        f"  P = {limits['period_nm']:.3f} nm\n"
        f"  D/P 实际训练范围 = [{limits['r_min']:.4f}, {limits['r_max']:.4f}]\n"
        f"  D 实际训练范围 = [{limits['d_min_nm']:.3f}, {limits['d_max_nm']:.3f}] nm\n"
        f"  最小间隙 gap >= {limits['gap_min_nm']:.3f} nm\n"
        "EMT 条件:\n"
        f"  n_eff_max(lambda_min) = {emt['neff_max']:.4f}\n"
        f"  lambda_min / n_eff_max = {emt['emt_limit_nm']:.3f} nm\n"
        f"  P < lambda_min / n_eff_max: {status}, margin = {emt['margin_nm']:.3f} nm"
    )


def quarter_wave_ar_thickness(lambda0_nm: float = 530.0) -> list[float]:
    """给 AR 层一个四分之一波长厚度初值。

    层序固定为：
        空气 / L-H / SU-8 / EMT / H-L / 熔石英
    L 是 SiO2，H 是 TiO2。
    返回顺序:
        [top_L, top_H, bottom_H, bottom_L]
    """

    n_l = 1.46
    n_h = 2.35
    h_l = lambda0_nm / (4.0 * n_l)
    h_h = lambda0_nm / (4.0 * n_h)
    return [h_l, h_h, h_h, h_l]


def _expand_layer_n(n_value: torch.Tensor, n_struct: int, n_wl: int, device: torch.device) -> torch.Tensor:
    """把某一层折射率整理成 [结构数, 波长数]。"""

    n = n_value.to(device=device)
    if n.ndim == 0:
        n = n.expand(n_struct, n_wl)
    elif n.ndim == 1:
        n = n[None, :].expand(n_struct, n_wl)
    elif n.ndim == 2:
        if n.shape != (n_struct, n_wl):
            raise ValueError(f"折射率层形状应为 {(n_struct, n_wl)}，实际为 {tuple(n.shape)}")
    else:
        raise ValueError(f"折射率层维度过多: {n.ndim}")
    return n


def _expand_layer_d(d_value: torch.Tensor, n_struct: int, device: torch.device) -> torch.Tensor:
    """把某一层厚度整理成 [结构数]。"""

    d = d_value.to(device=device, dtype=torch.float32)
    if d.ndim == 0:
        d = d.expand(n_struct)
    elif d.ndim == 1:
        if d.shape[0] != n_struct:
            raise ValueError(f"厚度层长度应为 {n_struct}，实际为 {d.shape[0]}")
    else:
        raise ValueError(f"厚度层维度过多: {d.ndim}")
    return d


def tmm_transmission_unpolarized(
    n_layers: list[torch.Tensor],
    d_layers_nm: list[torch.Tensor],
    wl_nm: torch.Tensor,
    alpha_deg: torch.Tensor | float,
) -> torch.Tensor:
    """PyTorch 可微 TMM，返回非偏振透过率。

    n_layers:
        包含入射介质和基底，长度 = 内部有限厚度层数 + 2。
    d_layers_nm:
        只包含内部有限厚度层，不包含空气和基底。
    返回:
        [Nangle, Nstructure, Nwl]。

    当前把 s 偏振和 p 偏振各算一次，最后取平均。
    """

    if len(d_layers_nm) != len(n_layers) - 2:
        raise ValueError("d_layers_nm 必须只给内部层厚度，不包含空气和基底。")

    wl = wl_nm.to(dtype=torch.float32)
    device = wl.device
    n_wl = wl.numel()

    n_struct = 1
    for n in n_layers:
        if isinstance(n, torch.Tensor) and n.ndim == 2:
            n_struct = max(n_struct, n.shape[0])

    n_stack = torch.stack([_expand_layer_n(n, n_struct, n_wl, device) for n in n_layers], dim=0)
    d_stack = torch.stack([_expand_layer_d(d, n_struct, device) for d in d_layers_nm], dim=0)

    alpha = torch.as_tensor(alpha_deg, dtype=torch.float32, device=device).reshape(-1)
    alpha_rad = alpha * math.pi / 180.0

    q = n_stack[0][None, :, :] * torch.sin(alpha_rad)[:, None, None]
    n_for_all_layers = n_stack[:, None, :, :]
    cos_theta = torch.sqrt(1.0 - (q[None, :, :, :] / n_for_all_layers).square())
    cos_theta = torch.where(cos_theta.real < 0, -cos_theta, cos_theta)

    def one_pol(pol: str) -> torch.Tensor:
        if pol == "s":
            eta = n_for_all_layers * cos_theta
        elif pol == "p":
            eta = n_for_all_layers / cos_theta
        else:
            raise ValueError(pol)

        shape = eta[0].shape
        m11 = torch.ones(shape, dtype=eta.dtype, device=device)
        m12 = torch.zeros(shape, dtype=eta.dtype, device=device)
        m21 = torch.zeros(shape, dtype=eta.dtype, device=device)
        m22 = torch.ones(shape, dtype=eta.dtype, device=device)

        wl_b = wl[None, None, :]
        imaginary = torch.tensor(1j, dtype=eta.dtype, device=device)

        for layer_idx in range(1, len(n_layers) - 1):
            d_nm = d_stack[layer_idx - 1][None, :, None]
            delta = 2.0 * math.pi / wl_b * n_stack[layer_idx][None, :, :] * d_nm * cos_theta[layer_idx]
            cos_delta = torch.cos(delta)
            sin_delta = torch.sin(delta)
            eta_j = eta[layer_idx]

            a = cos_delta
            b = imaginary * sin_delta / eta_j
            c = imaginary * eta_j * sin_delta
            d = cos_delta

            new11 = m11 * a + m12 * c
            new12 = m11 * b + m12 * d
            new21 = m21 * a + m22 * c
            new22 = m21 * b + m22 * d
            m11, m12, m21, m22 = new11, new12, new21, new22

        eta0 = eta[0]
        etas = eta[-1]
        b_total = m11 + m12 * etas
        c_total = m21 + m22 * etas
        denom = torch.abs(eta0 * b_total + c_total).square()
        trans = 4.0 * eta0.real * etas.real / denom
        return trans.real

    t_s = one_pol("s")
    t_p = one_pol("p")
    return 0.5 * (t_s + t_p)


def build_ar_emt_transmission(
    ratio: torch.Tensor,
    h_c_nm: torch.Tensor,
    t_r_nm: torch.Tensor,
    ar_thickness_nm: torch.Tensor,
    wl_nm: torch.Tensor,
    alpha_deg: torch.Tensor | float,
) -> torch.Tensor:
    """根据结构参数计算 16 个滤光片的透过谱。

    层序：
        air / top SiO2 / top TiO2 / residual SU-8 /
        EMT cavity / bottom TiO2 / bottom SiO2 / fused silica
    """

    wl = wl_nm.to(dtype=torch.float32)
    ratio = ratio.to(device=wl.device, dtype=torch.float32)
    ar = ar_thickness_nm.to(device=wl.device, dtype=torch.float32)

    if ar.numel() != 4:
        raise ValueError("ar_thickness_nm 必须是 4 个数: top_L, top_H, bottom_H, bottom_L。")

    n_eff = emt_neff_from_ratio(ratio, wl)
    n_struct = ratio.numel()

    def const_layer(name: str) -> torch.Tensor:
        return material_n(name, wl)[None, :].expand(n_struct, wl.numel())

    n_layers = [
        const_layer("air"),
        const_layer("sio2"),
        const_layer("tio2"),
        const_layer("su8"),
        n_eff,
        const_layer("tio2"),
        const_layer("sio2"),
        const_layer("fused_silica"),
    ]
    d_layers = [ar[0], ar[1], t_r_nm, h_c_nm, ar[2], ar[3]]
    return tmm_transmission_unpolarized(n_layers, d_layers, wl, alpha_deg)


class AREMTModel(nn.Module):
    """AR-EMT 可训练模型。

    前半部分是物理编码器：
        ρ -> D/P -> D -> f -> n_eff -> TMM -> 16 通道测量。
    后半部分是 MLP 解码器：
        16 通道测量 -> 151 维重建光谱。

    注意：
        当前输入是一条光谱，不是二维图像 patch，所以没有卷积，也不学习空间结构。
    """

    def __init__(
        self,
        wl_nm: torch.Tensor,
        config: GeometryConfig,
        n_channels: int = 16,
        hidden_dim: int = 500,
        h_c_range: tuple[float, float] = (300.0, 1200.0),
        t_r_range: tuple[float, float] = (0.0, 150.0),
        ar_range: tuple[float, float] = (5.0, 160.0),
    ) -> None:
        super().__init__()
        self.config = config
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.h_c_range = h_c_range
        self.t_r_range = t_r_range
        self.ar_range = ar_range

        limits = geometry_limits(config)
        self.r_min = limits["r_min"]
        self.r_max = limits["r_max"]

        self.register_buffer("wl_nm", wl_nm.detach().clone().to(dtype=torch.float32))

        # 16 个通道的 D/P 初始值均匀铺开，避免一开始所有通道完全一样。
        ratio_init = torch.linspace(
            self.r_min + 0.02 * (self.r_max - self.r_min),
            self.r_max - 0.02 * (self.r_max - self.r_min),
            n_channels,
        )
        self.rho = nn.Parameter(raw_from_bounded(ratio_init, self.r_min, self.r_max))

        # 这些厚度是全局共享的，16 个滤光片一起训练同一组厚度。
        self.raw_h_c = nn.Parameter(raw_from_bounded(600.0, *h_c_range))
        self.raw_t_r = nn.Parameter(raw_from_bounded(50.0, *t_r_range))
        self.raw_ar = nn.Parameter(raw_from_bounded(quarter_wave_ar_thickness(), *ar_range))

        # 输出层不加 Sigmoid：现在目标是按位深缩放后的强度，允许网络直接回归数值。
        self.decoder = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, wl_nm.numel()),
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        """把 raw 参数转成有物理意义的参数。"""

        ratio = bounded_value(self.rho, self.r_min, self.r_max)
        h_c = bounded_value(self.raw_h_c, *self.h_c_range)
        t_r = bounded_value(self.raw_t_r, *self.t_r_range)
        ar = bounded_value(self.raw_ar, *self.ar_range)
        return {"ratio": ratio, "h_c_nm": h_c, "t_r_nm": t_r, "ar_nm": ar}

    def transmission(
        self,
        alpha_deg: torch.Tensor | float,
        ratio_override: torch.Tensor | None = None,
        h_c_override: torch.Tensor | None = None,
        t_r_override: torch.Tensor | None = None,
        ar_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算当前结构在指定入射角下的透过谱。"""

        params = self.physical_parameters()
        ratio = params["ratio"] if ratio_override is None else ratio_override
        h_c = params["h_c_nm"] if h_c_override is None else h_c_override
        t_r = params["t_r_nm"] if t_r_override is None else t_r_override
        ar = params["ar_nm"] if ar_override is None else ar_override
        return build_ar_emt_transmission(ratio, h_c, t_r, ar, self.wl_nm, alpha_deg)

    def measure(self, spectra: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
        """用透过谱把输入光谱压缩成 16 通道测量。

        spectra:
            [B, 151]，B 是 batch 大小。
        transmission:
            [16, 151] 或 [B, 16, 151]。
        返回:
            [B, 16]。

        这里保持积分和，不除以 151。
        """

        if transmission.ndim == 2:
            return spectra @ transmission.T
        if transmission.ndim == 3 and transmission.shape[0] == spectra.shape[0]:
            return torch.einsum("bn,bmn->bm", spectra, transmission)
        raise ValueError(f"透过谱 shape={tuple(transmission.shape)} 与光谱 batch={spectra.shape[0]} 不匹配。")

    def forward(self, spectra: torch.Tensor, alpha_deg: torch.Tensor, noise_max: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """完整前向传播：光谱 -> 光学测量 -> 重建光谱。"""

        t = self.transmission(alpha_deg)
        if t.shape[0] == 1:
            meas = self.measure(spectra, t[0])
        else:
            meas = self.measure(spectra, t)

        if self.training and noise_max > 0:
            sigma = torch.rand(meas.shape[0], 1, device=meas.device) * noise_max
            meas = meas + sigma * torch.randn_like(meas)

        pred = self.decoder(meas)
        return pred, t


def metric_mse_psnr_sam(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """计算 MSE、PSNR、SAM。

    PSNR 这里仍按峰值 1 计算，因为数据准备会把 CAVE PNG 按位深缩放到 0-1。
    """

    with torch.no_grad():
        mse = torch.mean((pred - target).square())
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
        dot = torch.sum(pred * target, dim=1)
        denom = torch.linalg.norm(pred, dim=1) * torch.linalg.norm(target, dim=1) + 1e-12
        cos_val = torch.clamp(dot / denom, -1.0, 1.0)
        sam = torch.mean(torch.arccos(cos_val))
    return {"mse": float(mse.cpu()), "psnr": float(psnr.cpu()), "sam": float(sam.cpu())}


def evaluate_fixed_angle(
    model: AREMTModel,
    spectra: torch.Tensor,
    angle_deg: float,
    batch_size: int = 4096,
) -> dict[str, float]:
    """在固定入射角下评估重建精度。

    评估时分块跑，避免一次性把所有测试光谱放进显存。
    """

    device = next(model.parameters()).device
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        t = model.transmission(torch.tensor([angle_deg], device=device))[0]
        for start in range(0, spectra.shape[0], batch_size):
            batch = spectra[start:start + batch_size].to(device)
            meas = model.measure(batch, t)
            pred = model.decoder(meas)
            preds.append(pred.cpu())
            targets.append(batch.cpu())
    return metric_mse_psnr_sam(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def tor_percent(t_matrix: torch.Tensor) -> float:
    """计算通道区分度 tor。

    做法：
    1. 16 个通道两两比较；
    2. 每一对都算 mean(abs(T_i - T_j))*100；
    3. 取最小值。

    它只用于观察滤光片是否太像，不参与 loss。
    """

    t = t_matrix.detach().cpu()
    n_ch = t.shape[0]
    diffs = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            diffs.append(torch.mean(torch.abs(t[i] - t[j])) * 100.0)
    return float(torch.min(torch.stack(diffs))) if diffs else 0.0


def structure_rows(model: AREMTModel) -> list[dict[str, float]]:
    """导出每个通道的结构参数，方便保存 CSV。"""

    params = model.physical_parameters()
    ratio = params["ratio"].detach().cpu()
    period = model.config.period_nm
    wl = model.wl_nm.detach().cpu()
    neff = emt_neff_from_ratio(ratio, wl).real.detach().cpu()
    rows = []
    for idx, r in enumerate(ratio):
        d_nm = float(r) * period
        gap_nm = period - d_nm
        f = float(ratio_to_fill_factor(r))
        rows.append(
            {
                "channel": idx,
                "ratio_D_over_P": float(r),
                "D_nm": d_nm,
                "gap_nm": gap_nm,
                "fill_factor": f,
                "neff_min": float(neff[idx].min()),
                "neff_max": float(neff[idx].max()),
            }
        )
    return rows
