"""AR-EMT 项目的共用代码（核心库）。

===================== 这套代码到底在干嘛（先看这段）=====================
我们想造一个“计算光谱仪”：
  1) 一束光（一条光谱 S(λ)，151 个波长点）打到一片“超表面滤光片阵列”上；
  2) 阵列里有 16 个小滤光片，每个滤光片的透过谱 T_m(λ) 都不一样；
  3) 每个滤光片后面接一个探测器，读到的数就是“光谱 × 透过谱再求和”，
     也就是把 151 维的光谱压缩成 16 个数（16 个测量值 y_m）；
  4) 再用一个小神经网络（解码器 MLP）把这 16 个数还原回 151 维光谱 Ŝ(λ)。

“训练”同时优化两样东西：
  - 物理结构（决定 16 条透过谱长什么样）；
  - 解码器（决定怎么把 16 个数还原成光谱）。
让还原出来的 Ŝ(λ) 尽量接近真实的 S(λ)。

这个文件只放“多个脚本都会用到”的公共零件：
  1. 几何约束：D/P、D、间隙 gap、EMT 适用条件；
  2. 材料折射率：TiO2、SiO2、SU-8、空气、熔石英；
  3. 可微 TMM：用 PyTorch 写的传输矩阵法，所以结构参数能反向传播（能被训练）；
  4. AR-EMT 模型：物理编码器 + MLP 解码器；
  5. 常用工具：噪声、通道去相关、MSE/PSNR/SAM 指标、结构参数导出。

单位约定：
  - 长度默认都是 nm；
  - 光谱 shape 默认是 [样本数, 波长点数]；
  - 透过谱 shape 默认是 [通道数, 波长点数]，即 [16, 151]；
  - 数据是“按图像位深缩放后的相对强度”，不做逐条光谱最大值归一化（保留明暗关系）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


# =============================================================================
# 第 1 部分：几何约束
# 决定 16 个滤光片的柱径 D（相对周期 P 的比值 D/P）能取多大、多小。
# =============================================================================


@dataclass
class GeometryConfig:
    """几何约束配置（就是一堆可调数字，打包在一起方便传来传去）。

    period_nm:
        超表面周期 P（相邻柱子中心间距）。当前固定 180 nm，不参与训练。
    ratio_min_design / ratio_max_design:
        设计上希望的 D/P 范围。默认希望 0.1 ~ 0.9。
    g_min_nm:
        最小间隙 gap = P - D。工艺上两根柱子不能挨太近，这会限制 D/P 的上限。
    d_min_nm:
        最小柱径 D。太细的柱子做不出来，默认 60 nm。
    enforce_d_min:
        是否启用 D >= d_min_nm。关掉的话，下限就只由 ratio_min_design 说了算。
    lambda_min_nm:
        检查 EMT 条件用的最短波长。当前光谱从 400 nm 开始。
    """

    period_nm: float = 180.0
    ratio_min_design: float = 0.1
    ratio_max_design: float = 0.9
    g_min_nm: float = 40.0
    d_min_nm: float = 60.0
    enforce_d_min: bool = True
    lambda_min_nm: float = 400.0


def geometry_limits(config: GeometryConfig) -> dict[str, float]:
    """把“设计希望的范围”和“工艺能做到的范围”取交集，得到真正能训练的 D/P 区间。

    设计希望 D/P 在 [0.1, 0.9]，但还得同时满足：
    - D >= d_min_nm；
    - gap = P - D >= g_min_nm。

    举例 P=180、D_min=60、gap_min=40：
    - r_min = max(0.1, 60/180)   = 0.3333；
    - r_max = min(0.9, 1-40/180) = 0.7778。
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


# =============================================================================
# 第 2 部分：把“无界的训练参数”映射到“有物理意义的有界数值”
# 优化器喜欢在 -∞~+∞ 上自由移动；但厚度、D/P 必须落在合理区间。
# 用 sigmoid 做“重参数化”：优化器动 raw，我们用 raw 算出真正的物理量。
# =============================================================================


def bounded_value(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """把无界的训练变量 raw 压进区间 [low, high]。

    sigmoid(raw) 落在 (0,1)，再线性拉伸到 [low, high]。
    好处：训练时怎么更新都不会跑出制造边界，也不用额外写裁剪。
    """

    return low + (high - low) * torch.sigmoid(raw)


def raw_from_bounded(value: torch.Tensor | float | Iterable[float], low: float, high: float) -> torch.Tensor:
    """bounded_value 的逆运算：给一个想要的物理初值，反推出对应的 raw 初值。

    例：想让 h_c 从 600 nm 开始，但真正被优化的是 raw_h_c；
    这个函数就负责把 600 nm 换算成对应的 raw_h_c。
    （logit 就是 sigmoid 的反函数。）
    """

    value_tensor = torch.as_tensor(value, dtype=torch.float32)
    x = (value_tensor - low) / (high - low)
    x = torch.clamp(x, 1e-5, 1.0 - 1e-5)  # 夹一下，避免 logit(0)/logit(1) 变成无穷
    return torch.logit(x)


# =============================================================================
# 第 3 部分：材料折射率 与 EMT 等效折射率
# =============================================================================


def complex_dtype_from_real(real_dtype: torch.dtype) -> torch.dtype:
    """根据实数精度选对应的复数精度。

    TMM 里相位是复数（e^{iδ}），所以折射率、矩阵都要用复数存。
    """

    if real_dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def material_n(name: str, wl_nm: torch.Tensor) -> torch.Tensor:
    """返回某种材料的折射率 n(λ)，输出是复数张量（虚部当前为 0，即不吸收）。

    现在先用最简单的“透明、几乎不色散”模型，方便复现和调试：
    - TiO2（高折射率 H）：n = 2.35；
    - SiO2 / 熔石英（低折射率 L）：n = 1.46；
    - SU-8（光刻胶）：一个简单的 Cauchy 色散公式；
    - air（空气）：n = 1。

    以后若拿到真实椭偏（ellipsometry）数据，把这里换成按波长插值即可，
    其余代码都不用动。
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


def ratio_to_fill_factor(ratio: torch.Tensor) -> torch.Tensor:
    """由 D/P 算“圆柱在正方晶胞里占的面积比”，叫填充因子 f。

    圆柱截面积 = π·(D/2)^2；晶胞面积 = P^2；
    所以 f = π·D^2 / (4·P^2) = π·(D/P)^2 / 4。
    """

    return math.pi * ratio.square() / 4.0


def emt_neff_from_ratio(
    ratio: torch.Tensor,
    wl_nm: torch.Tensor,
    pillar_material: str = "tio2",
    fill_material: str = "su8",
    mode: str = "volume",
) -> torch.Tensor:
    """由 D/P 算 EMT 腔层的“等效折射率” n_eff。

    ratio:
        [M]，M 是滤光片通道数（当前 16）。
    返回:
        [M, Nwl]，每个通道、每个波长一个 n_eff（复数张量）。

    物理图像：
        EMT 腔既不是纯 TiO2，也不是纯 SU-8，而是“TiO2 柱子嵌在 SU-8 里”。
        当结构比波长小很多时，光“看不清”细节，只感受到一个平均折射率，
        这个平均值由填充因子 f 决定——柱子越粗（f 越大），n_eff 越接近 TiO2。

    两种混合公式（默认 volume 体积平均，够用）：
        volume: eps_eff = (1-f)·eps_SU8 + f·eps_TiO2
        mg2d  : Maxwell-Garnett（二维圆柱），更精细，需要时可切换
    （eps 是介电常数，等于 n^2。）
    """

    ratio = torch.as_tensor(ratio, dtype=wl_nm.dtype, device=wl_nm.device)
    if ratio.ndim == 0:
        ratio = ratio[None]

    f = ratio_to_fill_factor(ratio)[:, None]          # [M,1]
    n_fill = material_n(fill_material, wl_nm)[None, :]  # [1,Nwl]
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
    """检查 EMT 近似是否还成立：要求 周期 P < λ_min / n_eff_max。

    只是个“提醒”，不进 loss。如果 P 太大（相对波长），
    结构就会开始衍射，EMT“看成一层均匀介质”的假设就不准了。
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
    """把几何约束和 EMT 检查拼成一段人话报告，训练/评估开始时打印出来看一眼。"""

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
    """给 4 个 AR（增透）层一个“四分之一波长”厚度初值。

    增透膜的经典做法：让每层厚度约等于 λ0/(4n)，这样反射光相消、透过变高。
    层序固定为：空气 / L-H / SU-8 / EMT / H-L / 熔石英（L=SiO2，H=TiO2）。
    返回顺序：[top_L, top_H, bottom_H, bottom_L]。
    """

    n_l = 1.46
    n_h = 2.35
    h_l = lambda0_nm / (4.0 * n_l)
    h_h = lambda0_nm / (4.0 * n_h)
    return [h_l, h_h, h_h, h_l]


# =============================================================================
# 第 4 部分：可微 TMM（传输矩阵法）
# 给定“每层折射率 + 每层厚度 + 入射角”，算出多层膜的透过率 T(λ)。
# 关键点：全用 PyTorch 张量运算写，所以对“结构参数”可求导 → 能被训练。
# 下面这段数学没改过，保留原样，只补了注释。
# =============================================================================


def _expand_layer_n(n_value: torch.Tensor, n_struct: int, n_wl: int, device: torch.device) -> torch.Tensor:
    """把某一层的折射率整理成统一形状 [结构数, 波长数]，方便后面堆叠。"""

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
    """把某一层的厚度整理成统一形状 [结构数]。"""

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
    """PyTorch 可微 TMM，返回非偏振透过率（s 偏振和 p 偏振各算一次再平均）。

    n_layers:
        每一层的折射率，包含最上面的入射介质(空气)和最下面的基底(熔石英)，
        长度 = 中间有限厚度层数 + 2。
    d_layers_nm:
        只包含中间那些有限厚度层的厚度，不含空气和基底（它们当作半无限厚）。
    返回:
        [Nangle, Nstructure, Nwl]，即 [角度数, 通道数, 波长数]。

    直觉：每一层用一个 2x2“特征矩阵”描述光在里面走一趟的相位和振幅变化，
    把所有层的矩阵乘起来，就得到整叠膜的总响应，再换算成透过率。
    """

    if len(d_layers_nm) != len(n_layers) - 2:
        raise ValueError("d_layers_nm 必须只给内部层厚度，不包含空气和基底。")

    wl = wl_nm.to(dtype=torch.float32)
    device = wl.device
    n_wl = wl.numel()

    # 有多少个“结构”（这里就是 16 个滤光片；它们只有 EMT 层的 n_eff 不同）
    n_struct = 1
    for n in n_layers:
        if isinstance(n, torch.Tensor) and n.ndim == 2:
            n_struct = max(n_struct, n.shape[0])

    n_stack = torch.stack([_expand_layer_n(n, n_struct, n_wl, device) for n in n_layers], dim=0)
    d_stack = torch.stack([_expand_layer_d(d, n_struct, device) for d in d_layers_nm], dim=0)

    # 入射角 → 弧度。可以一次算多个角度。
    alpha = torch.as_tensor(alpha_deg, dtype=torch.float32, device=device).reshape(-1)
    alpha_rad = alpha * math.pi / 180.0

    # 斯涅尔定律：横向波矢 q = n0·sin(θ0) 在所有层里守恒，
    # 于是每层的 cosθ 都能由 q 和该层折射率算出来。
    q = n_stack[0][None, :, :] * torch.sin(alpha_rad)[:, None, None]
    n_for_all_layers = n_stack[:, None, :, :]
    cos_theta = torch.sqrt(1.0 - (q[None, :, :, :] / n_for_all_layers).square())
    cos_theta = torch.where(cos_theta.real < 0, -cos_theta, cos_theta)  # 选物理上正确的那个根

    def one_pol(pol: str) -> torch.Tensor:
        # η 是“光学导纳”，s 和 p 偏振公式不同
        if pol == "s":
            eta = n_for_all_layers * cos_theta
        elif pol == "p":
            eta = n_for_all_layers / cos_theta
        else:
            raise ValueError(pol)

        # 从单位矩阵开始，逐层把特征矩阵乘进去
        shape = eta[0].shape
        m11 = torch.ones(shape, dtype=eta.dtype, device=device)
        m12 = torch.zeros(shape, dtype=eta.dtype, device=device)
        m21 = torch.zeros(shape, dtype=eta.dtype, device=device)
        m22 = torch.ones(shape, dtype=eta.dtype, device=device)

        wl_b = wl[None, None, :]
        imaginary = torch.tensor(1j, dtype=eta.dtype, device=device)

        for layer_idx in range(1, len(n_layers) - 1):  # 只乘中间有限厚度层
            d_nm = d_stack[layer_idx - 1][None, :, None]
            # 相位厚度 δ = 2π/λ · n · d · cosθ
            delta = 2.0 * math.pi / wl_b * n_stack[layer_idx][None, :, :] * d_nm * cos_theta[layer_idx]
            cos_delta = torch.cos(delta)
            sin_delta = torch.sin(delta)
            eta_j = eta[layer_idx]

            # 单层特征矩阵 [[cosδ, i·sinδ/η], [i·η·sinδ, cosδ]]
            a = cos_delta
            b = imaginary * sin_delta / eta_j
            c = imaginary * eta_j * sin_delta
            d = cos_delta

            new11 = m11 * a + m12 * c
            new12 = m11 * b + m12 * d
            new21 = m21 * a + m22 * c
            new22 = m21 * b + m22 * d
            m11, m12, m21, m22 = new11, new12, new21, new22

        # 用总矩阵和首末介质的导纳，换算成透过率
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
    """把结构参数拼成完整层序，调用 TMM 得到 16 个滤光片的透过谱。

    固定层序（从上到下）：
        空气 / top SiO2 / top TiO2 / 残余 SU-8 / EMT 腔 / bottom TiO2 / bottom SiO2 / 熔石英
    只有 EMT 腔那一层的折射率随通道(D/P)变化，其它层 16 个通道共享。
    """

    wl = wl_nm.to(dtype=torch.float32)
    ratio = ratio.to(device=wl.device, dtype=torch.float32)
    ar = ar_thickness_nm.to(device=wl.device, dtype=torch.float32)

    if ar.numel() != 4:
        raise ValueError("ar_thickness_nm 必须是 4 个数: top_L, top_H, bottom_H, bottom_L。")

    n_eff = emt_neff_from_ratio(ratio, wl)  # [16, Nwl]，每个通道不同
    n_struct = ratio.numel()

    def const_layer(name: str) -> torch.Tensor:
        # 16 个通道共享的常数折射率层，扩成 [16, Nwl]
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


# =============================================================================
# 第 5 部分：训练时会用到的“小工具函数”
# 噪声、通道去相关、光谱角损失。它们让建模更贴近现实、重建更稳。
# =============================================================================


def add_measurement_noise(
    meas: torch.Tensor,
    rel_sigma: float = 0.0,
    abs_sigma: float = 0.0,
) -> torch.Tensor:
    """给 16 通道测量值加噪声（只在训练时用，评估/推理保持干净）。

    为什么必须加噪声？
        真实探测器一定有噪声。如果只用“干净测量”训练，解码器会把
        16→151 这个本来很病态的还原过程“背”得非常好，纸面 PSNR 很漂亮；
        但一上真实设备、测量一抖，重建就崩。训练时故意加噪，
        等于逼模型学会“抗噪的、稳健的”还原方式（也顺带起正则化作用）。

    两种噪声：
        rel_sigma —— 相对(光度/散粒)噪声：大小正比于信号本身。
                     好处是与数据的绝对尺度无关，最稳妥、最好调。
                     y += y · rel_sigma · N(0,1)
        abs_sigma —— 绝对(读出)噪声：固定大小，模拟探测器本底读出噪声。
                     需要按你测量值的典型尺度来设（比如 0.02）。
                     y += abs_sigma · N(0,1)

    想更物理一点（真正的散粒噪声正比于 sqrt(信号)）：把下面
        meas * rel_sigma
    改成
        torch.sqrt(torch.relu(meas)) * rel_sigma
    即可。这里默认用相对噪声，是因为它不挑数据尺度、对小白最省心。
    """

    if rel_sigma <= 0.0 and abs_sigma <= 0.0:
        return meas

    out = meas
    if rel_sigma > 0.0:
        out = out + meas * rel_sigma * torch.randn_like(meas)
    if abs_sigma > 0.0:
        out = out + abs_sigma * torch.randn_like(meas)
    return out


def measurement_matrix_coherence(transmission: torch.Tensor) -> torch.Tensor:
    """衡量 16 条透过曲线彼此有多“像”（相关），返回一个标量，越小越好。

    为什么要管这个？
        重建靠的是 16 个滤光片“看到不同的东西”。如果两个滤光片透过谱几乎一样，
        它们提供的信息就重复了，等于只有更少的有效通道 → 16→151 更难还原。
        所以我们希望这 16 条曲线尽量“形状各不相同、互相补充”。

    做法（可导，能进 loss）：
        1. 把每条透过曲线归一化成单位长度（只看形状，不看亮度）；
        2. 两两算余弦相似度，得到一个 16×16 的相似度矩阵，对角线都是 1；
        3. 取“非对角元素的均方”作为惩罚。越小 → 越正交 → 通道越互补。

    注意用的是“归一化后”的曲线，所以它只惩罚形状雷同，
    不会为了降这个值而把透过率整体压到 0（亮度另有 t_target 约束管）。
    """

    t = transmission
    if t.ndim == 3:      # [A,16,151] 取第 0 个角度当代表
        t = t[0]
    t = t.real if torch.is_complex(t) else t

    t_norm = t / (t.norm(dim=1, keepdim=True) + 1e-8)  # 每行单位化
    gram = t_norm @ t_norm.t()                          # [16,16]，对角=1
    n = gram.shape[0]
    off = gram - torch.eye(n, device=gram.device, dtype=gram.dtype)  # 去掉对角
    return off.square().sum() / (n * (n - 1))


def sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """光谱角损失（可导版），返回平均弧度。

    MSE 只看逐点数值差，容易把尖锐的谱峰“抹平”；
    SAM(Spectral Angle) 把每条光谱当成一个向量，只比较“方向/形状”差异，
    对峰位、峰形更敏感。加一点点 SAM，有助于保住谱形。
    """

    dot = torch.sum(pred * target, dim=1)
    denom = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    cos_val = torch.clamp(dot / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.mean(torch.arccos(cos_val))


# =============================================================================
# 第 6 部分：AR-EMT 模型（物理编码器 + MLP 解码器）
# =============================================================================


class AREMTModel(nn.Module):
    """AR-EMT 可训练模型。

    前半（物理编码器）：
        ρ(raw) → D/P → D → 填充因子 f → n_eff → TMM → 16 条透过谱 → 16 个测量值。
    后半（MLP 解码器）：
        16 个测量值 → 还原出 151 维光谱。

    要点：
        当前输入是“一条光谱”，不是二维图像 patch，所以没有卷积、不学空间结构。
        它学的是“单条光谱经过 16 个物理滤光片后，怎么还原回去”。
    """

    def __init__(
        self,
        wl_nm: torch.Tensor,
        config: GeometryConfig,
        n_channels: int = 16,
        hidden_dims: tuple[int, ...] = (512, 256),  # 解码器隐藏层：可自由加深/加宽（改了要重新训练）
        h_c_range: tuple[float, float] = (300.0, 1200.0),
        t_r_range: tuple[float, float] = (0.0, 150.0),
        ar_range: tuple[float, float] = (5.0, 160.0),
    ) -> None:
        super().__init__()
        self.config = config
        self.n_channels = n_channels
        self.hidden_dims = hidden_dims
        self.h_c_range = h_c_range
        self.t_r_range = t_r_range
        self.ar_range = ar_range

        limits = geometry_limits(config)
        self.r_min = limits["r_min"]
        self.r_max = limits["r_max"]

        # 波长网格存成 buffer：跟着模型 .to(device) 走，但不是可训练参数
        self.register_buffer("wl_nm", wl_nm.detach().clone().to(dtype=torch.float32))

        # ---- 可训练的物理参数（都存 raw，用 sigmoid 映射到物理区间）----
        # 16 个通道的 D/P 初值在可行区间里均匀铺开，避免一开始 16 个滤光片全一样
        ratio_init = torch.linspace(
            self.r_min + 0.02 * (self.r_max - self.r_min),
            self.r_max - 0.02 * (self.r_max - self.r_min),
            n_channels,
        )
        self.rho = nn.Parameter(raw_from_bounded(ratio_init, self.r_min, self.r_max))

        # 下面三组厚度是“全局共享”的：16 个滤光片用同一套厚度，只有 D/P 各不相同
        self.raw_h_c = nn.Parameter(raw_from_bounded(600.0, *h_c_range))              # EMT 腔厚
        self.raw_t_r = nn.Parameter(raw_from_bounded(50.0, *t_r_range))               # 残余 SU-8 厚
        self.raw_ar = nn.Parameter(raw_from_bounded(quarter_wave_ar_thickness(), *ar_range))  # 4 个 AR 层厚

        # ---- 解码器 MLP：16 → ...隐藏层... → 151 ----
        # 结尾用 Softplus 把输出压成“非负”：光强不可能为负，这样重建物理上才说得通。
        # （旧版最后一层是纯线性，会输出负值。若你想换回纯线性，删掉那行 Softplus 即可。）
        decoder_layers: list[nn.Module] = []
        in_dim = n_channels
        for h_dim in hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.LeakyReLU(0.01))
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, wl_nm.numel()))
        decoder_layers.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder_layers)

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        """把 raw 参数换算成有物理意义的量（D/P、各层厚度，单位 nm）。"""

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
        """算当前结构在指定入射角下的 16 条透过谱，返回 [角度数, 16, 151]。

        那几个 *_override 参数是给“制造误差评估”用的：
        评估时想临时把结构参数扰动一下（模拟加工误差），但不改动模型本身，
        就把扰动后的值传进来覆盖。平时训练不用管它们。
        """

        params = self.physical_parameters()
        ratio = params["ratio"] if ratio_override is None else ratio_override
        h_c = params["h_c_nm"] if h_c_override is None else h_c_override
        t_r = params["t_r_nm"] if t_r_override is None else t_r_override
        ar = params["ar_nm"] if ar_override is None else ar_override
        return build_ar_emt_transmission(ratio, h_c, t_r, ar, self.wl_nm, alpha_deg)

    def measure(self, spectra: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
        """用透过谱把输入光谱压成 16 个测量值：y_m = Σ_λ S(λ)·T_m(λ)。

        spectra:      [B, 151]
        transmission: [16, 151]（所有样本共用一组滤光片）
                      或 [B, 16, 151]（每个样本入射角不同 → 每个样本一组滤光片）
        返回:         [B, 16]

        这里是“积分求和”，不除以 151——因为数据用的是绝对强度，
        我们想保留“光进来多少、探测器收多少”的真实数量关系。
        """

        if transmission.ndim == 2:
            return spectra @ transmission.T
        if transmission.ndim == 3 and transmission.shape[0] == spectra.shape[0]:
            return torch.einsum("bn,bmn->bm", spectra, transmission)
        raise ValueError(f"透过谱 shape={tuple(transmission.shape)} 与光谱 batch={spectra.shape[0]} 不匹配。")

    def forward(self, spectra: torch.Tensor, alpha_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """干净的完整前向：光谱 → 测量 → 重建光谱。返回 (重建, 透过谱)。

        注意：这里不加噪声。训练脚本里会把“加噪声”这一步单独、显式地写出来
        （measure → 加噪 → decoder），这样噪声在哪一步一目了然，方便你调。
        """

        t = self.transmission(alpha_deg)
        meas = self.measure(spectra, t[0] if t.shape[0] == 1 else t)
        pred = self.decoder(meas)
        return pred, t


# =============================================================================
# 第 7 部分：评估指标 与 结构参数导出
# =============================================================================


def metric_mse_psnr_sam(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """一次算三个常用指标：MSE、PSNR、SAM（都只用于看结果，不进 loss）。

    - MSE：逐点均方误差，越小越好。
    - PSNR：峰值信噪比(dB)，越大越好。这里按峰值=1 计算（数据已缩放到 0~1）。
    - SAM：光谱角(弧度)，衡量谱形差异，越小越好。
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
    """在固定入射角下评估重建精度（干净、无噪声）。

    分块跑，避免一次性把所有测试光谱塞进显存。
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
    """通道区分度 tor（只用于观察，不进 loss）。

    做法：16 个通道两两比较，每对算 mean(|T_i - T_j|)·100，取最小值。
    数值越大，说明“最像的那一对”也还挺不一样，即整体区分度好。
    （这是个直观指标；训练里真正推动区分度的是 measurement_matrix_coherence。）
    """

    t = t_matrix.detach().cpu()
    n_ch = t.shape[0]
    diffs = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            diffs.append(torch.mean(torch.abs(t[i] - t[j])) * 100.0)
    return float(torch.min(torch.stack(diffs))) if diffs else 0.0


def structure_rows(model: AREMTModel) -> list[dict[str, float]]:
    """导出每个通道的结构参数（D/P、D、gap、填充因子、n_eff 范围），方便存 CSV。"""

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
