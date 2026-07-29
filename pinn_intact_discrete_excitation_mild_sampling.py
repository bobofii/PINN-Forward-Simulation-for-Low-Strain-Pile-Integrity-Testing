# -*- coding: utf-8 -*-
"""
PINN Forward Simulation for Low-Strain Integrity Testing (LIT)
1D wave propagation in an intact pile (完整桩)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from timeit import default_timer

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None

# =========================
# 0) Plot font
# =========================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) Device & Seed
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

RESULT_DIR = "Results_Intact_DiscreteExcitation_MildSampling"
os.makedirs(RESULT_DIR, exist_ok=True)

# ABAQUS 数据文件路径：默认与本脚本放在同一目录
ABAQUS_RPT_PATH = "abaqus_1d_intact_400H_top_velocity.rpt"

# =========================
# 2) Intact pile parameters
# =========================
L = 13.0           # 桩长 (m)
D_normal = 0.5     # 完整桩直径 (m)

# 材料参数 (C40混凝土)
rho0 = 2400.0      # 密度 (kg/m^3)
E0 = 32500e6       # 弹性模量 (Pa)

# 计算波速
c0 = np.sqrt(E0 / rho0)

# 时间参数
T_obs = 10e-3      # 观察时间 10 ms
T_pulse = 1e-3     # 脉冲宽度 1 ms
p0 = 750.0         # 激振应力峰值 (N/m^2)

# =========================
# 3) Non-dimensionalization
# =========================
x_ref = L
t_ref = L / c0
t_star_max = T_obs / t_ref
T_pulse_star = T_pulse / t_ref
U_ref = p0 * L / E0

# 理论关键时刻，无量纲时间
# 波速归一化后为 1，因此：
# - 入射波前到达桩底：t*=1
# - 桩顶接收桩底反射波前：t*=2
# - 入射脉冲峰值：t*=T_pulse*/2
# - 桩底反射峰值到达桩顶：t*=2+T_pulse*/2
t_bottom_arrive_star = 1.0
t_bottom_reflect_star = 2.0

print("=== 完整桩参数 ===")
print(f"桩长: {L} m")
print(f"桩径: {D_normal} m")
print(f"材料密度: {rho0} kg/m^3")
print(f"弹性模量: {E0 / 1e6:.1f} MPa")
print(f"波速: {c0:.1f} m/s")
print(f"观察时间: {T_obs * 1e3:.1f} ms")
print(f"脉冲宽度: {T_pulse * 1e3:.1f} ms")
print(f"无量纲观察时间: {t_star_max:.4f}")
print(f"无量纲脉冲宽度: {T_pulse_star:.4f}")
print(f"桩底反射波前到达桩顶时间: {(2 * L / c0) * 1e3:.3f} ms")
print(f"桩底反射峰值到达桩顶时间: {((2 * L / c0) + T_pulse / 2.0) * 1e3:.3f} ms")
print("==================")

# =========================
# 4) Intact-pile material profiles
# =========================
def make_intact_profiles():
    """完整桩：截面、弹性模量和密度均为常数。"""
    A_normal = np.pi * (D_normal / 2.0) ** 2
    print(f"完整桩截面面积: {A_normal:.6f} m^2")
    print("无量纲截面 A*(x*) = 1.0")

    def A_star(x):
        return torch.ones_like(x)

    def E_star(x):
        return torch.ones_like(x)

    def rho_star(x):
        return torch.ones_like(x)

    return A_star, E_star, rho_star


A_star_fun, E_star_fun, rho_star_fun = make_intact_profiles()

# =========================
# 5) Discrete top-excitation data (CSV input)
# =========================
EXCITATION_CSV_PATH = "half_sine_discrete_excitation.csv"
AUTO_CREATE_EXCITATION_CSV = True
EXCITATION_SAMPLE_COUNT = 101


EXCITATION_INTERPOLATION = "linear"
EXCITATION_LOOKUP_POINTS = 2001


def get_base_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_local_path(path):
    if path is None or str(path).strip() == "":
        return None

    path = str(path)
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.join(get_base_dir(), path),
        os.path.join(os.getcwd(), path),
        path,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return os.path.join(get_base_dir(), path)


def create_example_excitation_csv(csv_path, n_points=EXCITATION_SAMPLE_COUNT):
    if n_points < 3:
        raise ValueError("EXCITATION_SAMPLE_COUNT 至少应为 3。")

    time_s = np.linspace(0.0, T_pulse, n_points, dtype=np.float64)
    stress_pa = p0 * np.sin(np.pi * time_s / T_pulse)
    stress_pa[0] = 0.0
    stress_pa[-1] = 0.0

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    np.savetxt(
        csv_path,
        np.column_stack([time_s, stress_pa]),
        delimiter=",",
        header="time_s,stress_pa",
        comments="",
        fmt=["%.10e", "%.10e"],
    )
    print(f"已自动生成离散激励示例文件: {csv_path}")


def read_excitation_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到离散激励文件: {csv_path}")

    try:
        data = np.genfromtxt(
            csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig"
        )
        if data.dtype.names is None or len(data.dtype.names) < 2:
            raise ValueError
        names = data.dtype.names
        time_s = np.atleast_1d(data[names[0]]).astype(np.float64)
        stress_pa = np.atleast_1d(data[names[1]]).astype(np.float64)
    except Exception:
        raw = np.genfromtxt(csv_path, delimiter=",", dtype=float, encoding="utf-8-sig")
        raw = np.atleast_2d(raw)
        if raw.shape[1] < 2:
            raise ValueError("离散激励 CSV 至少需要两列：time_s, stress_pa。")
        raw = raw[np.all(np.isfinite(raw[:, :2]), axis=1)]
        if raw.shape[0] < 2:
            raise ValueError("离散激励 CSV 有效数据点少于 2 个。")
        time_s = raw[:, 0].astype(np.float64)
        stress_pa = raw[:, 1].astype(np.float64)

    valid = np.isfinite(time_s) & np.isfinite(stress_pa)
    time_s = time_s[valid]
    stress_pa = stress_pa[valid]

    if time_s.size < 2:
        raise ValueError("离散激励数据少于两个有效时间点。")
    if np.any(time_s < 0.0):
        raise ValueError("离散激励时间不能为负数。")

    order = np.argsort(time_s)
    time_s = time_s[order]
    stress_pa = stress_pa[order]

    unique_t, inverse = np.unique(time_s, return_inverse=True)
    if unique_t.size != time_s.size:
        stress_sum = np.zeros_like(unique_t, dtype=np.float64)
        count = np.zeros_like(unique_t, dtype=np.float64)
        np.add.at(stress_sum, inverse, stress_pa)
        np.add.at(count, inverse, 1.0)
        time_s = unique_t
        stress_pa = stress_sum / count

    if time_s.size < 2 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("离散激励时间点必须严格递增。")

    return time_s, stress_pa


def build_excitation_lookup(time_s, stress_pa):
    mode = EXCITATION_INTERPOLATION.lower().strip()

    if mode == "pchip" and PchipInterpolator is not None and time_s.size >= 3:
        n_dense = max(int(EXCITATION_LOOKUP_POINTS), int(time_s.size))
        lookup_time_s = np.linspace(time_s[0], time_s[-1], n_dense, dtype=np.float64)
        interpolator = PchipInterpolator(time_s, stress_pa, extrapolate=False)
        lookup_stress_pa = np.asarray(interpolator(lookup_time_s), dtype=np.float64)
        lookup_stress_pa[0] = stress_pa[0]
        lookup_stress_pa[-1] = stress_pa[-1]
        used_mode = "PCHIP shape-preserving interpolation"
    else:
        if mode == "pchip" and PchipInterpolator is None:
            print("警告：未安装 SciPy，离散激励自动退回分段线性插值。")
        lookup_time_s = time_s.copy()
        lookup_stress_pa = stress_pa.copy()
        used_mode = "piecewise linear interpolation"

    if not np.all(np.isfinite(lookup_stress_pa)):
        raise ValueError("插值后的激励查找表包含非有限值。")

    return lookup_time_s, lookup_stress_pa, used_mode


def load_excitation_data():
    csv_path = resolve_local_path(EXCITATION_CSV_PATH)

    if not os.path.exists(csv_path):
        if AUTO_CREATE_EXCITATION_CSV:
            create_example_excitation_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"未找到 {csv_path}，请提供离散激励 CSV，"
                "或将 AUTO_CREATE_EXCITATION_CSV 设为 True。"
            )

    time_s, stress_pa = read_excitation_csv(csv_path)
    lookup_time_s, lookup_stress_pa, used_mode = build_excitation_lookup(time_s, stress_pa)

    lookup_time_star = lookup_time_s / t_ref
    lookup_stress_star = lookup_stress_pa / p0

    lookup_time_tensor = torch.tensor(
        lookup_time_star, dtype=torch.float32, device=device
    ).contiguous()
    lookup_stress_tensor = torch.tensor(
        lookup_stress_star, dtype=torch.float32, device=device
    ).contiguous()

    print("=== 离散桩顶激励数据 ===")
    print(f"文件: {csv_path}")
    print(f"原始离散点数: {time_s.size}")
    print(f"训练查找表点数: {lookup_time_s.size}")
    print(f"时间范围: {time_s[0] * 1e3:.6f} ~ {time_s[-1] * 1e3:.6f} ms")
    print(f"离散应力峰值: {np.max(np.abs(stress_pa)):.6f} Pa")
    print(f"插值方式: {used_mode}")
    print("数据时间范围外激励自动置零")
    print("========================")

    return (
        csv_path,
        time_s,
        stress_pa,
        lookup_time_tensor,
        lookup_stress_tensor,
        used_mode,
    )


(
    EXCITATION_CSV_RESOLVED,
    excitation_time_s,
    excitation_stress_pa,
    excitation_time_star_tensor,
    excitation_stress_star_tensor,
    excitation_interpolation_used,
) = load_excitation_data()


def p_star(t_star: torch.Tensor) -> torch.Tensor:
    """由离散激励查找表获得无量纲桩顶应力 p*(t*)。"""
    original_shape = t_star.shape
    t_flat = t_star.reshape(-1).contiguous()

    t_data = excitation_time_star_tensor
    p_data = excitation_stress_star_tensor

    idx_right = torch.searchsorted(t_data, t_flat, right=False)
    idx_right = torch.clamp(idx_right, 1, t_data.numel() - 1)
    idx_left = idx_right - 1

    t_left = t_data[idx_left]
    t_right = t_data[idx_right]
    p_left = p_data[idx_left]
    p_right = p_data[idx_right]

    ratio = (t_flat - t_left) / (t_right - t_left + 1.0e-12)
    p_interp = p_left + ratio * (p_right - p_left)

    outside = (t_flat < t_data[0]) | (t_flat > t_data[-1])
    p_interp = torch.where(outside, torch.zeros_like(p_interp), p_interp)
    return p_interp.reshape(original_shape)


def save_excitation_check_plot():
    t_dense_s = np.linspace(0.0, T_obs, 2400, dtype=np.float64)
    t_dense_star = torch.tensor(
        t_dense_s / t_ref, dtype=torch.float32, device=device
    ).view(-1, 1)

    with torch.no_grad():
        p_dense_pa = p0 * p_star(t_dense_star).cpu().numpy().reshape(-1)

    plt.figure(figsize=(11, 4.5))
    plt.plot(
        t_dense_s * 1e3,
        p_dense_pa,
        linewidth=1.8,
        label=excitation_interpolation_used,
    )
    plt.scatter(
        excitation_time_s * 1e3,
        excitation_stress_pa,
        s=18,
        label="Discrete excitation data",
        zorder=3,
    )
    plt.xlabel("Time t / ms")
    plt.ylabel(r"Excitation stress $p(t)$ / Pa")
    plt.title("Discrete pile-head excitation input")
    plt.xlim(0.0, T_obs * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "discrete_excitation_input.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"离散激励检查图已保存: {save_path}")

# =========================
# 6) PINN model
# =========================
class PINN_Intact(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden=4, activation="tanh"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2, hidden_dim))
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "sin":
            self.act = torch.sin
        else:
            raise ValueError("activation must be 'tanh' or 'sin'")


        self.history = {
            "total": [],
            "pde": [],
            "ic_u": [],
            "ic_v": [],
            "bc_top": [],
            "bc_bot": [],
            "w_ic_v": [],
            "lr": [],
            "val_epoch": [],
            "val_total": [],
            "val_pde": [],
            "val_ic_u": [],
            "val_ic_v": [],
            "val_bc_top": [],
            "val_bc_bot": [],
        }

    def forward(self, x_star, t_star):
        X = torch.cat([x_star, t_star], dim=1)
        h = X
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)

    def pde_residual(self, x_star, t_star):
        """无量纲一维纵向波动方程残差。"""
        x_star = x_star.clone().detach().requires_grad_(True)
        t_star = t_star.clone().detach().requires_grad_(True)
        u = self.forward(x_star, t_star)

        u_x = grad(u, x_star, torch.ones_like(u), create_graph=True)[0]
        u_t = grad(u, t_star, torch.ones_like(u), create_graph=True)[0]
        u_tt = grad(u_t, t_star, torch.ones_like(u_t), create_graph=True)[0]

        A = A_star_fun(x_star)
        E = E_star_fun(x_star)
        rho = rho_star_fun(x_star)

        flux = E * A * u_x
        flux_x = grad(flux, x_star, torch.ones_like(flux), create_graph=True)[0]
        return rho * A * u_tt - flux_x

    def bc_top_residual(self, t_star):
        t_star = t_star.clone().detach().requires_grad_(True)
        x0 = torch.zeros_like(t_star, device=t_star.device).requires_grad_(True)
        u = self.forward(x0, t_star)
        u_x = grad(u, x0, torch.ones_like(u), create_graph=True)[0]
        return u_x + p_star(t_star)

    def bc_bot_residual(self, t_star):
        t_star = t_star.clone().detach().requires_grad_(True)
        x1 = torch.ones_like(t_star, device=t_star.device).requires_grad_(True)
        u = self.forward(x1, t_star)
        u_x = grad(u, x1, torch.ones_like(u), create_graph=True)[0]
        return u_x

    def ic_residuals(self, x_star):
        x_star = x_star.clone().detach().requires_grad_(True)
        t0 = torch.zeros_like(x_star, device=x_star.device).requires_grad_(True)
        u = self.forward(x_star, t0)
        u_t = grad(u, t0, torch.ones_like(u), create_graph=True)[0]
        return u, u_t

    def loss(self, batch, weights, record=True, lr_value=None):
        w_pde = weights.get("pde", 50.0)
        w_ic_u = weights.get("ic_u", 100.0)
        w_ic_v = weights.get("ic_v", 100.0)
        w_bc_top = weights.get("bc_top", 100.0)
        w_bc_bot = weights.get("bc_bot", 100.0)

        x_pde, t_pde = batch["pde"]
        res_pde = self.pde_residual(x_pde, t_pde)
        loss_pde = torch.mean(res_pde ** 2)


        u_ic, v_ic = self.ic_residuals(batch["ic"])
        loss_ic_u = torch.mean(u_ic ** 2)
        loss_ic_v = torch.mean(v_ic ** 2)


        res_top = self.bc_top_residual(batch["bc_top_t"])
        loss_bc_top = torch.mean(res_top ** 2)

        res_bot = self.bc_bot_residual(batch["bc_bot_t"])
        loss_bc_bot = torch.mean(res_bot ** 2)

        total_loss = (
            w_pde * loss_pde
            + w_ic_u * loss_ic_u
            + w_ic_v * loss_ic_v
            + w_bc_top * loss_bc_top
            + w_bc_bot * loss_bc_bot
        )

        if record:
            self.history["total"].append(float(total_loss.detach().cpu()))
            self.history["pde"].append(float(loss_pde.detach().cpu()))
            self.history["ic_u"].append(float(loss_ic_u.detach().cpu()))
            self.history["ic_v"].append(float(loss_ic_v.detach().cpu()))
            self.history["bc_top"].append(float(loss_bc_top.detach().cpu()))
            self.history["bc_bot"].append(float(loss_bc_bot.detach().cpu()))
            self.history["w_ic_v"].append(float(w_ic_v))
            self.history["lr"].append(
                float("nan") if lr_value is None else float(lr_value)
            )

        return (
            total_loss,
            loss_pde,
            loss_ic_u,
            loss_ic_v,
            loss_bc_top,
            loss_bc_bot,
        )

# =========================
# 7) Sampling utilities
# =========================
def _shuffle_cat(*parts):
    parts = [p for p in parts if p is not None and p.numel() > 0]
    if len(parts) == 0:
        return torch.empty(0, 1, device=device)
    x = torch.cat(parts, dim=0)
    perm = torch.randperm(x.shape[0], device=device)
    return x[perm]


def _linspace_col(t0, t1, n):
    if n <= 0:
        return torch.empty(0, 1, device=device)
    return torch.linspace(float(t0), float(t1), n, device=device).view(-1, 1)


def _rand_window(t0, t1, n):
    if n <= 0:
        return torch.empty(0, 1, device=device)
    t0 = max(0.0, float(t0))
    t1 = min(float(t1), float(t_star_max))
    if t1 <= t0:
        return torch.full((n, 1), t0, device=device)
    return t0 + (t1 - t0) * torch.rand(n, 1, device=device)


def sample_uniform_x(n_total):
    return torch.rand(n_total, 1, device=device)


def sample_ic_x(
    n_ic,
    top_ratio=0.03125,
    near_top_ratio=0.03125,
    near_top_width=0.03,
):

    if n_ic < 1:
        raise ValueError("n_ic 必须大于 0。")

    n_top = max(1, int(round(n_ic * top_ratio)))
    n_near = max(1, int(round(n_ic * near_top_ratio)))

    if n_top + n_near > n_ic:
        n_top = max(1, n_ic // 2)
        n_near = max(0, n_ic - n_top)

    n_uniform = n_ic - n_top - n_near

    x_top = torch.zeros(n_top, 1, device=device)
    x_near = near_top_width * torch.rand(n_near, 1, device=device)
    x_uniform = torch.rand(n_uniform, 1, device=device)


    anchors = torch.tensor(
        [[0.25], [0.50], [0.75], [1.00]],
        device=device,
        dtype=torch.float32,
    )
    n_anchor = min(anchors.shape[0], n_uniform)
    if n_anchor > 0:
        x_uniform[:n_anchor] = anchors[:n_anchor]

    return _shuffle_cat(x_top, x_near, x_uniform)

# =========================
# 8) Mild fixed sampling
# =========================
def sample_time_with_windows(
    n_total,
    bottom_return_ratio=0.06,
    pulse_ratio=0.04,
    early_ratio=0.04,
):

    n_early = int(n_total * early_ratio)
    n_bottom_return = int(n_total * bottom_return_ratio)
    n_pulse = int(n_total * pulse_ratio)
    n_uniform = max(n_total - n_early - n_bottom_return - n_pulse, 0)

    t_uniform = torch.rand(n_uniform, 1, device=device) * t_star_max
    t_early = _rand_window(
        0.0,
        min(0.15 * T_pulse_star, t_star_max),
        n_early,
    )
    t_pulse = _rand_window(
        0.0,
        min(1.20 * T_pulse_star, t_star_max),
        n_pulse,
    )
    t_bottom_return = _rand_window(
        t_bottom_reflect_star - 0.12,
        t_bottom_reflect_star + T_pulse_star + 0.12,
        n_bottom_return,
    )

    return _shuffle_cat(t_uniform, t_early, t_pulse, t_bottom_return)


def get_excitation_boundary_times():
    t_csv = torch.tensor(
        excitation_time_s / t_ref,
        dtype=torch.float32,
        device=device,
    ).view(-1, 1)
    mask = (t_csv[:, 0] >= 0.0) & (t_csv[:, 0] <= t_star_max)
    t_csv = t_csv[mask].view(-1, 1)
    if t_csv.numel() == 0:
        return t_csv
    return torch.unique(t_csv.squeeze(1), sorted=True).view(-1, 1)


def sample_top_time_mild_discrete(
    n_bc,
    pulse_rand_ratio=0.06,
    early_ratio=0.03,
    reflection_ratio=0.04,
):

    t_csv = get_excitation_boundary_times()
    if t_csv.shape[0] > n_bc:
        # 极端情况下按索引均匀抽取，保证总点数不超过 n_bc。
        idx = torch.linspace(
            0, t_csv.shape[0] - 1, n_bc, device=device
        ).round().long()
        return t_csv[idx]

    n_pulse_rand = int(n_bc * pulse_rand_ratio)
    n_early = int(n_bc * early_ratio)
    n_reflection = int(n_bc * reflection_ratio)

    reflection_anchors = torch.tensor(
        [
            [t_bottom_reflect_star],
            [t_bottom_reflect_star + 0.50 * T_pulse_star],
        ],
        dtype=torch.float32,
        device=device,
    )
    reflection_anchors = reflection_anchors[
        (reflection_anchors[:, 0] >= 0.0)
        & (reflection_anchors[:, 0] <= t_star_max)
    ].view(-1, 1)

    n_uniform = max(
        n_bc
        - t_csv.shape[0]
        - n_pulse_rand
        - n_early
        - n_reflection
        - reflection_anchors.shape[0],
        0,
    )

    t_uniform = torch.rand(n_uniform, 1, device=device) * t_star_max
    t_pulse_rand = _rand_window(0.0, T_pulse_star, n_pulse_rand)
    t_early = _rand_window(
        0.0,
        min(0.15 * T_pulse_star, t_star_max),
        n_early,
    )
    t_reflection = _rand_window(
        t_bottom_reflect_star - 0.06,
        t_bottom_reflect_star + T_pulse_star + 0.06,
        n_reflection,
    )

    t_all = _shuffle_cat(
        t_uniform,
        t_csv,
        t_pulse_rand,
        t_early,
        t_reflection,
        reflection_anchors,
    )

    # 由于整数取整，理论上可能少于 n_bc；用全时域随机点补足。
    if t_all.shape[0] < n_bc:
        extra = torch.rand(n_bc - t_all.shape[0], 1, device=device) * t_star_max
        t_all = _shuffle_cat(t_all, extra)
    elif t_all.shape[0] > n_bc:
        t_all = t_all[:n_bc]

    return t_all


def sample_bottom_time_mild(n_bc):
    n_arrive = int(0.08 * n_bc)
    n_peak = int(0.04 * n_bc)

    anchor_values = [
        t_bottom_arrive_star,
        t_bottom_arrive_star + 0.50 * T_pulse_star,
        t_bottom_arrive_star + T_pulse_star,
    ]
    t_anchor = torch.tensor(
        anchor_values, device=device, dtype=torch.float32
    ).view(-1, 1)
    t_anchor = t_anchor[
        (t_anchor[:, 0] >= 0.0) & (t_anchor[:, 0] <= t_star_max)
    ].view(-1, 1)

    n_uniform = max(n_bc - n_arrive - n_peak - t_anchor.shape[0], 0)

    t_uniform = torch.rand(n_uniform, 1, device=device) * t_star_max
    t_arrive = _rand_window(
        t_bottom_arrive_star - 0.08,
        t_bottom_arrive_star + T_pulse_star + 0.08,
        n_arrive,
    )
    t_peak = _rand_window(
        t_bottom_arrive_star + 0.50 * T_pulse_star - 0.05,
        t_bottom_arrive_star + 0.50 * T_pulse_star + 0.05,
        n_peak,
    )

    return _shuffle_cat(t_uniform, t_arrive, t_peak, t_anchor)


def sample_points(
    n_pde=10000,
    n_ic=1600,
    n_bc=1500,
    fixed_ic=None,
):

    x_pde = sample_uniform_x(n_pde)
    t_pde = sample_time_with_windows(
        n_pde,
        bottom_return_ratio=0.06,
        pulse_ratio=0.04,
        early_ratio=0.04,
    )

    x_ic = sample_ic_x(n_ic) if fixed_ic is None else fixed_ic
    t_top = sample_top_time_mild_discrete(n_bc)
    t_bot = sample_bottom_time_mild(n_bc)

    return {
        "pde": (x_pde, t_pde),
        "ic": x_ic,
        "bc_top_t": t_top,
        "bc_bot_t": t_bot,
    }


def make_fixed_batch(n_pde=10000, n_ic=1600, n_bc=1500):

    fixed_ic = sample_ic_x(n_ic)
    return sample_points(
        n_pde=n_pde,
        n_ic=n_ic,
        n_bc=n_bc,
        fixed_ic=fixed_ic,
    )

# =========================
# 9) Training
# =========================
def weights_for_epoch(ep):

    if ep < 10000:
        w_ic_v = 150.0
    elif ep < 25000:
        w_ic_v = 120.0
    else:
        w_ic_v = 100.0

    return {
        "pde": 50.0,
        "ic_u": 100.0,
        "ic_v": w_ic_v,
        "bc_top": 100.0,
        "bc_bot": 100.0,
    }


def set_piecewise_lr(opt, ep):
    if ep < 10000:
        lr_now = 2.0e-3
    elif ep < 20000:
        lr_now = 1.0e-3
    elif ep < 30000:
        lr_now = 2.0e-4
    else:
        lr_now = 5.0e-5

    for pg in opt.param_groups:
        pg["lr"] = lr_now
    return lr_now


def _record_validation_history(model, epoch, val_info):
    h = model.history
    h["val_epoch"].append(int(epoch))
    for key in ("total", "pde", "ic_u", "ic_v", "bc_top", "bc_bot"):
        h[f"val_{key}"].append(float(val_info[key]))


def evaluate_on_batch(model, batch, weights):
    was_training = model.training
    model.eval()
    values = model.loss(batch, weights=weights, record=False)
    names = ("total", "pde", "ic_u", "ic_v", "bc_top", "bc_bot")
    info = {
        name: float(value.detach().cpu())
        for name, value in zip(names, values)
    }
    if was_training:
        model.train()
    return info


def train(model, adam_epochs=40000, lr=2e-3, lbfgs_max_iter=800, val_interval=500):
    model.train()
    print(
        f"==> Adam训练阶段 ({adam_epochs}轮次，全程固定采样，完整桩，含独立验证集) ..."
    )
    print("    损失项: PDE、IC位移、IC速度、桩顶BC、桩底BC")
    print("    已删除桩顶初始速度锚点及全部局部专项损失")
    print("    采样: 全域PDE为主 + 关键时段温和加密，不使用特征带专项采样")
    print("    IC: 在原有IC损失内增加 x*=0 和近桩顶点，不新增损失")
    print("    桩顶BC: CSV全部离散时刻 + 温和随机采样 -> 应力边界残差")
    print("    学习率: 2e-3 -> 1e-3 -> 2e-4 -> 5e-5")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 恢复原来较稳定的温和采样规模。
    train_batch = make_fixed_batch(n_pde=10000, n_ic=1600, n_bc=1500)

    # 验证集独立生成，采样原则与训练集一致。
    val_batch = make_fixed_batch(n_pde=4000, n_ic=800, n_bc=800)

    t1 = default_timer()
    for ep in range(adam_epochs):
        lr_now = set_piecewise_lr(opt, ep)
        weights = weights_for_epoch(ep)

        opt.zero_grad()
        values = model.loss(
            train_batch,
            weights=weights,
            record=True,
            lr_value=lr_now,
        )
        total_loss = values[0]
        total_loss.backward()
        opt.step()

        if ep % val_interval == 0:
            val_info = evaluate_on_batch(model, val_batch, weights)
            _record_validation_history(model, ep, val_info)

            (
                _,
                loss_pde,
                loss_ic_u,
                loss_ic_v,
                loss_bc_top,
                loss_bc_bot,
            ) = values

            print(
                f"[Adam-fixed] 轮次={ep:5d} "
                f"训练总损失={total_loss.item():.3e} | "
                f"验证总损失={val_info['total']:.3e}"
            )
            print(
                "       Train: "
                f"PDE={loss_pde.item():.3e} "
                f"IC位移={loss_ic_u.item():.3e} "
                f"IC速度={loss_ic_v.item():.3e} "
                f"桩顶={loss_bc_top.item():.3e} "
                f"桩底={loss_bc_bot.item():.3e}"
            )
            print(
                "       Val  : "
                f"PDE={val_info['pde']:.3e} "
                f"IC位移={val_info['ic_u']:.3e} "
                f"IC速度={val_info['ic_v']:.3e} "
                f"桩顶={val_info['bc_top']:.3e} "
                f"桩底={val_info['bc_bot']:.3e}"
            )
            print(
                f"       权重: PDE={weights['pde']:.1f} "
                f"ICu={weights['ic_u']:.1f} "
                f"ICv={weights['ic_v']:.1f} "
                f"topBC={weights['bc_top']:.1f} "
                f"bottomBC={weights['bc_bot']:.1f} "
                f"lr={lr_now:.3e}"
            )
            print("-" * 72)

    print("==> L-BFGS固定批次微调阶段 ...")
    weights_lbfgs = {
        "pde": 50.0,
        "ic_u": 100.0,
        "ic_v": 100.0,
        "bc_top": 100.0,
        "bc_bot": 100.0,
    }

    opt2 = torch.optim.LBFGS(
        model.parameters(),
        lr=0.3,
        max_iter=lbfgs_max_iter,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt2.zero_grad()
        total_loss, *_ = model.loss(
            train_batch, weights=weights_lbfgs, record=False
        )
        total_loss.backward()
        return total_loss

    opt2.step(closure)

    model.loss(train_batch, weights=weights_lbfgs, record=True, lr_value=0.0)
    val_info = evaluate_on_batch(model, val_batch, weights_lbfgs)
    _record_validation_history(model, adam_epochs + 1, val_info)
    print(
        f"[L-BFGS-end] 验证总损失={val_info['total']:.3e}, "
        f"PDE={val_info['pde']:.3e}, "
        f"IC位移={val_info['ic_u']:.3e}, "
        f"IC速度={val_info['ic_v']:.3e}, "
        f"桩顶={val_info['bc_top']:.3e}, "
        f"桩底={val_info['bc_bot']:.3e}"
    )

    t2 = default_timer()
    print(f"训练完成. 总耗时 = {t2 - t1:.2f} 秒")

# =========================
# 10) ABAQUS RPT parser and comparison plot
# =========================
def _parse_abaqus_float(token):
    token = token.strip().replace("D", "E").replace("d", "E")
    return float(token)


def read_abaqus_rpt_segments(rpt_path):
    if not os.path.exists(rpt_path):
        print(f"警告：未找到 ABAQUS 文件: {rpt_path}")
        return []

    float_pattern = re.compile(
        r"^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
        r"[\s,]+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
        r"[\s]*$"
    )

    segments = []
    cur_t = []
    cur_v = []
    last_t = None

    with open(rpt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = float_pattern.match(line)
            if m is None:
                if len(cur_t) >= 2:
                    segments.append((np.asarray(cur_t, dtype=float), np.asarray(cur_v, dtype=float)))
                cur_t, cur_v = [], []
                last_t = None
                continue

            t_val = _parse_abaqus_float(m.group(1))
            v_val = _parse_abaqus_float(m.group(2))

            # 如果时间回跳，说明进入了新数据块，必须切断，避免首尾相连。
            if last_t is not None and t_val < last_t - 1e-12:
                if len(cur_t) >= 2:
                    segments.append((np.asarray(cur_t, dtype=float), np.asarray(cur_v, dtype=float)))
                cur_t, cur_v = [], []

            cur_t.append(t_val)
            cur_v.append(v_val)
            last_t = t_val

    if len(cur_t) >= 2:
        segments.append((np.asarray(cur_t, dtype=float), np.asarray(cur_v, dtype=float)))


    clean_segments = []
    for t, v in segments:
        mask = np.isfinite(t) & np.isfinite(v)
        t = t[mask]
        v = v[mask]
        if len(t) < 2:
            continue


        keep = np.ones_like(t, dtype=bool)
        keep[1:] = np.abs(np.diff(t)) > 1e-15
        t = t[keep]
        v = v[keep]
        if len(t) >= 2:
            clean_segments.append((t, v))

    print(f"已读取 ABAQUS 数据段数量: {len(clean_segments)}")
    return clean_segments


def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def resolve_abaqus_path(path):
    if path is None or str(path).strip() == "":
        return None

    if os.path.isabs(path) and os.path.exists(path):
        return path

    candidates = [
        path,
        os.path.join(get_script_dir(), path),
        os.path.join(os.getcwd(), path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return path

# =========================
# 11) Visualization
# =========================
def plot_losses(model):
    h = model.history

    plt.figure(figsize=(11, 6))
    plt.semilogy(h["total"], label="Train total", linewidth=2)
    plt.semilogy(h["pde"], label="Train PDE", alpha=0.8)
    plt.semilogy(h["ic_u"], label="Train IC displacement", alpha=0.8)
    plt.semilogy(h["ic_v"], label="Train IC velocity", alpha=0.8)
    plt.semilogy(h["bc_top"], label="Train top BC", alpha=0.8)
    plt.semilogy(h["bc_bot"], label="Train bottom BC", alpha=0.8)

    if len(h["val_epoch"]) > 0:
        plt.semilogy(
            h["val_epoch"],
            h["val_total"],
            "--",
            label="Validation total",
            linewidth=2,
        )
        plt.semilogy(
            h["val_epoch"],
            h["val_pde"],
            "--",
            label="Validation PDE",
            alpha=0.9,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss, log scale")
    plt.title("Training and validation loss history - intact pile")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "loss_history_train_val.png"), dpi=300)
    plt.show()

    if len(h["val_epoch"]) > 0:
        plt.figure(figsize=(11, 5.5))
        for key, label in (
            ("val_total", "Validation total"),
            ("val_pde", "Validation PDE"),
            ("val_ic_u", "Validation IC displacement"),
            ("val_ic_v", "Validation IC velocity"),
            ("val_bc_top", "Validation top BC"),
            ("val_bc_bot", "Validation bottom BC"),
        ):
            plt.semilogy(h["val_epoch"], h[key], label=label, alpha=0.85)

        plt.xlabel("Epoch")
        plt.ylabel("Validation loss, log scale")
        plt.title("Validation loss history - intact pile")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULT_DIR, "validation_loss_history.png"), dpi=300
        )
        plt.show()


@torch.no_grad()
def predict_grid(model, nx=200, nt=300):
    x = torch.linspace(0, 1, nx, device=device).view(-1, 1)
    t = torch.linspace(0, t_star_max, nt, device=device).view(-1, 1)
    X, T = torch.meshgrid(x.squeeze(1), t.squeeze(1), indexing="ij")

    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    u_star = model(xf, tf).reshape(nx, nt)
    return X.cpu().numpy(), T.cpu().numpy(), u_star.cpu().numpy()


def plot_displacement_contour(X, T, u_star):
    u = U_ref * u_star * 1000.0
    x_dim = X * L
    t_dim = T * t_ref

    plt.figure(figsize=(12, 6))
    levels = np.linspace(np.min(u), np.max(u), 100)
    cs = plt.contourf(t_dim * 1e3, x_dim, u, levels=levels, cmap="RdBu_r")
    plt.colorbar(cs, label="Displacement u / mm")

    plt.xlabel("Time t / ms")
    plt.ylabel("Pile position x / m")
    plt.title("Displacement contour of intact pile")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "displacement_contour.png"), dpi=300)
    plt.show()


def plot_area_profile():
    x_plot = np.linspace(0, 1, 500)
    x_dim = x_plot * L
    A_normal = np.pi * (D_normal / 2.0) ** 2
    A_values = A_normal * np.ones_like(x_plot)

    plt.figure(figsize=(10, 4))
    plt.plot(x_dim, A_values, linewidth=2)
    plt.fill_between(x_dim, 0, A_values, alpha=0.3)
    plt.xlabel("Pile position x / m")
    plt.ylabel("Cross-sectional area A(x) / m²")
    plt.title("Cross-sectional area distribution of intact pile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "area_profile.png"), dpi=300)
    plt.show()


def top_velocity_timehistory(model, nt=1600):
    model.eval()
    t_star = torch.linspace(0, t_star_max, nt, device=device).view(-1, 1)

    x0 = torch.zeros_like(t_star, device=device).requires_grad_(True)
    t_req = t_star.clone().detach().requires_grad_(True)

    u_star = model(x0, t_req)
    u_t_star = grad(u_star, t_req, torch.ones_like(u_star), create_graph=False)[0]

    v_top_m_s = (U_ref / t_ref) * u_t_star.detach().cpu().numpy().flatten()
    t_ms = (t_ref * t_star.detach().cpu().numpy().flatten()) * 1e3
    return t_ms, v_top_m_s


def plot_top_velocity(model):
    t_ms, v_top_m_s = top_velocity_timehistory(model, nt=1600)
    v_top_mm_s = v_top_m_s * 1000.0

    plt.figure(figsize=(12, 5))
    plt.plot(t_ms, v_top_mm_s, linewidth=1.8, label="PINN")
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile-head velocity v_top / mm/s")
    plt.title("Pile-head velocity response of intact pile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "top_velocity.png"), dpi=300)
    plt.show()


def plot_pinn_abaqus_comparison(model, abaqus_rpt_path=ABAQUS_RPT_PATH, nt=1600):

    t_pinn_ms, v_pinn_m_s = top_velocity_timehistory(model, nt=nt)
    v_pinn_mm_s = v_pinn_m_s * 1000.0

    abaqus_path = resolve_abaqus_path(abaqus_rpt_path)
    abaqus_segments = read_abaqus_rpt_segments(abaqus_path)

    plt.figure(figsize=(14, 5.5))
    plt.plot(t_pinn_ms, v_pinn_mm_s, linewidth=2.0, label="PINN")

    if len(abaqus_segments) > 0:
        first_label = True
        for t_s, v_m_s in abaqus_segments:
            t_ms = t_s * 1000.0
            v_mm_s = v_m_s * 1000.0
            plt.plot(
                t_ms,
                v_mm_s,
                linestyle="--",
                linewidth=2.0,
                label="ABAQUS 1D" if first_label else None,
            )
            first_label = False
    else:
        print("未绘制 ABAQUS 曲线：没有读取到有效 ABAQUS 数据。")

    plt.xlabel("Time t / ms", fontsize=12)
    plt.ylabel(r"Pile-head velocity $v_{top}$ / mm/s", fontsize=12)
    plt.title("Comparison of pile-head velocity response between PINN and ABAQUS 1D", fontsize=14)
    plt.xlim(0.0, T_obs * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(RESULT_DIR, "comparison_pinn_abaqus_1d.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"PINN-ABAQUS 对比图已保存: {save_path}")


def save_top_velocity_csv(model, nt=1600, filename="pinn_intact_top_velocity.csv"):
    t_ms, v_top_m_s = top_velocity_timehistory(model, nt=nt)
    t_s = t_ms / 1000.0
    v_top_mm_s = v_top_m_s * 1000.0

    data = np.column_stack([t_s, t_ms, v_top_m_s, v_top_mm_s])
    save_path = os.path.join(RESULT_DIR, filename)
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header="time_s,time_ms,v_top_m_s,v_top_mm_s",
        comments="",
    )

    print(f"PINN桩顶速度时程数据已保存: {save_path}")
    print("CSV列: time_s, time_ms, v_top_m_s, v_top_mm_s")
    return save_path


def save_abaqus_csv(rpt_path=ABAQUS_RPT_PATH, filename="abaqus_1d_top_velocity_parsed.csv"):

    abaqus_path = resolve_abaqus_path(rpt_path)
    segments = read_abaqus_rpt_segments(abaqus_path)
    if len(segments) == 0:
        print("未保存 ABAQUS CSV：没有读取到有效数据。")
        return None

    rows = []
    for seg_id, (t_s, v_m_s) in enumerate(segments, start=1):
        t_ms = t_s * 1000.0
        v_mm_s = v_m_s * 1000.0
        seg_col = np.full_like(t_s, seg_id, dtype=float)
        rows.append(np.column_stack([seg_col, t_s, t_ms, v_m_s, v_mm_s]))
    data = np.vstack(rows)

    save_path = os.path.join(RESULT_DIR, filename)
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header="segment,time_s,time_ms,v_m_s,v_mm_s",
        comments="",
    )
    print(f"ABAQUS解析数据已保存: {save_path}")
    return save_path


def check_initial_velocity(model, nx=200):
    model.eval()
    x = torch.linspace(0, 1, nx, device=device).view(-1, 1).requires_grad_(True)
    t0 = torch.zeros_like(x, device=device).requires_grad_(True)
    u = model(x, t0)
    u_t = grad(u, t0, torch.ones_like(u), create_graph=False)[0]
    max_v_star = torch.max(torch.abs(u_t)).detach().cpu().item()
    top_v_star = u_t[0].detach().cpu().item()

    max_v_dim = (U_ref / t_ref) * max_v_star * 1000.0
    top_v_dim = (U_ref / t_ref) * top_v_star * 1000.0
    print(f"初始速度检查: max|u*_t(x,0)| = {max_v_star:.3e}, u*_t(0,0) = {top_v_star:.3e}")
    print(f"             max|v(x,0)| = {max_v_dim:.3e} mm/s, v(0,0) = {top_v_dim:.3e} mm/s")

# =========================
# 12) Main execution
# =========================
if __name__ == "__main__":
    print("====================================")
    print("PINN低应变完整性检测 - 完整桩模拟")
    print("设备:", device)
    print("====================================")

    model = PINN_Intact(hidden_dim=64, num_hidden=4, activation="tanh").to(device)


    save_excitation_check_plot()
    plot_area_profile()
    train(model, adam_epochs=40000, lr=2e-3, lbfgs_max_iter=800, val_interval=500)

    check_initial_velocity(model)
    plot_losses(model)

    X, T, u_star = predict_grid(model, nx=200, nt=300)
    plot_displacement_contour(X, T, u_star)
    plot_top_velocity(model)

    # 保存 PINN 桩顶速度 CSV
    save_top_velocity_csv(model, nt=1600, filename="pinn_intact_top_velocity.csv")

    # 可选：保存解析后的 ABAQUS CSV，方便检查 ABAQUS 数据是否被正确读取
    save_abaqus_csv(ABAQUS_RPT_PATH, filename="abaqus_1d_top_velocity_parsed.csv")

    # 绘制 PINN vs ABAQUS 1D 对比图：只保留两条曲线，不画理论峰值线
    plot_pinn_abaqus_comparison(model, abaqus_rpt_path=ABAQUS_RPT_PATH, nt=1600)

    # 保存训练后的模型权重
    model_path = os.path.join(RESULT_DIR, "pinn_intact_discrete_excitation_mild_sampling_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存: {model_path}")

    print(f"\n模拟完成！结果保存在 {RESULT_DIR}/ 文件夹中")
