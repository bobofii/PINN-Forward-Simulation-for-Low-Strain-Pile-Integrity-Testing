# -*- coding: utf-8 -*-
"""
PINN Forward Simulation for Low-Strain Integrity Testing (LIT)
Upper-pile free-end broken pile:
frozen intact-pile background field + one-domain scattering correction
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import re
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt

# =========================
# 0) Global settings
# =========================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

RESULT_DIR = "Results_BrokenPile_FreeEnd_BackgroundScattering_NoClip"
os.makedirs(RESULT_DIR, exist_ok=True)

# 是否在找不到完整桩权重时自动训练背景模型
TRAIN_BACKGROUND_IF_MISSING = True
FORCE_RETRAIN_BACKGROUND = False

# 完整桩背景训练配置（仅在找不到权重时使用）
BACKGROUND_ADAM_EPOCHS = 40000
BACKGROUND_LBFGS_MAX_ITER = 800
BACKGROUND_VAL_INTERVAL = 500

# 断桩散射场训练配置
SCATTER_ADAM_EPOCHS = 55000
SCATTER_LBFGS_MAX_ITER = 500
SCATTER_VAL_INTERVAL = 500

# =========================
# 1) Physical parameters
# =========================
# L 仍取原始完整桩长 13 m，以保持与既有完整桩背景权重的无量纲尺度一致。
L_ORIGINAL = 13.0
L = L_ORIGINAL
X_BREAK = 6.20
L_ACTUAL = X_BREAK

D_NORMAL = 0.50
RHO0 = 2400.0
E0 = 32500e6

T_OBS = 10e-3
T_PULSE = 1e-3
P0 = 750.0

A_NORMAL = np.pi * (D_NORMAL / 2.0) ** 2
F_MAX = P0 * A_NORMAL

C0 = np.sqrt(E0 / RHO0)
X_REF = L
T_REF = L / C0
U_REF = P0 * L / E0

T_STAR_MAX = T_OBS / T_REF
T_PULSE_STAR = T_PULSE / T_REF
X_BREAK_STAR = X_BREAK / L

# 完整桩背景模型的原始桩底关键时刻，供背景模型训练采样使用。
T_BOTTOM_ARRIVE_STAR = 1.0
T_BOTTOM_REFLECT_STAR = 2.0

# 断桩关键传播时刻。
T_BREAK_FIRST_ARRIVE_STAR = X_BREAK_STAR
T_BREAK_FIRST_REFLECT_TOP_STAR = 2.0 * X_BREAK_STAR
T_BREAK_SECOND_ARRIVE_STAR = 3.0 * X_BREAK_STAR
T_BREAK_SECOND_REFLECT_TOP_STAR = 4.0 * X_BREAK_STAR
T_BREAK_THIRD_ARRIVE_STAR = 5.0 * X_BREAK_STAR
T_BREAK_THIRD_REFLECT_TOP_STAR = 6.0 * X_BREAK_STAR

# 完整桩背景的桩底反射重新经过断面位置的时刻。
# 散射场需要在断端自由边界中抵消这部分不属于断桩工况的背景响应。
T_BACKGROUND_BOTTOM_PASS_BREAK_STAR = 2.0 - X_BREAK_STAR

T_BREAK_FIRST_ARRIVE = T_BREAK_FIRST_ARRIVE_STAR * T_REF
T_BREAK_FIRST_REFLECT_TOP = T_BREAK_FIRST_REFLECT_TOP_STAR * T_REF
T_BREAK_SECOND_REFLECT_TOP = T_BREAK_SECOND_REFLECT_TOP_STAR * T_REF

print("=" * 78)
print("上部桩自由端断桩：完整桩背景场 + 单域散射场 PINN")
print(f"设备: {device}")
print(f"原始桩长: {L_ORIGINAL:.3f} m")
print(f"实际 PINN/ABAQUS 计算长度: {L_ACTUAL:.3f} m")
print(f"断桩位置: x = {X_BREAK:.3f} m, x* = {X_BREAK_STAR:.6f}")
print(f"桩径: {D_NORMAL:.3f} m")
print(f"截面面积: {A_NORMAL:.7f} m^2")
print(f"弹性模量: {E0 / 1e6:.1f} MPa")
print(f"密度: {RHO0:.1f} kg/m^3")
print(f"波速: {C0:.3f} m/s")
print(f"荷载峰值应力: {P0:.3f} N/m^2")
print(f"ABAQUS 等效集中力峰值: {F_MAX:.3f} N")
print(f"第一断端反射到达桩顶: {T_BREAK_FIRST_REFLECT_TOP * 1e3:.3f} ms")
print(f"第二断端反射到达桩顶: {T_BREAK_SECOND_REFLECT_TOP * 1e3:.3f} ms")
print(f"无量纲观察时间: {T_STAR_MAX:.6f}")
print("断端边界: 总场自由端 u_x(x_b,t)=0")
print("梯度裁剪: 已完全取消")
print("=" * 78)

# =========================
# 2) Paths and helpers
# =========================
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


# =========================
# 3) Discrete excitation
# =========================
EXCITATION_CSV_PATH = "half_sine_discrete_excitation.csv"
AUTO_CREATE_EXCITATION_CSV = True
EXCITATION_SAMPLE_COUNT = 101


def create_example_excitation_csv(csv_path, n_points=EXCITATION_SAMPLE_COUNT):
    if n_points < 3:
        raise ValueError("EXCITATION_SAMPLE_COUNT 至少为 3。")
    time_s = np.linspace(0.0, T_PULSE, n_points, dtype=np.float64)
    stress_pa = P0 * np.sin(np.pi * time_s / T_PULSE)
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
    print(f"已自动生成离散激励文件: {csv_path}")


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
            raise ValueError("激励 CSV 至少需要两列：time_s, stress_pa。")
        raw = raw[np.all(np.isfinite(raw[:, :2]), axis=1)]
        time_s = raw[:, 0].astype(np.float64)
        stress_pa = raw[:, 1].astype(np.float64)

    valid = np.isfinite(time_s) & np.isfinite(stress_pa)
    time_s = time_s[valid]
    stress_pa = stress_pa[valid]
    if time_s.size < 2:
        raise ValueError("离散激励有效数据点少于 2。")
    if np.any(time_s < 0.0):
        raise ValueError("离散激励时间不能为负数。")

    order = np.argsort(time_s)
    time_s = time_s[order]
    stress_pa = stress_pa[order]

    unique_t, inverse = np.unique(time_s, return_inverse=True)
    if unique_t.size != time_s.size:
        sums = np.zeros_like(unique_t, dtype=np.float64)
        counts = np.zeros_like(unique_t, dtype=np.float64)
        np.add.at(sums, inverse, stress_pa)
        np.add.at(counts, inverse, 1.0)
        time_s = unique_t
        stress_pa = sums / counts

    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("离散激励时间点必须严格递增。")
    return time_s, stress_pa


def load_excitation_data():
    csv_path = resolve_local_path(EXCITATION_CSV_PATH)
    if not os.path.exists(csv_path):
        if AUTO_CREATE_EXCITATION_CSV:
            create_example_excitation_csv(csv_path)
        else:
            raise FileNotFoundError(f"未找到离散激励文件: {csv_path}")

    time_s, stress_pa = read_excitation_csv(csv_path)
    t_tensor = torch.tensor(time_s / T_REF, dtype=torch.float32, device=device).contiguous()
    p_tensor = torch.tensor(stress_pa / P0, dtype=torch.float32, device=device).contiguous()

    print("=== 离散激励 ===")
    print(f"文件: {csv_path}")
    print(f"点数: {time_s.size}")
    print(f"时间范围: {time_s[0] * 1e3:.6f} ~ {time_s[-1] * 1e3:.6f} ms")
    print(f"峰值应力: {np.max(np.abs(stress_pa)):.6f} Pa")
    print("插值方式: 分段线性")
    print("================")
    return csv_path, time_s, stress_pa, t_tensor, p_tensor


(
    EXCITATION_CSV_RESOLVED,
    EXCITATION_TIME_S,
    EXCITATION_STRESS_PA,
    EXCITATION_TIME_STAR_TENSOR,
    EXCITATION_STRESS_STAR_TENSOR,
) = load_excitation_data()


def p_star(t_star):
    original_shape = t_star.shape
    t_flat = t_star.reshape(-1).contiguous()
    t_data = EXCITATION_TIME_STAR_TENSOR
    p_data = EXCITATION_STRESS_STAR_TENSOR

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


def excitation_time_tensor():
    t = torch.tensor(
        EXCITATION_TIME_S / T_REF,
        dtype=torch.float32,
        device=device,
    ).view(-1, 1)
    mask = (t[:, 0] >= 0.0) & (t[:, 0] <= T_STAR_MAX)
    return torch.unique(t[mask].squeeze(1), sorted=True).view(-1, 1)


def save_excitation_plot():
    t_dense_s = np.linspace(0.0, T_OBS, 2400)
    t_dense_star = torch.tensor(
        t_dense_s / T_REF, dtype=torch.float32, device=device
    ).view(-1, 1)
    with torch.no_grad():
        p_dense = P0 * p_star(t_dense_star).cpu().numpy().reshape(-1)

    plt.figure(figsize=(11, 4.5))
    plt.plot(t_dense_s * 1e3, p_dense, linewidth=1.8, label="Linear interpolation")
    plt.scatter(
        EXCITATION_TIME_S * 1e3,
        EXCITATION_STRESS_PA,
        s=17,
        zorder=3,
        label="Discrete excitation data",
    )
    plt.xlabel("Time t / ms")
    plt.ylabel("Excitation stress p(t) / Pa")
    plt.title("Discrete pile-head excitation")
    plt.xlim(0.0, T_OBS * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "discrete_excitation_input.png"), dpi=300)
    plt.close()


# =========================
# 4) Common sampling helpers
# =========================
def _shuffle_cat(*parts):
    parts = [p for p in parts if p is not None and p.numel() > 0]
    if not parts:
        return torch.empty(0, 1, device=device)
    x = torch.cat(parts, dim=0)
    perm = torch.randperm(x.shape[0], device=device)
    return x[perm]


def _shuffle_pair(x, t):
    if x.shape[0] != t.shape[0]:
        raise ValueError("x 与 t 点数不一致。")
    perm = torch.randperm(x.shape[0], device=device)
    return x[perm], t[perm]


def _concat_pairs(*pairs):
    pairs = [pair for pair in pairs if pair is not None and pair[0].numel() > 0]
    if not pairs:
        return torch.empty(0, 1, device=device), torch.empty(0, 1, device=device)
    x = torch.cat([p[0] for p in pairs], dim=0)
    t = torch.cat([p[1] for p in pairs], dim=0)
    return _shuffle_pair(x, t)


def rand_window(t0, t1, n):
    if n <= 0:
        return torch.empty(0, 1, device=device)
    t0 = max(0.0, float(t0))
    t1 = min(float(t1), float(T_STAR_MAX))
    if t1 <= t0:
        return torch.full((n, 1), t0, device=device)
    return t0 + (t1 - t0) * torch.rand(n, 1, device=device)


def sample_x_interval(n, x0, x1, margin=1.0e-6):
    if n <= 0:
        return torch.empty(0, 1, device=device)
    lo = float(x0) + margin
    hi = float(x1) - margin
    if hi <= lo:
        raise ValueError("采样区间过窄。")
    return lo + (hi - lo) * torch.rand(n, 1, device=device)


def sample_x_near_boundary(n, boundary, side, width):
    if n <= 0:
        return torch.empty(0, 1, device=device)
    if side == "left":
        lo = max(0.0, boundary - width)
        hi = boundary
    elif side == "right":
        lo = boundary
        hi = min(1.0, boundary + width)
    else:
        raise ValueError("side 必须为 'left' 或 'right'。")
    return sample_x_interval(n, lo, hi)


def sample_time_around(centers, n_total, half_width):
    if n_total <= 0:
        return torch.empty(0, 1, device=device)
    centers = [float(c) for c in centers if 0.0 <= float(c) <= T_STAR_MAX]
    if not centers:
        return torch.rand(n_total, 1, device=device) * T_STAR_MAX
    base = n_total // len(centers)
    counts = [base] * len(centers)
    counts[-1] += n_total - base * len(centers)
    parts = []
    for center, n in zip(centers, counts):
        parts.append(rand_window(center - half_width, center + half_width, n))
    return _shuffle_cat(*parts)


# =========================
# 5) Intact background network
# =========================
class IntactPINN(nn.Module):

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
            "train_epoch": [],
            "total": [],
            "pde": [],
            "ic_u": [],
            "ic_v": [],
            "bc_top": [],
            "bc_bot": [],
            "val_epoch": [],
            "val_total": [],
            "val_pde": [],
            "val_ic_u": [],
            "val_ic_v": [],
            "val_bc_top": [],
            "val_bc_bot": [],
        }

    def forward(self, x_star, t_star):
        h = torch.cat([x_star, t_star], dim=1)
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)

    def pde_residual(self, x_star, t_star):
        x = x_star.clone().detach().requires_grad_(True)
        t = t_star.clone().detach().requires_grad_(True)
        u = self.forward(x, t)
        u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_tt = grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
        return u_tt - u_xx

    def ic_residuals(self, x_star):
        x = x_star.clone().detach().requires_grad_(True)
        t0 = torch.zeros_like(x).requires_grad_(True)
        u = self.forward(x, t0)
        u_t = grad(u, t0, torch.ones_like(u), create_graph=True)[0]
        return u, u_t

    def bc_top_residual(self, t_star):
        t = t_star.clone().detach().requires_grad_(True)
        x0 = torch.zeros_like(t).requires_grad_(True)
        u = self.forward(x0, t)
        u_x = grad(u, x0, torch.ones_like(u), create_graph=True)[0]
        return u_x + p_star(t)

    def bc_bot_residual(self, t_star):
        t = t_star.clone().detach().requires_grad_(True)
        x1 = torch.ones_like(t).requires_grad_(True)
        u = self.forward(x1, t)
        u_x = grad(u, x1, torch.ones_like(u), create_graph=True)[0]
        return u_x

    def loss(self, batch, weights, record=False, epoch=None):
        r_pde = self.pde_residual(*batch["pde"])
        u_ic, v_ic = self.ic_residuals(batch["ic"])
        r_top = self.bc_top_residual(batch["bc_top_t"])
        r_bot = self.bc_bot_residual(batch["bc_bot_t"])

        l_pde = torch.mean(r_pde ** 2)
        l_ic_u = torch.mean(u_ic ** 2)
        l_ic_v = torch.mean(v_ic ** 2)
        l_top = torch.mean(r_top ** 2)
        l_bot = torch.mean(r_bot ** 2)

        total = (
            weights["pde"] * l_pde
            + weights["ic_u"] * l_ic_u
            + weights["ic_v"] * l_ic_v
            + weights["bc_top"] * l_top
            + weights["bc_bot"] * l_bot
        )

        if record:
            self.history["train_epoch"].append(int(epoch))
            self.history["total"].append(float(total.detach().cpu()))
            self.history["pde"].append(float(l_pde.detach().cpu()))
            self.history["ic_u"].append(float(l_ic_u.detach().cpu()))
            self.history["ic_v"].append(float(l_ic_v.detach().cpu()))
            self.history["bc_top"].append(float(l_top.detach().cpu()))
            self.history["bc_bot"].append(float(l_bot.detach().cpu()))

        return {
            "total": total,
            "pde": l_pde,
            "ic_u": l_ic_u,
            "ic_v": l_ic_v,
            "bc_top": l_top,
            "bc_bot": l_bot,
        }


def sample_background_ic(n_ic):
    n_top = max(1, int(round(0.03125 * n_ic)))
    n_near = max(1, int(round(0.03125 * n_ic)))
    n_uniform = max(0, n_ic - n_top - n_near)
    x_top = torch.zeros(n_top, 1, device=device)
    x_near = 0.03 * torch.rand(n_near, 1, device=device)
    x_uniform = torch.rand(n_uniform, 1, device=device)
    anchors = torch.tensor([[0.25], [0.50], [0.75], [1.00]], device=device)
    n_anchor = min(anchors.shape[0], x_uniform.shape[0])
    if n_anchor > 0:
        x_uniform[:n_anchor] = anchors[:n_anchor]
    return _shuffle_cat(x_top, x_near, x_uniform)


def sample_background_pde(n_pde):
    x = torch.rand(n_pde, 1, device=device)
    n_early = int(0.04 * n_pde)
    n_pulse = int(0.04 * n_pde)
    n_bottom = int(0.06 * n_pde)
    n_uniform = n_pde - n_early - n_pulse - n_bottom
    t = _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        rand_window(0.0, 0.15 * T_PULSE_STAR, n_early),
        rand_window(0.0, 1.20 * T_PULSE_STAR, n_pulse),
        rand_window(
            T_BOTTOM_REFLECT_STAR - 0.12,
            T_BOTTOM_REFLECT_STAR + T_PULSE_STAR + 0.12,
            n_bottom,
        ),
    )
    return _shuffle_pair(x, t)


def sample_background_top(n_bc):
    t_csv = excitation_time_tensor()
    if t_csv.shape[0] > n_bc:
        idx = torch.linspace(0, t_csv.shape[0] - 1, n_bc, device=device).round().long()
        return t_csv[idx]

    n_pulse = int(0.06 * n_bc)
    n_early = int(0.03 * n_bc)
    n_ref = int(0.04 * n_bc)
    anchors = torch.tensor(
        [[T_BOTTOM_REFLECT_STAR], [T_BOTTOM_REFLECT_STAR + 0.5 * T_PULSE_STAR]],
        dtype=torch.float32,
        device=device,
    )
    anchors = anchors[(anchors[:, 0] >= 0.0) & (anchors[:, 0] <= T_STAR_MAX)].view(-1, 1)
    n_uniform = max(
        n_bc - t_csv.shape[0] - n_pulse - n_early - n_ref - anchors.shape[0], 0
    )
    t_all = _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        t_csv,
        rand_window(0.0, T_PULSE_STAR, n_pulse),
        rand_window(0.0, 0.15 * T_PULSE_STAR, n_early),
        rand_window(
            T_BOTTOM_REFLECT_STAR - 0.06,
            T_BOTTOM_REFLECT_STAR + T_PULSE_STAR + 0.06,
            n_ref,
        ),
        anchors,
    )
    if t_all.shape[0] < n_bc:
        t_all = _shuffle_cat(
            t_all,
            torch.rand(n_bc - t_all.shape[0], 1, device=device) * T_STAR_MAX,
        )
    return t_all[:n_bc]


def sample_background_bottom(n_bc):
    n_arrive = int(0.08 * n_bc)
    n_peak = int(0.04 * n_bc)
    anchors = torch.tensor(
        [
            [T_BOTTOM_ARRIVE_STAR],
            [T_BOTTOM_ARRIVE_STAR + 0.5 * T_PULSE_STAR],
            [T_BOTTOM_ARRIVE_STAR + T_PULSE_STAR],
        ],
        dtype=torch.float32,
        device=device,
    )
    n_uniform = max(n_bc - n_arrive - n_peak - anchors.shape[0], 0)
    return _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        rand_window(
            T_BOTTOM_ARRIVE_STAR - 0.08,
            T_BOTTOM_ARRIVE_STAR + T_PULSE_STAR + 0.08,
            n_arrive,
        ),
        rand_window(
            T_BOTTOM_ARRIVE_STAR + 0.5 * T_PULSE_STAR - 0.05,
            T_BOTTOM_ARRIVE_STAR + 0.5 * T_PULSE_STAR + 0.05,
            n_peak,
        ),
        anchors,
    )


def make_background_batch(n_pde, n_ic, n_bc):
    return {
        "pde": sample_background_pde(n_pde),
        "ic": sample_background_ic(n_ic),
        "bc_top_t": sample_background_top(n_bc),
        "bc_bot_t": sample_background_bottom(n_bc),
    }


def background_weights(epoch):
    if epoch < 10000:
        w_ic_v = 150.0
    elif epoch < 25000:
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


def set_background_lr(opt, epoch):
    if epoch < 10000:
        lr_now = 2.0e-3
    elif epoch < 20000:
        lr_now = 1.0e-3
    elif epoch < 30000:
        lr_now = 2.0e-4
    else:
        lr_now = 5.0e-5
    for group in opt.param_groups:
        group["lr"] = lr_now
    return lr_now


def evaluate_background(model, batch, weights):
    model.eval()
    values = model.loss(batch, weights, record=False)
    return {k: float(v.detach().cpu()) for k, v in values.items()}


def train_background_model(model):
    print("\n==> 未找到完整桩权重，开始训练完整桩背景模型 ...")
    train_batch = make_background_batch(10000, 1600, 1500)
    val_batch = make_background_batch(4000, 800, 800)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)

    best_val = float("inf")
    best_state = None
    start = default_timer()

    for epoch in range(BACKGROUND_ADAM_EPOCHS):
        model.train()
        set_background_lr(optimizer, epoch)
        weights = background_weights(epoch)

        optimizer.zero_grad()
        values = model.loss(train_batch, weights, record=True, epoch=epoch)
        values["total"].backward()
        optimizer.step()

        if epoch % BACKGROUND_VAL_INTERVAL == 0 or epoch == BACKGROUND_ADAM_EPOCHS - 1:
            val = evaluate_background(model, val_batch, weights)
            model.history["val_epoch"].append(epoch)
            for key in ("total", "pde", "ic_u", "ic_v", "bc_top", "bc_bot"):
                model.history[f"val_{key}"].append(val[key])

            if val["total"] < best_val:
                best_val = val["total"]
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"[背景 Adam] epoch={epoch:5d} "
                f"train={values['total'].item():.3e} val={val['total']:.3e} "
                f"PDE={val['pde']:.3e} top={val['bc_top']:.3e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    before_state = copy.deepcopy(model.state_dict())
    before_val = evaluate_background(model, val_batch, background_weights(BACKGROUND_ADAM_EPOCHS))

    print("==> 完整桩背景 L-BFGS 微调 ...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.3,
        max_iter=BACKGROUND_LBFGS_MAX_ITER,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    weights_final = background_weights(BACKGROUND_ADAM_EPOCHS)

    def closure():
        optimizer_lbfgs.zero_grad()
        values = model.loss(train_batch, weights_final, record=False)
        values["total"].backward()
        return values["total"]

    optimizer_lbfgs.step(closure)
    after_val = evaluate_background(model, val_batch, weights_final)

    if after_val["total"] > before_val["total"]:
        print("背景 L-BFGS 验证损失变差，恢复 Adam 最优模型。")
        model.load_state_dict(before_state)
    else:
        print("背景 L-BFGS 被接受。")

    elapsed = default_timer() - start
    print(f"完整桩背景训练完成，用时 {elapsed:.2f} s")
    return model


def background_model_candidates():
    name = "pinn_intact_discrete_excitation_mild_sampling_model.pth"
    base = get_base_dir()
    cwd = os.getcwd()
    return [
        os.path.join(base, name),
        os.path.join(cwd, name),
        os.path.join(base, "Results_Intact_DiscreteExcitation_MildSampling", name),
        os.path.join(cwd, "Results_Intact_DiscreteExcitation_MildSampling", name),
        os.path.join("/mnt/data", name),
        os.path.join(
            "/mnt/data", "Results_Intact_DiscreteExcitation_MildSampling", name
        ),
    ]


def load_or_train_background():
    model = IntactPINN(hidden_dim=64, num_hidden=4, activation="tanh").to(device)
    loaded_path = None

    if not FORCE_RETRAIN_BACKGROUND:
        for candidate in background_model_candidates():
            if os.path.exists(candidate):
                try:
                    state = torch.load(candidate, map_location=device)
                    if isinstance(state, dict) and "model_state_dict" in state:
                        state = state["model_state_dict"]
                    model.load_state_dict(state, strict=True)
                    loaded_path = candidate
                    break
                except Exception as exc:
                    print(f"无法加载背景权重 {candidate}: {exc}")

    if loaded_path is not None:
        print(f"已加载完整桩背景模型: {loaded_path}")
    else:
        if not TRAIN_BACKGROUND_IF_MISSING:
            raise FileNotFoundError(
                "未找到完整桩背景模型。请提供 "
                "pinn_intact_discrete_excitation_mild_sampling_model.pth，"
                "或将 TRAIN_BACKGROUND_IF_MISSING=True。"
            )
        model = train_background_model(model)
        loaded_path = os.path.join(
            RESULT_DIR, "pinn_intact_background_trained_inside_script.pth"
        )
        torch.save(model.state_dict(), loaded_path)
        print(f"完整桩背景模型已保存: {loaded_path}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, loaded_path



# =========================
# 6) One-domain broken-pile scattering model
# =========================
class ScatterNet(nn.Module):

    def __init__(self, hidden_dim=64, num_hidden=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2, hidden_dim))
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.act = torch.tanh
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)


        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x_star, t_star):
        xi = 2.0 * x_star / X_BREAK_STAR - 1.0
        tau = 2.0 * t_star / T_STAR_MAX - 1.0
        h = torch.cat([xi, tau], dim=1)
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)


class BackgroundScatteringBrokenPilePINN(nn.Module):
    def __init__(self, background_model):
        super().__init__()
        self.background = background_model
        self.net = ScatterNet(hidden_dim=64, num_hidden=4)

        self.history = {
            "train_epoch": [],
            "total": [],
            "pde": [],
            "ic_u": [],
            "ic_v": [],
            "bc_top": [],
            "bc_break": [],
            "val_epoch": [],
            "val_total": [],
            "val_pde": [],
            "val_ic_u": [],
            "val_ic_v": [],
            "val_bc_top": [],
            "val_bc_break": [],
        }

    def forward(self, x_star, t_star):
        return self.background(x_star, t_star) + self.net(x_star, t_star)

    def pde_residual(self, x_star, t_star):
        x = x_star.clone().detach().requires_grad_(True)
        t = t_star.clone().detach().requires_grad_(True)
        s = self.net(x, t)
        s_x = grad(s, x, torch.ones_like(s), create_graph=True)[0]
        s_t = grad(s, t, torch.ones_like(s), create_graph=True)[0]
        s_xx = grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
        s_tt = grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
        return s_tt - s_xx

    def ic_residuals(self, x_star):
        x = x_star.clone().detach().requires_grad_(True)
        t0 = torch.zeros_like(x).requires_grad_(True)
        s = self.net(x, t0)
        s_t = grad(s, t0, torch.ones_like(s), create_graph=True)[0]
        return s, s_t

    def bc_top_residual(self, t_star):
        t = t_star.clone().detach().requires_grad_(True)
        x0 = torch.zeros_like(t).requires_grad_(True)
        s = self.net(x0, t)
        s_x = grad(s, x0, torch.ones_like(s), create_graph=True)[0]
        return s_x

    def _background_ux_at_break(self, t_star):
        x_b = torch.full_like(t_star, X_BREAK_STAR).requires_grad_(True)
        t = t_star.clone().detach()
        u0 = self.background(x_b, t)
        u0_x = grad(u0, x_b, torch.ones_like(u0), create_graph=False)[0]
        return u0_x.detach()

    def bc_break_residual(self, t_star):
        t = t_star.clone().detach().requires_grad_(True)
        x_b = torch.full_like(t, X_BREAK_STAR).requires_grad_(True)
        s = self.net(x_b, t)
        s_x = grad(s, x_b, torch.ones_like(s), create_graph=True)[0]
        u0_x = self._background_ux_at_break(t)
        return s_x + u0_x

    def loss(self, batch, weights, record=False, epoch=None):
        r_pde = self.pde_residual(*batch["pde"])
        s_ic, v_ic = self.ic_residuals(batch["ic"])
        r_top = self.bc_top_residual(batch["bc_top_t"])
        r_break = self.bc_break_residual(batch["bc_break_t"])

        l_pde = torch.mean(r_pde ** 2)
        l_ic_u = torch.mean(s_ic ** 2)
        l_ic_v = torch.mean(v_ic ** 2)
        l_top = torch.mean(r_top ** 2)
        l_break = torch.mean(r_break ** 2)

        total = (
            weights["pde"] * l_pde
            + weights["ic_u"] * l_ic_u
            + weights["ic_v"] * l_ic_v
            + weights["bc_top"] * l_top
            + weights["bc_break"] * l_break
        )

        if record:
            self.history["train_epoch"].append(int(epoch))
            for key, value in (
                ("total", total),
                ("pde", l_pde),
                ("ic_u", l_ic_u),
                ("ic_v", l_ic_v),
                ("bc_top", l_top),
                ("bc_break", l_break),
            ):
                self.history[key].append(float(value.detach().cpu()))

        return {
            "total": total,
            "pde": l_pde,
            "ic_u": l_ic_u,
            "ic_v": l_ic_v,
            "bc_top": l_top,
            "bc_break": l_break,
        }


# =========================
# 7) Fixed sampling for broken-pile scattering field
# =========================
def valid_event_centers(centers):
    return [
        float(c) for c in centers
        if 0.0 <= float(c) <= float(T_STAR_MAX)
    ]


BREAK_ARRIVE_CENTERS = valid_event_centers(
    [
        T_BREAK_FIRST_ARRIVE_STAR,
        T_BREAK_SECOND_ARRIVE_STAR,
        T_BREAK_THIRD_ARRIVE_STAR,
        T_BACKGROUND_BOTTOM_PASS_BREAK_STAR,
    ]
)

TOP_REFLECT_CENTERS = valid_event_centers(
    [
        T_BREAK_FIRST_REFLECT_TOP_STAR,
        T_BREAK_SECOND_REFLECT_TOP_STAR,
        T_BREAK_THIRD_REFLECT_TOP_STAR,
        T_BOTTOM_REFLECT_STAR,
    ]
)


def uniform_broken_pair(n):
    x = torch.rand(n, 1, device=device) * X_BREAK_STAR
    t = torch.rand(n, 1, device=device) * T_STAR_MAX
    return _shuffle_pair(x, t)


def sample_event_pulse_band(centers, n_total, jitter_half_width=0.035):
    if n_total <= 0:
        return torch.empty(0, 1, device=device)

    centers = valid_event_centers(centers)
    if not centers:
        return torch.rand(n_total, 1, device=device) * T_STAR_MAX

    idx = torch.randint(0, len(centers), (n_total,), device=device)
    center_tensor = torch.tensor(
        centers, dtype=torch.float32, device=device
    )[idx].view(-1, 1)

    pulse_offset = torch.rand(n_total, 1, device=device) * T_PULSE_STAR
    jitter = (
        2.0 * torch.rand(n_total, 1, device=device) - 1.0
    ) * jitter_half_width
    t = center_tensor + pulse_offset + jitter
    return torch.clamp(t, 0.0, T_STAR_MAX)


def sample_scattered_characteristic_pair(n):
    if n <= 0:
        return (
            torch.empty(0, 1, device=device),
            torch.empty(0, 1, device=device),
        )

    x = torch.rand(n, 1, device=device) * X_BREAK_STAR
    branch = torch.randint(0, 5, (n, 1), device=device)

    t0 = torch.empty_like(x)
    t0 = torch.where(branch == 0, 2.0 * X_BREAK_STAR - x, t0)
    t0 = torch.where(branch == 1, 2.0 * X_BREAK_STAR + x, t0)
    t0 = torch.where(branch == 2, 4.0 * X_BREAK_STAR - x, t0)
    t0 = torch.where(branch == 3, 4.0 * X_BREAK_STAR + x, t0)
    t0 = torch.where(branch == 4, 6.0 * X_BREAK_STAR - x, t0)

    pulse_offset = torch.rand(n, 1, device=device) * T_PULSE_STAR
    jitter = (2.0 * torch.rand(n, 1, device=device) - 1.0) * 0.035
    t = t0 + pulse_offset + jitter

    mask = (t[:, 0] >= 0.0) & (t[:, 0] <= T_STAR_MAX)
    x = x[mask]
    t = t[mask]


    n_missing = n - x.shape[0]
    if n_missing > 0:
        x_fill, t_fill = uniform_broken_pair(n_missing)
        x = torch.cat([x, x_fill], dim=0)
        t = torch.cat([t, t_fill], dim=0)

    return _shuffle_pair(x, t)


def make_broken_pde(n_pde):
    n_uniform = int(0.62 * n_pde)
    n_top_early = int(0.08 * n_pde)
    n_break = int(0.14 * n_pde)
    n_characteristic = n_pde - n_uniform - n_top_early - n_break

    pair_uniform = uniform_broken_pair(n_uniform)

    x_top = sample_x_interval(
        n_top_early, 0.0, min(0.045, X_BREAK_STAR)
    )
    t_top = rand_window(0.0, 1.25 * T_PULSE_STAR, n_top_early)

    x_break = sample_x_interval(
        n_break,
        max(0.0, X_BREAK_STAR - 0.045),
        X_BREAK_STAR,
    )
    t_break = sample_event_pulse_band(
        BREAK_ARRIVE_CENTERS,
        n_break,
        jitter_half_width=0.05,
    )

    pair_char = sample_scattered_characteristic_pair(n_characteristic)

    return _concat_pairs(
        pair_uniform,
        (x_top, t_top),
        (x_break, t_break),
        pair_char,
    )


def sample_scatter_ic(n_ic):
    n_top = max(2, int(0.04 * n_ic))
    n_break = max(2, int(0.04 * n_ic))
    n_uniform = n_ic - n_top - n_break

    x = _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * X_BREAK_STAR,
        torch.rand(n_top, 1, device=device) * min(0.03, X_BREAK_STAR),
        X_BREAK_STAR
        - torch.rand(n_break, 1, device=device) * min(0.03, X_BREAK_STAR),
    )

    anchors = torch.tensor(
        [[0.0], [0.25 * X_BREAK_STAR], [0.50 * X_BREAK_STAR],
         [0.75 * X_BREAK_STAR], [X_BREAK_STAR]],
        dtype=torch.float32,
        device=device,
    )
    n_anchor = min(anchors.shape[0], x.shape[0])
    x[:n_anchor] = anchors[:n_anchor]
    return x


def sample_scatter_top(n_bc):
    n_early = int(0.06 * n_bc)
    n_reflect = int(0.22 * n_bc)
    n_background_cancel = int(0.06 * n_bc)
    n_uniform = n_bc - n_early - n_reflect - n_background_cancel

    return _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        rand_window(0.0, 1.20 * T_PULSE_STAR, n_early),
        sample_event_pulse_band(
            TOP_REFLECT_CENTERS,
            n_reflect,
            jitter_half_width=0.05,
        ),
        sample_event_pulse_band(
            [T_BOTTOM_REFLECT_STAR],
            n_background_cancel,
            jitter_half_width=0.05,
        ),
    )


def sample_scatter_break(n_bc):
    n_first = int(0.22 * n_bc)
    n_repeat = int(0.24 * n_bc)
    n_background_cancel = int(0.12 * n_bc)
    n_uniform = n_bc - n_first - n_repeat - n_background_cancel

    return _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        sample_event_pulse_band(
            [T_BREAK_FIRST_ARRIVE_STAR],
            n_first,
            jitter_half_width=0.045,
        ),
        sample_event_pulse_band(
            [
                T_BREAK_SECOND_ARRIVE_STAR,
                T_BREAK_THIRD_ARRIVE_STAR,
            ],
            n_repeat,
            jitter_half_width=0.05,
        ),
        sample_event_pulse_band(
            [T_BACKGROUND_BOTTOM_PASS_BREAK_STAR],
            n_background_cancel,
            jitter_half_width=0.05,
        ),
    )


def make_scatter_batch(n_pde=12000, n_ic=1800, n_top=1600, n_break=2200):
    return {
        "pde": make_broken_pde(n_pde),
        "ic": sample_scatter_ic(n_ic),
        "bc_top_t": sample_scatter_top(n_top),
        "bc_break_t": sample_scatter_break(n_break),
    }


def scatter_weights(epoch):
    # 断端反射完全由自由端边界产生，因此断端边界权重略高于桩顶齐次边界。
    if epoch < 15000:
        w_ic_v = 140.0
        w_break = 200.0
    elif epoch < 35000:
        w_ic_v = 120.0
        w_break = 180.0
    else:
        w_ic_v = 100.0
        w_break = 170.0

    return {
        "pde": 50.0,
        "ic_u": 100.0,
        "ic_v": w_ic_v,
        "bc_top": 120.0,
        "bc_break": w_break,
    }


def set_scatter_lr(optimizer, epoch):
    if epoch < 10000:
        lr_now = 1.5e-3
    elif epoch < 25000:
        lr_now = 8.0e-4
    elif epoch < 40000:
        lr_now = 2.0e-4
    else:
        lr_now = 5.0e-5

    for group in optimizer.param_groups:
        group["lr"] = lr_now
    return lr_now


def evaluate_scatter(model, batch, weights):
    model.eval()
    values = model.loss(batch, weights, record=False)
    return {
        key: float(value.detach().cpu())
        for key, value in values.items()
    }


def record_scatter_validation(model, epoch, values):
    h = model.history
    h["val_epoch"].append(int(epoch))
    for key in (
        "total", "pde", "ic_u", "ic_v", "bc_top", "bc_break"
    ):
        h[f"val_{key}"].append(float(values[key]))


def train_scatter_model(model):
    print("\n==> 开始训练断桩自由端散射场 ...")
    print("    背景网络已冻结；仅更新一个上部桩散射网络。")
    print("    实际散射计算域: 0 <= x <= 6.20 m。")
    print("    断端条件: u0_x + s_x = 0。")
    print("    采样: 全程固定，断端事件与反射特征带温和加密。")
    print("    梯度裁剪: 完全取消。")

    train_batch = make_scatter_batch(
        n_pde=12000,
        n_ic=1800,
        n_top=1600,
        n_break=2200,
    )
    val_batch = make_scatter_batch(
        n_pde=4500,
        n_ic=900,
        n_top=900,
        n_break=1200,
    )

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=1.5e-3,
    )

    best_val = float("inf")
    best_state = None
    start = default_timer()

    for epoch in range(SCATTER_ADAM_EPOCHS):
        model.train()
        model.background.eval()
        lr_now = set_scatter_lr(optimizer, epoch)
        weights = scatter_weights(epoch)

        optimizer.zero_grad()
        values = model.loss(
            train_batch,
            weights,
            record=True,
            epoch=epoch,
        )
        values["total"].backward()


        optimizer.step()

        if (
            epoch % SCATTER_VAL_INTERVAL == 0
            or epoch == SCATTER_ADAM_EPOCHS - 1
        ):
            val = evaluate_scatter(model, val_batch, weights)
            record_scatter_validation(model, epoch, val)

            if val["total"] < best_val:
                best_val = val["total"]
                best_state = copy.deepcopy(model.net.state_dict())
                torch.save(
                    best_state,
                    os.path.join(
                        RESULT_DIR,
                        "best_broken_scatter_adam_state.pth",
                    ),
                )

            print(
                f"[断桩散射 Adam] epoch={epoch:5d} "
                f"lr={lr_now:.1e} "
                f"train={values['total'].item():.3e} "
                f"val={val['total']:.3e}"
            )
            print(
                f"    Val PDE={val['pde']:.3e} "
                f"IC-u={val['ic_u']:.3e} "
                f"IC-v={val['ic_v']:.3e} "
                f"Top={val['bc_top']:.3e} "
                f"Break-free={val['bc_break']:.3e}"
            )

    if best_state is not None:
        model.net.load_state_dict(best_state)

    weights_final = scatter_weights(SCATTER_ADAM_EPOCHS)

    before_state = copy.deepcopy(model.net.state_dict())
    before_val = evaluate_scatter(
        model,
        val_batch,
        weights_final,
    )

    print("\n==> 开始验证集控制的 L-BFGS 微调 ...")
    optimizer_lbfgs = torch.optim.LBFGS(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.3,
        max_iter=SCATTER_LBFGS_MAX_ITER,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        values = model.loss(
            train_batch,
            weights_final,
            record=False,
        )
        values["total"].backward()
        return values["total"]

    optimizer_lbfgs.step(closure)

    after_val = evaluate_scatter(
        model,
        val_batch,
        weights_final,
    )

    accept_lbfgs = (
        after_val["total"] < before_val["total"]
        and after_val["pde"] <= 1.05 * before_val["pde"]
        and after_val["bc_break"] <= 1.05 * before_val["bc_break"]
    )

    if accept_lbfgs:
        print(
            "L-BFGS 被接受：验证总损失下降，"
            "且验证 PDE 与断端自由边界未明显恶化。"
        )
        final_choice = "LBFGS_ACCEPTED"
        final_val = after_val
    else:
        print(
            "L-BFGS 未通过验证条件，恢复 Adam 最优断桩散射模型。"
        )
        model.net.load_state_dict(before_state)
        final_choice = "ADAM_RESTORED"
        final_val = before_val

    elapsed = default_timer() - start
    print(f"断桩散射场训练完成，用时 {elapsed:.2f} s")

    summary_path = os.path.join(
        RESULT_DIR,
        "training_summary.txt",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("model=upper_pile_free_end_broken_pile\n")
        f.write("background_frozen=True\n")
        f.write("gradient_clipping=False\n")
        f.write(f"original_length_m={L_ORIGINAL:.12e}\n")
        f.write(f"actual_length_m={L_ACTUAL:.12e}\n")
        f.write(f"break_position_m={X_BREAK:.12e}\n")
        f.write(f"final_choice={final_choice}\n")
        f.write(
            f"before_lbfgs_val_total="
            f"{before_val['total']:.12e}\n"
        )
        f.write(
            f"final_val_total="
            f"{final_val['total']:.12e}\n"
        )
        f.write(
            f"before_lbfgs_val_pde="
            f"{before_val['pde']:.12e}\n"
        )
        f.write(
            f"final_val_pde="
            f"{final_val['pde']:.12e}\n"
        )
        f.write(
            f"before_lbfgs_val_bc_break="
            f"{before_val['bc_break']:.12e}\n"
        )
        f.write(
            f"final_val_bc_break="
            f"{final_val['bc_break']:.12e}\n"
        )

    return model


# =========================
# 8) Prediction and diagnostics
# =========================
def top_velocity_components(model, nt=2000):
    model.eval()
    t_star = torch.linspace(
        0.0,
        T_STAR_MAX,
        nt,
        device=device,
    ).view(-1, 1)

    # 完整桩背景速度
    x0_bg = torch.zeros_like(t_star).requires_grad_(True)
    t_bg = t_star.clone().detach().requires_grad_(True)
    u0 = model.background(x0_bg, t_bg)
    u0_t = grad(
        u0,
        t_bg,
        torch.ones_like(u0),
        create_graph=False,
    )[0]

    # 断桩散射修正速度
    x0_sc = torch.zeros_like(t_star).requires_grad_(True)
    t_sc = t_star.clone().detach().requires_grad_(True)
    s = model.net(x0_sc, t_sc)
    s_t = grad(
        s,
        t_sc,
        torch.ones_like(s),
        create_graph=False,
    )[0]

    factor = U_REF / T_REF
    v0_m_s = factor * u0_t.detach().cpu().numpy().reshape(-1)
    vs_m_s = factor * s_t.detach().cpu().numpy().reshape(-1)
    vt_m_s = v0_m_s + vs_m_s
    t_ms = (
        T_REF
        * t_star.detach().cpu().numpy().reshape(-1)
        * 1e3
    )
    return t_ms, v0_m_s, vs_m_s, vt_m_s


def check_initial_total_velocity(model, nx=300):
    x = torch.linspace(
        0.0,
        X_BREAK_STAR,
        nx,
        device=device,
    ).view(-1, 1).requires_grad_(True)
    t = torch.zeros_like(x).requires_grad_(True)

    u = model(x, t)
    u_t = grad(
        u,
        t,
        torch.ones_like(u),
        create_graph=False,
    )[0]

    v_mm_s = (
        (U_REF / T_REF)
        * u_t.detach().cpu().numpy().reshape(-1)
        * 1e3
    )

    print(
        f"初始总速度检查: "
        f"max|v(x,0)|={np.max(np.abs(v_mm_s)):.6e} mm/s, "
        f"v(0,0)={v_mm_s[0]:.6e} mm/s, "
        f"v(x_b,0)={v_mm_s[-1]:.6e} mm/s"
    )


def predict_total_grid(model, nx=220, nt=360):
    x = torch.linspace(
        0.0,
        X_BREAK_STAR,
        nx,
        device=device,
    )
    t = torch.linspace(
        0.0,
        T_STAR_MAX,
        nt,
        device=device,
    )
    X, T = torch.meshgrid(x, t, indexing="ij")
    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    with torch.no_grad():
        u = model(xf, tf).reshape(nx, nt)

    return (
        X.cpu().numpy(),
        T.cpu().numpy(),
        u.cpu().numpy(),
    )


def predict_total_velocity_grid(model, nx=180, nt=300):
    x = torch.linspace(
        0.0,
        X_BREAK_STAR,
        nx,
        device=device,
    )
    t = torch.linspace(
        0.0,
        T_STAR_MAX,
        nt,
        device=device,
    )
    X, T = torch.meshgrid(x, t, indexing="ij")
    xf = X.reshape(-1, 1).requires_grad_(True)
    tf = T.reshape(-1, 1).requires_grad_(True)

    u = model(xf, tf)
    u_t = grad(
        u,
        tf,
        torch.ones_like(u),
        create_graph=False,
    )[0]

    return (
        X.cpu().numpy(),
        T.cpu().numpy(),
        u_t.detach().cpu().numpy().reshape(nx, nt),
    )


def break_boundary_components(model, nt=1600):
    model.eval()
    t_star = torch.linspace(
        0.0,
        T_STAR_MAX,
        nt,
        device=device,
    ).view(-1, 1)

    x_bg = torch.full_like(
        t_star,
        X_BREAK_STAR,
    ).requires_grad_(True)
    t_bg = t_star.clone().detach()
    u0 = model.background(x_bg, t_bg)
    u0_x = grad(
        u0,
        x_bg,
        torch.ones_like(u0),
        create_graph=False,
    )[0]

    x_sc = torch.full_like(
        t_star,
        X_BREAK_STAR,
    ).requires_grad_(True)
    t_sc = t_star.clone().detach()
    s = model.net(x_sc, t_sc)
    s_x = grad(
        s,
        x_sc,
        torch.ones_like(s),
        create_graph=False,
    )[0]

    u0_x_np = u0_x.detach().cpu().numpy().reshape(-1)
    s_x_np = s_x.detach().cpu().numpy().reshape(-1)
    total_x_np = u0_x_np + s_x_np

    t_ms = (
        T_REF
        * t_star.detach().cpu().numpy().reshape(-1)
        * 1e3
    )
    return t_ms, u0_x_np, s_x_np, total_x_np


def plot_loss_histories(model):
    h = model.history
    epochs = h["train_epoch"]

    plt.figure(figsize=(13, 6.8))
    plt.semilogy(
        epochs,
        h["total"],
        label="Train total",
        linewidth=1.8,
    )
    plt.semilogy(epochs, h["pde"], label="Train PDE")
    plt.semilogy(
        epochs,
        h["ic_u"],
        label="Train IC displacement",
    )
    plt.semilogy(
        epochs,
        h["ic_v"],
        label="Train IC velocity",
    )
    plt.semilogy(
        epochs,
        h["bc_top"],
        label="Train top scattering BC",
    )
    plt.semilogy(
        epochs,
        h["bc_break"],
        label="Train break free-end BC",
    )

    if h["val_epoch"]:
        plt.semilogy(
            h["val_epoch"],
            h["val_total"],
            "--",
            linewidth=2.0,
            label="Validation total",
        )
        plt.semilogy(
            h["val_epoch"],
            h["val_pde"],
            "--",
            linewidth=1.5,
            label="Validation PDE",
        )
        plt.semilogy(
            h["val_epoch"],
            h["val_bc_break"],
            "--",
            linewidth=1.5,
            label="Validation break BC",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss, log scale")
    plt.title(
        "Training and validation loss - free-end broken pile"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "loss_history_train_val.png",
        ),
        dpi=300,
    )
    plt.close()

    if h["val_epoch"]:
        plt.figure(figsize=(13, 6.5))
        for key, label in (
            ("val_total", "Validation total"),
            ("val_pde", "Validation PDE"),
            ("val_ic_u", "Validation IC displacement"),
            ("val_ic_v", "Validation IC velocity"),
            ("val_bc_top", "Validation top BC"),
            ("val_bc_break", "Validation break free-end BC"),
        ):
            plt.semilogy(
                h["val_epoch"],
                h[key],
                label=label,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Validation loss, log scale")
        plt.title(
            "Validation loss history - free-end broken pile"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULT_DIR,
                "validation_loss_history.png",
            ),
            dpi=300,
        )
        plt.close()


def plot_top_velocity(model):
    t_ms, _, _, vt = top_velocity_components(
        model,
        nt=2000,
    )

    plt.figure(figsize=(13, 5.8))
    plt.plot(
        t_ms,
        vt * 1e3,
        linewidth=2.0,
        label="Total PINN",
    )
    plt.axvline(
        T_BREAK_FIRST_REFLECT_TOP * 1e3,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="First break reflection",
    )
    if T_BREAK_SECOND_REFLECT_TOP <= T_OBS:
        plt.axvline(
            T_BREAK_SECOND_REFLECT_TOP * 1e3,
            linestyle=":",
            linewidth=1.0,
            alpha=0.65,
            label="Second break reflection",
        )

    plt.xlabel("Time t / ms")
    plt.ylabel("Pile-head velocity / mm/s")
    plt.title("Pile-head velocity - free-end broken pile")
    plt.xlim(0.0, T_OBS * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "top_velocity_total.png",
        ),
        dpi=300,
    )
    plt.close()


def plot_break_boundary_check(model):
    t_ms, u0_x, s_x, total_x = break_boundary_components(
        model,
        nt=1600,
    )

    plt.figure(figsize=(13, 5.8))
    plt.plot(
        t_ms,
        u0_x,
        linewidth=1.5,
        label=r"Background $u_{0,x}$",
    )
    plt.plot(
        t_ms,
        s_x,
        linewidth=1.5,
        label=r"Scattering $s_x$",
    )
    plt.plot(
        t_ms,
        total_x,
        linewidth=2.0,
        label=r"Total $u_x=u_{0,x}+s_x$",
    )
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Time t / ms")
    plt.ylabel(r"Dimensionless axial gradient at $x_b$")
    plt.title("Free-end boundary check at the broken section")
    plt.xlim(0.0, T_OBS * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "break_free_boundary_check.png",
        ),
        dpi=300,
    )
    plt.close()

    csv_path = os.path.join(
        RESULT_DIR,
        "break_free_boundary_check.csv",
    )
    np.savetxt(
        csv_path,
        np.column_stack([t_ms, u0_x, s_x, total_x]),
        delimiter=",",
        header=(
            "time_ms,background_ux_star,"
            "scatter_sx_star,total_ux_star"
        ),
        comments="",
    )


def plot_displacement_contour(X, T, u_star):
    u_mm = U_REF * u_star * 1e3
    x_m = X * L
    t_ms = T * T_REF * 1e3

    u_min = float(np.min(u_mm))
    u_max = float(np.max(u_mm))
    if np.isclose(u_min, u_max):
        u_min -= 1.0e-12
        u_max += 1.0e-12

    levels = np.linspace(u_min, u_max, 100)

    plt.figure(figsize=(12.5, 6.2))
    cs = plt.contourf(
        t_ms,
        x_m,
        u_mm,
        levels=levels,
        cmap="RdBu_r",
    )
    plt.colorbar(
        cs,
        label="Displacement u(x,t) / mm",
    )
    plt.axhline(
        X_BREAK,
        linestyle="--",
        linewidth=1.2,
        label="Free broken end",
    )
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile position x / m")
    plt.title(
        "Displacement contour - free-end broken pile"
    )
    plt.ylim(0.0, X_BREAK)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "displacement_contour.png",
        ),
        dpi=300,
    )
    plt.close()


def plot_velocity_contour(X, T, u_t_star):
    v_mm_s = (
        (U_REF / T_REF)
        * u_t_star
        * 1e3
    )
    x_m = X * L
    t_ms = T * T_REF * 1e3

    vmax = np.percentile(
        np.abs(v_mm_s),
        99.5,
    )
    vmax = max(float(vmax), 1.0e-12)
    levels = np.linspace(-vmax, vmax, 100)

    plt.figure(figsize=(12.5, 6.2))
    cs = plt.contourf(
        t_ms,
        x_m,
        v_mm_s,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    plt.colorbar(
        cs,
        label="Velocity v(x,t) / mm/s",
    )
    plt.axhline(
        X_BREAK,
        linestyle="--",
        linewidth=1.2,
        label="Free broken end",
    )
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile position x / m")
    plt.title(
        "Velocity contour - free-end broken pile"
    )
    plt.ylim(0.0, X_BREAK)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "velocity_contour.png",
        ),
        dpi=300,
    )
    plt.close()


def save_top_velocity_csv(model):
    t_ms, v0, vs, vt = top_velocity_components(
        model,
        nt=2000,
    )

    data = np.column_stack(
        [
            t_ms / 1e3,
            t_ms,
            v0,
            vs,
            vt,
            v0 * 1e3,
            vs * 1e3,
            vt * 1e3,
        ]
    )

    path = os.path.join(
        RESULT_DIR,
        "pinn_broken_pile_free_end_top_velocity.csv",
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=(
            "time_s,time_ms,"
            "background_v_m_s,scatter_v_m_s,total_v_m_s,"
            "background_v_mm_s,scatter_v_mm_s,total_v_mm_s"
        ),
        comments="",
    )
    print(f"桩顶速度 CSV 已保存: {path}")
    return path


# =========================
# 9) ABAQUS RPT comparison
# =========================
def parse_abaqus_float(token):
    return float(
        token.strip().replace("D", "E").replace("d", "E")
    )


def read_abaqus_rpt_segments(rpt_path):
    """
    将 ABAQUS .rpt 按时间单调递增的连续段读取。
    绘图时逐段画线，避免不同 XY 数据段首尾相连。
    """
    if rpt_path is None or not os.path.exists(rpt_path):
        return []

    pattern = re.compile(
        r"^[\s]*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[EeDd][+-]?\d+)?)"
        r"[\s,]+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[EeDd][+-]?\d+)?)"
        r"[\s]*$"
    )

    segments = []
    cur_t = []
    cur_v = []
    last_t = None

    with open(
        rpt_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            match = pattern.match(line)

            if match is None:
                if len(cur_t) >= 2:
                    segments.append(
                        (
                            np.asarray(cur_t),
                            np.asarray(cur_v),
                        )
                    )
                cur_t = []
                cur_v = []
                last_t = None
                continue

            t_value = parse_abaqus_float(match.group(1))
            v_value = parse_abaqus_float(match.group(2))

            if (
                last_t is not None
                and t_value < last_t - 1.0e-12
            ):
                if len(cur_t) >= 2:
                    segments.append(
                        (
                            np.asarray(cur_t),
                            np.asarray(cur_v),
                        )
                    )
                cur_t = []
                cur_v = []

            cur_t.append(t_value)
            cur_v.append(v_value)
            last_t = t_value

    if len(cur_t) >= 2:
        segments.append(
            (
                np.asarray(cur_t),
                np.asarray(cur_v),
            )
        )

    clean = []
    for t, v in segments:
        mask = np.isfinite(t) & np.isfinite(v)
        t = t[mask]
        v = v[mask]

        if len(t) < 2:
            continue

        keep = np.ones_like(t, dtype=bool)
        keep[1:] = np.abs(np.diff(t)) > 1.0e-15
        t = t[keep]
        v = v[keep]

        if len(t) >= 2:
            clean.append((t, v))

    return clean


def find_abaqus_rpt():
    preferred_names = [
        "XY_pile_broken_freeend_1D.rpt",
        "XY_pile_broken_free_end_1D.rpt",
        "XY_pile_broken_1D.rpt",
        "abaqus_broken_pile_free_end_top_velocity.rpt",
    ]

    search_dirs = [
        get_base_dir(),
        os.getcwd(),
        "/mnt/data",
    ]

    for directory in search_dirs:
        for name in preferred_names:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate

    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue

        for name in os.listdir(directory):
            lower = name.lower()
            if (
                lower.endswith(".rpt")
                and (
                    "broken" in lower
                    or "break" in lower
                    or "断桩" in lower
                )
            ):
                return os.path.join(directory, name)

    return None


def plot_abaqus_comparison(model):
    rpt_path = find_abaqus_rpt()

    if rpt_path is None:
        print(
            "未找到断桩自由端 ABAQUS .rpt，跳过对比图。"
        )
        return

    segments = read_abaqus_rpt_segments(rpt_path)
    if not segments:
        print(
            f"未能从 {rpt_path} 读取有效 ABAQUS 数据，"
            "跳过对比图。"
        )
        return

    t_ms, _, _, vt = top_velocity_components(
        model,
        nt=2000,
    )

    plt.figure(figsize=(14, 5.8))
    first = True
    for t_s, v_m_s in segments:
        plt.plot(
            t_s * 1e3,
            v_m_s * 1e3,
            linewidth=1.8,
            label="ABAQUS 1D" if first else None,
        )
        first = False

    plt.plot(
        t_ms,
        vt * 1e3,
        "--",
        linewidth=2.0,
        label="PINN total",
    )
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile-head velocity v / mm/s")
    plt.title(
        "Free-end broken pile: ABAQUS vs background-scattering PINN"
    )
    plt.xlim(0.0, T_OBS * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "abaqus_vs_pinn_broken_pile_free_end.png",
        ),
        dpi=300,
    )
    plt.close()

    print(f"ABAQUS 对比文件: {rpt_path}")


# =========================
# 10) Save model
# =========================
def save_scatter_model(model, background_path):
    path = os.path.join(
        RESULT_DIR,
        "pinn_broken_pile_free_end_background_scattering_model.pth",
    )

    torch.save(
        {
            "scatter_net": model.net.state_dict(),
            "background_model_path": background_path,
            "parameters": {
                "original_length_m": L_ORIGINAL,
                "actual_model_length_m": L_ACTUAL,
                "break_position_m": X_BREAK,
                "diameter_m": D_NORMAL,
                "area_m2": A_NORMAL,
                "rho_kg_m3": RHO0,
                "E_pa": E0,
                "wave_speed_m_s": C0,
                "T_obs_s": T_OBS,
                "T_pulse_s": T_PULSE,
                "p0_pa": P0,
                "abaqus_equivalent_force_n": F_MAX,
                "break_boundary": "free_end_total_ux_zero",
                "gradient_clipping": False,
            },
        },
        path,
    )

    print(f"断桩散射模型已保存: {path}")
    return path


# =========================
# 11) Main
# =========================
if __name__ == "__main__":
    save_excitation_plot()

    background_model, background_path = load_or_train_background()

    model = BackgroundScatteringBrokenPilePINN(
        background_model
    ).to(device)

    model = train_scatter_model(model)

    check_initial_total_velocity(model)
    plot_loss_histories(model)
    plot_top_velocity(model)
    plot_break_boundary_check(model)

    X, T, U = predict_total_grid(
        model,
        nx=220,
        nt=360,
    )
    plot_displacement_contour(X, T, U)

    XV, TV, UT = predict_total_velocity_grid(
        model,
        nx=180,
        nt=300,
    )
    plot_velocity_contour(XV, TV, UT)

    save_top_velocity_csv(model)
    plot_abaqus_comparison(model)
    save_scatter_model(model, background_path)

    print("\n模拟完成。")
    print(f"结果目录: {RESULT_DIR}")
