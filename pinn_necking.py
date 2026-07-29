# -*- coding: utf-8 -*-
"""
PINN Forward Simulation for Low-Strain Integrity Testing (LIT)
Abrupt necking pile: intact-pile background field + three-subdomain scattering field

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

RESULT_DIR = "Results_AbruptNecking_BackgroundScattering_NoClip"
os.makedirs(RESULT_DIR, exist_ok=True)

# 是否在找不到完整桩权重时自动训练背景模型
TRAIN_BACKGROUND_IF_MISSING = True
FORCE_RETRAIN_BACKGROUND = False

# 完整桩背景训练配置（仅在找不到权重时使用）
BACKGROUND_ADAM_EPOCHS = 40000
BACKGROUND_LBFGS_MAX_ITER = 800
BACKGROUND_VAL_INTERVAL = 500

# 缩颈散射场训练配置
SCATTER_ADAM_EPOCHS = 55000
SCATTER_LBFGS_MAX_ITER = 500
SCATTER_VAL_INTERVAL = 500

# =========================
# 1) Physical parameters
# =========================
L = 13.0
D_NORMAL = 0.5
D_NECK = 0.3
X_NECK_CENTER = 6.25
NECK_LENGTH = 0.5

RHO0 = 2400.0
E0 = 32500e6

T_OBS = 10e-3
T_PULSE = 1e-3
P0 = 750.0

A_NORMAL = np.pi * (D_NORMAL / 2.0) ** 2
A_NECK = np.pi * (D_NECK / 2.0) ** 2
A1_STAR = 1.0
A2_STAR = A_NECK / A_NORMAL
A3_STAR = 1.0

C0 = np.sqrt(E0 / RHO0)
X_REF = L
T_REF = L / C0
U_REF = P0 * L / E0

T_STAR_MAX = T_OBS / T_REF
T_PULSE_STAR = T_PULSE / T_REF

X_NECK_CENTER_STAR = X_NECK_CENTER / L
NECK_LENGTH_STAR = NECK_LENGTH / L
X_NECK_START_STAR = X_NECK_CENTER_STAR - 0.5 * NECK_LENGTH_STAR
X_NECK_END_STAR = X_NECK_CENTER_STAR + 0.5 * NECK_LENGTH_STAR

T_NECK_START_ARRIVE_STAR = X_NECK_START_STAR
T_NECK_END_ARRIVE_STAR = X_NECK_END_STAR
T_NECK_START_REFLECT_STAR = 2.0 * X_NECK_START_STAR
T_NECK_END_REFLECT_STAR = 2.0 * X_NECK_END_STAR
T_BOTTOM_ARRIVE_STAR = 1.0
T_BOTTOM_REFLECT_STAR = 2.0

print("=" * 72)
print("突变缩颈桩：完整桩背景场 + 三子域散射场 PINN")
print(f"设备: {device}")
print(f"桩长: {L:.3f} m")
print(f"正常桩径: {D_NORMAL:.3f} m")
print(f"缩颈桩径: {D_NECK:.3f} m")
print(f"缩颈范围: {X_NECK_START_STAR * L:.3f} ~ {X_NECK_END_STAR * L:.3f} m")
print(f"面积比 A_neck/A_normal = {A2_STAR:.6f}")
print(f"波速: {C0:.3f} m/s")
print(f"无量纲观察时间: {T_STAR_MAX:.6f}")
print("梯度裁剪: 已完全取消")
print("=" * 72)

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
# 6) Three-subdomain scattering model
# =========================
class ScatterSubNet(nn.Module):


    def __init__(self, x_min, x_max, hidden_dim=64, num_hidden=4):
        super().__init__()
        self.x_min = float(x_min)
        self.x_max = float(x_max)
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
        xi = 2.0 * (x_star - self.x_min) / (self.x_max - self.x_min) - 1.0
        tau = 2.0 * t_star / T_STAR_MAX - 1.0
        h = torch.cat([xi, tau], dim=1)
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)


class BackgroundScatteringNeckingPINN(nn.Module):
    def __init__(self, background_model):
        super().__init__()
        self.background = background_model
        self.net1 = ScatterSubNet(0.0, X_NECK_START_STAR)
        self.net2 = ScatterSubNet(X_NECK_START_STAR, X_NECK_END_STAR)
        self.net3 = ScatterSubNet(X_NECK_END_STAR, 1.0)

        self.history = {
            "train_epoch": [],
            "total": [],
            "pde": [],
            "pde_1": [],
            "pde_2": [],
            "pde_3": [],
            "ic_u": [],
            "ic_v": [],
            "bc_top": [],
            "bc_bot": [],
            "interface": [],
            "interface_u": [],
            "interface_v": [],
            "interface_force": [],
            "val_epoch": [],
            "val_total": [],
            "val_pde": [],
            "val_pde_1": [],
            "val_pde_2": [],
            "val_pde_3": [],
            "val_ic_u": [],
            "val_ic_v": [],
            "val_bc_top": [],
            "val_bc_bot": [],
            "val_interface": [],
            "val_interface_u": [],
            "val_interface_v": [],
            "val_interface_force": [],
        }

    def scatter_piecewise(self, x_star, t_star):
        out = torch.empty_like(x_star)
        m1 = x_star < X_NECK_START_STAR
        m2 = (x_star >= X_NECK_START_STAR) & (x_star <= X_NECK_END_STAR)
        m3 = x_star > X_NECK_END_STAR

        if torch.any(m1):
            out[m1] = self.net1(
                x_star[m1].view(-1, 1), t_star[m1].view(-1, 1)
            ).view(-1)
        if torch.any(m2):
            out[m2] = self.net2(
                x_star[m2].view(-1, 1), t_star[m2].view(-1, 1)
            ).view(-1)
        if torch.any(m3):
            out[m3] = self.net3(
                x_star[m3].view(-1, 1), t_star[m3].view(-1, 1)
            ).view(-1)
        return out.view(-1, 1)

    def forward(self, x_star, t_star):
        return self.background(x_star, t_star) + self.scatter_piecewise(x_star, t_star)

    @staticmethod
    def _pde_single(net, x_star, t_star):
        x = x_star.clone().detach().requires_grad_(True)
        t = t_star.clone().detach().requires_grad_(True)
        s = net(x, t)
        s_x = grad(s, x, torch.ones_like(s), create_graph=True)[0]
        s_t = grad(s, t, torch.ones_like(s), create_graph=True)[0]
        s_xx = grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
        s_tt = grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
        return s_tt - s_xx

    def pde_residuals(self, batch):
        return (
            self._pde_single(self.net1, *batch["pde_1"]),
            self._pde_single(self.net2, *batch["pde_2"]),
            self._pde_single(self.net3, *batch["pde_3"]),
        )

    @staticmethod
    def _ic_single(net, x_star):
        x = x_star.clone().detach().requires_grad_(True)
        t0 = torch.zeros_like(x).requires_grad_(True)
        s = net(x, t0)
        s_t = grad(s, t0, torch.ones_like(s), create_graph=True)[0]
        return s, s_t

    def ic_residuals(self, batch):
        s1, v1 = self._ic_single(self.net1, batch["ic_1"])
        s2, v2 = self._ic_single(self.net2, batch["ic_2"])
        s3, v3 = self._ic_single(self.net3, batch["ic_3"])
        return torch.cat([s1, s2, s3], 0), torch.cat([v1, v2, v3], 0)

    def bc_top_residual(self, t_star):

        t = t_star.clone().detach().requires_grad_(True)
        x0 = torch.zeros_like(t).requires_grad_(True)
        s = self.net1(x0, t)
        return grad(s, x0, torch.ones_like(s), create_graph=True)[0]

    def bc_bot_residual(self, t_star):
        t = t_star.clone().detach().requires_grad_(True)
        x1 = torch.ones_like(t).requires_grad_(True)
        s = self.net3(x1, t)
        return grad(s, x1, torch.ones_like(s), create_graph=True)[0]

    def _background_ux(self, x_value, t_star):
        x = torch.full_like(t_star, float(x_value)).requires_grad_(True)
        t = t_star.clone().detach()
        u0 = self.background(x, t)
        u0_x = grad(u0, x, torch.ones_like(u0), create_graph=False)[0]
        return u0_x.detach()

    def interface_residuals(self, t_start_star, t_end_star):
        ts = t_start_star.clone().detach().requires_grad_(True)
        te = t_end_star.clone().detach().requires_grad_(True)

        # 界面 1：正常段 -> 缩颈段
        xs_l = torch.full_like(ts, X_NECK_START_STAR).requires_grad_(True)
        xs_r = torch.full_like(ts, X_NECK_START_STAR).requires_grad_(True)
        s1 = self.net1(xs_l, ts)
        s2s = self.net2(xs_r, ts)
        s1_x = grad(s1, xs_l, torch.ones_like(s1), create_graph=True, retain_graph=True)[0]
        s2s_x = grad(s2s, xs_r, torch.ones_like(s2s), create_graph=True, retain_graph=True)[0]
        s1_t = grad(s1, ts, torch.ones_like(s1), create_graph=True, retain_graph=True)[0]
        s2s_t = grad(s2s, ts, torch.ones_like(s2s), create_graph=True, retain_graph=True)[0]
        u0_x_s = self._background_ux(X_NECK_START_STAR, ts)

        r_u_s = s1 - s2s
        r_v_s = s1_t - s2s_t
        r_f_s = (
            A1_STAR * s1_x
            - A2_STAR * s2s_x
            + (A1_STAR - A2_STAR) * u0_x_s
        )

        # 界面 2：缩颈段 -> 正常段
        xe_l = torch.full_like(te, X_NECK_END_STAR).requires_grad_(True)
        xe_r = torch.full_like(te, X_NECK_END_STAR).requires_grad_(True)
        s2e = self.net2(xe_l, te)
        s3 = self.net3(xe_r, te)
        s2e_x = grad(s2e, xe_l, torch.ones_like(s2e), create_graph=True, retain_graph=True)[0]
        s3_x = grad(s3, xe_r, torch.ones_like(s3), create_graph=True, retain_graph=True)[0]
        s2e_t = grad(s2e, te, torch.ones_like(s2e), create_graph=True, retain_graph=True)[0]
        s3_t = grad(s3, te, torch.ones_like(s3), create_graph=True, retain_graph=True)[0]
        u0_x_e = self._background_ux(X_NECK_END_STAR, te)

        r_u_e = s2e - s3
        r_v_e = s2e_t - s3_t
        r_f_e = (
            A2_STAR * s2e_x
            - A3_STAR * s3_x
            + (A2_STAR - A3_STAR) * u0_x_e
        )

        return (
            torch.cat([r_u_s, r_u_e], 0),
            torch.cat([r_v_s, r_v_e], 0),
            torch.cat([r_f_s, r_f_e], 0),
        )

    def loss(self, batch, weights, record=False, epoch=None):
        r1, r2, r3 = self.pde_residuals(batch)
        l_pde_1 = torch.mean(r1 ** 2)
        l_pde_2 = torch.mean(r2 ** 2)
        l_pde_3 = torch.mean(r3 ** 2)

        pde_w1 = weights.get("pde_1", 0.40)
        pde_w2 = weights.get("pde_2", 0.20)
        pde_w3 = weights.get("pde_3", 0.40)
        l_pde = (
            pde_w1 * l_pde_1 + pde_w2 * l_pde_2 + pde_w3 * l_pde_3
        ) / (pde_w1 + pde_w2 + pde_w3)

        s_ic, v_ic = self.ic_residuals(batch)
        l_ic_u = torch.mean(s_ic ** 2)
        l_ic_v = torch.mean(v_ic ** 2)

        l_top = torch.mean(self.bc_top_residual(batch["bc_top_t"]) ** 2)
        l_bot = torch.mean(self.bc_bot_residual(batch["bc_bot_t"]) ** 2)

        r_if_u, r_if_v, r_if_f = self.interface_residuals(
            batch["interface_start_t"], batch["interface_end_t"]
        )
        l_if_u = torch.mean(r_if_u ** 2)
        l_if_v = torch.mean(r_if_v ** 2)
        l_if_f = torch.mean(r_if_f ** 2)
        l_interface = (
            weights.get("interface_u", 0.5) * l_if_u
            + weights.get("interface_v", 1.0) * l_if_v
            + weights.get("interface_force", 2.0) * l_if_f
        )

        total = (
            weights["pde"] * l_pde
            + weights["ic_u"] * l_ic_u
            + weights["ic_v"] * l_ic_v
            + weights["bc_top"] * l_top
            + weights["bc_bot"] * l_bot
            + weights["interface"] * l_interface
        )

        if record:
            h = self.history
            h["train_epoch"].append(int(epoch))
            for key, value in (
                ("total", total),
                ("pde", l_pde),
                ("pde_1", l_pde_1),
                ("pde_2", l_pde_2),
                ("pde_3", l_pde_3),
                ("ic_u", l_ic_u),
                ("ic_v", l_ic_v),
                ("bc_top", l_top),
                ("bc_bot", l_bot),
                ("interface", l_interface),
                ("interface_u", l_if_u),
                ("interface_v", l_if_v),
                ("interface_force", l_if_f),
            ):
                h[key].append(float(value.detach().cpu()))

        return {
            "total": total,
            "pde": l_pde,
            "pde_1": l_pde_1,
            "pde_2": l_pde_2,
            "pde_3": l_pde_3,
            "ic_u": l_ic_u,
            "ic_v": l_ic_v,
            "bc_top": l_top,
            "bc_bot": l_bot,
            "interface": l_interface,
            "interface_u": l_if_u,
            "interface_v": l_if_v,
            "interface_force": l_if_f,
        }


# =========================
# 7) Fixed mild sampling for scattering field
# =========================
def uniform_pair(n, x0, x1):
    return (
        sample_x_interval(n, x0, x1),
        torch.rand(n, 1, device=device) * T_STAR_MAX,
    )


def make_segment1_pde(n):
    n_interface = int(0.10 * n)
    n_return = int(0.08 * n)
    n_uniform = n - n_interface - n_return

    pair_uniform = uniform_pair(n_uniform, 0.0, X_NECK_START_STAR)

    x_if = sample_x_near_boundary(
        n_interface, X_NECK_START_STAR, "left", width=0.018
    )
    t_if = sample_time_around(
        [T_NECK_START_ARRIVE_STAR, T_NECK_START_REFLECT_STAR],
        n_interface,
        half_width=0.08,
    )

    x_return = sample_x_interval(n_return, 0.0, X_NECK_START_STAR)
    t_return = sample_time_around(
        [
            T_NECK_START_REFLECT_STAR,
            T_NECK_END_REFLECT_STAR,
            T_BOTTOM_REFLECT_STAR,
        ],
        n_return,
        half_width=0.12,
    )
    return _concat_pairs(pair_uniform, (x_if, t_if), (x_return, t_return))


def make_segment2_pde(n):
    n_start = int(0.11 * n)
    n_end = int(0.11 * n)
    n_uniform = n - n_start - n_end

    pair_uniform = uniform_pair(n_uniform, X_NECK_START_STAR, X_NECK_END_STAR)

    width = min(0.012, 0.30 * NECK_LENGTH_STAR)
    x_start = sample_x_near_boundary(
        n_start, X_NECK_START_STAR, "right", width=width
    )
    t_start = sample_time_around(
        [T_NECK_START_ARRIVE_STAR], n_start, half_width=0.08
    )

    x_end = sample_x_near_boundary(n_end, X_NECK_END_STAR, "left", width=width)
    t_end = sample_time_around([T_NECK_END_ARRIVE_STAR], n_end, half_width=0.08)

    return _concat_pairs(pair_uniform, (x_start, t_start), (x_end, t_end))


def make_segment3_pde(n):
    n_interface = int(0.10 * n)
    n_bottom = int(0.08 * n)
    n_uniform = n - n_interface - n_bottom

    pair_uniform = uniform_pair(n_uniform, X_NECK_END_STAR, 1.0)

    x_if = sample_x_near_boundary(
        n_interface, X_NECK_END_STAR, "right", width=0.018
    )
    t_if = sample_time_around(
        [T_NECK_END_ARRIVE_STAR], n_interface, half_width=0.08
    )

    x_bottom = sample_x_interval(
        n_bottom, max(X_NECK_END_STAR, 0.97), 1.0
    )
    t_bottom = sample_time_around(
        [T_BOTTOM_ARRIVE_STAR], n_bottom, half_width=0.12
    )
    return _concat_pairs(pair_uniform, (x_if, t_if), (x_bottom, t_bottom))


def sample_scatter_ic(n_total):
    n2 = int(0.24 * n_total)
    n1 = int(0.38 * n_total)
    n3 = n_total - n1 - n2

    x1 = sample_x_interval(n1, 0.0, X_NECK_START_STAR)
    x2 = sample_x_interval(n2, X_NECK_START_STAR, X_NECK_END_STAR)
    x3 = sample_x_interval(n3, X_NECK_END_STAR, 1.0)


    anchors1 = torch.tensor([[0.0], [X_NECK_START_STAR]], device=device)
    anchors2 = torch.tensor(
        [[X_NECK_START_STAR], [X_NECK_END_STAR]], device=device
    )
    anchors3 = torch.tensor([[X_NECK_END_STAR], [1.0]], device=device)
    x1[: anchors1.shape[0]] = anchors1
    x2[: anchors2.shape[0]] = anchors2
    x3[: anchors3.shape[0]] = anchors3
    return x1, x2, x3


def sample_scatter_top(n_bc):
    # 齐次散射边界以全时域固定采样为主，只温和覆盖关键时段。
    n_early = int(0.04 * n_bc)
    n_defect = int(0.06 * n_bc)
    n_bottom = int(0.05 * n_bc)
    n_uniform = n_bc - n_early - n_defect - n_bottom
    return _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        rand_window(0.0, 1.20 * T_PULSE_STAR, n_early),
        sample_time_around(
            [T_NECK_START_REFLECT_STAR, T_NECK_END_REFLECT_STAR],
            n_defect,
            half_width=0.10,
        ),
        sample_time_around(
            [T_BOTTOM_REFLECT_STAR], n_bottom, half_width=0.12
        ),
    )


def sample_scatter_bottom(n_bc):
    n_arrive = int(0.12 * n_bc)
    n_uniform = n_bc - n_arrive
    return _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        sample_time_around([T_BOTTOM_ARRIVE_STAR], n_arrive, half_width=0.12),
    )


def sample_interface_time(n_if, arrive_center, reflect_centers):
    n_arrive = int(0.15 * n_if)
    n_reflect = int(0.10 * n_if)
    n_uniform = n_if - n_arrive - n_reflect
    return _shuffle_cat(
        torch.rand(n_uniform, 1, device=device) * T_STAR_MAX,
        sample_time_around([arrive_center], n_arrive, half_width=0.08),
        sample_time_around(reflect_centers, n_reflect, half_width=0.10),
    )


def make_scatter_batch(n_pde=14000, n_ic=1800, n_bc=1500, n_if=1800):
    n2 = int(0.24 * n_pde)
    n1 = int(0.38 * n_pde)
    n3 = n_pde - n1 - n2
    ic1, ic2, ic3 = sample_scatter_ic(n_ic)

    return {
        "pde_1": make_segment1_pde(n1),
        "pde_2": make_segment2_pde(n2),
        "pde_3": make_segment3_pde(n3),
        "ic_1": ic1,
        "ic_2": ic2,
        "ic_3": ic3,
        "bc_top_t": sample_scatter_top(n_bc),
        "bc_bot_t": sample_scatter_bottom(n_bc),
        "interface_start_t": sample_interface_time(
            n_if,
            T_NECK_START_ARRIVE_STAR,
            [T_NECK_START_REFLECT_STAR, T_NECK_END_REFLECT_STAR],
        ),
        "interface_end_t": sample_interface_time(
            n_if,
            T_NECK_END_ARRIVE_STAR,
            [T_NECK_START_REFLECT_STAR, T_NECK_END_REFLECT_STAR],
        ),
    }


def scatter_weights(epoch):
    w_ic_v = 120.0 if epoch < 15000 else 100.0
    return {
        "pde": 50.0,
        "pde_1": 0.40,
        "pde_2": 0.20,
        "pde_3": 0.40,
        "ic_u": 100.0,
        "ic_v": w_ic_v,
        "bc_top": 100.0,
        "bc_bot": 100.0,
        "interface": 150.0,
        "interface_u": 0.5,
        "interface_v": 1.0,
        "interface_force": 2.0,
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
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def record_scatter_validation(model, epoch, values):
    h = model.history
    h["val_epoch"].append(int(epoch))
    for key in (
        "total", "pde", "pde_1", "pde_2", "pde_3",
        "ic_u", "ic_v", "bc_top", "bc_bot", "interface",
        "interface_u", "interface_v", "interface_force",
    ):
        h[f"val_{key}"].append(float(values[key]))


def train_scatter_model(model):
    print("\n==> 开始训练缩颈散射场 ...")
    print("    背景网络已冻结；仅更新 net1/net2/net3 散射网络。")
    print("    采样：全程固定、全域为主、关键时段温和加密。")
    print("    梯度裁剪：完全取消。")

    train_batch = make_scatter_batch(14000, 1800, 1500, 1800)
    val_batch = make_scatter_batch(5000, 900, 800, 900)

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1.5e-3)

    best_val = float("inf")
    best_state = None
    start = default_timer()

    for epoch in range(SCATTER_ADAM_EPOCHS):
        model.train()
        model.background.eval()
        lr_now = set_scatter_lr(optimizer, epoch)
        weights = scatter_weights(epoch)

        optimizer.zero_grad()
        values = model.loss(train_batch, weights, record=True, epoch=epoch)
        values["total"].backward()
        # 按用户要求：这里不调用任何梯度裁剪函数。
        optimizer.step()

        if epoch % SCATTER_VAL_INTERVAL == 0 or epoch == SCATTER_ADAM_EPOCHS - 1:
            val = evaluate_scatter(model, val_batch, weights)
            record_scatter_validation(model, epoch, val)

            if val["total"] < best_val:
                best_val = val["total"]
                best_state = {
                    "net1": copy.deepcopy(model.net1.state_dict()),
                    "net2": copy.deepcopy(model.net2.state_dict()),
                    "net3": copy.deepcopy(model.net3.state_dict()),
                }
                torch.save(
                    best_state,
                    os.path.join(RESULT_DIR, "best_scatter_adam_state.pth"),
                )

            print(
                f"[散射 Adam] epoch={epoch:5d} lr={lr_now:.1e} "
                f"train={values['total'].item():.3e} val={val['total']:.3e}"
            )
            print(
                f"    Val PDE={val['pde']:.3e} "
                f"PDE1={val['pde_1']:.3e} PDE2={val['pde_2']:.3e} "
                f"PDE3={val['pde_3']:.3e} IF={val['interface']:.3e}"
            )

    if best_state is not None:
        model.net1.load_state_dict(best_state["net1"])
        model.net2.load_state_dict(best_state["net2"])
        model.net3.load_state_dict(best_state["net3"])

    weights_final = scatter_weights(SCATTER_ADAM_EPOCHS)
    before_state = {
        "net1": copy.deepcopy(model.net1.state_dict()),
        "net2": copy.deepcopy(model.net2.state_dict()),
        "net3": copy.deepcopy(model.net3.state_dict()),
    }
    before_val = evaluate_scatter(model, val_batch, weights_final)

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
        values = model.loss(train_batch, weights_final, record=False)
        values["total"].backward()
        return values["total"]

    optimizer_lbfgs.step(closure)
    after_val = evaluate_scatter(model, val_batch, weights_final)

    accept_lbfgs = (
        after_val["total"] < before_val["total"]
        and after_val["pde"] <= 1.05 * before_val["pde"]
    )

    if accept_lbfgs:
        print("L-BFGS 被接受：验证总损失下降且验证 PDE 未明显恶化。")
        final_choice = "LBFGS_ACCEPTED"
    else:
        print("L-BFGS 未通过验证条件，恢复 Adam 最优散射模型。")
        model.net1.load_state_dict(before_state["net1"])
        model.net2.load_state_dict(before_state["net2"])
        model.net3.load_state_dict(before_state["net3"])
        after_val = before_val
        final_choice = "ADAM_RESTORED"

    elapsed = default_timer() - start
    print(f"散射场训练完成，用时 {elapsed:.2f} s")

    summary_path = os.path.join(RESULT_DIR, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"background_frozen=True\n")
        f.write(f"gradient_clipping=False\n")
        f.write(f"final_choice={final_choice}\n")
        f.write(f"before_lbfgs_val_total={before_val['total']:.12e}\n")
        f.write(f"after_or_final_val_total={after_val['total']:.12e}\n")
        f.write(f"before_lbfgs_val_pde={before_val['pde']:.12e}\n")
        f.write(f"after_or_final_val_pde={after_val['pde']:.12e}\n")

    return model


# =========================
# 8) Prediction and diagnostics
# =========================
def top_velocity_components(model, nt=1800):
    model.eval()
    t_star = torch.linspace(0.0, T_STAR_MAX, nt, device=device).view(-1, 1)

    # 背景速度
    xb = torch.zeros_like(t_star).requires_grad_(True)
    tb = t_star.clone().detach().requires_grad_(True)
    u0 = model.background(xb, tb)
    u0_t = grad(u0, tb, torch.ones_like(u0), create_graph=False)[0]

    # 散射速度
    xs = torch.zeros_like(t_star).requires_grad_(True)
    ts = t_star.clone().detach().requires_grad_(True)
    s = model.net1(xs, ts)
    s_t = grad(s, ts, torch.ones_like(s), create_graph=False)[0]

    factor = U_REF / T_REF
    v0_m_s = factor * u0_t.detach().cpu().numpy().reshape(-1)
    vs_m_s = factor * s_t.detach().cpu().numpy().reshape(-1)
    vt_m_s = v0_m_s + vs_m_s
    t_ms = T_REF * t_star.detach().cpu().numpy().reshape(-1) * 1e3
    return t_ms, v0_m_s, vs_m_s, vt_m_s


def check_initial_total_velocity(model, nx=300):
    x = torch.linspace(0.0, 1.0, nx, device=device).view(-1, 1).requires_grad_(True)
    t = torch.zeros_like(x).requires_grad_(True)
    u = model(x, t)
    u_t = grad(u, t, torch.ones_like(u), create_graph=False)[0]
    v_mm_s = (U_REF / T_REF) * u_t.detach().cpu().numpy().reshape(-1) * 1e3
    print(
        f"初始总速度检查: max|v(x,0)| = {np.max(np.abs(v_mm_s)):.6e} mm/s, "
        f"v(0,0) = {v_mm_s[0]:.6e} mm/s"
    )


def predict_total_grid(model, nx=220, nt=360):
    x = torch.linspace(0.0, 1.0, nx, device=device)
    t = torch.linspace(0.0, T_STAR_MAX, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    with torch.no_grad():
        u = model(xf, tf).reshape(nx, nt)
    return X.cpu().numpy(), T.cpu().numpy(), u.cpu().numpy()


def predict_total_velocity_grid(model, nx=180, nt=300):
    x = torch.linspace(0.0, 1.0, nx, device=device)
    t = torch.linspace(0.0, T_STAR_MAX, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")
    xf = X.reshape(-1, 1).requires_grad_(True)
    tf = T.reshape(-1, 1).requires_grad_(True)
    u = model(xf, tf)
    u_t = grad(u, tf, torch.ones_like(u), create_graph=False)[0]
    return (
        X.cpu().numpy(),
        T.cpu().numpy(),
        u_t.detach().cpu().numpy().reshape(nx, nt),
    )


def plot_loss_histories(model):
    h = model.history
    epochs = h["train_epoch"]

    plt.figure(figsize=(13, 6.8))
    plt.semilogy(epochs, h["total"], label="Train total", linewidth=1.8)
    plt.semilogy(epochs, h["pde"], label="Train PDE")
    plt.semilogy(epochs, h["ic_u"], label="Train IC displacement")
    plt.semilogy(epochs, h["ic_v"], label="Train IC velocity")
    plt.semilogy(epochs, h["bc_top"], label="Train top BC")
    plt.semilogy(epochs, h["bc_bot"], label="Train bottom BC")
    plt.semilogy(epochs, h["interface"], label="Train interface")
    if h["val_epoch"]:
        plt.semilogy(
            h["val_epoch"], h["val_total"], "--", linewidth=2.0,
            label="Validation total"
        )
        plt.semilogy(
            h["val_epoch"], h["val_pde"], "--", linewidth=1.5,
            label="Validation PDE"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss, log scale")
    plt.title("Training and validation loss - background + scattering PINN")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "loss_history_train_val.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(13, 6.5))
    plt.semilogy(epochs, h["pde_1"], label="Train PDE - segment 1")
    plt.semilogy(epochs, h["pde_2"], label="Train PDE - necking segment")
    plt.semilogy(epochs, h["pde_3"], label="Train PDE - segment 3")
    plt.semilogy(epochs, h["interface_u"], label="Interface displacement")
    plt.semilogy(epochs, h["interface_v"], label="Interface velocity")
    plt.semilogy(epochs, h["interface_force"], label="Interface force")
    plt.xlabel("Epoch")
    plt.ylabel("Loss, log scale")
    plt.title("Segment PDE and interface loss")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pde_interface_loss.png"), dpi=300)
    plt.close()

    if h["val_epoch"]:
        plt.figure(figsize=(13, 6.5))
        for key, label in (
            ("val_total", "Validation total"),
            ("val_pde", "Validation PDE"),
            ("val_ic_u", "Validation IC displacement"),
            ("val_ic_v", "Validation IC velocity"),
            ("val_bc_top", "Validation top BC"),
            ("val_bc_bot", "Validation bottom BC"),
            ("val_interface", "Validation interface"),
        ):
            plt.semilogy(h["val_epoch"], h[key], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss, log scale")
        plt.title("Validation loss history - abrupt necking pile")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULT_DIR, "validation_loss_history.png"), dpi=300
        )
        plt.close()


def plot_top_velocity(model):
    t_ms, v0, vs, vt = top_velocity_components(model, nt=1800)
    plt.figure(figsize=(13, 5.8))
    plt.plot(t_ms, vt * 1e3, linewidth=2.0, label="Total PINN")
    plt.plot(t_ms, v0 * 1e3, "--", linewidth=1.5, label="Frozen intact background")
    plt.plot(t_ms, vs * 1e3, ":", linewidth=1.5, label="Scattering correction")
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile-head velocity / mm/s")
    plt.title("Pile-head velocity - abrupt necking pile")
    plt.xlim(0.0, T_OBS * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "top_velocity_components.png"), dpi=300)
    plt.close()


def plot_displacement_contour(X, T, u_star):
    u_mm = U_REF * u_star * 1e3
    x_m = X * L
    t_ms = T * T_REF * 1e3
    plt.figure(figsize=(12.5, 6.2))
    levels = np.linspace(np.min(u_mm), np.max(u_mm), 100)
    cs = plt.contourf(t_ms, x_m, u_mm, levels=levels, cmap="RdBu_r")
    plt.colorbar(cs, label="Displacement u(x,t) / mm")
    plt.axhspan(
        X_NECK_START_STAR * L,
        X_NECK_END_STAR * L,
        alpha=0.18,
        label="Abrupt necking region",
    )
    plt.axhline(X_NECK_START_STAR * L, linestyle="--", linewidth=1.1)
    plt.axhline(X_NECK_END_STAR * L, linestyle="--", linewidth=1.1)
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile position x / m")
    plt.title("Displacement contour - abrupt necking pile")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "displacement_contour.png"), dpi=300)
    plt.close()


def plot_velocity_contour(X, T, u_t_star):
    v_mm_s = (U_REF / T_REF) * u_t_star * 1e3
    x_m = X * L
    t_ms = T * T_REF * 1e3
    vmax = np.percentile(np.abs(v_mm_s), 99.5)
    vmax = max(vmax, 1.0e-12)
    levels = np.linspace(-vmax, vmax, 100)

    plt.figure(figsize=(12.5, 6.2))
    cs = plt.contourf(
        t_ms, x_m, v_mm_s, levels=levels, cmap="RdBu_r", extend="both"
    )
    plt.colorbar(cs, label="Velocity v(x,t) / mm/s")
    plt.axhspan(
        X_NECK_START_STAR * L,
        X_NECK_END_STAR * L,
        alpha=0.18,
        label="Abrupt necking region",
    )
    plt.axhline(X_NECK_START_STAR * L, linestyle="--", linewidth=1.1)
    plt.axhline(X_NECK_END_STAR * L, linestyle="--", linewidth=1.1)
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile position x / m")
    plt.title("Velocity contour - abrupt necking pile")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "velocity_contour.png"), dpi=300)
    plt.close()


def save_top_velocity_csv(model):
    t_ms, v0, vs, vt = top_velocity_components(model, nt=1800)
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
    path = os.path.join(RESULT_DIR, "pinn_abrupt_necking_top_velocity.csv")
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=(
            "time_s,time_ms,background_v_m_s,scatter_v_m_s,total_v_m_s,"
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
    return float(token.strip().replace("D", "E").replace("d", "E"))


def read_abaqus_rpt_segments(rpt_path):
    if rpt_path is None or not os.path.exists(rpt_path):
        return []

    pattern = re.compile(
        r"^[\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
        r"[\s,]+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
        r"[\s]*$"
    )

    segments = []
    cur_t, cur_v = [], []
    last_t = None
    with open(rpt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line)
            if match is None:
                if len(cur_t) >= 2:
                    segments.append((np.asarray(cur_t), np.asarray(cur_v)))
                cur_t, cur_v, last_t = [], [], None
                continue

            t_value = parse_abaqus_float(match.group(1))
            v_value = parse_abaqus_float(match.group(2))
            if last_t is not None and t_value < last_t - 1.0e-12:
                if len(cur_t) >= 2:
                    segments.append((np.asarray(cur_t), np.asarray(cur_v)))
                cur_t, cur_v = [], []
            cur_t.append(t_value)
            cur_v.append(v_value)
            last_t = t_value

    if len(cur_t) >= 2:
        segments.append((np.asarray(cur_t), np.asarray(cur_v)))

    clean = []
    for t, v in segments:
        mask = np.isfinite(t) & np.isfinite(v)
        t, v = t[mask], v[mask]
        if len(t) < 2:
            continue
        keep = np.ones_like(t, dtype=bool)
        keep[1:] = np.abs(np.diff(t)) > 1.0e-15
        t, v = t[keep], v[keep]
        if len(t) >= 2:
            clean.append((t, v))
    return clean


def find_abaqus_rpt():
    preferred_names = [
        "XY_pile_abrupt_necking_1D.rpt",
        "XY_pile_abrupt_necking_1D(2).rpt",
        "abaqus_abrupt_necking_top_velocity.rpt",
    ]
    search_dirs = [get_base_dir(), os.getcwd(), "/mnt/data"]
    for directory in search_dirs:
        for name in preferred_names:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate

    # 最后再搜索文件名中包含 necking 的 rpt
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            lower = name.lower()
            if lower.endswith(".rpt") and ("neck" in lower or "缩颈" in lower):
                return os.path.join(directory, name)
    return None


def plot_abaqus_comparison(model):
    rpt_path = find_abaqus_rpt()
    if rpt_path is None:
        print("未找到突变缩颈 ABAQUS .rpt，跳过对比图。")
        return

    segments = read_abaqus_rpt_segments(rpt_path)
    if not segments:
        print(f"未能从 {rpt_path} 读取有效 ABAQUS 数据，跳过对比图。")
        return

    t_ms, _, _, vt = top_velocity_components(model, nt=1800)
    plt.figure(figsize=(14, 5.8))
    first = True
    for t_s, v_m_s in segments:
        plt.plot(
            t_s * 1e3,
            v_m_s * 1e3,
            linewidth=1.8,
            label="ABAQUS 1D Truss" if first else None,
        )
        first = False
    plt.plot(t_ms, vt * 1e3, "--", linewidth=2.0, label="PINN total")
    plt.xlabel("Time t / ms")
    plt.ylabel("Pile-head velocity v / mm/s")
    plt.title("Abrupt necking pile: ABAQUS vs background-scattering PINN")
    plt.xlim(0.0, T_OBS * 1e3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "abaqus_vs_pinn_abrupt_necking.png"), dpi=300)
    plt.close()
    print(f"ABAQUS 对比文件: {rpt_path}")


# =========================
# 10) Save model
# =========================
def save_scatter_model(model, background_path):
    path = os.path.join(RESULT_DIR, "pinn_abrupt_necking_background_scattering_model.pth")
    torch.save(
        {
            "net1": model.net1.state_dict(),
            "net2": model.net2.state_dict(),
            "net3": model.net3.state_dict(),
            "background_model_path": background_path,
            "parameters": {
                "L": L,
                "D_normal": D_NORMAL,
                "D_neck": D_NECK,
                "x_neck_center": X_NECK_CENTER,
                "neck_length": NECK_LENGTH,
                "rho0": RHO0,
                "E0": E0,
                "T_obs": T_OBS,
                "T_pulse": T_PULSE,
                "p0": P0,
            },
        },
        path,
    )
    print(f"缩颈散射模型已保存: {path}")
    return path


# =========================
# 11) Main
# =========================
if __name__ == "__main__":
    save_excitation_plot()

    background_model, background_path = load_or_train_background()
    model = BackgroundScatteringNeckingPINN(background_model).to(device)

    model = train_scatter_model(model)

    check_initial_total_velocity(model)
    plot_loss_histories(model)
    plot_top_velocity(model)

    X, T, U = predict_total_grid(model, nx=220, nt=360)
    plot_displacement_contour(X, T, U)

    XV, TV, UT = predict_total_velocity_grid(model, nx=180, nt=300)
    plot_velocity_contour(XV, TV, UT)

    save_top_velocity_csv(model)
    plot_abaqus_comparison(model)
    save_scatter_model(model, background_path)

    print("\n模拟完成。")
    print(f"结果目录: {RESULT_DIR}")
