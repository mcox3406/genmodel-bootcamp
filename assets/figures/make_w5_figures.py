# Generates the Week 5 figures (couplings, optimal transport, and data-dependent
# bases): independent versus OT pairings and their endpoint costs, the
# entropic-OT plan under a regularization sweep, conditional-velocity ambiguity,
# and Gaussian versus physically-informed base distributions. Reuses
# the palette and helpers from make_figures.py and the same Gaussian -> three-mode
# toy as the Week 3/4 figures for visual continuity.
# Usage: python make_w5_figures.py   (requires matplotlib + numpy, no scipy)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from make_figures import (ACCENT, ACCENT_MID, ACCENT_SOFT, FG, MID, DIM, MUTED,
                          LINE, STREAM, DENS, clean, save)


def int_log_ticks(ax, xticks, axis="x"):
    """Replace matplotlib's cluttered log minor-tick labels with clean integer
    ticks at the data points."""
    getattr(ax, f"set_{axis}ticks")(xticks)
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(ScalarFormatter())
    a.set_minor_formatter(NullFormatter())
    getattr(ax, f"set_{axis}ticklabels")([str(t) for t in xticks])

BLUE = "#2f6fb3"
BLUE_SOFT = "#cfe0f0"
GREEN = "#1f9d6b"
GREEN_SOFT = "#c9e9db"
GOLD = "#c98a1f"

# ============================================================================
# The shared Week 3/4/5 toy: a Gaussian base and a molecule-like three-mode
# target, joined by the straight interpolant x_t = (1-t) x0 + t x1. Week 5 keeps
# the interpolant fixed and varies the *coupling* pi(x0, x1): independent pairing
# versus an exact empirical optimal-transport assignment. Sinkhorn is used only
# where the soft, entropically regularized plan itself is displayed.
# ============================================================================
MODES = np.array([[0.0, 1.15], [-1.0, -0.6], [1.0, -0.6]])
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])
S0 = 0.13                                   # per-mode std of the "atoms"
SRC = 0.9                                   # base std (isotropic Gaussian)


def sample_target(n, rng):
    k = rng.choice(len(MODES), size=n, p=WEIGHTS)
    return MODES[k] + S0 * rng.standard_normal((n, 2))


def sample_source(n, rng):
    return SRC * rng.standard_normal((n, 2))


# ---------------------------------------------------------------------------
# Entropic optimal transport by Sinkhorn iteration, on a squared-Euclidean cost.
# Returns the coupling matrix P (rows = source, cols = target) with uniform
# marginals. No scipy: plain log-domain-free scaling, which is fine at these
# small batch sizes and regularizations.
# ---------------------------------------------------------------------------
def sinkhorn(x0, x1, reg=0.05, iters=400):
    C = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    C = C / (C.max() + 1e-12)
    K = np.exp(-C / reg)
    n, m = len(x0), len(x1)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(iters):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    return u[:, None] * K * v[None, :]


def _hungarian_square(cost):
    """Return a minimum-cost one-to-one assignment for a square cost matrix.

    This is the shortest-augmenting-path form of the Hungarian algorithm.  It
    keeps this figure script NumPy-only while, unlike rowwise argmax of a soft
    Sinkhorn plan, preserving both empirical marginals exactly.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("the figure assignment requires equal-size point sets")
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            unused = np.flatnonzero(~used[1:]) + 1
            cur = cost[i0 - 1, unused - 1] - u[i0] - v[unused]
            improve = cur < minv[unused]
            minv[unused[improve]] = cur[improve]
            way[unused[improve]] = j0
            j1 = unused[np.argmin(minv[unused])]
            delta = minv[j1]
            used_idx = np.flatnonzero(used)
            u[p[used_idx]] += delta
            v[used_idx] -= delta
            minv[unused] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.empty(n, dtype=int)
    assignment[p[1:] - 1] = np.arange(n)
    return assignment


def ot_assignment(x0, x1):
    """Exact equal-weight empirical OT assignment for squared Euclidean cost."""
    cost = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    return _hungarian_square(cost)


def transport_cost(x0, x1):
    return ((x1 - x0) ** 2).sum(1).mean()


# ---------------------------------------------------------------------------
# The marginal velocity of the straight interpolant under a given coupling,
# approximated on the toy by a posterior-weighted average over paired samples.
# Used only to draw curved/straight marginal streamlines for the two couplings.
# ---------------------------------------------------------------------------
def marginal_velocity(P, pairs_x0, pairs_x1, t, h=0.16):
    """Nadaraya-Watson estimate of E[x1 - x0 | x_t = x] from a cloud of paired
    endpoints. pairs are (N,2); P is (M,2) query points at time t."""
    xt = (1 - t) * pairs_x0 + t * pairs_x1
    v = pairs_x1 - pairs_x0
    d2 = ((P[:, None, :] - xt[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * h * h))
    w /= w.sum(1, keepdims=True) + 1e-12
    return (w[..., None] * v[None, :, :]).sum(1)


def velocity_ambiguity(pairs_x0, pairs_x1, t, h=0.22):
    """Leave-one-out kernel estimate of E Var[x1-x0 | x_t].

    This is an illustrative finite-sample regression diagnostic, not a trained
    model metric or a theorem about ODE solver error.
    """
    xt = (1 - t) * pairs_x0 + t * pairs_x1
    velocity = pairs_x1 - pairs_x0
    d2 = ((xt[:, None, :] - xt[None, :, :]) ** 2).sum(-1)
    weights = np.exp(-d2 / (2 * h * h))
    np.fill_diagonal(weights, 0.0)
    weights /= weights.sum(1, keepdims=True) + 1e-12
    conditional_mean = weights @ velocity
    return ((velocity - conditional_mean) ** 2).sum(1).mean()


def integrate(P0, pairs_x0, pairs_x1, n_steps=60):
    ts = np.linspace(1e-2, 1 - 1e-2, n_steps)
    x = P0.copy()
    traj = [x.copy()]
    for i in range(n_steps - 1):
        dt = ts[i + 1] - ts[i]
        x = x + dt * marginal_velocity(x, pairs_x0, pairs_x1, ts[i])
        traj.append(x.copy())
    return np.array(traj)


def target_density(P):
    d2 = ((P[:, None, :] - MODES[None, :, :]) ** 2).sum(-1)
    s2 = S0 ** 2 + 0.02
    return (WEIGHTS[None, :] * np.exp(-d2 / (2 * s2))).sum(1)


def dens_panel(ax, lim=2.3):
    g = np.linspace(-lim, lim, 240)
    X, Y = np.meshgrid(g, g)
    Pd = np.stack([X.ravel(), Y.ravel()], 1)
    Z = target_density(Pd).reshape(X.shape)
    ax.imshow(Z, extent=[-lim, lim, -lim, lim], origin="lower", cmap=DENS, aspect="auto")


def frame(ax, lim=2.3, ticks=False):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    clean(ax, ticks=ticks)


def count_crossings(x0, x1):
    """Number of crossing segment pairs among the straight lines x0[i]->x1[i]."""
    def ccw(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > \
               (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
    n = len(x0)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            A, B, C, D = x0[i], x1[i], x0[j], x1[j]
            if (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D)):
                c += 1
    return c


# ============================================================================
# Overview (main page): independent pairing | exact OT assignment | the measured
# conditional-path energy of those same two empirical couplings.
# ============================================================================
def fig_overview():
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    n = 9
    x0 = sample_source(n, rng)
    x1 = sample_target(n, rng)

    # Panel 1: independent coupling (as sampled), crossing lines.
    ax = axes[0]
    dens_panel(ax)
    cr_i = count_crossings(x0, x1)
    for a, b in zip(x0, x1):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=1.1, alpha=0.85, zorder=3)
    ax.scatter(x0[:, 0], x0[:, 1], s=24, color=DIM, zorder=4, label=r"base $x_0$")
    ax.scatter(x1[:, 0], x1[:, 1], s=28, color=ACCENT, zorder=4, label=r"data $x_1$")
    ax.set_title(f"independent  ({cr_i} segment intersections)", color=FG)
    ax.legend(loc="lower center", fontsize=7, frameon=False, ncol=2)
    frame(ax)

    # Panel 2: exact equal-weight empirical OT assignment.
    ax = axes[1]
    dens_panel(ax)
    perm = ot_assignment(x0, x1)
    x1o = x1[perm]
    cr_o = count_crossings(x0, x1o)
    for a, b in zip(x0, x1o):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.3, alpha=0.9, zorder=3)
    ax.scatter(x0[:, 0], x0[:, 1], s=24, color=DIM, zorder=4)
    ax.scatter(x1o[:, 0], x1o[:, 1], s=28, color=ACCENT, zorder=4)
    ax.set_title(f"exact OT  ({cr_o} segment intersection{'s' if cr_o != 1 else ''})", color=FG)
    frame(ax)

    # Panel 3: a quantity computed from exactly the displayed endpoint pairs.
    ax = axes[2]
    values = [transport_cost(x0, x1), transport_cost(x0, x1o)]
    ax.bar([0, 1], values, color=[MUTED, BLUE], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["independent", "exact OT"], fontsize=8)
    ax.set_ylabel(r"conditional-path energy  $\mathbb{E}|x_1-x_0|^2$")
    ax.set_title("OT minimizes endpoint displacement", color=FG)
    for i, value in enumerate(values):
        ax.text(i, value + 0.03 * max(values), f"{value:.2f}", ha="center", fontsize=8, color=FG)
    clean(ax, ticks=True)

    save(fig, "w5-overview.png")


# ============================================================================
# Problem 1: exact empirical assignment and endpoint cost (left) | kernel
# estimates of the marginal fields induced by two empirical couplings (right).
# ============================================================================
def fig_p1():
    rng = np.random.default_rng(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    n = 8
    x0 = sample_source(n, rng)
    x1 = sample_target(n, rng)

    ax1.set_title("independent vs OT pairing of the same points", color=FG)
    dens_panel(ax1)
    for a, b in zip(x0, x1):
        ax1.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=1.0, alpha=0.6, zorder=2)
    perm = ot_assignment(x0, x1)
    for a, b in zip(x0, x1[perm]):
        ax1.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.5, alpha=0.95, zorder=3)
    ax1.scatter(x0[:, 0], x0[:, 1], s=24, color=DIM, zorder=4)
    ax1.scatter(x1[:, 0], x1[:, 1], s=28, color=ACCENT, zorder=4)
    ax1.plot([], [], color=MUTED, lw=1.4, label=f"independent ({transport_cost(x0, x1):.2f})")
    ax1.plot([], [], color=BLUE, lw=1.6, label=f"OT ({transport_cost(x0, x1[perm]):.2f})")
    ax1.legend(loc="lower center", fontsize=7.2, frameon=False, ncol=2, title="mean $|x_1-x_0|^2$")
    frame(ax1)

    # marginal streamlines at t = 0.5 for the two couplings, from denser clouds
    N = 900
    a0 = sample_source(N, rng)
    a1 = sample_target(N, rng)
    pm = ot_assignment(a0[:400], a1[:400])
    lim = 2.3
    g = np.linspace(-lim, lim, 20)
    X, Y = np.meshgrid(g, g)
    Pq = np.stack([X.ravel(), Y.ravel()], 1)
    U_ind = marginal_velocity(Pq, a0, a1, 0.5)
    U_ot = marginal_velocity(Pq, a0[:400], a1[:400][pm], 0.5)
    ax2.streamplot(X, Y, U_ind[:, 0].reshape(X.shape), U_ind[:, 1].reshape(X.shape),
                   color=STREAM, density=1.0, linewidth=0.6, arrowsize=0.6)
    ax2.streamplot(X, Y, U_ot[:, 0].reshape(X.shape), U_ot[:, 1].reshape(X.shape),
                   color=BLUE, density=1.0, linewidth=0.8, arrowsize=0.7)
    ax2.scatter(MODES[:, 0], MODES[:, 1], s=60, color=ACCENT, zorder=4)
    ax2.set_title("kernel estimate of $b_{1/2}$ under two couplings", color=FG)
    frame(ax2)

    save(fig, "w5p1.png")


# ============================================================================
# Problem 2: Sinkhorn convergence to the marginals (left) | minibatch OT cost
# and finite-batch dependence of endpoint cost (right).
# ============================================================================
def fig_p2():
    rng = np.random.default_rng(2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    # Panel 1: entropic OT plan as a heatmap for three regularizations.
    x0 = sample_source(24, rng)
    x1 = sample_target(24, rng)
    order = np.argsort(np.arctan2(x1[:, 1], x1[:, 0]))
    x1 = x1[order]
    reg_list = [0.3, 0.05, 0.008]
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    for i, reg in enumerate(reg_list):
        P = sinkhorn(x0, x1, reg=reg)
        axin = ax1.inset_axes([0.02 + i * 0.335, 0.12, 0.30, 0.76])
        axin.imshow(P, cmap=DENS, aspect="auto")
        axin.set_title(fr"$\varepsilon={reg}$", fontsize=8, color=MID)
        axin.set_xticks([]); axin.set_yticks([])
        for s in axin.spines.values():
            s.set_color(LINE)
    ax1.set_title("entropic OT plan sharpens as $\\varepsilon\\to0$", color=FG)
    ax1.axis("off")

    # Panel 2: minibatch OT transport cost vs batch size, approaching population.
    batch = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    rng2 = np.random.default_rng(9)
    cost_ind, cost_ot = [], []
    for bs in batch:
        ci, co = [], []
        for _ in range(40):
            a = sample_source(bs, rng2)
            b = sample_target(bs, rng2)
            ci.append(transport_cost(a, b))
            pm = ot_assignment(a, b)
            co.append(transport_cost(a, b[pm]))
        cost_ind.append(np.mean(ci))
        cost_ot.append(np.mean(co))
    ax2.semilogx(batch, cost_ind, "o-", color=MUTED, lw=1.6, ms=4, label="independent")
    ax2.semilogx(batch, cost_ot, "s-", color=BLUE, lw=1.6, ms=4, label="minibatch OT")
    ax2.axhline(cost_ot[-1], color=DIM, ls=":", lw=0.9)
    ax2.text(2.2, cost_ot[-1] * 1.06, "largest-batch estimate (visual guide)",
             fontsize=6.6, color=DIM)
    ax2.set_xlabel("minibatch size")
    ax2.set_ylabel(r"mean transport cost $|x_1-x_0|^2$")
    ax2.set_title("minibatch OT: cost depends on batch size", color=FG)
    ax2.legend(fontsize=7.5, frameon=False)
    clean(ax2, ticks=True)
    int_log_ticks(ax2, [2, 8, 32, 128])

    save(fig, "w5p2.png")


# ============================================================================
# Problem 3: kernel-estimated marginal trajectories (left) and a measured
# conditional-velocity ambiguity diagnostic (right). No neural network is fit.
# ============================================================================
def fig_p3():
    rng = np.random.default_rng(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    # trajectories under the two couplings from the same base points
    N = 700
    a0 = sample_source(N, rng)
    a1 = sample_target(N, rng)
    pairs0 = a0[:350]
    pairs1_ind = a1[:350]
    pm = ot_assignment(pairs0, pairs1_ind)
    pairs1_ot = pairs1_ind[pm]
    start = sample_source(10, rng)
    tr_ind = integrate(start.copy(), pairs0, pairs1_ind)
    tr_ot = integrate(start.copy(), pairs0, pairs1_ot)
    dens_panel(ax1)
    for j in range(start.shape[0]):
        ax1.plot(tr_ind[:, j, 0], tr_ind[:, j, 1], color=MUTED, lw=1.0, alpha=0.8, zorder=3)
        ax1.plot(tr_ot[:, j, 0], tr_ot[:, j, 1], color=BLUE, lw=1.1, alpha=0.9, zorder=3)
    ax1.scatter(start[:, 0], start[:, 1], s=16, color=DIM, zorder=4)
    ax1.plot([], [], color=MUTED, lw=1.4, label="independent pairs")
    ax1.plot([], [], color=BLUE, lw=1.6, label="exact OT pairs")
    ax1.set_title("kernel-estimated trajectories, same starts", color=FG)
    ax1.legend(loc="lower center", fontsize=7.2, frameon=False, ncol=2)
    frame(ax1)

    ts = np.linspace(0.1, 0.9, 9)
    ambiguity_ind = [velocity_ambiguity(pairs0, pairs1_ind, t) for t in ts]
    ambiguity_ot = [velocity_ambiguity(pairs0, pairs1_ot, t) for t in ts]
    ax2.plot(ts, ambiguity_ind, "o-", color=MUTED, lw=1.6, ms=4, label="independent")
    ax2.plot(ts, ambiguity_ot, "s-", color=BLUE, lw=1.6, ms=4, label="exact OT")
    ax2.set_xlabel("interpolation time $t$")
    ax2.set_ylabel(r"kernel estimate of $\mathbb{E}\,\mathrm{Var}[\dot X_t\mid X_t]$")
    ax2.set_title("conditional-velocity ambiguity", color=FG)
    ax2.legend(fontsize=7.5, frameon=False)
    clean(ax2, ticks=True)

    save(fig, "w5p3.png")


# ---------- torus helpers (shared with Week 3/4 Problem 4 geometry) ----------
def wrap(u):
    return (u + 0.5) % 1.0 - 0.5


def to_cell(u):
    return u % 1.0


def torus_assignment(x0, x1):
    """Exact equal-weight assignment using minimum-image squared distance."""
    displacement = wrap(x1[None, :, :] - x0[:, None, :])
    return _hungarian_square((displacement ** 2).sum(-1))


# ============================================================================
# Problem 4: Gaussian base + independent coupling (left) vs a data-dependent
# base + OT coupling (center) on the torus, and the transport-cost saving
# (right). The base carries the composition/coarse structure; only the short
# high-fidelity correction is learned.
# ============================================================================
def fig_p4():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    sites = np.array([[0.5, 0.5], [0.08, 0.9], [0.9, 0.12], [0.5, 0.06], [0.28, 0.42]])

    def draw_cell(ax, title):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); clean(ax)
        for v in (0, 1):
            ax.axvline(v, color=LINE, lw=1); ax.axhline(v, color=LINE, lw=1)
        ax.set_title(title, color=FG)

    def torus_lines(ax, a_pts, b_pts, color, lw):
        tot = 0.0
        for a, b in zip(a_pts, b_pts):
            d = wrap(b - a)
            tot += (d ** 2).sum()
            ax.plot([a[0], a[0] + d[0]], [a[1], a[1] + d[1]], color=color, lw=lw, alpha=0.9)
        return tot / len(a_pts)

    # Panel 1: Gaussian/uniform base, independent coupling: long crossing moves.
    ax = axes[0]
    draw_cell(ax, "uniform base, independent coupling")
    base = rng.random((len(sites), 2))
    c_ind = torus_lines(ax, base, sites, MUTED, 1.1)
    ax.scatter(base[:, 0], base[:, 1], s=24, color=MUTED, zorder=4, label="uniform base")
    ax.scatter(sites[:, 0], sites[:, 1], s=42, color=ACCENT, marker="*", zorder=4, label="high-fidelity")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6.8, frameon=False, ncol=2)

    # Panel 2: informed coarse base plus exact minimum-image OT assignment.
    ax = axes[1]
    draw_cell(ax, "coarse base, OT coupling")
    coarse = to_cell(sites + 0.07 * rng.standard_normal(sites.shape))
    perm = torus_assignment(coarse, sites)
    c_ot = torus_lines(ax, coarse, sites[perm], BLUE, 1.7)
    ax.scatter(coarse[:, 0], coarse[:, 1], s=24, color=BLUE, zorder=4, label="coarse / low-fidelity")
    ax.scatter(sites[:, 0], sites[:, 1], s=42, color=ACCENT, marker="*", zorder=4, label="high-fidelity")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6.8, frameon=False, ncol=2)

    # Panel 3: transport cost bar comparison.
    ax = axes[2]
    labels = ["uniform base\nindependent", "coarse base\nOT"]
    vals = [c_ind, c_ot]
    ax.bar([0, 1], vals, color=[MUTED, BLUE], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8, color=MID)
    ax.set_ylabel("mean squared displacement on the torus")
    ax.set_title("an informed base shortens the correction", color=FG)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=FG)
    clean(ax, ticks=True)
    ax.set_xlim(-0.6, 1.6)

    save(fig, "w5p4.png")


# ============================================================================
# Notes figures
# ============================================================================
def fig_notes_couplings():
    rng = np.random.default_rng(11)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    n = 10
    x0 = sample_source(n, rng)
    x1 = sample_target(n, rng)

    dens_panel(ax1)
    cr = count_crossings(x0, x1)
    for a, b in zip(x0, x1):
        ax1.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=1.1, alpha=0.85, zorder=3)
    ax1.scatter(x0[:, 0], x0[:, 1], s=22, color=DIM, zorder=4)
    ax1.scatter(x1[:, 0], x1[:, 1], s=26, color=ACCENT, zorder=4)
    ax1.set_title(f"independent: {cr} segment intersections, cost {transport_cost(x0, x1):.2f}", color=FG)
    frame(ax1)

    dens_panel(ax2)
    perm = ot_assignment(x0, x1)
    x1o = x1[perm]
    cro = count_crossings(x0, x1o)
    for a, b in zip(x0, x1o):
        ax2.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.3, alpha=0.9, zorder=3)
    ax2.scatter(x0[:, 0], x0[:, 1], s=22, color=DIM, zorder=4)
    ax2.scatter(x1o[:, 0], x1o[:, 1], s=26, color=ACCENT, zorder=4)
    ax2.set_title(f"exact OT: {cro} segment intersection{'s' if cro != 1 else ''}, "
                  f"cost {transport_cost(x0, x1o):.2f}", color=FG)
    frame(ax2)
    save(fig, "w5n-couplings.png")


def fig_notes_sinkhorn():
    rng = np.random.default_rng(12)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    x0 = sample_source(28, rng)
    x1 = sample_target(28, rng)
    order = np.argsort(np.arctan2(x1[:, 1] - x1[:, 1].mean(), x1[:, 0] - x1[:, 0].mean()))
    x1 = x1[order]
    for ax, reg, lab in zip(axes, [0.5, 0.05, 0.006],
                            [r"$\varepsilon=0.5$ (blurred)",
                             r"$\varepsilon=0.05$",
                             r"$\varepsilon=0.006$ (sharp plan)"]):
        P = sinkhorn(x0, x1, reg=reg)
        ax.imshow(P, cmap=DENS, aspect="auto")
        ax.set_title(lab, color=FG, fontsize=9.5)
        ax.set_xlabel("target index", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("source index", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(LINE)
    save(fig, "w5n-sinkhorn.png")


def fig_notes_straight():
    rng = np.random.default_rng(13)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    N = 700
    a0 = sample_source(N, rng)
    a1 = sample_target(N, rng)
    pm = ot_assignment(a0[:350], a1[:350])
    start = sample_source(11, rng)
    tr_ind = integrate(start.copy(), a0, a1)
    tr_ot = integrate(start.copy(), a0[:350], a1[:350][pm])
    dens_panel(ax1)
    for j in range(start.shape[0]):
        ax1.plot(tr_ind[:, j, 0], tr_ind[:, j, 1], color=MUTED, lw=1.0, alpha=0.85, zorder=3)
    ax1.scatter(start[:, 0], start[:, 1], s=16, color=DIM, zorder=4)
    ax1.set_title("independent: kernel-estimated marginal flow", color=FG)
    frame(ax1)
    dens_panel(ax2)
    for j in range(start.shape[0]):
        ax2.plot(tr_ot[:, j, 0], tr_ot[:, j, 1], color=BLUE, lw=1.1, alpha=0.9, zorder=3)
    ax2.scatter(start[:, 0], start[:, 1], s=16, color=DIM, zorder=4)
    ax2.set_title("exact OT: kernel-estimated marginal flow", color=FG)
    frame(ax2)
    save(fig, "w5n-straight.png")


def fig_notes_bases():
    rng = np.random.default_rng(14)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    # Left: Gaussian base far from data, long transport.
    dens_panel(ax1)
    x0 = sample_source(9, rng)
    x1 = sample_target(9, rng)
    perm = ot_assignment(x0, x1)
    for a, b in zip(x0, x1[perm]):
        ax1.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=1.1, alpha=0.85)
    ax1.scatter(x0[:, 0], x0[:, 1], s=22, color=DIM, zorder=4, label=r"Gaussian base $\rho_0$")
    ax1.scatter(x1[:, 0], x1[:, 1], s=26, color=ACCENT, zorder=4, label="data")
    ax1.set_title(f"Gaussian base: cost {transport_cost(x0, x1[perm]):.2f}", color=FG)
    ax1.legend(loc="lower center", fontsize=7, frameon=False, ncol=2)
    frame(ax1)

    # Right: data-dependent base sitting near the data modes, short transport.
    dens_panel(ax2)
    # base = perturbed data modes (a physics-informed / low-fidelity ensemble)
    k = rng.choice(len(MODES), size=9, p=WEIGHTS)
    base = MODES[k] + 0.28 * rng.standard_normal((9, 2))
    x1b = sample_target(9, rng)
    perm = ot_assignment(base, x1b)
    for a, b in zip(base, x1b[perm]):
        ax2.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.3, alpha=0.9)
    ax2.scatter(base[:, 0], base[:, 1], s=22, color=BLUE, zorder=4, label=r"informed base $\rho_0$")
    ax2.scatter(x1b[:, 0], x1b[:, 1], s=26, color=ACCENT, zorder=4, label="data")
    ax2.set_title(f"data-dependent base: cost {transport_cost(base, x1b[perm]):.2f}", color=FG)
    ax2.legend(loc="lower center", fontsize=7, frameon=False, ncol=2)
    frame(ax2)
    save(fig, "w5n-bases.png")


if __name__ == "__main__":
    fig_overview()
    fig_p1()
    fig_p2()
    fig_p3()
    fig_p4()
    fig_notes_couplings()
    fig_notes_sinkhorn()
    fig_notes_straight()
    fig_notes_bases()
    print("all Week 5 figures written")
