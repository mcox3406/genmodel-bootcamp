# Generates the Week 3 figures (flow matching, rectified flow, couplings, and
# Riemannian/torus flow matching) in the site color theme. Reuses the palette
# and helpers from make_figures.py.
# Usage: python make_w3_figures.py   (requires matplotlib + numpy, no scipy)
import numpy as np
import matplotlib.pyplot as plt
from make_figures import (ACCENT, ACCENT_MID, ACCENT_SOFT, FG, MID, DIM, MUTED,
                          LINE, STREAM, DENS, clean, save)

BLUE = "#2f6fb3"
BLUE_SOFT = "#cfe0f0"
GREEN = "#1f9d6b"
GREEN_SOFT = "#c9e9db"
GOLD = "#c98a1f"

# ============================================================================
# The Week 3 toy: a Gaussian source and a molecule-like three-mode target,
# joined by the straight conditional (OT) interpolant x_t = (1-t) x0 + t x1.
# Because the target is a finite point cloud, the marginal velocity field is a
# softmax-weighted average of per-sample conditional velocities, available in
# closed form. This makes every flow-matching figure exact rather than trained,
# mirroring the closed-form mixture score used in the Week 2 figures.
# ============================================================================
MODES = np.array([[0.0, 1.15], [-1.0, -0.6], [1.0, -0.6]])
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])
S0 = 0.14                                   # per-mode std of the "atoms"
SRC = 0.9                                   # source std (isotropic Gaussian)


def sample_target(n, rng):
    k = rng.choice(len(MODES), size=n, p=WEIGHTS)
    return MODES[k] + S0 * rng.standard_normal((n, 2))


def sample_source(n, rng):
    return SRC * rng.standard_normal((n, 2))


def clip_norm(v, vmax):
    n = np.hypot(v[:, 0], v[:, 1])
    s = np.minimum(1.0, vmax / np.maximum(n, 1e-12))
    return v * s[:, None]


def marginal_velocity(x, t, data, eps=1e-2):
    """Exact marginal FM velocity for the straight interpolant with a Gaussian
    source and a finite target cloud `data`. Given x1=data_k, the conditional
    path is x_t ~ N(t x1, s(t)^2 I) with s(t)^2 = (1-t)^2 SRC^2 + t^2 S0^2, and
    the conditional velocity is u(x|x1) = (x1 - x)/(1-t) shifted by the source
    mean-zero draw. We use the posterior over x1 to average conditional
    velocities: u(x,t) = sum_k w_k (x1_k - x)/(1-t)."""
    tt = np.clip(t, 0.0, 1.0 - eps)
    var = (1 - tt) ** 2 * SRC ** 2 + tt ** 2 * S0 ** 2
    d2 = ((x[:, None, :] - tt * data[None, :, :]) ** 2).sum(-1)     # (M, K)
    logw = -d2 / (2 * var)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)                  # (M, K)
    x1_hat = (w[..., None] * data[None, :, :]).sum(1)              # E[x1 | x_t]
    return (x1_hat - x) / (1 - tt)


def integrate(x, data, n_steps=60, t0=0.0, t1=0.99, vmax=14.0):
    """Heun integration of the marginal-velocity ODE from t0 to t1. The velocity
    is norm-capped so the (1-t) singularity of the straight interpolant does not
    make coarse-step trajectories overshoot the plotting window."""
    ts = np.linspace(t0, t1, n_steps + 1)
    traj = [x.copy()]
    for i in range(n_steps):
        t, tn = ts[i], ts[i + 1]
        h = tn - t
        k1 = clip_norm(marginal_velocity(x, t, data), vmax)
        xe = x + h * k1
        k2 = clip_norm(marginal_velocity(xe, tn, data), vmax)
        x = x + 0.5 * h * (k1 + k2)
        traj.append(x.copy())
    return np.array(traj)                                          # (steps+1, M, 2)


def target_density(P, sigma_extra=0.0):
    var = S0 ** 2 + sigma_extra ** 2
    d2 = ((P[:, None, :] - MODES[None, :, :]) ** 2).sum(-1)
    log_comp = np.log(WEIGHTS) - np.log(2 * np.pi * var) - d2 / (2 * var)
    m = log_comp.max(1, keepdims=True)
    return np.exp(m[:, 0]) * np.exp(log_comp - m).sum(1)


def dens_panel(ax, sigma_extra=0.0, lim=2.2):
    g = np.linspace(-lim, lim, 220)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    Z = target_density(P, sigma_extra).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=14, cmap=DENS)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")


# ---------- Sinkhorn coupling (pure numpy) for the OT-vs-independent figure ----------
def sinkhorn_plan(x0, x1, eps=0.05, iters=200):
    C = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    C = C / C.max()
    K = np.exp(-C / eps)
    n, m = len(x0), len(x1)
    u = np.ones(n) / n; v = np.ones(m) / m
    a = np.ones(n) / n; b = np.ones(m) / m
    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    return u[:, None] * K * v[None, :]                            # transport plan


def ot_match(x0, x1, eps=0.05):
    """Return a permutation matching each source to a target via the entropic
    OT plan (argmax over the row), then greedily de-duplicate for a clean 1-1
    picture."""
    P = sinkhorn_plan(x0, x1, eps=eps)
    order = np.argsort(-P.max(1))
    taken = set(); match = np.full(len(x0), -1, int)
    for i in order:
        for j in np.argsort(-P[i]):
            if j not in taken:
                match[i] = j; taken.add(j); break
    return match


# ============================================================================
# Overview figure (main week page): the straight conditional interpolant, the
# marginal velocity field it induces, and few-step sampling.
# ============================================================================
def fig_overview():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    # Panel 1: conditional interpolant paths as straight arrows, noise -> data.
    # Noise samples sit on a loose ring around the target; each is joined to a
    # data point on one of the three modes, so the straight conditional paths
    # (and their crossings) read cleanly instead of as a tangle.
    ax = axes[0]
    dens_panel(ax)
    n_arr = 9
    ang = np.linspace(0, 2 * np.pi, n_arr, endpoint=False) + 0.35
    x0 = np.stack([1.85 * np.cos(ang), 1.85 * np.sin(ang)], 1) + 0.10 * rng.standard_normal((n_arr, 2))
    kk = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2])          # assign each to a mode
    x1 = MODES[kk] + 0.12 * rng.standard_normal((n_arr, 2))
    for i in range(n_arr):
        ax.annotate("", xy=(x1[i, 0], x1[i, 1]), xytext=(x0[i, 0], x0[i, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1, alpha=0.75,
                                    shrinkA=4, shrinkB=4), zorder=2)
    ax.scatter(x0[:, 0], x0[:, 1], s=20, color=MID, zorder=3, label="noise $x_0$")
    ax.scatter(x1[:, 0], x1[:, 1], s=20, color=ACCENT, zorder=3, label="data $x_1$")
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.set_aspect("equal")
    clean(ax); ax.set_title("Conditional paths  $x_t=(1-t)x_0+t\\,x_1$", color=FG)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.3)

    # Panel 2: the marginal velocity field at an intermediate time.
    ax = axes[1]
    data = sample_target(500, rng)
    lim = 2.2
    g = np.linspace(-lim, lim, 19)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    U = marginal_velocity(P, 0.5, data)
    ax.quiver(X, Y, U[:, 0].reshape(X.shape), U[:, 1].reshape(X.shape),
              color=STREAM, width=0.004, scale=42)
    ax.scatter(MODES[:, 0], MODES[:, 1], s=34, color=ACCENT, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    clean(ax); ax.set_title("Marginal velocity $u_t(x)=\\mathbb{E}[\\dot x_t\\mid x_t]$", color=FG)

    # Panel 3: few-step Euler sampling landing on the modes.
    ax = axes[2]
    dens_panel(ax)
    xs = sample_source(120, rng)
    traj = integrate(xs, data, n_steps=8)
    for i in range(len(xs)):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=GREEN, lw=0.5, alpha=0.3, zorder=2)
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, color=FG, zorder=3)
    clean(ax); ax.set_title("Few-step sampling (8 steps)", color=FG)

    save(fig, "w3-overview.png")


# ============================================================================
# Problem 1: conditional vs marginal vector field (CFM unbiasedness).
# ============================================================================
def fig_p1():
    rng = np.random.default_rng(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    data = sample_target(500, rng)
    t = 0.5

    # Left: several conditional velocities at one point x, averaging to the
    # marginal velocity there.
    x = np.array([[0.1, 0.2]])
    dens_panel(ax1)
    # draw the conditional velocity (x1 - x)/(1-t) toward one sample per mode
    pick = MODES + S0 * rng.standard_normal(MODES.shape)
    for k in range(len(pick)):
        v = (pick[k] - x[0]) / (1 - t)
        ax1.arrow(x[0, 0], x[0, 1], 0.20 * v[0], 0.20 * v[1], color=BLUE,
                  width=0.008, head_width=0.07, alpha=0.55, length_includes_head=True, zorder=3)
    um = marginal_velocity(x, t, data)[0]
    ax1.arrow(x[0, 0], x[0, 1], 0.20 * um[0], 0.20 * um[1], color=ACCENT,
              width=0.02, head_width=0.13, length_includes_head=True, zorder=5)
    ax1.scatter([x[0, 0]], [x[0, 1]], s=34, color=FG, zorder=6)
    clean(ax1)
    ax1.set_title("Per-sample $u_t(x\\mid x_1)$ (blue) average to $u_t(x)$ (red)", color=FG, fontsize=9)

    # Right: the exact marginal field recovered as that conditional average.
    lim = 2.2
    g = np.linspace(-lim, lim, 21)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    U = marginal_velocity(P, t, data)
    spd = np.hypot(U[:, 0], U[:, 1]).reshape(X.shape)
    ax2.streamplot(X, Y, U[:, 0].reshape(X.shape), U[:, 1].reshape(X.shape),
                   color=spd, cmap="cividis", density=1.1, linewidth=0.8, arrowsize=0.7)
    ax2.scatter(MODES[:, 0], MODES[:, 1], s=34, color=ACCENT, zorder=3)
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_aspect("equal")
    clean(ax2); ax2.set_title("Marginal field at $t=0.5$", color=FG)

    save(fig, "w3p1.png")


# ============================================================================
# Problem 2: straightness, curvature, and rectified flow (reflow).
# ============================================================================
def fig_p2():
    rng = np.random.default_rng(7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    data = sample_target(500, rng)

    # Left: round-0 marginal trajectories are curved and cross.
    dens_panel(ax1)
    xs = sample_source(24, rng)
    traj = integrate(xs, data, n_steps=60)
    for i in range(len(xs)):
        ax1.plot(traj[:, i, 0], traj[:, i, 1], color=BLUE, lw=0.9, alpha=0.7, zorder=2)
    ax1.scatter(xs[:, 0], xs[:, 1], s=12, color=MID, zorder=3)
    ax1.scatter(traj[-1, :, 0], traj[-1, :, 1], s=12, color=ACCENT, zorder=3)
    clean(ax1); ax1.set_title("Round 0: curved marginal paths", color=FG)

    # Right: reflow uses the induced coupling (x0, G(x0)) with straight lines.
    dens_panel(ax2)
    x1 = traj[-1]
    for i in range(len(xs)):
        ax2.plot([xs[i, 0], x1[i, 0]], [xs[i, 1], x1[i, 1]],
                 color=GREEN, lw=0.9, alpha=0.8, zorder=2)
    ax2.scatter(xs[:, 0], xs[:, 1], s=12, color=MID, zorder=3)
    ax2.scatter(x1[:, 0], x1[:, 1], s=12, color=ACCENT, zorder=3)
    clean(ax2); ax2.set_title("Reflow target: straightened paths", color=FG)

    save(fig, "w3p2.png")


# ============================================================================
# Problem 3: independent vs OT (minibatch) coupling.
# ============================================================================
def fig_p3():
    rng = np.random.default_rng(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    n = 60
    x0 = sample_source(n, rng)
    x1 = sample_target(n, rng)

    # Left: independent (random) pairing -> long, crossing straight paths.
    dens_panel(ax1)
    perm = rng.permutation(n)
    cost_ind = 0.0
    for i in range(n):
        j = perm[i]
        ax1.plot([x0[i, 0], x1[j, 0]], [x0[i, 1], x1[j, 1]],
                 color=BLUE, lw=0.7, alpha=0.55, zorder=2)
        cost_ind += ((x0[i] - x1[j]) ** 2).sum()
    ax1.scatter(x0[:, 0], x0[:, 1], s=10, color=MID, zorder=3)
    ax1.scatter(x1[:, 0], x1[:, 1], s=10, color=ACCENT, zorder=3)
    clean(ax1)
    ax1.set_title(f"Independent pairing   $\\overline{{\\|x_1-x_0\\|^2}}$={cost_ind/n:.2f}", color=FG, fontsize=9)

    # Right: OT (Sinkhorn) pairing -> short, nearly non-crossing paths.
    dens_panel(ax2)
    match = ot_match(x0, x1, eps=0.03)
    cost_ot = 0.0
    for i in range(n):
        j = match[i]
        ax2.plot([x0[i, 0], x1[j, 0]], [x0[i, 1], x1[j, 1]],
                 color=GREEN, lw=0.7, alpha=0.7, zorder=2)
        cost_ot += ((x0[i] - x1[j]) ** 2).sum()
    ax2.scatter(x0[:, 0], x0[:, 1], s=10, color=MID, zorder=3)
    ax2.scatter(x1[:, 0], x1[:, 1], s=10, color=ACCENT, zorder=3)
    clean(ax2)
    ax2.set_title(f"OT pairing   $\\overline{{\\|x_1-x_0\\|^2}}$={cost_ot/n:.2f}", color=FG, fontsize=9)

    save(fig, "w3p3.png")


# ============================================================================
# Problem 4: Riemannian / torus flow matching for fractional coordinates.
# ============================================================================
def wrap(u):
    return u - np.round(u)                                        # minimum image to [-0.5, 0.5)


def to_cell(u):
    return u - np.floor(u)                                        # into [0, 1)


def fig_p4():
    rng = np.random.default_rng(9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    # A few "Wyckoff-like" target sites in the unit cell, some near the edges so
    # the geodesic must wrap across the periodic boundary.
    sites = np.array([[0.12, 0.5], [0.5, 0.88], [0.85, 0.15]])
    x0 = to_cell(rng.random((14, 2)))
    kk = rng.choice(len(sites), 14)
    x1 = to_cell(sites[kk] + 0.03 * rng.standard_normal((14, 2)))

    # Left: naive Euclidean straight line ignores periodicity (leaves the cell,
    # takes the long way around).
    for ax, geodesic, col, title in (
            (ax1, False, ACCENT, "Euclidean interpolant (ignores periodicity)"),
            (ax2, True, GREEN, "Toroidal geodesic (minimum image)")):
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=LINE, lw=1.2))
        for i in range(len(x0)):
            if geodesic:
                disp = wrap(x1[i] - x0[i])
                ts = np.linspace(0, 1, 40)
                path = to_cell(x0[i][None, :] + ts[:, None] * disp[None, :])
                # break the line where it wraps for a clean plot
                seg = np.abs(np.diff(path, axis=0)).max(1) > 0.5
                path_plot = path.copy()
                path_plot[1:][seg] = np.nan
                ax.plot(path_plot[:, 0], path_plot[:, 1], color=col, lw=1.0, alpha=0.8)
            else:
                ax.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]],
                        color=col, lw=1.0, alpha=0.7)
        ax.scatter(x0[:, 0], x0[:, 1], s=14, color=MID, zorder=3)
        ax.scatter(sites[:, 0], sites[:, 1], s=48, marker="*", color=ACCENT, zorder=4)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect("equal")
        clean(ax); ax.set_title(title, color=FG, fontsize=9)

    save(fig, "w3p4.png")


# ============================================================================
# Notes figure: the probability path interpolating source to data.
# ============================================================================
def fig_notes_path():
    rng = np.random.default_rng(2)
    data = sample_target(600, rng)
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4))
    for ax, t in zip(axes, [0.0, 0.35, 0.7, 1.0]):
        lim = 2.4
        g = np.linspace(-lim, lim, 200)
        X, Y = np.meshgrid(g, g)
        P = np.stack([X.ravel(), Y.ravel()], 1)
        # marginal density of x_t = (1-t) x0 + t x1 under the point-cloud target
        var = (1 - t) ** 2 * SRC ** 2 + t ** 2 * S0 ** 2
        d2 = ((P[:, None, :] - t * data[None, :, :]) ** 2).sum(-1)
        logc = -d2 / (2 * var)
        m = logc.max(1, keepdims=True)
        Z = (np.exp(m[:, 0]) * np.exp(logc - m).sum(1)).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=14, cmap=DENS)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        clean(ax); ax.set_title(f"$t={t:.2f}$", color=FG)
    axes[0].set_ylabel("noise $\\to$ data", color=MID)
    save(fig, "w3n-path.png")


# ============================================================================
# Notes figure: straightness and the step-count / error tradeoff.
# ============================================================================
def fig_notes_straight():
    rng = np.random.default_rng(5)
    data = sample_target(500, rng)
    ref = integrate(sample_source(4000, rng), data, n_steps=200)[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    # Left: one curved marginal path vs the straight chord it would be after
    # rectification, to define curvature.
    dens_panel(ax1)
    # pick, among several source draws, the trajectory that deviates most from
    # its own start->end chord, so "curvature" is visible.
    cand = sample_source(40, np.random.default_rng(11))
    trc = integrate(cand, data, n_steps=80)
    dev = np.zeros(len(cand))
    for i in range(len(cand)):
        a, b = trc[0, i], trc[-1, i]
        ab = b - a; L = np.hypot(*ab) + 1e-9
        proj = a + np.clip(((trc[:, i] - a) @ ab) / L ** 2, 0, 1)[:, None] * ab
        dev[i] = np.hypot(*(trc[:, i] - proj).T).max()
    j = int(np.argmax(dev))
    tr = trc[:, j:j + 1]
    ax1.plot(tr[:, 0, 0], tr[:, 0, 1], color=BLUE, lw=1.6, label="marginal path")
    ax1.plot([tr[0, 0, 0], tr[-1, 0, 0]], [tr[0, 0, 1], tr[-1, 0, 1]],
             color=GREEN, lw=1.4, ls="--", label="straight chord")
    ax1.scatter([tr[0, 0, 0], tr[-1, 0, 0]], [tr[0, 0, 1], tr[-1, 0, 1]],
                s=20, color=FG, zorder=3)
    clean(ax1); ax1.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax1.set_title("Curvature = gap to the chord", color=FG)

    # Right: sampling error vs number of Euler steps (few-step benefits from
    # straightness).
    steps = [1, 2, 4, 8, 16, 32, 64]
    errs = []
    xs = sample_source(2000, np.random.default_rng(21))
    for ns in steps:
        end = integrate(xs, data, n_steps=ns)[-1]
        # energy-distance-like proxy: difference of mean pairwise stats to ref
        e = np.abs(end.mean(0) - ref.mean(0)).sum() + \
            np.abs(end.std(0) - ref.std(0)).sum()
        errs.append(e)
    ax2.plot(steps, errs, "-o", color=ACCENT, lw=1.6, ms=4)
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xlabel("Euler steps (NFE)"); ax2.set_ylabel("distribution error (proxy)")
    clean(ax2, ticks=True); ax2.set_title("Fewer steps suffice as paths straighten", color=FG, fontsize=9)

    save(fig, "w3n-straight.png")


if __name__ == "__main__":
    fig_overview()
    fig_p1()
    fig_p2()
    fig_p3()
    fig_p4()
    fig_notes_path()
    fig_notes_straight()
    print("done")
