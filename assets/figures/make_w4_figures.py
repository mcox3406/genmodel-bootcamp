# Generates the Week 4 figures (stochastic interpolants: the interpolant with a
# latent noise term, the velocity/score, the ODE-SDE sampler family, recovery of
# flow matching and diffusion, and data-dependent couplings on the torus) in the
# site color theme. Reuses the palette and helpers from make_figures.py and the
# same Gaussian -> three-mode toy as the Week 3 figures for visual continuity.
# Usage: python make_w4_figures.py   (requires matplotlib + numpy, no scipy)
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
# The shared Week 3/4 toy: a Gaussian base and a molecule-like three-mode
# target, now joined by the stochastic interpolant x_t = (1-t) x0 + t x1 +
# gamma_t z with a latent term. gamma_t = G * sqrt(t (1-t)) is zero at both ends
# and widest at t = 1/2. Because the target is a finite point cloud, the
# marginal velocity and score are closed-form posterior-weighted averages, so
# every figure is exact rather than trained (mirroring the Week 2/3 figures).
# ============================================================================
MODES = np.array([[0.0, 1.15], [-1.0, -0.6], [1.0, -0.6]])
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])
S0 = 0.13                                   # per-mode std of the "atoms"
SRC = 0.9                                   # base std (isotropic Gaussian)
GAMMA = 0.42                                # latent amplitude


def gamma_t(t):
    return GAMMA * np.sqrt(np.clip(t * (1 - t), 0, None))


def sample_target(n, rng):
    k = rng.choice(len(MODES), size=n, p=WEIGHTS)
    return MODES[k] + S0 * rng.standard_normal((n, 2))


def sample_source(n, rng):
    return SRC * rng.standard_normal((n, 2))


def _post_stats(x, t):
    """Posterior over the target mode k given x_t = x for the straight
    interpolant with a Gaussian base folded into an effective variance plus the
    latent width. Returns per-point E[x1|x], and the effective std used."""
    tt = np.clip(t, 1e-3, 1 - 1e-3)
    # var of x_t | mode k = (1-t)^2 SRC^2 + t^2 S0^2 + gamma_t^2
    var = (1 - tt) ** 2 * SRC ** 2 + tt ** 2 * S0 ** 2 + gamma_t(tt) ** 2
    d2 = ((x[:, None, :] - tt * MODES[None, :, :]) ** 2).sum(-1)
    logw = -d2 / (2 * var) + np.log(WEIGHTS)[None, :]
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw); w /= w.sum(1, keepdims=True)
    x1_hat = (w[..., None] * MODES[None, :, :]).sum(1)
    return x1_hat, var, tt


def marginal_velocity(x, t):
    """Approximate marginal interpolant velocity b_t(x) for the toy. Using the
    affine relation, b_t(x) ~ (E[x1|x] - x) * (something) but for a clean field
    we use b_t(x) = (x1_hat - x) / (1 - t) + drift from the base mean 0, which
    reproduces the straight-line marginal flow toward the modes."""
    x1_hat, _, tt = _post_stats(x, t)
    return (x1_hat - x) / (1 - tt)


def marginal_score(x, t):
    """Marginal score s_t(x) = grad log rho_t(x) for the Gaussian-mixture
    marginal with effective variance var(t)."""
    x1_hat, var, tt = _post_stats(x, t)
    return -(x - tt * x1_hat) / var


def clip_norm(v, vmax):
    n = np.hypot(v[:, 0], v[:, 1])
    s = np.minimum(1.0, vmax / np.maximum(n, 1e-12))
    return v * s[:, None]


def integrate_ode(x, n_steps=70, t0=1e-2, t1=0.985):
    ts = np.linspace(t0, t1, n_steps)
    traj = [x.copy()]
    for i in range(n_steps - 1):
        dt = ts[i + 1] - ts[i]
        x = x + dt * clip_norm(marginal_velocity(x, ts[i]), 16.0)
        traj.append(x.copy())
    return np.array(traj)


def integrate_sde(x, eps, rng, n_steps=70, t0=1e-2, t1=0.985):
    ts = np.linspace(t0, t1, n_steps)
    traj = [x.copy()]
    for i in range(n_steps - 1):
        dt = ts[i + 1] - ts[i]
        drift = marginal_velocity(x, ts[i]) + eps * marginal_score(x, ts[i])
        x = x + dt * clip_norm(drift, 16.0) + np.sqrt(2 * eps * dt) * rng.standard_normal(x.shape)
        traj.append(x.copy())
    return np.array(traj)


def target_density(P):
    d2 = ((P[:, None, :] - MODES[None, :, :]) ** 2).sum(-1)
    s2 = S0 ** 2 + 0.02
    return (WEIGHTS[None, :] * np.exp(-d2 / (2 * s2))).sum(1)


def dens_panel(ax, lim=2.3):
    g = np.linspace(-lim, lim, 240)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    Z = target_density(P).reshape(X.shape)
    ax.imshow(Z, extent=[-lim, lim, -lim, lim], origin="lower", cmap=DENS, aspect="auto")


def frame(ax, lim=2.3, ticks=False):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    clean(ax, ticks=ticks)


# ============================================================================
# Overview (main page): interpolant with latent band | ODE vs SDE trajectories
# | the epsilon dial from ODE to diffusion SDE.
# ============================================================================
def fig_overview():
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    # Panel 1: interpolant paths with a latent-width band.
    ax = axes[0]
    dens_panel(ax)
    x1 = sample_target(6, rng)
    x0 = sample_source(6, rng)
    ts = np.linspace(0, 1, 60)
    for a, b in zip(x0, x1):
        mid = (1 - ts)[:, None] * a + ts[:, None] * b
        ax.plot(mid[:, 0], mid[:, 1], color=BLUE, lw=1.1, alpha=0.8, zorder=3)
        w = gamma_t(ts)
        # perpendicular band
        d = b - a; L = np.hypot(*d) + 1e-9
        perp = np.array([-d[1], d[0]]) / L
        upper = mid + (w[:, None]) * perp
        lower = mid - (w[:, None]) * perp
        ax.fill(np.concatenate([upper[:, 0], lower[::-1, 0]]),
                np.concatenate([upper[:, 1], lower[::-1, 1]]),
                color=BLUE_SOFT, alpha=0.5, lw=0, zorder=2)
    ax.scatter(x0[:, 0], x0[:, 1], s=22, color=MUTED, zorder=4, label=r"base $x_0$")
    ax.scatter(x1[:, 0], x1[:, 1], s=26, color=ACCENT, zorder=4, label=r"data $x_1$")
    ax.set_title(r"interpolant  $x_t=\alpha_t x_0+\beta_t x_1+\gamma_t z$", color=FG)
    ax.legend(loc="lower center", fontsize=7, frameon=False, ncol=2)
    frame(ax)

    # Panel 2: ODE (eps=0) vs SDE trajectories from the same base points.
    ax = axes[1]
    dens_panel(ax)
    start = sample_source(9, rng)
    traj0 = integrate_ode(start.copy())
    for j in range(start.shape[0]):
        ax.plot(traj0[:, j, 0], traj0[:, j, 1], color=DIM, lw=1.0, alpha=0.9, zorder=3)
    trajS = integrate_sde(start.copy(), eps=0.55, rng=rng)
    for j in range(start.shape[0]):
        ax.plot(trajS[:, j, 0], trajS[:, j, 1], color=GREEN, lw=0.9, alpha=0.7, zorder=3)
    ax.scatter(start[:, 0], start[:, 1], s=18, color=MUTED, zorder=4)
    ax.plot([], [], color=DIM, lw=1.2, label=r"ODE  $\epsilon_t=0$")
    ax.plot([], [], color=GREEN, lw=1.2, label=r"SDE  $\epsilon_t>0$")
    ax.set_title("one field, two samplers", color=FG)
    ax.legend(loc="lower center", fontsize=7, frameon=False, ncol=2)
    frame(ax)

    # Panel 3: the epsilon dial (schematic).
    ax = axes[2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xs = np.linspace(0.1, 0.9, 200)
    ax.plot(xs, 0.5 + 0 * xs, color=LINE, lw=6, solid_capstyle="round", zorder=1)
    for frac, lab, col in [(0.0, "flow matching\n(ODE)", DIM),
                           (0.5, "interpolant\nSDE", GREEN),
                           (1.0, "diffusion\n(reverse SDE)", ACCENT)]:
        xp = 0.1 + 0.8 * frac
        ax.scatter([xp], [0.5], s=90, color=col, zorder=3, edgecolor="white", linewidth=1.4)
        ax.annotate(lab, (xp, 0.5), (xp, 0.5 + (0.22 if frac == 0.5 else -0.26)),
                    ha="center", va="center", fontsize=8, color=col)
    ax.annotate(r"diffusion coefficient $\epsilon_t$", (0.5, 0.5), (0.5, 0.12),
                ha="center", fontsize=9, color=MID)
    ax.annotate("", (0.92, 0.5), (0.08, 0.5),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_title("one dial: ODE to SDE", color=FG)
    clean(ax); ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    save(fig, "w4-overview.png")


# ============================================================================
# Problem 1: conditional velocities from endpoint pairs averaging to the
# marginal velocity (left) | the exact marginal velocity field at t=0.5 (right).
# ============================================================================
def fig_p1():
    rng = np.random.default_rng(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    t = 0.5
    x = np.array([[0.15, 0.2]])
    # several endpoint pairs consistent-ish with x_t = x: sample x1 from modes,
    # back out x0, draw the per-sample velocity x1 - x0.
    ax1.set_title(r"conditional velocities average to $b_t(x)$", color=FG)
    dens_panel(ax1)
    ax1.scatter(x[:, 0], x[:, 1], s=40, color=FG, zorder=5)
    band = gamma_t(t)
    circ = plt.Circle((x[0, 0], x[0, 1]), band, color=BLUE_SOFT, alpha=0.5, zorder=2)
    ax1.add_patch(circ)
    for k in range(len(MODES)):
        x1 = MODES[k] + S0 * rng.standard_normal((3, 2))
        for row in x1:
            x0 = (x[0] - t * row) / (1 - t)
            v = clip_norm((row - x[0])[None, :], 1.6)[0]
            ax1.arrow(x[0, 0], x[0, 1], v[0] * 0.5, v[1] * 0.5,
                      color=BLUE, width=0.006, head_width=0.06, alpha=0.7, zorder=3)
    bm = clip_norm(marginal_velocity(x, t), 1.6)[0]
    ax1.arrow(x[0, 0], x[0, 1], bm[0] * 0.5, bm[1] * 0.5, color=ACCENT,
              width=0.014, head_width=0.11, zorder=4)
    ax1.plot([], [], color=BLUE, lw=1.4, label=r"per-pair $\dot x_t$")
    ax1.plot([], [], color=ACCENT, lw=1.8, label=r"marginal $b_t(x)$")
    ax1.legend(loc="lower center", fontsize=7.5, frameon=False, ncol=2)
    frame(ax1)

    lim = 2.3
    g = np.linspace(-lim, lim, 22)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    U = marginal_velocity(P, t)
    ax2.streamplot(X, Y, U[:, 0].reshape(X.shape), U[:, 1].reshape(X.shape),
                   color=STREAM, density=1.1, linewidth=0.7, arrowsize=0.7)
    ax2.scatter(MODES[:, 0], MODES[:, 1], s=60, color=ACCENT, zorder=4)
    ax2.set_title(r"marginal velocity field $b_{t}$ at $t=0.5$", color=FG)
    frame(ax2)

    save(fig, "w4p1.png")


# ============================================================================
# Problem 2: the score field (left) | ODE vs SDE runs from one base point,
# with a small Fokker-Planck cancellation inset (right).
# ============================================================================
def fig_p2():
    rng = np.random.default_rng(2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4))

    t = 0.5
    lim = 2.3
    g = np.linspace(-lim, lim, 20)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    S = marginal_score(P, t)
    ax1.quiver(X, Y, S[:, 0].reshape(X.shape), S[:, 1].reshape(X.shape),
               color=GREEN, scale=42, width=0.004, alpha=0.85)
    ax1.scatter(MODES[:, 0], MODES[:, 1], s=60, color=ACCENT, zorder=4)
    ax1.set_title(r"score field $s_t=\nabla\log\rho_t$ at $t=0.5$", color=FG)
    frame(ax1)

    dens_panel(ax2)
    start = np.array([[-0.1, -1.9], [-0.1, -1.9], [-0.1, -1.9]])
    traj0 = integrate_ode(start[:1].copy())
    ax2.plot(traj0[:, 0, 0], traj0[:, 0, 1], color=DIM, lw=1.6, zorder=4, label=r"$\epsilon_t=0$ (ODE)")
    for eps, col, a in [(0.35, GREEN, 0.85), (0.9, GOLD, 0.8)]:
        tr = integrate_sde(start[:1].copy(), eps=eps, rng=rng)
        ax2.plot(tr[:, 0, 0], tr[:, 0, 1], color=col, lw=1.2, alpha=a,
                 zorder=3, label=fr"$\epsilon_t={eps}$")
    ax2.set_title("same marginals, different paths", color=FG)
    ax2.legend(loc="upper left", fontsize=7, frameon=False)
    frame(ax2)
    # cancellation inset
    axin = ax2.inset_axes([0.58, 0.06, 0.4, 0.2])
    axin.text(0.5, 0.5, r"$-\nabla\!\cdot(\epsilon\rho s)+\epsilon\Delta\rho=0$",
              ha="center", va="center", fontsize=7.5, color=MID)
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_color(LINE)

    save(fig, "w4p2.png")


# ============================================================================
# Problem 3: metrics vs the diffusion coefficient (left) | cost-vs-quality for
# ODE and best SDE (right). Schematic curves in the toy's spirit.
# ============================================================================
def fig_p3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    eps = np.linspace(0, 1.4, 60)
    # U-shaped quality: ODE decent, moderate eps best, large eps worse at fixed NFE
    good = 0.34 - 0.5 * eps * np.exp(-2.1 * eps) + 0.11 * eps ** 2
    imperfect = 0.62 - 0.9 * eps * np.exp(-1.8 * eps) + 0.16 * eps ** 2
    ax1.plot(eps, good, color=BLUE, lw=1.8, label="well-trained field")
    ax1.plot(eps, imperfect, color=ACCENT, lw=1.8, label="degraded field")
    for arr, col in [(good, BLUE), (imperfect, ACCENT)]:
        k = int(np.argmin(arr))
        ax1.scatter([eps[k]], [arr[k]], s=42, color=col, zorder=5, edgecolor="white")
    ax1.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax1.text(0.02, ax1.get_ylim()[1] * 0.96, "ODE", fontsize=7.5, color=DIM, va="top")
    ax1.set_xlabel(r"diffusion coefficient $\epsilon$")
    ax1.set_ylabel("energy distance to reference")
    ax1.set_title("quality vs the sampler dial", color=FG)
    ax1.legend(fontsize=7.5, frameon=False)
    clean(ax1, ticks=True)

    nfe = np.array([2, 4, 8, 16, 32, 64, 128])
    ode = 0.09 + 0.9 / nfe ** 0.9
    sde = 0.045 + 1.7 / nfe ** 0.95
    ax2.loglog(nfe, ode, "o-", color=DIM, lw=1.5, ms=4, label=r"ODE ($\epsilon=0$)")
    ax2.loglog(nfe, sde, "s-", color=GREEN, lw=1.5, ms=4, label="best SDE")
    ax2.set_xlabel("function evaluations")
    ax2.set_ylabel("energy distance")
    ax2.set_title("cost vs quality", color=FG)
    ax2.legend(fontsize=7.5, frameon=False)
    clean(ax2, ticks=True)

    save(fig, "w4p3.png")


# ---------- torus helpers (shared with Week 3 Problem 4 geometry) ----------
def wrap(u):
    return (u + 0.5) % 1.0 - 0.5


def to_cell(u):
    return u % 1.0


# ============================================================================
# Problem 4: independent coupling (left) | data-dependent coupling (center) |
# the trained bridge relaxing a held-out structure (right), all on the torus.
# ============================================================================
def fig_p4():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    # relaxed sites, some near boundaries so wrapping matters
    sites = np.array([[0.5, 0.5], [0.05, 0.9], [0.92, 0.12], [0.5, 0.06]])

    def draw_cell(ax, title):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); clean(ax)
        for v in (0, 1):
            ax.axvline(v, color=LINE, lw=1); ax.axhline(v, color=LINE, lw=1)
        ax.set_title(title, color=FG)

    # Panel 1: independent coupling (uniform base -> relaxed sites)
    ax = axes[0]
    draw_cell(ax, "independent coupling")
    base = rng.random((len(sites), 2))
    for a, b in zip(base, sites):
        d = wrap(b - a)
        ax.plot([a[0], a[0] + d[0]], [a[1], a[1] + d[1]], color=MUTED, lw=1.0, alpha=0.8)
    ax.scatter(base[:, 0], base[:, 1], s=26, color=MUTED, zorder=4, label="uniform base")
    ax.scatter(sites[:, 0], sites[:, 1], s=40, color=ACCENT, marker="*", zorder=4, label="relaxed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False, ncol=2)

    # Panel 2: data-dependent coupling (unrelaxed near its own relaxed target)
    ax = axes[1]
    draw_cell(ax, "data-dependent coupling")
    unrel = to_cell(sites + 0.06 * rng.standard_normal(sites.shape))
    for a, b in zip(unrel, sites):
        d = wrap(b - a)
        ax.plot([a[0], a[0] + d[0]], [a[1], a[1] + d[1]], color=BLUE, lw=1.6, alpha=0.9)
    ax.scatter(unrel[:, 0], unrel[:, 1], s=26, color=BLUE, zorder=4, label="unrelaxed")
    ax.scatter(sites[:, 0], sites[:, 1], s=40, color=ACCENT, marker="*", zorder=4, label="relaxed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False, ncol=2)

    # Panel 3: the trained bridge relaxing a held-out configuration
    ax = axes[2]
    draw_cell(ax, "the trained bridge relaxes a held-out input")
    held = to_cell(sites + 0.09 * rng.standard_normal(sites.shape))
    ts = np.linspace(0, 1, 40)
    for a, b in zip(held, sites):
        d = wrap(b - a)
        path = to_cell(a[None, :] + ts[:, None] * d[None, :])
        # split at wraps for clean plotting
        seg = np.abs(np.diff(path, axis=0)).max(1) > 0.5
        xs = path[:, 0].copy(); ys = path[:, 1].copy()
        xs[np.where(seg)[0]] = np.nan
        ax.plot(xs, ys, color=GREEN, lw=1.5, alpha=0.9, zorder=3)
    ax.scatter(held[:, 0], held[:, 1], s=26, color=GREEN, zorder=4, label="held-out $x_0$")
    ax.scatter(sites[:, 0], sites[:, 1], s=40, color=ACCENT, marker="*", zorder=4, label="target $x_1$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False, ncol=2)

    save(fig, "w4p4.png")


# ============================================================================
# Notes figures
# ============================================================================
def fig_notes_interpolant():
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    dens_panel(ax, lim=2.4)
    x1 = sample_target(7, rng)
    x0 = sample_source(7, rng)
    ts = np.linspace(0, 1, 60)
    for a, b in zip(x0, x1):
        mid = (1 - ts)[:, None] * a + ts[:, None] * b
        ax.plot(mid[:, 0], mid[:, 1], color=BLUE, lw=1.0, alpha=0.75, zorder=3)
        w = gamma_t(ts); d = b - a; L = np.hypot(*d) + 1e-9
        perp = np.array([-d[1], d[0]]) / L
        up = mid + w[:, None] * perp; lo = mid - w[:, None] * perp
        ax.fill(np.concatenate([up[:, 0], lo[::-1, 0]]),
                np.concatenate([up[:, 1], lo[::-1, 1]]),
                color=BLUE_SOFT, alpha=0.45, lw=0, zorder=2)
    ax.scatter(x0[:, 0], x0[:, 1], s=20, color=MUTED, zorder=4, label=r"base $x_0$ ($t=0$)")
    ax.scatter(x1[:, 0], x1[:, 1], s=24, color=ACCENT, zorder=4, label=r"data $x_1$ ($t=1$)")
    ax.set_title(r"stochastic interpolant: latent band $\gamma_t z$ widest at $t=1/2$", color=FG)
    ax.legend(loc="lower center", fontsize=7.5, frameon=False, ncol=2)
    frame(ax, lim=2.4)
    save(fig, "w4n-interpolant.png")


def fig_notes_family():
    rng = np.random.default_rng(12)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    start = sample_source(11, rng)
    for ax, eps, lab, col in zip(axes, [0.0, 0.4, 1.0],
                                 [r"$\epsilon_t=0$ (ODE)", r"$\epsilon_t$ moderate", r"$\epsilon_t$ large"],
                                 [DIM, GREEN, GOLD]):
        dens_panel(ax)
        if eps == 0:
            tr = integrate_ode(start.copy())
        else:
            tr = integrate_sde(start.copy(), eps=eps, rng=np.random.default_rng(99))
        for j in range(start.shape[0]):
            ax.plot(tr[:, j, 0], tr[:, j, 1], color=col, lw=1.0, alpha=0.85, zorder=3)
        ax.scatter(start[:, 0], start[:, 1], s=14, color=MUTED, zorder=4)
        ax.set_title(lab, color=FG)
        frame(ax)
    save(fig, "w4n-family.png")


def fig_notes_recover():
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); clean(ax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    # three axes as labeled bars
    rows = [
        (0.80, r"latent width $\gamma_t$", "deterministic", "stochastic"),
        (0.55, "learned target", "velocity", "score"),
        (0.30, r"diffusion coeff. $\epsilon_t$", "ODE", "SDE"),
    ]
    for y, lab, left, right in rows:
        ax.plot([0.28, 0.9], [y, y], color=LINE, lw=5, solid_capstyle="round")
        ax.text(0.26, y, lab, ha="right", va="center", fontsize=9, color=MID)
        ax.text(0.28, y + 0.06, left, ha="center", fontsize=7.5, color=DIM)
        ax.text(0.9, y + 0.06, right, ha="center", fontsize=7.5, color=DIM)
    # place three methods
    pts = [("flow matching", [0.28, 0.28, 0.28], DIM),
           ("diffusion", [0.9, 0.9, 0.9], ACCENT),
           ("stochastic\ninterpolant", [0.62, 0.62, 0.62], GREEN)]
    ys = [0.80, 0.55, 0.30]
    for name, xs, col in pts:
        for xp, y in zip(xs, ys):
            ax.scatter([xp], [y], s=70, color=col, zorder=5, edgecolor="white", linewidth=1.2)
    ax.scatter([], [], s=70, color=DIM, label="flow matching")
    ax.scatter([], [], s=70, color=ACCENT, label="diffusion")
    ax.scatter([], [], s=70, color=GREEN, label="full interpolant")
    ax.legend(loc="upper center", bbox_to_anchor=(0.58, 0.14), ncol=3, fontsize=7.5, frameon=False)
    ax.set_title("three independent choices on one path", color=FG)
    save(fig, "w4n-recover.png")


def fig_notes_couplings():
    rng = np.random.default_rng(21)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    sites = np.array([[0.5, 0.55], [0.08, 0.88], [0.9, 0.15], [0.55, 0.08], [0.3, 0.4]])

    def draw_cell(ax, title):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); clean(ax)
        for v in (0, 1):
            ax.axvline(v, color=LINE, lw=1); ax.axhline(v, color=LINE, lw=1)
        ax.set_title(title, color=FG)

    draw_cell(ax1, "independent: noise to data")
    base = rng.random((len(sites), 2))
    for a, b in zip(base, sites):
        d = wrap(b - a)
        ax1.plot([a[0], a[0] + d[0]], [a[1], a[1] + d[1]], color=MUTED, lw=1.0, alpha=0.8)
    ax1.scatter(base[:, 0], base[:, 1], s=22, color=MUTED, zorder=4)
    ax1.scatter(sites[:, 0], sites[:, 1], s=36, color=ACCENT, marker="*", zorder=4)

    draw_cell(ax2, "data-dependent: ensemble to ensemble")
    unrel = to_cell(sites + 0.06 * rng.standard_normal(sites.shape))
    for a, b in zip(unrel, sites):
        d = wrap(b - a)
        ax2.plot([a[0], a[0] + d[0]], [a[1], a[1] + d[1]], color=BLUE, lw=1.6, alpha=0.9)
    ax2.scatter(unrel[:, 0], unrel[:, 1], s=22, color=BLUE, zorder=4)
    ax2.scatter(sites[:, 0], sites[:, 1], s=36, color=ACCENT, marker="*", zorder=4)
    save(fig, "w4n-couplings.png")


if __name__ == "__main__":
    fig_overview()
    fig_p1()
    fig_p2()
    fig_p3()
    fig_p4()
    fig_notes_interpolant()
    fig_notes_family()
    fig_notes_recover()
    fig_notes_couplings()
    print("all Week 4 figures written")
