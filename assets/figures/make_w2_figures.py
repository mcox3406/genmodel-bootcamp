# Generates the Week 2 problem figures (diffusion / score models) in the site
# color theme. Reuses the palette and helpers from make_figures.py.
# Usage: python make_w2_figures.py   (requires matplotlib + numpy)
import numpy as np
import matplotlib.pyplot as plt
from make_figures import (ACCENT, ACCENT_MID, ACCENT_SOFT, FG, MID, DIM, MUTED,
                          LINE, STREAM, DENS, clean, save)

BLUE = "#2f6fb3"
BLUE_SOFT = "#cfe0f0"
GREEN = "#1f9d6b"


# ============================================================================
# Shared building blocks: a mixture-of-Gaussians "molecule-like" data
# distribution whose variance-exploding (EDM-style) perturbed marginals, score,
# and samples are all available in closed form. This is the toy that makes every
# diffusion figure exact rather than trained.
# ============================================================================
# Three modes at the vertices of a triangle, like a triatomic molecule.
MODES = np.array([[0.0, 1.15], [-1.0, -0.6], [1.0, -0.6]])
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])
S0 = 0.16                                  # clean per-mode std (the "atoms")


def mog_var(sigma):
    return S0 ** 2 + sigma ** 2            # VE perturbed per-mode variance


def mog_logp(P, sigma):
    var = mog_var(sigma)
    d2 = ((P[:, None, :] - MODES[None, :, :]) ** 2).sum(-1)        # (M, K)
    log_comp = np.log(WEIGHTS) - np.log(2 * np.pi * var) - d2 / (2 * var)
    m = log_comp.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(log_comp - m).sum(1)))


def mog_score(P, sigma):
    var = mog_var(sigma)
    d = MODES[None, :, :] - P[:, None, :]                          # (M, K, 2)
    d2 = (d ** 2).sum(-1)
    logw = np.log(WEIGHTS) - d2 / (2 * var)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * d).sum(1) / var                        # (M, 2)


def mog_sample(n, sigma, rng):
    k = rng.choice(len(MODES), size=n, p=WEIGHTS)
    return MODES[k] + np.sqrt(mog_var(sigma)) * rng.standard_normal((n, 2))


def pf_ode_reverse(x, sigmas, score_fn):
    """Deterministic probability-flow ODE for the VE/EDM process:
    dx = -sigma * score(x, sigma) d(sigma), integrated high -> low sigma (Heun)."""
    traj = [x.copy()]
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        d = -s * score_fn(x, s)
        x_eu = x + (s_next - s) * d
        if s_next > 0:
            d2 = -s_next * score_fn(x_eu, s_next)
            x = x + (s_next - s) * 0.5 * (d + d2)
        else:
            x = x_eu
        traj.append(x.copy())
    return np.array(traj)


def reverse_sde(x, sigmas, score_fn, rng, noise_scale=1.0):
    """Euler-Maruyama on the reverse VE SDE: dx = -2 sigma score d(sigma) +
    sqrt(2 sigma) dW, integrated high -> low sigma. Trajectories are noisy;
    noise_scale<1 tempers the injected Brownian term for legible figures."""
    traj = [x.copy()]
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        dsig = s_next - s                                          # negative
        x = x - 2 * s * score_fn(x, s) * dsig
        if s_next > 0:
            x = x + noise_scale * np.sqrt(2 * s * (-dsig)) * rng.standard_normal(x.shape)
        traj.append(x.copy())
    return np.array(traj)


# ============================================================================
# Muller-Brown potential: the classic 2D chemistry test surface (three basins,
# one barrier). Used for the Boltzmann-sampling lab.
# ============================================================================
MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
MB_a = np.array([-1.0, -1.0, -6.5, 0.7])
MB_b = np.array([0.0, 0.0, 11.0, 0.6])
MB_c = np.array([-10.0, -10.0, -6.5, 0.7])
MB_x0 = np.array([1.0, 0.0, -0.5, -1.0])
MB_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def muller_brown(X, Y):
    V = np.zeros_like(X)
    for k in range(4):
        dx, dy = X - MB_x0[k], Y - MB_y0[k]
        V += MB_A[k] * np.exp(MB_a[k] * dx ** 2 + MB_b[k] * dx * dy + MB_c[k] * dy ** 2)
    return V


# ---------- Problem 1: forward OU noising of a Boltzmann density + score=force ----------
def fig_p1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # 1D double-well Boltzmann density along a reaction coordinate.
    x = np.linspace(-2.6, 2.6, 400)
    U = lambda q: (q ** 2 - 1) ** 2                         # double well
    beta = 1.6
    p0 = np.exp(-beta * U(x)); p0 /= np.trapezoid(p0, x)

    # OU "heating": x_t = e^{-t} x_0 + sqrt(1-e^{-2t}) z  -> marginal is a
    # convolution that relaxes to the standard Gaussian as t grows.
    def ou_density(t):
        a = np.exp(-t); v = 1 - a ** 2
        rng = np.random.default_rng(0)
        # sample-free: convolve p0 (on grid) with N(0, v), then rescale mean by a
        xs = a * x
        dens = np.zeros_like(x)
        if v < 1e-4:
            # near-identity
            return np.interp(x, xs, p0 / a, left=0, right=0)
        ker = np.exp(-(x[None, :] - xs[:, None]) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
        dens = (p0[:, None] * (x[1] - x[0]) * ker).sum(0)
        return dens / np.trapezoid(dens, x)

    cols = [ACCENT, "#d2627a", "#b9a0c6", BLUE]
    for t, col in zip([0.0, 0.25, 0.7, 2.5], cols):
        ax1.plot(x, ou_density(t), color=col, lw=1.9, label=f"t = {t:g}")
    ax1.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=DIM, lw=1.0,
             ls=(0, (4, 3)), label=r"$\mathcal{N}(0,1)$")
    ax1.set_title("forward OU process relaxes a Boltzmann density to noise")
    ax1.set_xlabel("reaction coordinate  q"); ax1.set_ylabel("density")
    ax1.legend(frameon=False, fontsize=8, loc="upper right"); clean(ax1, ticks=True)

    # 2D: score of the smoothed density = (smoothed) force field -beta grad U.
    xs = np.linspace(-2.2, 2.2, 240)
    X, Y = np.meshgrid(xs, np.linspace(-1.6, 1.6, 200))
    Uxy = (X ** 2 - 1) ** 2 + 1.4 * Y ** 2
    dens = np.exp(-1.4 * Uxy)
    ax2.contourf(X, Y, dens, levels=16, cmap=DENS)
    g = np.linspace(-2.0, 2.0, 17); gy = np.linspace(-1.4, 1.4, 13)
    GX, GY = np.meshgrid(g, gy)
    # score = -beta grad U  (the physical force, up to temperature)
    Fx = -1.4 * (4 * GX * (GX ** 2 - 1))
    Fy = -1.4 * (2.8 * GY)
    n = np.hypot(Fx, Fy) + 1e-9
    ax2.quiver(GX, GY, Fx / n, Fy / n, color=DIM, width=0.004, scale=34,
               pivot="mid", alpha=0.85)
    ax2.scatter([-1, 1], [0, 0], s=42, color=FG, zorder=5)
    ax2.set_title(r"score $\nabla\log p = -\beta\,\nabla U$ is the force field")
    ax2.set_aspect("equal"); clean(ax2)
    fig.tight_layout(); save(fig, "w2p1.png")


# ---------- Problem 2: EDM design space (preconditioning + noise schedule) ----------
def fig_p2():
    sd = 0.5                                                 # sigma_data
    sig = np.logspace(-3, 2, 400)
    c_skip = sd ** 2 / (sig ** 2 + sd ** 2)
    c_out = sig * sd / np.sqrt(sig ** 2 + sd ** 2)
    c_in = 1.0 / np.sqrt(sig ** 2 + sd ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax1.plot(sig, c_skip, color=ACCENT, lw=2.0, label=r"$c_{\mathrm{skip}}$")
    ax1.plot(sig, c_out, color=BLUE, lw=2.0, label=r"$c_{\mathrm{out}}$")
    ax1.plot(sig, c_in, color=GREEN, lw=2.0, label=r"$c_{\mathrm{in}}$")
    ax1.axvline(sd, color=DIM, lw=1.0, ls=(0, (4, 3)))
    ax1.text(sd * 1.1, 1.6, r"$\sigma=\sigma_{\mathrm{data}}$", color=DIM, fontsize=8.5)
    ax1.set_xscale("log"); ax1.set_xlabel(r"noise level  $\sigma$")
    ax1.set_ylabel("preconditioning coefficient")
    ax1.set_title("EDM preconditioning balances signal and noise")
    ax1.legend(frameon=False, fontsize=9, loc="center left")
    ax1.grid(True, which="both", color=LINE, lw=0.5, alpha=0.5); clean(ax1, ticks=True)

    # training noise distribution (lognormal) vs the SNR of two molecular length
    # scales: a stiff bond (small sigma matters) and a soft torsion (large sigma).
    lnsig = np.linspace(-7, 5, 400)
    P_mean, P_std = -1.2, 1.2
    pdf = np.exp(-(lnsig - P_mean) ** 2 / (2 * P_std ** 2)) / (P_std * np.sqrt(2 * np.pi))
    ax2.fill_between(lnsig, pdf, color=ACCENT_SOFT, zorder=1)
    ax2.plot(lnsig, pdf, color=ACCENT, lw=2.0, zorder=2,
             label=r"training noise  $\ln\sigma\sim\mathcal{N}(-1.2,1.2^2)$")
    for ls, lab, col in [(np.log(0.02), "stiff bond\n($\\sigma\\!\\sim\\!0.02$)", BLUE),
                         (np.log(1.0), "torsion\n($\\sigma\\!\\sim\\!1$)", GREEN)]:
        ax2.axvline(ls, color=col, lw=1.4, ls=(0, (3, 2)))
        ax2.text(ls + 0.15, 0.17, lab, color=col, fontsize=8, ha="left", va="center")
    ax2.set_xlabel(r"$\ln\sigma$"); ax2.set_ylabel("training-time density")
    ax2.set_title("the schedule must cover every physical length scale")
    ax2.legend(frameon=False, fontsize=8, loc="upper left"); clean(ax2, ticks=True)
    fig.tight_layout(); save(fig, "w2p2.png")


# ---------- Problem 3: Muller-Brown potential + its Boltzmann distribution ----------
def fig_p3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.3))

    xs = np.linspace(-1.6, 1.1, 300)
    ys = np.linspace(-0.4, 2.05, 300)
    X, Y = np.meshgrid(xs, ys)
    V = muller_brown(X, Y)
    Vc = np.clip(V, -150, 100)
    ax1.contourf(X, Y, Vc, levels=30, cmap="terrain_r", alpha=0.92)
    ax1.contour(X, Y, Vc, levels=18, colors="white", linewidths=0.4, alpha=0.6)
    minima = np.array([[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]])
    ax1.scatter(minima[:, 0], minima[:, 1], s=60, color=ACCENT, edgecolor="white",
                lw=1.2, zorder=5)
    for i, (mx, my) in enumerate(minima):
        ax1.text(mx + 0.06, my + 0.06, f"basin {i+1}", color=FG, fontsize=8.5)
    ax1.set_title("Muller-Brown potential energy surface")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_aspect("auto"); clean(ax1, ticks=True)

    # Boltzmann samples p(x) ~ exp(-V/kT) via grid inverse-CDF (the target the
    # score model must learn to sample).
    kT = 25.0
    logp = -(V - V.min()) / kT
    p = np.exp(logp); p /= p.sum()
    rng = np.random.default_rng(2)
    idx = rng.choice(p.size, size=4000, p=p.ravel())
    iy, ix = np.unravel_index(idx, p.shape)
    jit = 0.5 * (xs[1] - xs[0])
    sx = xs[ix] + rng.uniform(-jit, jit, idx.size)
    sy = ys[iy] + rng.uniform(-jit, jit, idx.size)
    ax2.hexbin(sx, sy, gridsize=46, cmap=DENS, extent=(-1.6, 1.1, -0.4, 2.05))
    ax2.scatter(minima[:, 0], minima[:, 1], s=42, color=FG, zorder=5)
    ax2.set_title(r"target Boltzmann samples  $p\propto e^{-V/k_BT}$")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_aspect("auto"); clean(ax2, ticks=True)
    fig.tight_layout(); save(fig, "w2p3.png")


# ---------- Problem 4: E(3)-equivariant molecular diffusion ----------
def fig_p4():
    rng = np.random.default_rng(5)
    # a small 3D "molecule": a planar 5-ring projected, used to show denoising
    # and the rotational-equivariance commuting square.
    ang = np.linspace(0, 2 * np.pi, 6)[:-1] + 0.3
    mol = np.c_[np.cos(ang), 0.62 * np.sin(ang)]
    mol -= mol.mean(0)                                       # zero center of mass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.3))

    # panel 1: one reverse-denoising trajectory from noise to the molecule
    sigmas = np.concatenate([np.linspace(2.2, 0.05, 7), [0.0]])

    def mol_score(P, s):                                     # score toward the fixed ring
        var = 0.05 ** 2 + s ** 2
        return (mol - P) / var

    x = mol + 2.2 * rng.standard_normal(mol.shape)
    traj = pf_ode_reverse(x, sigmas, mol_score)
    for j in range(len(mol)):
        ax1.plot(traj[:, j, 0], traj[:, j, 1], color=STREAM, lw=0.8, zorder=1)
    ax1.scatter(traj[0, :, 0], traj[0, :, 1], s=42, color=MUTED, zorder=3, label="noise  $x_T$")
    ax1.scatter(mol[:, 0], mol[:, 1], s=70, color=ACCENT, zorder=4, label="molecule  $x_0$")
    # draw the recovered bonds
    for i in range(len(mol)):
        j = (i + 1) % len(mol)
        ax1.plot([mol[i, 0], mol[j, 0]], [mol[i, 1], mol[j, 1]], color=ACCENT, lw=1.4, zorder=3)
    ax1.set_title("reverse diffusion denoises a point cloud into a molecule")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax1.set_aspect("equal"); clean(ax1)

    # panel 2: equivariance commuting square. An E(3)-equivariant denoiser must
    # be built from relative positions only, like one EGNN message-passing step,
    # so D(Qx) = Q D(x) holds identically. A denoiser that secretly knew a
    # canonical orientation would break this square.
    th = np.deg2rad(70)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    noisy = mol + 0.55 * rng.standard_normal(mol.shape)

    def denoise(P, eta=0.6, tau=0.8):                       # equivariant graph smoothing
        d = P[:, None, :] - P[None, :, :]                  # (n, n, 2) relative positions
        w = np.exp(-(d ** 2).sum(-1) / tau)
        np.fill_diagonal(w, 0.0)
        w = w / (w.sum(1, keepdims=True) + 1e-9)
        return P + eta * (w[..., None] * (-d)).sum(1)      # pull toward neighbors

    D = denoise(noisy)
    QD = D @ Q.T
    DQ = denoise(noisy @ Q.T)
    ax2.scatter(noisy[:, 0], noisy[:, 1], s=34, color=MUTED, label=r"noisy $x$", zorder=3)
    ax2.scatter((noisy @ Q.T)[:, 0], (noisy @ Q.T)[:, 1], s=34, color="#9bbad8",
                label=r"rotated $Qx$", zorder=3)
    ax2.scatter(QD[:, 0], QD[:, 1], s=150, facecolors="none", edgecolors=ACCENT, lw=1.5,
                label=r"$Q\,D(x)$", zorder=4)
    ax2.scatter(DQ[:, 0], DQ[:, 1], s=30, color=BLUE, label=r"$D(Qx)$", zorder=5)
    ax2.set_title(r"equivariant denoiser:  $D(Qx)=Q\,D(x)$")
    ax2.set_aspect("equal"); ax2.set_xlim(-2.4, 2.4); ax2.set_ylim(-2.4, 2.4)
    ax2.legend(frameon=False, fontsize=8, loc="upper left", ncol=2,
               columnspacing=1.0, handletextpad=0.3)
    ax2.text(0.5, -0.09, "blue dots land inside the red circles, so the square commutes",
             transform=ax2.transAxes, ha="center", va="top", color=DIM, fontsize=8.5)
    clean(ax2)
    fig.tight_layout(); save(fig, "w2p4.png")


if __name__ == "__main__":
    fig_p1()
    fig_p2()
    fig_p3()
    fig_p4()
