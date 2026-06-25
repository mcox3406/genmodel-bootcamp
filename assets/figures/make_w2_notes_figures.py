# Figures for the Week 2 Notes and overview pages (diffusion / score models).
# Reuses the palette from make_figures.py and the mixture-of-Gaussians toy from
# make_w2_figures.py, so every score field and reverse trajectory is exact.
# Usage: python make_w2_notes_figures.py
import numpy as np
import matplotlib.pyplot as plt
from make_figures import (ACCENT, ACCENT_MID, ACCENT_SOFT, FG, MID, DIM, MUTED,
                          LINE, STREAM, DENS, clean, save)
from make_w2_figures import (MODES, mog_logp, mog_score, mog_sample, mog_var,
                             pf_ode_reverse, reverse_sde, BLUE, BLUE_SOFT, GREEN)


def density_grid(sigma, lim=3.2, n=240):
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    dens = np.exp(mog_logp(P, sigma)).reshape(X.shape)
    return X, Y, dens


# ---------- Overview: forward noising, the score, reverse generation ----------
def fig_overview():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.95))
    rng = np.random.default_rng(0)
    LIM = 3.2

    # panel 1: forward process. Clean data (red) blurs toward a wide Gaussian.
    X, Y, d_lo = density_grid(0.05, LIM)
    ax1.contourf(X, Y, d_lo, levels=14, cmap=DENS)
    pts = mog_sample(600, 1.4, rng)
    ax1.scatter(pts[:, 0], pts[:, 1], s=4, color=MID, alpha=0.35, edgecolors="none")
    ax1.set_title(r"forward:  add noise  $x_0\to x_T$", color=FG)
    ax1.annotate("", xy=(2.4, -2.4), xytext=(0.3, 0.3),
                 arrowprops=dict(arrowstyle="->", color=DIM, lw=1.3))
    ax1.text(0.5, -0.085, "data dissolves into a Gaussian", transform=ax1.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)

    # panel 2: the score field at an intermediate noise level.
    sig = 0.7
    X, Y, dens = density_grid(sig, LIM)
    ax2.contourf(X, Y, dens, levels=14, cmap=DENS)
    g = np.linspace(-2.7, 2.7, 16)
    GX, GY = np.meshgrid(g, g)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    S = mog_score(G, sig)
    n = np.hypot(S[:, 0], S[:, 1]) + 1e-9
    ax2.quiver(GX, GY, (S[:, 0] / n).reshape(GX.shape), (S[:, 1] / n).reshape(GX.shape),
               color=DIM, width=0.004, scale=34, pivot="mid", alpha=0.85)
    ax2.scatter(MODES[:, 0], MODES[:, 1], s=30, color=FG, zorder=5)
    ax2.set_title(r"the score  $\nabla_x\log p_t(x)$", color=FG)
    ax2.text(0.5, -0.085, "what the network learns", transform=ax2.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)

    # panel 3: reverse generation. PF-ODE trajectories from noise to the modes.
    sigmas = np.concatenate([np.geomspace(3.0, 0.04, 60), [0.0]])
    x = 3.0 * rng.standard_normal((900, 2))
    traj = pf_ode_reverse(x, sigmas, mog_score)
    sub = rng.choice(900, 22, replace=False)
    for j in sub:
        ax3.plot(traj[:, j, 0], traj[:, j, 1], color=STREAM, lw=0.5, alpha=0.7, zorder=1)
    ax3.scatter(traj[-1, :, 0], traj[-1, :, 1], s=4, color=ACCENT, alpha=0.5,
                edgecolors="none", zorder=2)
    ax3.scatter(MODES[:, 0], MODES[:, 1], s=30, color=FG, zorder=5)
    ax3.set_title(r"reverse:  denoise  $x_T\to x_0$", color=FG)
    ax3.text(0.5, -0.085, "integrate the probability-flow ODE", transform=ax3.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal"); clean(ax)
    fig.tight_layout(); save(fig, "w2-overview.png")


# ---------- Forward process: the perturbation kernel across noise scales ----------
def fig_forward():
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.2))
    rng = np.random.default_rng(1)
    sigs = [0.05, 0.3, 0.9, 2.5]
    for ax, s in zip(axes, sigs):
        pts = mog_sample(1400, s, rng)
        ax.scatter(pts[:, 0], pts[:, 1], s=5, color=ACCENT, alpha=0.45, edgecolors="none")
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
        snr = 1.0 / s ** 2
        ax.set_title(rf"$\sigma={s:g}$   (SNR $\approx$ {snr:.0f})" if snr >= 1
                     else rf"$\sigma={s:g}$   (SNR $\approx$ {snr:.2f})")
        clean(ax)
    axes[0].text(-3.1, 2.7, "clean data", color=MID, fontsize=9)
    axes[-1].text(-3.1, 2.7, "≈ pure noise", color=MID, fontsize=9)
    fig.suptitle(r"the forward perturbation kernel  $p_\sigma(x)=p_{\mathrm{data}}*\mathcal{N}(0,\sigma^2 I)$",
                 y=1.02, fontsize=11, color=FG)
    fig.tight_layout(); save(fig, "w2n-forward.png")


# ---------- Denoising score matching: target direction = score ----------
def fig_dsm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    rng = np.random.default_rng(3)
    sig = 0.6

    # left: a clean point x0, several noisy draws x_t, and the DSM target vectors
    # (x0 - x_t)/sigma^2 that the network regresses onto.
    x0 = MODES[0]
    X, Y, dens = density_grid(sig, 3.2)
    ax1.contourf(X, Y, dens, levels=12, cmap=DENS)
    noisy = x0 + sig * rng.standard_normal((12, 2))
    for xt in noisy:
        tgt = (x0 - xt) / sig ** 2
        ax1.arrow(xt[0], xt[1], 0.12 * tgt[0], 0.12 * tgt[1], color=BLUE,
                  width=0.012, head_width=0.09, length_includes_head=True, zorder=4)
        ax1.scatter(*xt, s=18, color=MID, zorder=3)
    ax1.scatter(*x0, s=90, color=ACCENT, edgecolor="white", lw=1.2, zorder=5)
    ax1.text(x0[0] + 0.12, x0[1] + 0.12, r"$x_0$", color=ACCENT, fontsize=11)
    ax1.set_title(r"DSM target  $\frac{x_0-x_t}{\sigma^2}$  points back to the clean sample")
    ax1.set_xlim(-2.2, 2.4); ax1.set_ylim(-1.4, 3.0); ax1.set_aspect("equal"); clean(ax1)

    # right: averaging those per-sample targets recovers the true score.
    g = np.linspace(-2.6, 2.6, 15)
    GX, GY = np.meshgrid(g, g)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    S = mog_score(G, sig)
    n = np.hypot(S[:, 0], S[:, 1]) + 1e-9
    X, Y, dens = density_grid(sig, 3.2)
    ax2.contourf(X, Y, dens, levels=12, cmap=DENS)
    ax2.quiver(GX, GY, (S[:, 0] / n).reshape(GX.shape), (S[:, 1] / n).reshape(GX.shape),
               color=DIM, width=0.004, scale=32, pivot="mid", alpha=0.9)
    ax2.scatter(MODES[:, 0], MODES[:, 1], s=30, color=FG, zorder=5)
    ax2.set_title(r"its conditional mean is the score  $\mathbb{E}[\,\cdot\mid x_t]=\nabla\log p_\sigma$")
    ax2.set_xlim(-3.0, 3.0); ax2.set_ylim(-3.0, 3.0); ax2.set_aspect("equal"); clean(ax2)
    fig.tight_layout(); save(fig, "w2n-dsm.png")


# ---------- Reverse SDE (stochastic) vs probability-flow ODE (deterministic) ----------
def fig_sde_ode():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    rng = np.random.default_rng(7)
    sigmas = np.concatenate([np.geomspace(3.0, 0.03, 60), [0.0]])
    x_init = 3.0 * rng.standard_normal((6, 2))

    for ax, title, kind in [(ax1, r"probability-flow ODE (deterministic)", "ode"),
                            (ax2, r"reverse SDE (stochastic)", "sde")]:
        X, Y, dens = density_grid(0.05, 3.2)
        ax.contourf(X, Y, dens, levels=12, cmap=DENS)
        if kind == "ode":
            traj = pf_ode_reverse(x_init.copy(), sigmas, mog_score)
        else:
            traj = reverse_sde(x_init.copy(), sigmas, mog_score, np.random.default_rng(11),
                               noise_scale=0.5)
        for j in range(x_init.shape[0]):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color=BLUE if kind == "ode" else ACCENT,
                    lw=0.9, alpha=0.8, zorder=2)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=22, color=MUTED, zorder=3)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=28, color=FG, zorder=4)
        ax.scatter(MODES[:, 0], MODES[:, 1], s=80, facecolors="none", edgecolors=DIM,
                   lw=1.0, zorder=3)
        ax.set_title(title); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal"); clean(ax)
    ax1.text(0.5, -0.085, "smooth paths, same marginals", transform=ax1.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)
    ax2.text(0.5, -0.085, "noisy paths, same marginals", transform=ax2.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)
    fig.tight_layout(); save(fig, "w2n-sde-ode.png")


# ---------- EDM design space: preconditioning + noise schedule ----------
def fig_edm():
    sd = 0.5
    sig = np.logspace(-3, 2, 400)
    c_skip = sd ** 2 / (sig ** 2 + sd ** 2)
    c_out = sig * sd / np.sqrt(sig ** 2 + sd ** 2)
    c_in = 1.0 / np.sqrt(sig ** 2 + sd ** 2)
    weight = 1.0 / c_out ** 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    ax1.plot(sig, c_skip, color=ACCENT, lw=2.0, label=r"$c_{\mathrm{skip}}$")
    ax1.plot(sig, c_out, color=BLUE, lw=2.0, label=r"$c_{\mathrm{out}}$")
    ax1.plot(sig, c_in, color=GREEN, lw=2.0, label=r"$c_{\mathrm{in}}$")
    ax1.set_xscale("log"); ax1.set_xlabel(r"noise level  $\sigma$")
    ax1.set_ylabel("coefficient"); ax1.set_title("EDM preconditioning")
    ax1.legend(frameon=False, fontsize=9, loc="center left")
    ax1.grid(True, which="both", color=LINE, lw=0.5, alpha=0.5); clean(ax1, ticks=True)

    # the EDM noise schedule (sampling) as a geometric-ish curve, plus the
    # lognormal training distribution behind it.
    rho = 7.0; N = 18
    smin, smax = 0.02, 80.0
    i = np.arange(N)
    sched = (smax ** (1 / rho) + i / (N - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    ax2.plot(i, sched, color=ACCENT, marker="o", ms=5, lw=1.4, label=r"sampling schedule ($\rho=7$)")
    ax2.set_yscale("log"); ax2.set_xlabel("sampling step  i"); ax2.set_ylabel(r"$\sigma_i$")
    ax2.set_title("noise schedule: dense where it matters")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2.grid(True, which="both", color=LINE, lw=0.5, alpha=0.5); clean(ax2, ticks=True)
    fig.tight_layout(); save(fig, "w2n-edm.png")


if __name__ == "__main__":
    fig_overview()
    fig_forward()
    fig_dsm()
    fig_sde_ode()
    fig_edm()
