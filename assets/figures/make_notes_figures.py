# Figures for the Week 1 Notes page. Reuses the palette and helpers from
# make_figures.py. Run with a matplotlib + numpy environment.
import numpy as np
import make_figures as base
from make_figures import ACCENT, ACCENT_MID, FG, MID, DIM, MUTED, LINE, STREAM, DENS, clean, save
import matplotlib.pyplot as plt

BLUE = "#2f6fb3"
BLUE_SOFT = "#cfe0f0"


# ---------- Simulating a flow: Euler vs Heun vs RK4 ----------
def fig_solvers():
    a, w = 0.35, 1.9                      # contraction + rotation, exact via matrix exp
    def vel(P):
        x, y = P[..., 0], P[..., 1]
        return np.stack([-a * x - w * y, w * x - a * y], -1)
    def exact(t, x0):
        r = np.exp(-a * t)
        return r * np.array([np.cos(w * t) * x0[0] - np.sin(w * t) * x0[1],
                             np.sin(w * t) * x0[0] + np.cos(w * t) * x0[1]])
    x0 = np.array([2.0, 0.0]); T = 2.6

    def integrate(method, N):
        h = T / N; X = x0.astype(float).copy(); pts = [X.copy()]
        for _ in range(N):
            if method == "euler":
                X = X + h * vel(X)
            elif method == "heun":
                k1 = vel(X); k2 = vel(X + h * k1); X = X + h / 2 * (k1 + k2)
            else:
                k1 = vel(X); k2 = vel(X + h / 2 * k1); k3 = vel(X + h / 2 * k2); k4 = vel(X + h * k3)
                X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            pts.append(X.copy())
        return np.array(pts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ts = np.linspace(0, T, 400)
    ex = np.array([exact(t, x0) for t in ts])
    ax1.plot(ex[:, 0], ex[:, 1], color=FG, lw=1.8, label="exact", zorder=5)
    for m, col, lab in [("euler", ACCENT, "Euler"), ("heun", BLUE, "Heun"), ("rk4", "#1f9d6b", "RK4")]:
        p = integrate(m, 12)
        ax1.plot(p[:, 0], p[:, 1], color=col, lw=1.1, marker="o", ms=3.2, label=f"{lab}, N=12", alpha=0.9)
    ax1.scatter(*x0, s=30, color=FG, zorder=6)
    ax1.set_aspect("equal"); ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("same step count, different accuracy"); clean(ax1)

    Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    xe = exact(T, x0)
    for m, col, lab in [("euler", ACCENT, "Euler (order 1)"), ("heun", BLUE, "Heun (order 2)"), ("rk4", "#1f9d6b", "RK4 (order 4)")]:
        errs = [np.linalg.norm(integrate(m, int(N))[-1] - xe) for N in Ns]
        ax2.loglog(Ns, errs, color=col, marker="o", ms=4, lw=1.1, label=lab)
    ax2.set_xlabel("number of steps  N"); ax2.set_ylabel("final error at  t = T")
    ax2.legend(frameon=False, fontsize=8); ax2.set_title("global error vs step count")
    ax2.grid(True, which="both", color=LINE, lw=0.5, alpha=0.6); clean(ax2, ticks=True)
    fig.tight_layout(); save(fig, "w1n-solvers.png")


# ---------- Instantaneous change of variables: tracked vs analytic log-density ----------
def fig_logdensity():
    # contraction u(x) = -theta x in 2D. div u = -2 theta, so d/dt log p_t(X_t) = 2 theta.
    th = 0.9
    rng = np.random.default_rng(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    P0 = rng.standard_normal((1400, 2))
    T = 1.8
    for tt, al in [(0.0, 0.16), (0.9, 0.28), (T, 0.5)]:
        Pt = np.exp(-th * tt) * P0
        ax1.scatter(Pt[:, 0], Pt[:, 1], s=4, color=ACCENT, alpha=al, edgecolors="none")
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3); ax1.set_aspect("equal")
    ax1.set_title(r"contraction concentrates the density"); clean(ax1)
    ax1.text(2.0, 2.4, r"$t=0$", color=MID, fontsize=9)
    ax1.text(0.55, 0.75, r"$t=T$", color=ACCENT, fontsize=9)

    # track log p along trajectories by integrating d/dt log p = -div u = 2 theta
    ts = np.linspace(0, T, 200)
    starts = P0[:6]
    for x0 in starts:
        Xt = np.exp(-th * ts)[:, None] * x0
        logp_analytic = (-np.log(2 * np.pi) + 2 * th * ts) - 0.5 * (Xt ** 2).sum(1) * 1.0
        # analytic: p_t = N(0, e^{-2 th t} I); compute exactly
        var = np.exp(-2 * th * ts)
        logp_exact = -np.log(2 * np.pi) - np.log(var) - 0.5 * (Xt ** 2).sum(1) / var
        logp_track = logp_exact[0] + 2 * th * ts                      # integrate -div u
        ax2.plot(ts, logp_exact, color=BLUE, lw=2.4, alpha=0.5)
        ax2.plot(ts, logp_track, color=ACCENT, lw=1.0, ls=(0, (4, 2)))
    ax2.plot([], [], color=BLUE, lw=2.4, alpha=0.5, label="analytic  " + r"$\log p_t(X_t)$")
    ax2.plot([], [], color=ACCENT, lw=1.0, ls=(0, (4, 2)), label=r"tracked via $-\nabla\!\cdot u$")
    ax2.set_xlabel("time  t"); ax2.set_ylabel(r"$\log p_t(X_t)$")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_title("the log-density identity holds along trajectories")
    clean(ax2, ticks=True)
    fig.tight_layout(); save(fig, "w1n-logdensity.png")


# ---------- Hutchinson trace estimator ----------
def fig_hutchinson():
    rng = np.random.default_rng(4)
    D = 24
    A = rng.standard_normal((D, D))
    np.fill_diagonal(A, rng.standard_normal(D) * 5.0)   # large diagonal exposes Rademacher's edge
    true = np.trace(A)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    M = 4000
    for kind, col in [("Rademacher", ACCENT), ("Gaussian", BLUE)]:
        ests = np.empty(M)
        run = 0.0
        for i in range(M):
            eps = rng.choice([-1.0, 1.0], D) if kind == "Rademacher" else rng.standard_normal(D)
            run += eps @ A @ eps
            ests[i] = run / (i + 1)
        ax1.plot(np.arange(1, M + 1), ests, color=col, lw=1.0, label=kind)
    ax1.axhline(true, color=FG, lw=1.1, ls=(0, (5, 3)))
    ax1.text(M, true, r"  $\mathrm{tr}(A)$", va="center", color=MID, fontsize=9)
    ax1.set_xscale("log"); ax1.set_xlabel("number of probe vectors")
    ax1.set_ylabel(r"running estimate of $\mathrm{tr}(A)$")
    ax1.legend(frameon=False, fontsize=8.5); ax1.set_title("unbiased, and it converges")
    ax1.grid(True, which="both", color=LINE, lw=0.5, alpha=0.5); clean(ax1, ticks=True)

    # single-probe variance, Rademacher vs Gaussian
    K = 20000
    epsR = rng.choice([-1.0, 1.0], (K, D))
    epsG = rng.standard_normal((K, D))
    vR = np.einsum("ki,ij,kj->k", epsR, A, epsR)
    vG = np.einsum("ki,ij,kj->k", epsG, A, epsG)
    ax2.hist(vG, bins=60, color=BLUE_SOFT, edgecolor=BLUE, lw=0.5, density=True, label=f"Gaussian  (var {vG.var():.0f})")
    ax2.hist(vR, bins=60, color="#f6d4da", edgecolor=ACCENT, lw=0.5, density=True, alpha=0.8,
             label=f"Rademacher  (var {vR.var():.0f})")
    ax2.axvline(true, color=FG, lw=1.1, ls=(0, (5, 3)))
    ax2.set_xlabel(r"single-probe value $\epsilon^\top A\,\epsilon$"); ax2.set_ylabel("density")
    ax2.legend(frameon=False, fontsize=8.5); ax2.set_title("Rademacher probes have lower variance")
    clean(ax2, ticks=True)
    fig.tight_layout(); save(fig, "w1n-hutchinson.png")


# ---------- Invariance vs equivariance ----------
def fig_equivariance():
    rng = np.random.default_rng(2)
    pts = []
    while len(pts) < 6:
        p = rng.uniform(-1.3, 1.3, 2)
        if all(np.linalg.norm(p - q) > 0.7 for q in pts):
            pts.append(p)
    X = np.array(pts)
    ang = np.deg2rad(80)
    Q = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    def flow(P, A, T=1.0, steps=60):
        h = T / steps; Z = P.astype(float).copy()
        for _ in range(steps):
            Z = Z + h * (Z @ A.T)
        return Z
    A_eq = np.array([[-0.35, 0.0], [0.0, -0.35]])      # contraction: rotation-equivariant
    A_neq = np.array([[0.0, 0.9], [0.0, 0.0]])         # shear: not rotation-equivariant

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, A, title, ok in [(ax1, A_eq, r"equivariant field   $u(Qx)=Q\,u(x)$", True),
                             (ax2, A_neq, r"non-equivariant field   $u(Qx)\neq Q\,u(x)$", False)]:
        phiX = flow(X, A)
        QphiX = phiX @ Q.T                              # rotate, then... (flow then rotate)
        phiQX = flow(X @ Q.T, A)                        # rotate first, then flow
        ax.scatter(X[:, 0], X[:, 1], s=34, color=MUTED, label=r"$x$", zorder=3)
        ax.scatter((X @ Q.T)[:, 0], (X @ Q.T)[:, 1], s=34, color="#9bbad8", label=r"$Qx$", zorder=3)
        ax.scatter(QphiX[:, 0], QphiX[:, 1], s=130, facecolors="none", edgecolors=ACCENT, lw=1.4,
                   label=r"$Q\,\phi(x)$", zorder=4)
        ax.scatter(phiQX[:, 0], phiQX[:, 1], s=30, color=BLUE, label=r"$\phi(Qx)$", zorder=5)
        ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.set_title(title); clean(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.3)
    ax1.text(0.5, -0.10, "blue dots land inside the red circles, so the square commutes",
             transform=ax1.transAxes, ha="center", va="top", color=DIM, fontsize=8.5)
    ax2.text(0.5, -0.10, "blue dots miss the red circles, so order matters",
             transform=ax2.transAxes, ha="center", va="top", color=DIM, fontsize=8.5)
    fig.tight_layout(); save(fig, "w1n-equivariance.png")


if __name__ == "__main__":
    fig_solvers()
    fig_logdensity()
    fig_hutchinson()
    fig_equivariance()
