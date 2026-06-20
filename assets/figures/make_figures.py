# Generates the Week 1 problem figures in the site color theme.
# Usage: python make_figures.py   (requires matplotlib + numpy)
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = pathlib.Path(__file__).parent

ACCENT = "#b3132a"
ACCENT_MID = "#e58a99"
ACCENT_SOFT = "#fdecef"
FG = "#18181b"
MID = "#3f3f46"
DIM = "#6b6b71"
MUTED = "#9a9aa0"
LINE = "#e5e5e8"
STREAM = "#bcbcc2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": FG,
    "axes.labelcolor": MID,
    "xtick.color": DIM,
    "ytick.color": DIM,
})

DENS = LinearSegmentedColormap.from_list("dens", ["#ffffff", ACCENT_SOFT, "#f5c1ca", ACCENT_MID])


def clean(ax, ticks=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


# ---------- Week 1 overview: the three objects that carry the week ----------
def fig_w1_overview():
    # one autonomous velocity field drives all three panels
    def UV(X, Y):
        r2 = X * X + Y * Y
        k = 1.15 * np.exp(-0.5 * r2)          # localized swirl
        sh = 0.34                              # symmetric shear (stretches the density)
        return -k * Y + sh * Y - 0.10 * X, k * X + sh * X - 0.10 * Y

    def vel2(P):
        u, v = UV(P[:, 0], P[:, 1])
        return np.stack([u, v], 1)

    def flow(P, T=1.0, steps=70):
        h = T / steps
        X = P.astype(float).copy()
        for _ in range(steps):
            k1 = vel2(X); k2 = vel2(X + 0.5 * h * k1)
            k3 = vel2(X + 0.5 * h * k2); k4 = vel2(X + h * k3)
            X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    LIM = 2.6
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.95))

    # panel 1: the velocity field
    xs = np.linspace(-LIM, LIM, 34)
    X, Y = np.meshgrid(xs, xs)
    U, V = UV(X, Y)
    ax1.streamplot(X, Y, U, V, color=STREAM, density=1.05, linewidth=0.75, arrowsize=0.8)
    ax1.set_title(r"velocity field  $u_t(x)$", color=FG)
    ax1.text(0.5, -0.085, "the vector a network learns", transform=ax1.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)

    # panel 2: the flow map as a warped grid (diffeomorphism)
    g = np.linspace(-2.15, 2.15, 11)
    tt = np.linspace(-2.15, 2.15, 90)
    for gx in g:
        line = np.c_[np.full_like(tt, gx), tt]
        ax2.plot(line[:, 0], line[:, 1], color=LINE, lw=0.6, zorder=1)
        w = flow(line)
        ax2.plot(w[:, 0], w[:, 1], color=ACCENT, lw=0.9, alpha=0.7, zorder=2)
    for gy in g:
        line = np.c_[tt, np.full_like(tt, gy)]
        ax2.plot(line[:, 0], line[:, 1], color=LINE, lw=0.6, zorder=1)
        w = flow(line)
        ax2.plot(w[:, 0], w[:, 1], color=ACCENT, lw=0.9, alpha=0.7, zorder=2)
    ax2.set_title(r"flow map  $\psi_{0,1}$  warps space", color=FG)
    ax2.text(0.5, -0.085, "a diffeomorphism (gray grid to red)", transform=ax2.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)

    # panel 3: the pushforward of a density
    rng = np.random.default_rng(0)
    P0 = 0.62 * rng.standard_normal((45000, 2))
    P1 = flow(P0)
    ax3.hexbin(P1[:, 0], P1[:, 1], gridsize=58, cmap=DENS, extent=(-LIM, LIM, -LIM, LIM),
               linewidths=0.0)
    th = np.linspace(0, 2 * np.pi, 120)
    for s in (0.62, 1.24):
        ax3.plot(s * np.cos(th), s * np.sin(th), color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=3)
    ax3.text(0.0, 0.0, r"$p_0$", color=DIM, fontsize=9.5, ha="center", va="center", zorder=4)
    ax3.text(-1.85, 1.7, r"$p_1=(\psi_{0,1})_\# p_0$", color=ACCENT, fontsize=10, zorder=4)
    ax3.set_title(r"pushforward transports the density", color=FG)
    ax3.text(0.5, -0.085, "evolves by the continuity equation", transform=ax3.transAxes,
             ha="center", va="top", color=DIM, fontsize=9)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal"); clean(ax)
    fig.tight_layout()
    save(fig, "w1-overview.png")


# ---------- Problem 1: smoothed score field + benzene relabeling ----------
def fig_p1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    rng = np.random.default_rng(3)
    ang = np.linspace(0.3, 2.7, 9)
    Z = np.c_[1.5 * np.cos(ang), 1.5 * np.sin(ang) - 0.5] + 0.12 * rng.standard_normal((9, 2))
    sig = 0.45

    xs = np.linspace(-2.7, 2.7, 220)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X, Y], -1)
    d2 = ((P[..., None, :] - Z) ** 2).sum(-1)          # (ny, nx, N)
    q = np.exp(-d2 / (2 * sig**2)).mean(-1)
    ax1.contourf(X, Y, q, levels=14, cmap=DENS)
    ax1.contour(X, Y, q, levels=14, colors="#eccdd3", linewidths=0.5)

    g = np.linspace(-2.45, 2.45, 15)
    GX, GY = np.meshgrid(g, g)
    G = np.stack([GX, GY], -1)
    gd2 = ((G[..., None, :] - Z) ** 2).sum(-1)
    W = np.exp(-gd2 / (2 * sig**2))
    W = W / W.sum(-1, keepdims=True)
    S = (W[..., None] * (Z - G[..., None, :])).sum(-2) / sig**2
    n = np.hypot(S[..., 0], S[..., 1]) + 1e-12
    ax1.quiver(GX, GY, S[..., 0] / n, S[..., 1] / n, color=DIM, width=0.0042,
               scale=30, pivot="mid", alpha=0.85)
    ax1.scatter(Z[:, 0], Z[:, 1], s=22, color=FG, zorder=5)
    ax1.set_title(r"smoothed density $q_\sigma$ and its score $\nabla\log q_\sigma$")
    ax1.set_aspect("equal")
    clean(ax1)

    def benzene(ax, cx, labels):
        a = np.deg2rad(np.arange(90, -270, -60))
        C = np.c_[0.85 * np.cos(a) + cx, 0.85 * np.sin(a)]
        H = np.c_[1.55 * np.cos(a) + cx, 1.55 * np.sin(a)]
        for i in range(6):
            j = (i + 1) % 6
            ax.plot(*zip(C[i], C[j]), color=MID, lw=1.4, zorder=1)
            ax.plot(*zip(C[i], H[i]), color=MUTED, lw=1.0, zorder=1)
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + 0.5 * np.cos(th), 0.5 * np.sin(th), color=MUTED, lw=0.8, ls=(0, (3, 3)))
        for i in range(6):
            ax.scatter(*C[i], s=190, color="white", edgecolor=FG, lw=1.1, zorder=2)
            ax.text(*C[i], str(labels[i]), ha="center", va="center", fontsize=7.5,
                    color=FG, zorder=3)
            ax.scatter(*H[i], s=150, color="white", edgecolor=MUTED, lw=1.0, zorder=2)
            ax.text(*H[i], str(labels[i] + 6), ha="center", va="center", fontsize=6.5,
                    color=DIM, zorder=3)

    benzene(ax2, -1.85, [1, 2, 3, 4, 5, 6])
    benzene(ax2, 1.85, [4, 1, 6, 2, 5, 3])
    ax2.text(0, 0, "=", ha="center", va="center", fontsize=20, color=ACCENT)
    ax2.set_xlim(-3.7, 3.7)
    ax2.set_ylim(-2.1, 2.3)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(r"one benzene, two of its $6!\,\cdot\,6! = 518{,}400$ labelings")
    fig.tight_layout()
    save(fig, "w1p1.png")


# ---------- Problem 2: three analytic flows acting on a circle ----------
def fig_p2():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    xs = np.linspace(-2.6, 2.6, 30)
    X, Y = np.meshgrid(xs, xs)
    th = np.linspace(0, 2 * np.pi, 200)
    circ = np.c_[1.5 * np.cos(th), 1.5 * np.sin(th)]
    mk_ang = np.deg2rad([90, 210, 330])
    P0 = np.c_[1.5 * np.cos(mk_ang), 1.5 * np.sin(mk_ang)]
    mcol = [ACCENT, FG, MUTED]

    def base(ax, U, V, title):
        ax.streamplot(X, Y, U, V, color=STREAM, density=0.9, linewidth=0.7, arrowsize=0.7)
        ax.plot(circ[:, 0], circ[:, 1], color=DIM, lw=1.1, ls=(0, (4, 3)))
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_title(title)
        clean(ax)

    # contraction
    ax = axes[0]
    base(ax, -X, -Y, r"contraction  $u(x)=-\theta x$")
    e = np.exp(-1.0)
    ax.plot(e * circ[:, 0], e * circ[:, 1], color=ACCENT, lw=1.6)
    for p, c in zip(P0, mcol):
        ax.plot([p[0], e * p[0]], [p[1], e * p[1]], color=c, lw=1.0, alpha=0.8)
        ax.scatter(*p, s=26, color=c, zorder=5)
        ax.scatter(*(e * p), s=26, color=c, zorder=5, marker="s")

    # translation
    ax = axes[1]
    c_vec = np.array([1.0, 0.45])
    base(ax, np.ones_like(X) * c_vec[0], np.ones_like(Y) * c_vec[1],
         r"translation  $u(x)=c$")
    ax.plot(circ[:, 0] + c_vec[0], circ[:, 1] + c_vec[1], color=ACCENT, lw=1.6)
    for p, c in zip(P0, mcol):
        q = p + c_vec
        ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=1.0, alpha=0.8)
        ax.scatter(*p, s=26, color=c, zorder=5)
        ax.scatter(*q, s=26, color=c, zorder=5, marker="s")

    # rotation
    ax = axes[2]
    base(ax, -Y, X, r"rotation  $u(x)=Ax$")
    ax.plot(circ[:, 0], circ[:, 1], color=ACCENT, lw=1.6, alpha=0.65)
    w = np.deg2rad(75)
    R = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    for p, c in zip(P0, mcol):
        a0 = np.arctan2(p[1], p[0])
        arc = np.linspace(a0, a0 + w, 40)
        ax.plot(1.5 * np.cos(arc), 1.5 * np.sin(arc), color=c, lw=1.0, alpha=0.8)
        ax.scatter(*p, s=26, color=c, zorder=5)
        ax.scatter(*(R @ p), s=26, color=c, zorder=5, marker="s")

    fig.tight_layout()
    save(fig, "w1p2.png")


# ---------- Problem 3: Euler compositions + Riemann sum of divergence ----------
def fig_p3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    a, b = -0.5, 2.0
    x0 = np.array([1.9, 0.0])
    T = 2.2

    def exact(t):
        r = np.exp(a * t)
        return r * np.array([np.cos(b * t) * x0[0] - np.sin(b * t) * x0[1],
                             np.sin(b * t) * x0[0] + np.cos(b * t) * x0[1]])

    ts = np.linspace(0, T, 400)
    ex = np.array([exact(t) for t in ts])
    ax1.plot(ex[:, 0], ex[:, 1], color=FG, lw=1.7, label="exact flow", zorder=3)

    def euler(K):
        h = T / K
        x = x0.copy()
        pts = [x.copy()]
        for _ in range(K):
            u = np.array([a * x[0] - b * x[1], b * x[0] + a * x[1]])
            x = x + h * u
            pts.append(x.copy())
        return np.array(pts)

    e5 = euler(5)
    e20 = euler(20)
    ax1.plot(e5[:, 0], e5[:, 1], color=ACCENT, lw=1.2, marker="o", ms=4, label="Euler, K = 5")
    ax1.plot(e20[:, 0], e20[:, 1], color=ACCENT_MID, lw=1.1, marker="o", ms=2.5, label="Euler, K = 20")
    ax1.scatter(*x0, s=34, color=FG, zorder=5)
    ax1.annotate(r"$x_0$", x0 + [0.1, 0.12], color=MID)
    ax1.set_aspect("equal")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("Euler compositions approach the exact flow")
    clean(ax1)

    # 1D: u(x) = -tanh(x), divergence u'(x) = -sech^2(x)
    x0s, Ts = 1.6, 2.0
    up = lambda x: -1.0 / np.cosh(x) ** 2
    u = lambda x: -np.tanh(x)

    def riemann(K):
        h = Ts / K
        x, s = x0s, 0.0
        for _ in range(K):
            s += h * up(x)
            x = x + h * u(x)
        return s

    # dense reference via RK4 on the joint (x, integral) system
    def ref(K=20000):
        h = Ts / K
        x, s = x0s, 0.0
        for _ in range(K):
            k1x, k1s = u(x), up(x)
            k2x, k2s = u(x + h / 2 * k1x), up(x + h / 2 * k1x)
            k3x, k3s = u(x + h / 2 * k2x), up(x + h / 2 * k2x)
            k4x, k4s = u(x + h * k3x), up(x + h * k3x)
            x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            s += h / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
        return s

    Ks = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    vals = [riemann(int(K)) for K in Ks]
    r = ref()
    ax2.axhline(r, color=FG, lw=1.1, ls=(0, (5, 3)))
    ax2.text(2.1, r + 0.008, r"$\int_0^T \nabla\!\cdot u\, dt$", va="bottom", color=MID, fontsize=9)
    ax2.semilogx(Ks, vals, color=ACCENT, marker="o", ms=5, lw=1.1)
    ax2.set_xlabel("number of Euler steps  K")
    ax2.set_ylabel(r"$\sum_k h\, \nabla\!\cdot u(x_k)$")
    ax2.set_title("the log-det Riemann sum converges")
    ax2.set_xlim(1.6, 900)
    clean(ax2, ticks=True)

    fig.tight_layout()
    save(fig, "w1p3.png")


# ---------- Problem 4: the four (plus one) velocity fields ----------
def fig_p4():
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
    xs = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(xs, xs)
    R2 = X**2 + Y**2
    fields = [
        (-X, -Y, r"$u_1(x) = -\theta x$"),
        (np.ones_like(X), 0.5 * np.ones_like(Y), r"$u_2(x) = c$"),
        (-Y, X, r"$u_3(x) = (-x_2,\ x_1)$"),
        (-3 * Y * np.exp(-R2), 3 * X * np.exp(-R2), r"$u_4(x) = \alpha\,(-x_2,\ x_1)\,e^{-\|x\|^2}$"),
    ]
    for ax, (U, V, title) in zip(axes.flat, fields):
        ax.streamplot(X, Y, U, V, color=STREAM, density=1.0, linewidth=0.75, arrowsize=0.75)
        ax.set_title(title)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        clean(ax)
    fig.tight_layout()
    save(fig, "w1p4.png")


# ---------- Problem 5: base, two moons, straight conditional paths ----------
def fig_p5():
    rng = np.random.default_rng(11)
    n = 320
    t = rng.uniform(0, np.pi, n)
    upper = np.c_[np.cos(t), np.sin(t)]
    lower = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    moons = np.vstack([upper, lower])
    moons += 0.07 * rng.standard_normal(moons.shape)
    moons = 1.7 * (moons - moons.mean(0))

    base = rng.standard_normal((n, 2))

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    idx = rng.choice(n, 30, replace=False)
    midx = rng.choice(2 * n, 30, replace=False)
    for i, j in zip(idx, midx):
        ax.plot([base[i, 0], moons[j, 0]], [base[i, 1], moons[j, 1]],
                color=MID, lw=0.7, alpha=0.3, zorder=1)
    ax.scatter(base[:, 0], base[:, 1], s=7, color=STREAM, zorder=2, label=r"$x_0 \sim \mathcal{N}(0, I)$")
    ax.scatter(moons[:, 0], moons[:, 1], s=7, color=ACCENT, zorder=3, label=r"$x_1 \sim p_{\mathrm{data}}$")
    ax.legend(frameon=False, fontsize=9, loc="upper left", handletextpad=0.1)
    ax.set_title(r"straight conditional paths $x_t = (1-t)\,x_0 + t\,x_1$ cross")
    ax.set_aspect("equal")
    clean(ax)
    fig.tight_layout()
    save(fig, "w1p5.png")


# ---------- Problem 6: three candidate fields on a point cloud ----------
def fig_p6():
    rng = np.random.default_rng(7)
    # rejection-sample a well-spread cloud
    pts = []
    while len(pts) < 7:
        p = rng.uniform(-1.6, 1.6, 2)
        if all(np.linalg.norm(p - q) > 0.75 for q in pts):
            pts.append(p)
    R = np.array(pts) + np.array([0.35, 0.2])   # offset so COM != origin
    com = R.mean(0)

    u1 = -R
    u2 = R - com
    d = R[:, None, :] - R[None, :, :]
    w = np.exp(-(d**2).sum(-1))
    np.fill_diagonal(w, 0)
    u3 = (w[..., None] * (-d)).sum(1)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7))
    titles = [r"$u^{(1)}_i = -r_i$", r"$u^{(2)}_i = r_i - \bar r$",
              r"$u^{(3)}_i = \sum_j e^{-\|r_i - r_j\|^2}(r_j - r_i)$"]
    scales = [0.42, 0.55, 1.0]
    for ax, U, title, s in zip(axes, [u1, u2, u3], titles, scales):
        for a, b in [(i, j) for i in range(7) for j in range(i + 1, 7)]:
            ax.plot(*zip(R[a], R[b]), color=LINE, lw=0.6, zorder=1)
        ax.quiver(R[:, 0], R[:, 1], s * U[:, 0], s * U[:, 1], color=ACCENT,
                  angles="xy", scale_units="xy", scale=1, width=0.009, zorder=3)
        ax.scatter(R[:, 0], R[:, 1], s=52, color=FG, zorder=4)
        ax.plot(0, 0, marker="+", ms=11, color=DIM, mew=1.4, zorder=2)
        ax.plot(*com, marker="x", ms=8, color=MID, mew=1.4, zorder=2)
        ax.set_xlim(-2.4, 2.6)
        ax.set_ylim(-2.2, 2.4)
        ax.set_aspect("equal")
        ax.set_title(title)
        clean(ax)
    axes[0].text(0.08, -0.18, "origin", color=DIM, fontsize=8)
    axes[0].text(com[0] + 0.08, com[1] + 0.06, "center of mass", color=MID, fontsize=8)
    fig.tight_layout()
    save(fig, "w1p6.png")


if __name__ == "__main__":
    fig_p1()
    fig_p2()
    fig_p3()
    fig_p4()
    fig_p5()
    fig_p6()
