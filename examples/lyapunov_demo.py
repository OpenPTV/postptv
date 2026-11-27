"""
Lyapunov-exponent demo (issue #65): generates the figures used in
``doc/lyapunov.rst``.

* ``lyapunov_synthetic.png`` - noisy synthetic pairs with a known exponent.
* ``lyapunov_real.png`` - real tracers from ``data/tracers`` (Guala et al.,
  two-phase turbulence experiment, 500 fps, see ``data/readme.txt``).
* ``eulerian_ftle_synthetic.png`` - Eulerian FTLE of a steady Taylor-Green
  flow, scattered noisy particles vs. the exact reference, for 3 radii.
* ``eulerian_ftle_real.png`` - Eulerian FTLE of the real tracers, with its
  quality diagnostics.
* ``eulerian_ftle_ridges.png`` - ridges extracted from the synthetic
  Taylor-Green FTLE, against the known separatrices.

Run from the repository root::

    uv run python examples/lyapunov_demo.py [output_dir]

The output directory defaults to ``doc/_static``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowtracks import lyapunov
from flowtracks.io import trajectories_ptvis
from flowtracks.trajectory import Trajectory

ROOT = Path(__file__).resolve().parent.parent


def synthetic_pairs(lam, dt, nframes, npairs, noise, rng):
    """Pairs separating as r0*exp(lam*t), plus Gaussian position noise."""
    frames = np.arange(nframes)
    t = frames * dt
    trajs = []
    for k in range(npairs):
        r0 = rng.uniform(4e-3, 7e-3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        # Far apart, so that unrelated particles never pair up by accident.
        origin = rng.uniform(-0.5, 0.5, size=3)
        a = np.tile(origin, (nframes, 1))
        b = origin + (r0 * np.exp(lam * t))[:, None] * d
        for i, p in enumerate((a, b)):
            p = p + rng.normal(scale=noise, size=p.shape)
            trajs.append(Trajectory(p, np.zeros_like(p), frames.astype(float),
                                    2 * k + i))
    return trajs


def figure_synthetic(outdir):
    rng = np.random.default_rng(1)
    lam, dt, noise = 5.0, 1 / 500, 0.1e-3
    trajs = synthetic_pairs(lam, dt, 60, 150, noise, rng)
    r_min, r_max = 3e-3, 8e-3

    pairs, f0 = lyapunov.find_pairs(trajs, r_min, r_max, min_overlap=10)
    seps = lyapunov.pair_separations(trajs, pairs, f0)
    ftle = lyapunov.finite_time_lyapunov(
        trajs, r_min, r_max, dt=dt, fit_lags=(1, 40), min_overlap=10)
    fsle = lyapunov.finite_size_lyapunov(
        trajs, r_min, r_max, ratio=1.5, num_scales=8, dt=dt, min_overlap=10)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for lg, r in seps[:25]:
        ax[0].semilogy(lg * dt, r * 1e3, color="C0", alpha=0.3, lw=0.8)
    ax[0].set(xlabel="time since pair start, $\\tau$ [s]",
              ylabel="separation $r$ [mm]", title="(a) pair separations")

    tau = ftle.lag.values
    ax[1].plot(tau, ftle.mean_log_growth, "o", ms=3, label="data")
    ax[1].plot(tau, lam * tau, "k--", label=f"true $\\lambda$={lam:g} 1/s")
    ax[1].plot(tau, float(ftle.lyapunov) * tau, "C3-", lw=1,
               label=f"fit $\\lambda$={float(ftle.lyapunov):.2f} 1/s")
    ax[1].set(xlabel="$\\tau$ [s]", ylabel="$\\langle \\ln (r/r_0)\\rangle$",
              title="(b) finite-time exponent")
    ax[1].legend(frameon=False)

    ax[2].semilogx(fsle.delta * 1e3, fsle.fsle, "o-", label="FSLE")
    ax[2].axhline(lam, color="k", ls="--", label="true $\\lambda$")
    ax[2].set(xlabel="$\\delta$ [mm]", ylabel="$\\lambda(\\delta)$ [1/s]",
              title="(c) finite-size exponent", ylim=(0, 2 * lam))
    ax[2].legend(frameon=False)

    path = outdir / "lyapunov_synthetic.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path, "FTLE =", float(ftle.lyapunov))
    return path


def figure_real(outdir):
    dt = 1 / 500
    trajs = trajectories_ptvis(
        str(ROOT / "data/tracers/ptv_is.%d"), first=10001, last=10200,
        frate=500, traj_min_len=8)
    r_min, r_max = 2e-3, 4e-3
    fit = (8, 20)  # skip the initial ballistic/noise-dominated lags

    pairs, f0 = lyapunov.find_pairs(trajs, r_min, r_max, min_overlap=8)
    seps = lyapunov.pair_separations(trajs, pairs, f0, max_lag=20)
    ftle = lyapunov.finite_time_lyapunov(
        trajs, r_min, r_max, dt=dt, max_lag=20, fit_lags=fit, min_overlap=8)
    fsle = lyapunov.finite_size_lyapunov(
        trajs, r_min, r_max, ratio=1.3, num_scales=8, dt=dt, min_overlap=8)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ix in rng.choice(len(seps), 40, replace=False):
        lg, r = seps[ix]
        if len(lg) > 8:
            ax[0].semilogy(lg * dt, r * 1e3, color="C0", alpha=0.3, lw=0.8)
    ax[0].set(xlabel="time since pair start, $\\tau$ [s]",
              ylabel="separation $r$ [mm]",
              title=f"(a) {len(seps)} tracer pairs (40 shown)")

    tau = ftle.lag.values
    lam = float(ftle.lyapunov)
    ax[1].plot(tau, ftle.mean_log_growth, "o", ms=3)
    ax[1].axvspan(fit[0] * dt, fit[1] * dt, color="C1", alpha=0.15,
                  label="fit window")
    sel = (np.arange(len(tau)) >= fit[0]) & (np.arange(len(tau)) <= fit[1])
    b = np.polyfit(tau[sel], ftle.mean_log_growth.values[sel], 1)
    ax[1].plot(tau[sel], np.polyval(b, tau[sel]), "C3-",
               label=f"slope = {lam:.1f} 1/s")
    ax[1].set(xlabel="$\\tau$ [s]", ylabel="$\\langle \\ln (r/r_0)\\rangle$",
              title="(b) finite-time exponent")
    ax[1].legend(frameon=False)

    ok = fsle['count'].values > 0
    ax[2].semilogx(fsle.delta.values[ok] * 1e3, fsle.fsle.values[ok], "o-")
    ax[2].set(xlabel="$\\delta$ [mm]", ylabel="$\\lambda(\\delta)$ [1/s]",
              title="(c) finite-size exponent")
    ax3 = ax[2].twinx()
    ax3.bar(fsle.delta.values[ok] * 1e3, fsle['count'].values[ok],
            width=0.15 * fsle.delta.values[ok] * 1e3, color="0.8", zorder=0)
    ax3.set_ylabel("pairs per scale", color="0.5")
    ax[2].set_zorder(ax3.get_zorder() + 1)
    ax[2].patch.set_visible(False)

    path = outdir / "lyapunov_real.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path, "FTLE =", lam)
    return path


def _taylor_green_map(x0, k, amp, T, nsteps=200):
    """Flow map of u=(-sin kx cos ky, cos kx sin ky, 0)*amp, by RK4."""
    def vel(p):
        out = np.zeros_like(p)
        out[:, 0] = -amp * np.sin(k * p[:, 0]) * np.cos(k * p[:, 1])
        out[:, 1] = amp * np.cos(k * p[:, 0]) * np.sin(k * p[:, 1])
        return out
    p, h = x0.copy(), T / nsteps
    for _ in range(nsteps):
        k1 = vel(p)
        k2 = vel(p + 0.5 * h * k1)
        k3 = vel(p + 0.5 * h * k2)
        k4 = vel(p + h * k3)
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


def figure_eulerian_synthetic(outdir):
    """Eulerian FTLE of a steady Taylor-Green flow vs the exact reference."""
    rng = np.random.default_rng(2)
    L, amp, T, sigma = 0.04, 0.05, 0.4, 20e-6
    k = 2 * np.pi / L
    n_frames = 10
    dt = T / n_frames

    npart = 40000
    x0 = np.column_stack([rng.uniform(-0.01, L + 0.01, npart),
                          rng.uniform(-0.01, L + 0.01, npart),
                          rng.uniform(-0.005, 0.005, npart)])
    x1 = _taylor_green_map(x0, k, amp, T) + rng.normal(scale=sigma,
                                                      size=x0.shape)
    x0 = x0 + rng.normal(scale=sigma, size=x0.shape)
    trajs = [Trajectory(np.array([a, b]), np.zeros((2, 3)),
                        np.array([0., n_frames]), i)
             for i, (a, b) in enumerate(zip(x0, x1))]

    gx = np.linspace(0.002, L - 0.002, 60)
    grid = (gx, gx, np.array([0.0]))

    # Exact reference: central differences of the exact (noise-free) map.
    nodes = np.stack(np.meshgrid(gx, gx, [0.0], indexing="ij"),
                     -1).reshape(-1, 3)
    h = 1e-6
    fmat = np.empty((len(nodes), 3, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        fmat[:, :, i] = (_taylor_green_map(nodes + e, k, amp, T)
                         - _taylor_green_map(nodes - e, k, amp, T)) / (2 * h)
    lam = np.linalg.eigvalsh(np.einsum("nji,njk->nik", fmat, fmat))[:, -1]
    ref = (0.5 * np.log(lam) / T).reshape(60, 60)

    radii = (1.5e-3, 3e-3, 6e-3)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
    vmax = np.nanmax(ref)
    ext = [gx[0] * 1e3, gx[-1] * 1e3] * 2
    im = ax[0].imshow(ref.T, origin="lower", extent=ext, vmin=0, vmax=vmax)
    ax[0].set_title("exact reference")
    for a, R in zip(ax[1:], radii):
        res = lyapunov.eulerian_ftle(trajs, grid, 0, n_frames, radius=R,
                                     dt=dt, position_error=sigma)
        f = res.ftle.values[:, :, 0]
        a.imshow(f.T, origin="lower", extent=ext, vmin=0, vmax=vmax)
        ok = res.valid.values[:, :, 0]
        err = np.sqrt(np.nanmean((f[ok] - ref[ok]) ** 2))
        a.set_title(f"estimate, R={R * 1e3:g} mm\n"
                    f"valid {ok.mean():.0%}, rms error {err:.2f} 1/s",
                    fontsize=10)
    for a in ax:
        a.set(xlabel="x [mm]")
    ax[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, label="FTLE [1/s]", shrink=0.8)
    path = outdir / "eulerian_ftle_synthetic.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)
    return path


def figure_ridges_synthetic(outdir):
    """Ridges of the Taylor-Green FTLE, against the known separatrices."""
    rng = np.random.default_rng(2)
    L, amp, T, sigma = 0.04, 0.05, 0.4, 20e-6
    k = 2 * np.pi / L
    n_frames = 10
    dt = T / n_frames
    npart = 40000
    x0 = np.column_stack([rng.uniform(-0.01, L + 0.01, npart),
                          rng.uniform(-0.01, L + 0.01, npart),
                          rng.uniform(-0.005, 0.005, npart)])
    x1 = _taylor_green_map(x0, k, amp, T) + rng.normal(scale=sigma,
                                                      size=x0.shape)
    x0 = x0 + rng.normal(scale=sigma, size=x0.shape)
    trajs = [Trajectory(np.array([a, b]), np.zeros((2, 3)),
                        np.array([0., n_frames]), i)
             for i, (a, b) in enumerate(zip(x0, x1))]
    g = np.linspace(0.002, L - 0.002, 60)
    grid = (g, g, np.array([0.0]))
    X, Y = np.meshgrid(g, g, indexing="ij")
    lines = np.array([0.0, L / 2, L])
    dist = np.minimum(np.abs(X[..., None] - lines).min(-1),
                      np.abs(Y[..., None] - lines).min(-1))
    ext = [g[0] * 1e3, g[-1] * 1e3] * 2

    radii = (1.5e-3, 3e-3, 6e-3)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for a, R in zip(ax, radii):
        res = lyapunov.eulerian_ftle(trajs, grid, 0, n_frames, radius=R,
                                     dt=dt, position_error=sigma)
        rid = lyapunov.ftle_ridges(res, smooth=1.0, min_percentile=80)
        f = res.ftle.values[:, :, 0]
        r = rid.ridge.values[:, :, 0]
        a.imshow(f.T, origin="lower", extent=ext, vmin=0, vmax=7.5)
        for c in (L / 2 * 1e3,):
            a.axvline(c, color="w", ls="--", lw=0.8)
            a.axhline(c, color="w", ls="--", lw=0.8)
        a.plot(X[r] * 1e3, Y[r] * 1e3, "r.", ms=4)
        if r.any():
            stats = (f"{r.sum()} ridge nodes, median error "
                     f"{np.median(dist[r]) * 1e3:.1f} mm")
        else:
            stats = "no ridges found"
        a.set_title(f"R={R * 1e3:g} mm, valid "
                    f"{res.valid.values.mean():.0%}\n{stats}", fontsize=10)
        a.set(xlabel="x [mm]")
    ax[0].set_ylabel("y [mm]")
    fig.suptitle("Ridges (red) of the estimated FTLE; dashed: exact "
                 "separatrices x = L/2 and y = L/2", fontsize=10)
    path = outdir / "eulerian_ftle_ridges.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)
    return path


def figure_eulerian_real(outdir):
    """Eulerian FTLE of the real tracers, with quality diagnostics."""
    dt = 1 / 500
    trajs = trajectories_ptvis(
        str(ROOT / "data/tracers/ptv_is.%d"), first=10001, last=10200,
        frate=500, traj_min_len=3)
    t0, radius, sigma = 10050, 6e-3, 20e-6
    grid = (np.linspace(-0.01, 0.01, 17), np.linspace(-0.008, 0.008, 13),
            np.linspace(-0.08, -0.06, 9))

    n = 6
    res = lyapunov.eulerian_ftle(trajs, grid, t0, t0 + n, radius=radius,
                                 dt=dt, position_error=sigma)
    iz = res.sizes["z"] // 2

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
    f = res.ftle.values[:, :, iz]
    ext = [grid[0][0] * 1e3, grid[0][-1] * 1e3,
           grid[1][0] * 1e3, grid[1][-1] * 1e3]
    im = ax[0].imshow(f.T, origin="lower", extent=ext)
    ax[0].set(xlabel="x [mm]", ylabel="y [mm]",
              title=f"(a) FTLE, z={grid[2][iz] * 1e3:.0f} mm\n"
                    "(blank = not valid)")
    fig.colorbar(im, ax=ax[0], label="FTLE [1/s]", shrink=0.8)

    dets = res.det_F.values[res.valid.values]
    ax[1].hist(dets, bins=30, color="C0")
    ax[1].axvline(1, color="k", ls="--")
    ax[1].set(xlabel="det F", ylabel="valid nodes",
              title="(b) incompressibility check\n(should peak at 1)")

    ns = [2, 3, 4, 6, 8]
    med, lo, hi, frac, floor, ntraj = [], [], [], [], [], []
    for m in ns:
        r = lyapunov.eulerian_ftle(trajs, grid, t0, t0 + m, radius=radius,
                                   dt=dt, position_error=sigma)
        v = r.ftle.values[r.valid.values]
        ntraj.append(r.attrs["n_trajectories_used"])
        frac.append(r.valid.values.mean())
        floor.append(float(r.ftle_noise_floor.where(r.valid).mean())
                     if v.size else np.nan)
        med.append(np.median(v) if v.size else np.nan)
        lo.append(np.percentile(v, 25) if v.size else np.nan)
        hi.append(np.percentile(v, 75) if v.size else np.nan)
    T = np.array(ns) * dt * 1e3
    ax[2].fill_between(T, lo, hi, alpha=0.25, label="25-75%")
    ax[2].plot(T, med, "o-", label="median FTLE")
    ax[2].plot(T, floor, "k--", label="noise floor")
    ax[2].set(xlabel="window T [ms]", ylabel="FTLE [1/s]",
              title="(c) dependence on the window")
    ax[2].set_ylim(0, None)
    ax[2].legend(frameon=False, loc="center right")

    ax[3].plot(T, np.array(frac) * 100, "o-", color="C3")
    ax[3].set(xlabel="window T [ms]", ylabel="valid nodes [%]",
              title="(d) survivorship: valid nodes vs T")
    ax3 = ax[3].twinx()
    ax3.plot(T, ntraj, "s--", color="0.5")
    ax3.set_ylabel("trajectories spanning window", color="0.4")

    path = outdir / "eulerian_ftle_real.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path, "valid", float(res.valid.mean()),
          "median FTLE", float(res.ftle.median()))
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "doc" / "_static"
    out.mkdir(parents=True, exist_ok=True)
    figure_synthetic(out)
    figure_real(out)
    figure_eulerian_synthetic(out)
    figure_eulerian_real(out)
    figure_ridges_synthetic(out)
