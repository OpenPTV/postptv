# /// script
# requires-python = ">=3.10,<3.15"
#
# [tool.marimo-studio]
# default = "story"
#
# [tool.marimo-studio.cells]
# cell-2 = {ref = "cell:v1:ac7f064b08a58e69079ee5680f455c832cfa141de83e4b9b817756d70aa985cd:adfc04e80bed0075bec088f0372d1d253414dfe06e774b90a4a3cfe507d85e6c:0"}
# cell-3 = {ref = "cell:v1:7efbdd1e07ffb5fe906f694fb966327a64f136d5661beefd6c4c1f43eeeb2cff:7efbdd1e07ffb5fe906f694fb966327a64f136d5661beefd6c4c1f43eeeb2cff:0"}
# cell-4 = {ref = "cell:v1:1d4b8f55054fd14a4767b97e76689241d73568496ebe75b876cc042eae65e6c6:c59e7ec0d967ac935100add462589deb5afea6697ca10f1ce38d30049ea13a53:0"}
# cell-5 = {ref = "cell:v1:f8c3d4e9fa91e0dddf91c21fed58ebf80104f54ebdea33b90ed3f86350b2eb04:f8c3d4e9fa91e0dddf91c21fed58ebf80104f54ebdea33b90ed3f86350b2eb04:0"}
# cell-6 = {ref = "cell:v1:a52dc786f6841c46f9a02592e727549718e14e92ab19551cd81b088a91a8e678:16635c56fff37b6a70fc8d46597ffab9ca0d5a9c73fb5f62a185872283a5ae27:0"}
# cell-7 = {ref = "cell:v1:02e3b277df9b44f7f5cc833e24e3d4ae9feed09db8f0960aed8f9d86f2b8360e:02e3b277df9b44f7f5cc833e24e3d4ae9feed09db8f0960aed8f9d86f2b8360e:0"}
# cell-8 = {ref = "cell:v1:3035af51fb597edc9f19aab11ef0fdd2ea858bbeaf8729ac79851990efc31d5d:4d2b39fd3119993108263c5da62cda194c900771d4ba6ec75002785e7c857d3a:0"}
# cell-9 = {ref = "cell:v1:a5b90591d9f19bebf9689fd54d2dc8d4a6b5d1ef6058a3a50cd90c6520f95bd8:7eb65be666c706abcc663d25a480e767f3ee4285e7c0ae8107e9181d3361cec0:0"}
# cell-10 = {ref = "cell:v1:d7a06cb38390b221d13d211235e59e350a1ee4c6359083297e4105c30c209223:22015848cc1979e59b420f5551ae55b2d677f4b88bfbebb7fe2ec89921aede12:0"}
# cell-11 = {ref = "cell:v1:4606f32a48260fd605679e9e740590fbdff83ab1878086e9d249548fcbb1a1f3:4606f32a48260fd605679e9e740590fbdff83ab1878086e9d249548fcbb1a1f3:0"}
# cell-12 = {ref = "cell:v1:10a84ed0af14f488655f6e96ed88f9e2332cb5759a3fa6ba57321c96a763ad9c:bb80d7794098a12f0862db6a6b07814e5e398374bef65f03621c9024966e477f:0"}
# cell-13 = {ref = "cell:v1:b06eb8407f141b3ef4ee68816afa8fd4a74fbfbcf4f5d19df2eb0f1310918101:b06eb8407f141b3ef4ee68816afa8fd4a74fbfbcf4f5d19df2eb0f1310918101:0"}
# cell-14 = {ref = "cell:v1:6d1229e3b501814826dc18ea6ec29c6f6a19575dbc4b0acd8faf3edf970cb597:5fd11b3352de5cdd57cb58db5af6e6515c23798eb9efa19e9abde9392168ebb5:0"}
# cell-15 = {ref = "cell:v1:e3c2b6e4c6bc8b533a7cf17f088ebbd06fac02e7a578d08d5166e0e714277a1a:e3c2b6e4c6bc8b533a7cf17f088ebbd06fac02e7a578d08d5166e0e714277a1a:0"}
# cell-16 = {ref = "cell:v1:ccb0f332b006d3e86343cfc5104992322f4f9c45fa05172fc6c48f976d0d4a71:26d6fc87b0c0bd8d34b471c1682fcde01dd0372d5d7dfa93751f05233178c154:0"}
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flowtracks.trajectory import Trajectory
    from flowtracks.smoothing import savitzky_golay
    from flowtracks.stitching import stitch_trajectories

    return Trajectory, mo, np, plt, savitzky_golay, stitch_trajectories


@app.cell
def _(mo):
    mo.md("""
    # flowtracks basics — from particles to fields

    **flowtracks** (`pip install flowtracks`) is the post-processing stage of a 3D-PTV pipeline:
    it turns per-frame particle positions into **trajectories** (Lagrangian),
    then into **Eulerian fields**, phase averages, and ParaView exports.

    This notebook uses small **synthetic** trajectories so it runs anywhere —
    the same calls work on real OpenPTV output.
    """)
    return


@app.cell
def _(mo):
    n_traj_ui = mo.ui.slider(start=10, stop=120, step=10, value=40, label="Number of trajectories")
    segs_ui = mo.ui.slider(start=20, stop=100, step=10, value=50, label="Max frames per trajectory")
    noise_ui = mo.ui.slider(start=0.0, stop=5.0, step=0.5, value=1.5, label="Position noise [mm]")
    mo.vstack([mo.md("### Controls — everything below reacts"), n_traj_ui, segs_ui, noise_ui])
    return n_traj_ui, noise_ui, segs_ui


@app.cell
def _(Trajectory, mo, n_traj_ui, noise_ui, np, segs_ui):
    _rng = np.random.default_rng(7)
    _n_traj = int(n_traj_ui.value)
    _n_t = int(segs_ui.value)
    _noise_mm = float(noise_ui.value)
    fps = 100.0
    _dt = 1.0 / fps

    trajs = []
    _min_len = max(8, _n_t - 30)
    for _i in range(_n_traj):
        _n_i = int(_rng.integers(_min_len, _n_t + 1))
        _frames = np.arange(_n_i)
        _t = _frames * _dt  # physical time [s] for the synthetic flow
        # helical drift + per-track offset: stands in for a vortex-ish lab flow
        _phase = _rng.uniform(0, 2 * np.pi)
        _r = _rng.uniform(8, 25)  # mm
        _omega = _rng.uniform(1.5, 4.0)  # rad/s
        _cx, _cy = _rng.uniform(-40, 40, 2)
        _x = _cx + _r * np.cos(_omega * _t + _phase) + 30 * _t
        _y = _cy + _r * np.sin(_omega * _t + _phase) + 10 * _t
        _z = _rng.uniform(-30, 30) + 5 * np.sin(2 * _omega * _t + _phase)
        _pos_mm = np.stack([_x, _y, _z], axis=1)
        _pos_noisy = _pos_mm + _rng.normal(0, _noise_mm, _pos_mm.shape)
        _pos_m = _pos_noisy / 1000.0  # flowtracks uses SI: metres
        _vel_m = np.gradient(_pos_m, _dt, axis=0)
        trajs.append(Trajectory(_pos_m, _vel_m, _frames, trajid=_i))

    speeds = np.concatenate([np.linalg.norm(_t.velocity(), axis=1) for _t in trajs]) * 1000.0  # mm/s
    _lengths = np.array([len(_t) for _t in trajs])
    summary = {
        "n_trajectories": len(trajs),
        "frames_per_track": _n_t,
        "min_len": int(_lengths.min()),
        "max_len": int(_lengths.max()),
        "mean_speed_mms": float(speeds.mean()),
        "max_speed_mms": float(speeds.max()),
    }
    mo.md(f"""
    ### 1 · Lagrangian data model — `Trajectory(pos, velocity, time, trajid)`

    Each `Trajectory` holds `pos` (t×3, metres), `velocity` (t×3, m/s), `time`, and a `trajid`,
    plus optional per-sample properties. `io.trajectories()` / `Scene` / `ZarrScene` build these from disk;
    here we synthesize **{summary['n_trajectories']} tracks × {summary['min_len']}–{summary['max_len']} frames**,
    mean speed **{summary['mean_speed_mms']:.1f} mm/s**.
    """)
    return fps, speeds, summary, trajs


@app.cell
def _(np, plt, speeds, trajs):
    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3.2))
    _lens = [len(_t) for _t in trajs]
    _ax[0].hist(_lens, bins=10, color="#2563eb", alpha=0.8)
    _ax[0].set_xlabel("frames per trajectory")
    _ax[0].set_ylabel("count")
    _ax[0].set_title("Track-length distribution")
    _ax[1].hist(np.asarray(speeds), bins=30, color="#059669", alpha=0.8)
    _ax[1].set_xlabel("speed [mm/s]")
    _ax[1].set_title("Lagrangian speed PDF")
    _fig.tight_layout()
    stats_fig = _fig
    stats_fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2 · 3D view — what raw tracks look like

    Real data comes from `flowtracks.io.trajectories("ptv_is.%d", ...)` (OpenPTV files),
    `Scene` (HDF5 `/particles` table) or `ZarrScene` (`trajectories/pos|vel|time|trajid` —
    the same layout `openptv2` writes, so no conversion step).
    """)
    return


@app.cell
def _(np, plt, trajs):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _fig3d = plt.figure(figsize=(7, 5))
    _ax3 = _fig3d.add_subplot(111, projection="3d")
    for _tr in trajs[:25]:
        _p = _tr.pos() * 1000.0
        _v = np.linalg.norm(_tr.velocity(), axis=1)
        _ax3.plot(_p[:, 0], _p[:, 1], _p[:, 2], alpha=0.7, linewidth=1)
        _ax3.scatter(_p[::10, 0], _p[::10, 1], _p[::10, 2], c=_v[::10], cmap="viridis", s=8)
    _ax3.set_xlabel("x [mm]")
    _ax3.set_ylabel("y [mm]")
    _ax3.set_zlabel("z [mm]")
    _ax3.set_title("Synthetic trajectories (colour ≈ speed)")
    _fig3d.tight_layout()
    traj_plot = _fig3d
    traj_plot
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3 · Smoothing — `flowtracks.smoothing.savitzky_golay`

    Differentiation amplifies noise, so velocity/acceleration come from a Savitzky–Golay fit
    over a sliding window (`fps`, `window_size` odd, `order < window_size − 1`).
    Tracks shorter than the window are dropped (or shrunk to `min_window`).
    """)
    return


@app.cell
def _(fps, mo, plt, savitzky_golay, trajs):
    smoothed = savitzky_golay(trajs, fps=fps, window_size=7, order=2)
    _kept = len(smoothed)
    # before/after on track 0, x-component
    _raw0 = trajs[0].pos()[:, 0] * 1000.0
    _sm0 = smoothed[0].pos()[:, 0] * 1000.0 if _kept else _raw0
    _fig2, _ax2 = plt.subplots(figsize=(8, 3.2))
    _ax2.plot(_raw0, ".", ms=3, alpha=0.5, label=f"raw ({len(trajs)} tracks in)")
    _ax2.plot(_sm0, "-", lw=1.5, label=f"smoothed ({_kept} kept)")
    _ax2.set_xlabel("frame")
    _ax2.set_ylabel("x [mm]")
    _ax2.set_title("Savitzky–Golay smoothing, track 0")
    _ax2.legend()
    _fig2.tight_layout()
    smooth_fig = _fig2
    mo.vstack([mo.md(f"Smoothed **{_kept}/{len(trajs)}** tracks (window 7, order 2)."), smooth_fig])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4 · Stitching — `flowtracks.stitching.stitch_trajectories`

    Occlusions split one physical path into segments. Stitching re-links candidates
    across short frame gaps by proximity + velocity continuity (linear-sum assignment).
    """)
    return


@app.cell
def _(Trajectory, fps, mo, stitch_trajectories, trajs):
    # demo: cut track 0 into two halves with a 2-frame gap
    _t0 = trajs[0]
    _ta = Trajectory(_t0.pos()[:15], _t0.velocity()[:15], _t0.time()[:15], trajid=1000)
    _tb = Trajectory(_t0.pos()[17:], _t0.velocity()[17:], _t0.time()[17:], trajid=1001)
    _broken = [_ta, _tb] + list(trajs[1:5])
    stitched = stitch_trajectories(_broken, fps=fps, max_gap=3, max_distance=0.05, max_vel_diff=0.5)
    _msg = (
        f"Broken set: **{len(_broken)}** segments → stitched: **{len(stitched)}** "
        f"(track 0 halves {'re-linked' if len(stitched) < len(_broken) else 'kept separate'})."
    )
    mo.md(_msg)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 5 · Eulerian gridding — `flowtracks.eulerian`

    `eulerian_grid()` / `eulerian_windowed()` bin Lagrangian samples into an
    `xarray.Dataset` on a regular grid (mean velocity, counts, masks), then
    `turbulent_statistics()` / `derived_fields()` add fluctuations, TKE, vorticity.
    Below: the same idea in a few lines of numpy, then the identical field as a quiver slice.
    """)
    return


@app.cell
def _(mo, np, plt, trajs):
    _all_p = np.concatenate([_t.pos() for _t in trajs]) * 1000.0  # mm
    _all_v = np.concatenate([_t.velocity() for _t in trajs]) * 1000.0  # mm/s
    _nx, _ny = 12, 10
    _xe = np.linspace(_all_p[:, 0].min(), _all_p[:, 0].max(), _nx + 1)
    _ye = np.linspace(_all_p[:, 1].min(), _all_p[:, 1].max(), _ny + 1)
    _u = np.full((_ny, _nx), np.nan)
    _vv = np.full((_ny, _nx), np.nan)
    _cnt = np.zeros((_ny, _nx), dtype=int)
    _ix = np.clip(np.digitize(_all_p[:, 0], _xe) - 1, 0, _nx - 1)
    _iy = np.clip(np.digitize(_all_p[:, 1], _ye) - 1, 0, _ny - 1)
    for _jj in range(_ny):
        for _ii in range(_nx):
            _m = (_ix == _ii) & (_iy == _jj)
            _cnt[_jj, _ii] = int(_m.sum())
            if _m.sum() > 3:
                _u[_jj, _ii] = _all_v[_m, 0].mean()
                _vv[_jj, _ii] = _all_v[_m, 1].mean()
    _xc = 0.5 * (_xe[:-1] + _xe[1:])
    _yc = 0.5 * (_ye[:-1] + _ye[1:])
    _X, _Y = np.meshgrid(_xc, _yc)
    _fig4, _ax4 = plt.subplots(figsize=(7, 4.5))
    _ax4.scatter(_all_p[:, 0], _all_p[:, 1], s=2, alpha=0.15, color="grey")
    _q = _ax4.quiver(_X, _Y, _u, _vv, np.sqrt(_u**2 + _vv**2), cmap="coolwarm")
    plt.colorbar(_q, ax=_ax4, label="|U| [mm/s]")
    _ax4.set_xlabel("x [mm]")
    _ax4.set_ylabel("y [mm]")
    _ax4.set_title(f"Eulerian mean-velocity grid ({_nx}×{_ny}, {int(_cnt.sum())} samples)")
    _fig4.tight_layout()
    euler_fig = _fig4
    _info = f"Grid **{_nx}×{_ny}**, **{int(_cnt.sum())}** samples → **{int((_cnt > 3).sum())}** filled cells."
    mo.vstack([mo.md(_info), euler_fig])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 6 · Phase averaging & the pipeline

    For periodic flows `phase_average` bins samples by cycle phase, then averages
    matching phases across cycles — the periodic mean + fluctuations.
    `pipeline.py` wraps each stage (`ptv_is_to_lagrangian → lagrangian_to_eulerian →
    phase_average_all_sets → …`) so the cloud orchestrator can fan runs out;
    `writers.write_eulerian_series` / `write_trajectories_vtp` export ParaView
    (`.vti/.vtr/.vtp/.pvd`), and `io.save_zarr_trajectories` keeps the
    Lagrangian store analysis-ready.

    **Two backends, one interface:** `Scene` (HDF5/PyTables, `read_where` queries)
    and `ZarrScene` (Zarr `trajectories/` group, numpy filtering) both serve
    `scene.collect(...)` — newer `eulerian`/`phase_average` code accepts either.
    """)
    return


@app.cell
def _(np, plt):
    # phase-average demo: sine base flow + noise, binned by phase
    _rng2 = np.random.default_rng(11)
    _ph = _rng2.uniform(0, 2 * np.pi, 2000)
    _sig = 50 * np.sin(_ph) + _rng2.normal(0, 8, _ph.shape)
    _nb = 12
    _edges = np.linspace(0, 2 * np.pi, _nb + 1)
    _pc = 0.5 * (_edges[:-1] + _edges[1:])
    _pm = np.array([_sig[(_ph >= _edges[_k]) & (_ph < _edges[_k + 1])].mean() for _k in range(_nb)])
    _fig5, _ax5 = plt.subplots(figsize=(8, 3.2))
    _ax5.scatter(_ph, _sig, s=3, alpha=0.15, color="grey", label="samples")
    _ax5.plot(_pc, _pm, "o-", color="#dc2626", label="phase mean")
    _ax5.set_xlabel("phase [rad]")
    _ax5.set_ylabel("u [mm/s]")
    _ax5.set_title("Phase averaging: periodic mean from scattered samples")
    _ax5.legend()
    _fig5.tight_layout()
    phase_fig = _fig5
    phase_fig
    return


@app.cell
def _(mo, summary):
    mo.md(f"""
    ### Takeaway

    flowtracks = **read** (`io`/`Scene`/`ZarrScene`) → **clean** (`stitching`, `smoothing.savitzky_golay`)
    → **grid** (`eulerian`) → **average** (`phase_average`) → **export** (`writers`, VTK/Zarr).

    Demo store: {summary['n_trajectories']} synthetic tracks, mean {summary['mean_speed_mms']:.1f} mm/s —
    move the sliders at the top and every figure recomputes.
    """)
    return


if __name__ == "__main__":
    app.run()
