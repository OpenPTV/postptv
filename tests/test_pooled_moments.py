"""Per-voxel moments (eulerian_grid(moments=True)) and exact pooling across
realizations (combine_moments / moment_statistics / run_pooled_analysis_ds).

Reference for every check is the definition itself: binning ALL samples of
ALL realizations together, or a plain per-voxel loop over the samples.
"""

import numpy as np
import pytest
import xarray as xr

from flowtracks import eulerian as E
from flowtracks.writers import write_eulerian_series

GRID = dict(min_x=0.0, max_x=1.0, stepx=4, min_y=0.0, max_y=1.0, stepy=3,
            min_z=0.0, max_z=1.0, stepz=2)
# 5 phases of 19 frames (deltat=9 -> zaman 19) over a 100-frame period
KW = dict(n_frames_in_period=100, n_phases=5, phase_zero_frame=0, deltat=9,
          min_count=0)


class FakeScene:
    def __init__(self, pos, vel, time, trajid):
        self.d = {"pos": pos, "velocity": vel, "time": time, "trajid": trajid}

    def collect(self, keys):
        return [self.d[k] for k in keys]


def make_scene(seed, n=6000, tid_offset=0, mean=(1.0, 0.2, -0.1)):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, (n, 3))
    pos[:40, 0] = 0.25  # exactly on an interior voxel edge
    vel = rng.normal(mean, (0.3, 0.1, 0.05), (n, 3))
    time = rng.integers(1, 201, n)
    tid = rng.integers(1, 300, n) + tid_offset
    return FakeScene(pos, vel, time, tid)


def concat_scenes(*scenes):
    return FakeScene(*(np.concatenate([s.d[k] for s in scenes])
                       for k in ("pos", "velocity", "time", "trajid")))


def brute_force(scene):
    """Per-voxel loop: half-open voxels, phase = ((t-1) % 100) // 19."""
    pos, vel, time, tid = (scene.d[k] for k in ("pos", "velocity", "time", "trajid"))
    ix = np.minimum((pos[:, 0] * 4).astype(int), 3)
    iy = np.minimum((pos[:, 1] * 3).astype(int), 2)
    iz = np.minimum((pos[:, 2] * 2).astype(int), 1)
    ti = np.clip(((time - 1) % 100) // 19, 0, 4)
    out = {}
    for key in set(zip(ix, iy, iz, ti)):
        sel = (ix == key[0]) & (iy == key[1]) & (iz == key[2]) & (ti == key[3])
        out[key] = (vel[sel], tid[sel])
    return out


@pytest.mark.parametrize("chunk", [None, 7])
def test_moments_match_brute_force(chunk):
    sc = make_scene(1)
    ds = E.eulerian_grid(sc, GRID, 1, 200, moments=True, chunk_frames=chunk, **KW)
    ref = brute_force(sc)
    assert int(ds[E.N_VAR].sum()) == sum(len(v) for v, _ in ref.values())
    for key, (v, t) in ref.items():
        d = v - v.mean(0)
        assert ds[E.N_VAR].values[key] == len(v)
        assert ds[E.NTRAJ_VAR].values[key] == len(np.unique(t))
        np.testing.assert_allclose([ds[m].values[key] for m in E.M1_VARS], v.mean(0))
        for (a, b), name in zip(E.PAIRS, E.M2_VARS):
            i, j = "uvw".index(a), "uvw".index(b)
            np.testing.assert_allclose(ds[name].values[key], (d[:, i] * d[:, j]).sum(),
                                       rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("rule", ["half-open", "ceil"])
def test_moments_use_the_same_voxels_as_the_means(rule):
    ds = E.eulerian_grid(make_scene(2), GRID, 1, 200, moments=True,
                         voxel_rule=rule, **KW)
    np.testing.assert_array_equal(ds[E.N_VAR].values, ds[E.COUNT_VAR].values)
    for v, m in zip(E.VEL_VARS, E.M1_VARS):
        np.testing.assert_allclose(ds[m].values, ds[v].values, equal_nan=True)


def test_moments_off_by_default_leaves_output_unchanged():
    sc = make_scene(3)
    plain = E.eulerian_grid(sc, GRID, 1, 200, **KW)
    assert not set(E.MOMENT_VARS) & set(plain.data_vars)
    withm = E.eulerian_grid(sc, GRID, 1, 200, moments=True, **KW)
    xr.testing.assert_equal(plain, withm[list(plain.data_vars)])
    assert set(withm.attrs) - set(plain.attrs) == {"moments"}


def test_moments_refuse_smoothing():
    with pytest.raises(ValueError, match="smoothing_sigma"):
        E.eulerian_grid(make_scene(4), GRID, 1, 200, moments=True,
                        smoothing_sigma=1.0, **KW)


def _stack(dss, names):
    return xr.concat(dss, dim=xr.DataArray(names, dims="set", name="set"))


def test_combine_moments_equals_binning_all_samples_together():
    a, b, c = make_scene(5), make_scene(6, n=300, tid_offset=1000), make_scene(7, tid_offset=2000)
    per = [E.eulerian_grid(s, GRID, 1, 200, moments=True, **KW)[E.MOMENT_VARS] for s in (a, b, c)]
    pooled = E.combine_moments(_stack(per, ["wp1", "wp2", "wp3"]))
    together = E.eulerian_grid(concat_scenes(a, b, c), GRID, 1, 200, moments=True, **KW)
    for v in E.MOMENT_VARS:
        np.testing.assert_allclose(pooled[v].values, together[v].values,
                                   rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=v)


def test_centred_moments_survive_low_turbulence_intensity():
    """mean 10 m/s, std 1e-4 m/s: sum(u^2)/N - mean^2 loses the variance in
    float64; the centred moments, pooled over two sets, keep it."""
    rng = np.random.default_rng(8)
    n = 20000
    pos = np.full((n, 3), 0.1)
    vel = np.column_stack([10.0 + 1e-4 * rng.standard_normal(n), np.zeros(n), np.zeros(n)])
    time = np.full(n, 5)
    halves = [FakeScene(pos[s], vel[s], time[s], np.arange(n)[s])
              for s in (slice(0, n // 2), slice(n // 2, n))]
    per = [E.eulerian_grid(h, GRID, 1, 200, moments=True, **KW)[E.MOMENT_VARS] for h in halves]
    st = E.moment_statistics(E.combine_moments(_stack(per, ["a", "b"])))
    got = float(st["u_ins_u_ins"].values[0, 0, 0, 0])
    np.testing.assert_allclose(got, np.var(vel[:, 0], ddof=1), rtol=1e-6)


def test_moment_statistics_masks_and_ddof():
    ds = E.eulerian_grid(make_scene(9), GRID, 1, 200, moments=True, **KW)
    st = E.moment_statistics(ds, min_count=60, ddof=1)
    n = ds[E.N_VAR]
    np.testing.assert_array_equal(st[E.VALID_VAR].values, (n >= 60).values)
    ok = st[E.VALID_VAR].values
    np.testing.assert_allclose(st["u_ins_v_ins"].values[ok],
                               (ds["uv_m2"] / (n - 1)).values[ok])
    assert np.isnan(st["u_rms"].values[~ok]).all()


def _recipe(sets, **extra):
    return {"sets": sets, "shift": 0, "mask": [], "average": {"weighting": "counts"},
            "derived": {"fields": ["MKE", "TKE"], "rho": 1.0, "mu": 1.0}, **extra}


def test_pooled_analysis_single_set_has_real_turbulent_stresses():
    """The two-stage path gives identically zero stresses for one realization
    (variance of one mean); the pooled path gives the in-bin variance."""
    ds = {"wp1": E.eulerian_grid(make_scene(10), GRID, 1, 200, moments=True, **KW)}
    two = E.run_post_analysis_ds(ds, _recipe(["wp1"]))
    pooled = E.run_pooled_analysis_ds(ds, _recipe(["wp1"]), min_count=2)
    assert np.nanmax(np.abs(two["u_ins_u_ins"].values)) == 0.0
    assert np.nanmin(pooled["u_ins_u_ins"].values) > 0.0
    np.testing.assert_allclose(np.nanmean(pooled["u_rms"].values), 0.3, rtol=0.1)
    assert "TKE" in pooled and np.nanmin(pooled["TKE"].values) > 0


def test_pooled_analysis_phase_shift_and_count_mask_from_recipe():
    a, b = make_scene(11), make_scene(12, tid_offset=1000)
    ds = {s: E.eulerian_grid(sc, GRID, 1, 200, moments=True, **KW)
          for s, sc in (("wp1", a), ("wp2", b))}
    rec = _recipe(["wp1", "wp2"], shift=2, mask=[{"method": "count", "min_count": 80}])
    out = E.run_pooled_analysis_ds(ds, rec)
    assert out.attrs["min_count"] == 80
    together = E.eulerian_grid(concat_scenes(a, b), GRID, 1, 200, moments=True, **KW)
    ref = E.moment_statistics(E.shift_phase(together, 2), min_count=80)
    for v in ["u_ins_mean", "u_ins_w_ins", E.N_VAR, E.VALID_VAR]:
        np.testing.assert_allclose(out[v].values, ref[v].values, equal_nan=True, err_msg=v)


def test_pooled_analysis_needs_moments():
    ds = {"wp1": E.eulerian_grid(make_scene(13), GRID, 1, 200, **KW)}
    with pytest.raises(ValueError, match="moments=True"):
        E.run_pooled_analysis_ds(ds, _recipe(["wp1"]))


def test_writer_groups_vectors_tensors_and_writes_bools(tmp_path):
    pv = pytest.importorskip("pyvista")

    ds = {"wp1": E.eulerian_grid(make_scene(14), GRID, 1, 200, moments=True, **KW)}
    out = E.run_pooled_analysis_ds(ds, _recipe(["wp1"]), min_count=2)
    tensor = ["u_ins_u_ins", "v_ins_v_ins", "w_ins_w_ins",
              "u_ins_v_ins", "v_ins_w_ins", "u_ins_w_ins"]  # VTK xx yy zz xy yz xz
    write_eulerian_series(out, tmp_path, tensors={"reynolds_stress": tensor},
                          vectors={"velocity_m1": E.M1_VARS}, keep_nan=False)
    g = pv.read(sorted(tmp_path.glob("*.vti"))[0])
    assert g.point_data["reynolds_stress"].shape == (g.n_points, 6)
    assert g.point_data["velocity_m1"].shape == (g.n_points, 3)
    assert "u_ins_u_ins" not in g.point_data and "u_m1" not in g.point_data
    assert g.point_data[E.VALID_VAR].dtype in (np.bool_, bool, np.uint8)
    snap = out.isel(phase=0)
    np.testing.assert_allclose(g.point_data["reynolds_stress"][:, 3],
                               np.nan_to_num(snap["u_ins_v_ins"].values).ravel(order="F"))


def test_cf_metadata_fills_in_but_never_overwrites(tmp_path):
    ds = E.eulerian_grid(make_scene(15), GRID, 1, 200, moments=True, **KW)
    ds["u_ins_mean"].attrs.update(units="m frame-1", long_name="bin average of u")
    E.save_zarr(ds, tmp_path / "e.zarr")
    back = xr.open_zarr(tmp_path / "e.zarr")
    assert back["u_ins_mean"].attrs["units"] == "m frame-1"
    assert back["u_ins_mean"].attrs["long_name"] == "bin average of u"
    assert back["v_ins_mean"].attrs["units"] == "m s-1"  # still filled in
    assert back[E.N_VAR].attrs["units"] == "1"


def test_writer_keep_nan(tmp_path):
    pv = pytest.importorskip("pyvista")

    ds = E.eulerian_grid(make_scene(16, n=200), GRID, 1, 200, moments=True,
                         **{**KW, "min_count": 5})
    assert ds["u_ins_mean"].isnull().any()  # some voxels have no valid data
    write_eulerian_series(ds, tmp_path / "zero", keep_nan=False)
    write_eulerian_series(ds, tmp_path / "nan", keep_nan=True)
    zero = pv.read(sorted((tmp_path / "zero").glob("*.vti"))[0]).point_data
    nan = pv.read(sorted((tmp_path / "nan").glob("*.vti"))[0]).point_data
    ref = ds["u_ins_mean"].isel(phase=0).values.ravel(order="F")
    np.testing.assert_array_equal(np.isnan(nan["velocity"][:, 0]), np.isnan(ref))
    assert not np.isnan(zero["velocity"]).any()
    assert np.issubdtype(nan[E.N_VAR].dtype, np.integer)  # counts keep their type


@pytest.mark.parametrize("chunk", [None, 5])
def test_moments_empty_scene(chunk):
    """Empty scene (0 particles) yields N=0, n_traj=0, m1=NaN, m2=0.0."""
    empty = FakeScene(np.empty((0, 3)), np.empty((0, 3)),
                      np.empty((0,), dtype=int), np.empty((0,), dtype=int))
    ds = E.eulerian_grid(empty, GRID, 1, 200, moments=True, chunk_frames=chunk, **KW)
    assert (ds[E.N_VAR].values == 0).all()
    assert (ds[E.NTRAJ_VAR].values == 0).all()
    for m in E.M1_VARS:
        assert np.isnan(ds[m].values).all()
    for m in E.M2_VARS:
        np.testing.assert_array_equal(ds[m].values, 0.0)


def test_combine_moments_with_empty_and_disjoint_sets():
    """Empty runs and disjoint spatial supports never poison m1/m2 with NaN."""
    empty = FakeScene(np.empty((0, 3)), np.empty((0, 3)),
                      np.empty((0,), dtype=int), np.empty((0,), dtype=int))
    # Run A only populates voxel (0, 0, 0); Run B only populates voxel (3, 2, 1)
    rng = np.random.default_rng(20)
    n = 40
    vel_a = rng.normal((1.0, 2.0, 3.0), 0.2, (n, 3))
    vel_b = rng.normal((-1.0, 0.5, 4.0), 0.3, (n, 3))
    sc_a = FakeScene(np.full((n, 3), 0.05), vel_a, np.full(n, 5), np.arange(1, n + 1))
    sc_b = FakeScene(np.full((n, 3), 0.95), vel_b, np.full(n, 5), np.arange(101, 101 + n))

    per = [E.eulerian_grid(s, GRID, 1, 200, moments=True, **KW)[E.MOMENT_VARS]
           for s in (empty, sc_a, sc_b)]
    pooled = E.combine_moments(_stack(per, ["empty", "a", "b"]))
    together = E.eulerian_grid(concat_scenes(sc_a, sc_b), GRID, 1, 200, moments=True, **KW)

    for v in E.MOMENT_VARS:
        np.testing.assert_allclose(pooled[v].values, together[v].values,
                                   rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=v)
    # Combining two all-empty runs also stays clean
    ds_empty = per[0]
    pooled_empty = E.combine_moments(_stack([ds_empty, ds_empty], ["e1", "e2"]))
    assert (pooled_empty[E.N_VAR].values == 0).all()
    assert all(np.isnan(pooled_empty[m].values).all() for m in E.M1_VARS)
    assert all((pooled_empty[m].values == 0.0).all() for m in E.M2_VARS)


def test_moment_statistics_single_sample_ddof():
    """N=1 voxel is masked for ddof=1 (no div-by-zero) and valid with var=0 for ddof=0."""
    sc = FakeScene(np.array([[0.1, 0.1, 0.1]]), np.array([[2.5, -1.0, 0.5]]),
                   np.array([5]), np.array([42]))
    ds = E.eulerian_grid(sc, GRID, 1, 200, moments=True, **KW)

    st1 = E.moment_statistics(ds, min_count=1, ddof=1)
    assert not bool(st1[E.VALID_VAR].values[0, 0, 0, 0])
    assert np.isnan(st1["u_ins_mean"].values[0, 0, 0, 0])
    assert np.isnan(st1["u_ins_u_ins"].values[0, 0, 0, 0])

    st0 = E.moment_statistics(ds, min_count=1, ddof=0)
    assert bool(st0[E.VALID_VAR].values[0, 0, 0, 0])
    assert st0["u_ins_mean"].values[0, 0, 0, 0] == pytest.approx(2.5)
    assert st0["u_ins_u_ins"].values[0, 0, 0, 0] == pytest.approx(0.0)
    assert st0["u_rms"].values[0, 0, 0, 0] == pytest.approx(0.0)


def test_chunk_frames_deduplicates_trajectory_and_handles_empty_chunks():
    """One trajectory crossing chunk boundaries in one voxel counts once in n_traj,
    and empty intermediate chunks merge cleanly."""
    # Frames 1, 2 (chunk 1), frames 3, 4 in another voxel (voxel 0 empty in chunk 2),
    # frames 5, 6 back in voxel 0 (chunk 3) with the SAME trajid=7
    pos = np.array([
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9],
        [0.9, 0.9, 0.9],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
    ])
    vel = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])
    time = np.array([1, 2, 3, 4, 5, 6])
    tid = np.array([7, 7, 99, 99, 7, 7])
    sc = FakeScene(pos, vel, time, tid)

    unchunked = E.eulerian_grid(sc, GRID, 1, 200, moments=True, **KW)
    chunked = E.eulerian_grid(sc, GRID, 1, 200, moments=True, chunk_frames=2, **KW)

    assert int(chunked[E.N_VAR].values[0, 0, 0, 0]) == 4
    assert int(chunked[E.NTRAJ_VAR].values[0, 0, 0, 0]) == 1
    for v in E.MOMENT_VARS:
        np.testing.assert_allclose(chunked[v].values, unchunked[v].values,
                                   rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=v)


def test_moments_with_qc_filters_trajid_in_lockstep():
    """QC filtering drops bad samples from pos/vel/time AND trajid together."""
    pos = np.full((6, 3), 0.1)
    vel = np.array([
        [1.0, 0.0, 0.0],       # keep (tid 1)
        [2.0, 0.0, 0.0],       # keep (tid 2)
        [100.0, 0.0, 0.0],     # drop: > max_speed (tid 3)
        [0.01, 0.0, 0.0],      # drop: < min_speed (tid 4)
        [np.nan, 0.0, 0.0],    # drop: non-finite (tid 5)
        [3.0, 0.0, 0.0],       # keep (tid 1)
    ])
    time = np.array([1, 2, 3, 4, 5, 6])
    tid = np.array([1, 2, 3, 4, 5, 1])
    sc = FakeScene(pos, vel, time, tid)

    ds = E.eulerian_grid(
        sc, GRID, 1, 200, moments=True,
        qc={"min_speed": 0.5, "max_speed": 10.0, "finite_only": True},
        **KW,
    )
    assert int(ds[E.N_VAR].values[0, 0, 0, 0]) == 3
    assert int(ds[E.NTRAJ_VAR].values[0, 0, 0, 0]) == 2  # only tids 1 and 2 survived
    assert ds["u_m1"].values[0, 0, 0, 0] == pytest.approx(2.0)
    assert ds["uu_m2"].values[0, 0, 0, 0] == pytest.approx(2.0)  # (-1)^2 + 0^2 + 1^2


def test_combine_moments_permutation_and_hierarchical_invariance():
    """combine_moments is invariant to set order and associative across stages."""
    a = make_scene(21, n=800)
    b = make_scene(22, n=500, tid_offset=1000, mean=(0.5, -0.3, 0.4))
    c = make_scene(23, n=1200, tid_offset=2000, mean=(-0.2, 0.7, -0.5))
    da, db, dc = [E.eulerian_grid(s, GRID, 1, 200, moments=True, **KW)[E.MOMENT_VARS]
                  for s in (a, b, c)]

    abc = E.combine_moments(_stack([da, db, dc], ["a", "b", "c"]))
    cba = E.combine_moments(_stack([dc, db, da], ["c", "b", "a"]))
    ab_then_c = E.combine_moments(
        _stack([E.combine_moments(_stack([da, db], ["a", "b"])), dc], ["ab", "c"])
    )
    for other in (cba, ab_then_c):
        for v in E.MOMENT_VARS:
            np.testing.assert_allclose(other[v].values, abc[v].values,
                                       rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=v)


def test_reynolds_stress_tensor_psd_and_cauchy_schwarz():
    """Pooled Reynolds stress tensor is positive semi-definite everywhere valid."""
    a, b = make_scene(24, n=4000), make_scene(25, n=4000, tid_offset=1000)
    per = [E.eulerian_grid(s, GRID, 1, 200, moments=True, **KW)[E.MOMENT_VARS] for s in (a, b)]
    st = E.moment_statistics(E.combine_moments(_stack(per, ["a", "b"])), min_count=5)

    ok = st[E.VALID_VAR].values
    assert ok.any()
    uu, vv, ww = st["u_ins_u_ins"].values[ok], st["v_ins_v_ins"].values[ok], st["w_ins_w_ins"].values[ok]
    uv, uw, vw = st["u_ins_v_ins"].values[ok], st["u_ins_w_ins"].values[ok], st["v_ins_w_ins"].values[ok]

    assert (uu >= 0).all() and (vv >= 0).all() and (ww >= 0).all()
    assert (uv**2 <= uu * vv + 1e-12).all()
    assert (uw**2 <= uu * ww + 1e-12).all()
    assert (vw**2 <= vv * ww + 1e-12).all()

    tensors = np.stack([
        np.stack([uu, uv, uw], axis=-1),
        np.stack([uv, vv, vw], axis=-1),
        np.stack([uw, vw, ww], axis=-1),
    ], axis=-2)
    evals = np.linalg.eigvalsh(tensors)
    assert (evals >= -1e-12).all()


def test_pooled_analysis_non_count_roi_mask_propagates():
    """Non-count masks (e.g. ROI array mask) mask means, stresses, rms and valid,
    while keeping raw n_samples/n_traj intact."""
    ds = {"wp1": E.eulerian_grid(make_scene(26), GRID, 1, 200, moments=True, **KW)}
    roi = np.ones((4, 3, 2), dtype=bool)
    roi[0, :, :] = False
    rec = _recipe(["wp1"], mask=[{"method": "count", "min_count": 2},
                                 {"method": "array", "mask": roi}])
    out = E.run_pooled_analysis_ds(ds, rec)

    assert not out[E.VALID_VAR].values[0, :, :, :].any()
    for v in ["u_ins_mean", "v_ins_mean", "w_ins_mean", "u_ins_u_ins", "u_rms"]:
        assert np.isnan(out[v].values[0, :, :, :]).all(), v
    assert (out[E.N_VAR].values[0, :, :, :] > 0).any()
    assert (out[E.NTRAJ_VAR].values[0, :, :, :] > 0).any()


def test_moments_file_path_branch_zarr(tmp_path):
    """Passing a Zarr path string (no .collect) exercises trajectories() with moments=True."""
    from flowtracks.io import save_zarr_trajectories
    from flowtracks.trajectory import Trajectory

    t1 = Trajectory(
        np.array([[0.1, 0.1, 0.1], [0.12, 0.12, 0.12], [0.15, 0.15, 0.15]]),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([1, 2, 3]),
        trajid=10,
    )
    t2 = Trajectory(
        np.array([[0.1, 0.1, 0.1], [0.11, 0.11, 0.11]]),
        np.array([[4.0, 1.0, 0.0], [5.0, 1.0, 0.0]]),
        np.array([1, 2]),
        trajid=20,
    )
    zpath = tmp_path / "run.zarr"
    save_zarr_trajectories([t1, t2], zpath, group="trajectories")

    ds = E.eulerian_grid(str(zpath), GRID, 1, 200, moments=True, **KW)
    assert int(ds[E.N_VAR].values[0, 0, 0, 0]) == 5
    assert int(ds[E.NTRAJ_VAR].values[0, 0, 0, 0]) == 2
    assert ds["u_m1"].values[0, 0, 0, 0] == pytest.approx(3.0)
    assert ds["uu_m2"].values[0, 0, 0, 0] == pytest.approx(10.0)


def test_moments_zarr_roundtrip_before_pooling(tmp_path):
    """Per-WP moment datasets saved to Zarr and reloaded pool identically to in-memory."""
    a, b = make_scene(27), make_scene(28, tid_offset=1000)
    mem = {s: E.eulerian_grid(sc, GRID, 1, 200, moments=True, **KW)
           for s, sc in (("wp1", a), ("wp2", b))}
    reloaded = {}
    for s, ds in mem.items():
        p = tmp_path / f"{s}.zarr"
        E.save_zarr(ds, p)
        reloaded[s] = xr.open_zarr(p).load()

    rec = _recipe(["wp1", "wp2"], mask=[{"method": "count", "min_count": 10}])
    out_mem = E.run_pooled_analysis_ds(mem, rec)
    out_disk = E.run_pooled_analysis_ds(reloaded, rec)
    for v in out_mem.data_vars:
        np.testing.assert_allclose(out_disk[v].values, out_mem[v].values,
                                   rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=v)


def test_writer_tensor_keep_nan_and_missing_group_skipped(tmp_path):
    """Default keep_nan=True preserves NaNs in grouped tensors; missing group members are skipped."""
    pv = pytest.importorskip("pyvista")

    ds = {"wp1": E.eulerian_grid(make_scene(29, n=200), GRID, 1, 200, moments=True, **KW)}
    out = E.run_pooled_analysis_ds(ds, _recipe(["wp1"]), min_count=10)
    assert out["u_ins_u_ins"].isnull().any()

    tensor = ["u_ins_u_ins", "v_ins_v_ins", "w_ins_w_ins",
              "u_ins_v_ins", "v_ins_w_ins", "u_ins_w_ins"]
    write_eulerian_series(
        out, tmp_path / "nan",
        tensors={"reynolds_stress": tensor, "missing_tensor": ["u_ins_u_ins", "does_not_exist"]},
        vectors={"missing_vec": ["u_m1", "v_m1", "nope"]},
    )
    g_nan = pv.read(sorted((tmp_path / "nan").glob("*.vti"))[0])
    assert "reynolds_stress" in g_nan.point_data
    assert "missing_tensor" not in g_nan.point_data
    assert "missing_vec" not in g_nan.point_data
    assert "u_m1" in g_nan.point_data  # ungrouped because missing_vec was skipped
    ref_uu = out["u_ins_u_ins"].isel(phase=0).values.ravel(order="F")
    np.testing.assert_array_equal(np.isnan(g_nan.point_data["reynolds_stress"][:, 0]),
                                  np.isnan(ref_uu))

    write_eulerian_series(out, tmp_path / "zero", tensors={"reynolds_stress": tensor},
                          keep_nan=False)
    g_zero = pv.read(sorted((tmp_path / "zero").glob("*.vti"))[0])
    assert not np.isnan(g_zero.point_data["reynolds_stress"]).any()
