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
