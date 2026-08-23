import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('vtk')

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))


from flowtracks import vtk_export as sample_vtkcode


def test_import_sample_vtkcode():
    pass


def test_write_structured_grid_vtk(tmp_path):
    vtkmod = sample_vtkcode

    nx, ny, nz = 3, 4, 2
    x = np.random.rand(nx, ny, nz)
    y = np.random.rand(nx, ny, nz)
    z = np.random.rand(nx, ny, nz)
    fields = {
        'velocity': (np.random.rand(nx * ny * nz, 3), 'vector'),
        'pressure': (np.random.rand(nx * ny * nz), 'scalar'),
    }
    out = tmp_path / 'test_out.vtk'
    try:
        vtkmod.write_structured_grid_vtk(str(out), x, y, z, fields)
        assert out.exists()
        content = out.read_text(errors='ignore')
        assert 'STRUCTURED_GRID' in content
        assert 'velocity' in content
        assert 'pressure' in content
    finally:
        if out.exists():
            out.unlink()


def test_main_handles_zero_velocity_without_divide_warning(tmp_path, monkeypatch):
    import warnings
    import h5py
    import yaml
    from flowtracks import vtk_export as sample_vtkcode
    from helpers import write_grid_h5, write_phase_averaged_h5, write_turb_stats_h5

    grid = {'stepx': 2, 'stepy': 2, 'stepz': 2,
            'min_x': -0.01, 'max_x': 0.02, 'min_y': 0.0, 'max_y': 0.05,
            'min_z': -0.03, 'max_z': 0.03}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({
        'traj_min_length': 20, 'first': 100001, 'last': 100010, 'frate': 5000,
        'hb': 70, 'data_path': '.', 'set_names': ['wp1'],
        'ptv_res_path': 'test/res/', 'h5_path': '.', 'delta_t': 90,
        'rho': 1000, 'mu': 0.001, 'shift': 0,
    }))
    (tmp_path / 'grid.yaml').write_text(yaml.safe_dump(grid))

    rng = np.random.default_rng(5)
    shape = (2, 2, 2, 1)
    u = rng.normal(size=shape)
    v = rng.normal(size=shape)
    w = rng.normal(size=shape)
    par = rng.integers(10, 100, size=shape).astype(np.int64)
    write_grid_h5(tmp_path / 'wp1_grid.h5', grid, u, v, w, par)
    with h5py.File(tmp_path / 'wp1_grid.h5', 'a') as f:
        f.create_dataset('u_fluct', data=u)
        f.create_dataset('v_fluct', data=v)
        f.create_dataset('w_fluct', data=w)
    # Zero-velocity frame would have triggered a divide-by-zero in PRT.
    ua = u.copy(); ua[..., 0] = 0
    va = v.copy(); va[..., 0] = 0
    wa = w.copy(); wa[..., 0] = 0
    write_phase_averaged_h5(tmp_path / 'phase_averaged.h5', ua, va, wa)
    z = np.zeros(shape)
    write_turb_stats_h5(tmp_path / 'turbulent_statistics_phase_averaged.h5',
                        z, z, z, z, z, z, z, z, z)

    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        sample_vtkcode.main()
    divide_warnings = [w for w in recorded
                       if issubclass(w.category, RuntimeWarning)
                       and 'divide' in str(w.message).lower()]
    assert not divide_warnings, f"unexpected divide warnings: {divide_warnings}"
    assert (tmp_path / 'vtk_output').exists()


def _reference_max_eigenvalue_spread(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    """Independent per-voxel reference of the original in-loop computation."""
    for arr in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
        np.nan_to_num(arr, copy=False)
    shape = p1.shape
    out = np.zeros(shape)
    for idx in np.ndindex(shape):
        A = np.array([[p1[idx], p2[idx], p3[idx]],
                     [p4[idx], p5[idx], p6[idx]],
                     [p7[idx], p8[idx], p9[idx]]])
        r = np.linalg.eigvalsh(A)
        out[idx] = 0.5 * (np.max(r) - np.min(r))
    return out


def test_max_eigenvalue_spread_matches_reference():
    from flowtracks import vtk_export as sample_vtkcode
    rng = np.random.default_rng(42)
    shape = (4, 5, 6)
    comps = [rng.normal(size=shape) for _ in range(9)]

    expected = _reference_max_eigenvalue_spread(*comps)
    got = sample_vtkcode.max_eigenvalue_spread(*comps)
    assert got.shape == shape
    assert np.allclose(got, expected)


def test_max_eigenvalue_spread_handles_nan():
    from flowtracks import vtk_export as sample_vtkcode
    rng = np.random.default_rng(7)
    shape = (3, 2, 2)
    comps = [rng.normal(size=shape) for _ in range(9)]
    # Inject NaNs in a couple of cells.
    comps[0][0, 0, 0] = np.nan
    comps[4][1, 1, 1] = np.nan

    expected = _reference_max_eigenvalue_spread(*comps)
    got = sample_vtkcode.max_eigenvalue_spread(*comps)
    assert np.allclose(got, expected, equal_nan=True)
