"""Pytest wrappers around the flowtracks.vortex / flowtracks.wall_shear
self-checks (see each module's __main__ block for the physical reasoning)."""
import numpy as np
import pytest

from flowtracks.vortex import vortex_identification
from flowtracks.wall_shear import (
    near_wall_shear_stress, occupancy_isosurface, running_tawss_osi_rrt,
    tawss_osi_rrt, vertex_normals,
)


def test_vortex_identification_solid_body_rotation():
    omega = 2.0
    n = 8
    x = y = z = np.linspace(-1, 1, n)
    dx = dy = dz = x[1] - x[0]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u, v, w = -omega * yy, omega * xx, np.zeros_like(xx)

    q, lambda2, enstrophy = vortex_identification(u, v, w, dx, dy, dz)
    interior = slice(2, -2)
    assert np.allclose(enstrophy[interior, interior, interior], 2 * omega**2, atol=1e-6)
    assert np.all(q[interior, interior, interior] > 0)
    assert np.all(lambda2[interior, interior, interior] < 0)


def test_occupancy_isosurface_sphere_and_vertex_normals():
    vtk = pytest.importorskip("vtk")
    n = 12
    x = y = z = np.linspace(-1, 1, n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    occ = (xx**2 + yy**2 + zz**2 < 0.5**2).astype(float)

    vertices, faces = occupancy_isosurface(occ, x, y, z, level=0.5)
    r = np.linalg.norm(vertices, axis=1)
    assert np.allclose(r.mean(), 0.5, atol=0.05)

    normals = vertex_normals(vertices, faces)
    cos_radial = np.sum(normals * vertices, axis=1) / (r + 1e-12)
    assert cos_radial.mean() > 0.9  # normals point outward


def test_near_wall_shear_stress_constant_gradient():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    normals = np.tile([0, 0, -1.0], (4, 1))

    n = 5
    x = y = z = np.linspace(-0.5, 1.5, n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    mu = 0.5
    u, v, w = zz.copy(), np.zeros_like(zz), np.zeros_like(zz)

    tau_vec, tau_mag = near_wall_shear_stress(vertices, normals, x, y, z, u, v, w, mu, offset=0.1)
    assert np.allclose(tau_vec[:, 0], mu, atol=1e-9)
    assert np.allclose(tau_mag, mu, atol=1e-9)


def test_tawss_osi_rrt_steady_shear_has_zero_osi():
    n_points, n_phase = 4, 6
    tau_vec = np.tile([1.0, 0.0, 0.0], (n_points, n_phase, 1))
    tau_mag = np.ones((n_points, n_phase))

    tawss, osi, rrt = tawss_osi_rrt(tau_vec, tau_mag)
    assert np.allclose(tawss, 1.0)
    assert np.allclose(osi, 0.0, atol=1e-9)

    run_tawss, run_osi, run_rrt = running_tawss_osi_rrt(tau_vec, tau_mag)
    assert np.allclose(run_tawss[:, -1], tawss)
    assert np.allclose(run_osi[:, -1], osi, atol=1e-9)


def test_tawss_osi_reversing_flow_has_high_osi():
    """Shear that fully reverses sign each half-cycle should give OSI -> 0.5."""
    n_points, n_phase = 3, 8
    sign = np.where(np.arange(n_phase) % 2 == 0, 1.0, -1.0)
    tau_vec = np.zeros((n_points, n_phase, 3))
    tau_vec[..., 0] = sign
    tau_mag = np.ones((n_points, n_phase))

    tawss, osi, rrt = tawss_osi_rrt(tau_vec, tau_mag)
    assert np.allclose(osi, 0.5, atol=1e-9)
