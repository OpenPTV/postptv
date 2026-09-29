"""flowtracks.flux against exact answers: Poiseuille flow in straight pipes.

u(r) = 2 U (1 - r^2 / R^2) along +x, so the flow rate is Q = pi R^2 U
(positions in mm and velocities in m/s give Q in mL/s).
"""

import numpy as np
import pytest

from flowtracks.flux import (
    Domain,
    centerline,
    domain_boundary,
    field_flux,
    fluid_domain,
    section_flux,
    section_lumen,
)

R, LENGTH = 6.0, 80.0


def pipe_samples(n, u_mean, frames=100, y0=0.0, seed=0):
    """n particle samples uniform in a pipe along x (axis at y=y0, z=0),
    Poiseuille velocity with mean u_mean(t), times uniform in [0, frames)."""
    rng = np.random.default_rng(seed)
    r = R * np.sqrt(rng.random(n))
    th = rng.uniform(0, 2 * np.pi, n)
    t = rng.integers(0, frames, n).astype(float)
    pos = np.column_stack([rng.uniform(0, LENGTH, n), y0 + r * np.cos(th), r * np.sin(th)])
    u = 2 * u_mean(t) * (1 - (r / R) ** 2)
    return pos, np.column_stack([u, np.zeros(n), np.zeros(n)]), t


def q_exact(u):
    return np.pi * R**2 * u


def section(pos, dom, x=40.0, y0=0.0, slab=4.0):
    return section_lumen(pos, dom, np.array([x, y0, 0.0]), np.array([1.0, 0, 0]),
                         slab=slab, cell=1.5, rmax=15.0)


def test_domain_covers_the_pipe():
    pos, _, _ = pipe_samples(200000, lambda t: 0.5 + 0 * t)
    dom = fluid_domain(pos, voxel=1.0)
    # the domain is deliberately generous (smoothed occupancy, closed): it only
    # decides which region a plane cuts; section areas come from the particles
    assert np.pi * R**2 * LENGTH < dom.volume < 1.6 * np.pi * R**2 * LENGTH
    assert dom.inside(np.array([[40.0, 0, 0]]))[0]
    assert not dom.inside(np.array([[40.0, 2 * R, 0]]))[0]


def test_centerline_follows_the_pipe_axis():
    pytest.importorskip("skimage")
    pos, _, _ = pipe_samples(200000, lambda t: 0.5 + 0 * t)
    cl = centerline(fluid_domain(pos, voxel=1.0), spacing=10.0)
    assert len(cl.stations) >= 4
    assert np.all(np.abs(cl.stations[:, 1:]) < 1.0)                       # on the axis
    assert np.all(np.abs(np.abs(cl.normals[:, 0]) - 1) < 0.02)            # normals along x


def test_dense_steady_flow_rate_is_exact():
    pos, vel, t = pipe_samples(400000, lambda t: 0.5 + 0 * t)
    sec = section(pos, fluid_domain(pos, voxel=1.0))
    assert sec.area == pytest.approx(np.pi * R**2, rel=0.08)
    f = section_flux(sec, pos, vel, t, edges=[0, 100])
    assert f["coverage"][0] > 0.95
    assert f["area_corrected"][0] == pytest.approx(q_exact(0.5), rel=0.03)
    assert f["measured"][0] <= f["area_corrected"][0] + 1e-9


def test_pulsatile_flow_rate_follows_the_waveform():
    wave = lambda t: np.where(t < 50, 0.2, 0.8)                           # noqa: E731
    pos, vel, t = pipe_samples(400000, wave)
    sec = section(pos, fluid_domain(pos, voxel=1.0))
    f = section_flux(sec, pos, vel, t, edges=[0, 50, 100])
    assert f["area_corrected"] == pytest.approx(q_exact(np.array([0.2, 0.8])), rel=0.05)


def test_sparse_sampling_shows_as_coverage_with_measured_as_lower_bound():
    pos, vel, t = pipe_samples(3000, lambda t: 0.5 + 0 * t)
    sec = section(pos, fluid_domain(pos, voxel=3.0))       # sparse data: coarser domain grid
    assert sec is not None
    f = section_flux(sec, pos, vel, t, edges=[0, 100])
    assert f["coverage"][0] < 0.95
    assert f["measured"][0] < q_exact(0.5)
    assert f["area_corrected"][0] == pytest.approx(q_exact(0.5), rel=0.2)


def test_plane_through_two_branches_keeps_its_own_lumen():
    """A U-bend: two parallel pipes 20 mm apart joined at the far end, one
    connected conduit. A section centred on one leg (rmax 15 mm reaches into
    the other leg) must keep only its own lumen and flow."""
    a = pipe_samples(300000, lambda t: 0.5 + 0 * t, y0=0.0, seed=1)
    b = pipe_samples(300000, lambda t: 1.5 + 0 * t, y0=20.0, seed=2)
    rng = np.random.default_rng(3)                     # the bend: a box joining the legs
    bend = np.column_stack([rng.uniform(LENGTH - 8, LENGTH, 60000), rng.uniform(0, 20, 60000),
                            rng.uniform(-R / 2, R / 2, 60000)])
    zero = np.zeros((len(bend), 3))
    pos, vel, t = (np.concatenate([x, y, z]) for x, y, z in zip(a, b, (bend, zero, np.zeros(len(bend)))))
    dom = fluid_domain(pos, voxel=1.0)
    assert dom.inside(np.array([[40.0, 0, 0], [40.0, 20.0, 0]])).all()   # both legs, one domain
    sec = section(pos, dom, y0=0.0)
    assert sec.area == pytest.approx(np.pi * R**2, rel=0.1)
    f = section_flux(sec, pos, vel, t, edges=[0, 100])
    assert f["area_corrected"][0] == pytest.approx(q_exact(0.5), rel=0.05)


def test_field_flux_is_exact():
    def velocity(p, t):
        r2 = p[:, 1] ** 2 + p[:, 2] ** 2
        u = 2 * (0.5 + 0.1 * t) * np.clip(1 - r2 / R**2, 0, None)
        return np.column_stack([u, np.zeros(len(p)), np.zeros(len(p))])

    q, area = field_flux(velocity, lambda p: p[:, 1] ** 2 + p[:, 2] ** 2 < R**2,
                         np.array([40.0, 0, 0]), np.array([1.0, 0, 0]), times=[0.0, 1.0],
                         rmax=10.0, h=0.1)
    assert q == pytest.approx(q_exact(np.array([0.5, 0.6])), rel=0.01)
    assert area == pytest.approx(np.pi * R**2, rel=0.01)


def test_domain_boundary_of_a_cube():
    """A 4x4x4 fluid cube inside a 6x6x6 grid: 6 faces x 16 boundary faces,
    outward normals, centres on the cube's surface."""
    mask = np.zeros((6, 6, 6), bool)
    mask[1:5, 1:5, 1:5] = True
    b = domain_boundary(Domain(mask, np.zeros(3), 2.0))
    assert len(b["cell"]) == 6 * 16
    assert b["area"] == 4.0
    for ax in range(3):
        for side in (1, -1):
            sel = (b["axis"] == ax) & (b["side"] == side)
            assert sel.sum() == 16
            assert np.allclose(b["normal"][sel][:, ax], side)
            assert np.allclose(b["centre"][sel][:, ax], 2.0 if side < 0 else 10.0)
    full = domain_boundary(Domain(np.ones((3, 3, 3), bool), np.zeros(3), 1.0))
    assert len(full["cell"]) == 6 * 9            # the grid edge counts as a boundary
