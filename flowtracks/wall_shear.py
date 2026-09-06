"""Near-wall shear stress and cyclic wall-shear indices from a velocity field.

General-purpose: works with any triangulated surface -- an explicit wall
mesh you already have, or one extracted from an occupancy/indicator field
with occupancy_isosurface() when no explicit mesh exists. TAWSS/OSI/RRT are
standard hemodynamic wall-shear indices (time-averaged WSS, oscillatory
shear index, relative residence time); nothing here is specific to any one
dataset or organ.

Vectorized throughout: normals via scatter-add over faces, near-wall
sampling via one batched grid interpolation call, TAWSS/OSI/RRT via
cumulative sums over phase -- no per-vertex or per-phase Python loops.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals: area-weighted average of adjacent face normals.

    vertices: (n_vertices, 3), faces: (n_faces, 3) integer indices.
    """
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    vnorm = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vnorm, faces[:, k], face_normals)
    norms = np.linalg.norm(vnorm, axis=1, keepdims=True)
    return vnorm / np.where(norms == 0, 1, norms)


def occupancy_isosurface(occupancy: np.ndarray, x: np.ndarray, y: np.ndarray,
                         z: np.ndarray, level: float = 0.5) -> tuple:
    """Extract a triangulated surface from an occupancy/indicator field via
    marching cubes, for use as a wall/interface proxy when no explicit mesh
    is available (e.g. the boundary of a masked flow region).

    Requires the optional `vtk` dependency (flowtracks[vtk]). x, y, z must
    be uniformly spaced (a vtkImageData assumption).
    """
    import vtk
    from vtk.util import numpy_support

    spacing = (float(x[1] - x[0]), float(y[1] - y[0]), float(z[1] - z[0]))
    origin = (float(x[0]), float(y[0]), float(z[0]))
    nx, ny, nz = occupancy.shape

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    scalars = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(occupancy, dtype=np.float64).ravel(order="F"), deep=True)
    image.GetPointData().SetScalars(scalars)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, level)
    mc.Update()
    poly = mc.GetOutput()

    vertices = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    faces = numpy_support.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def near_wall_shear_stress(points: np.ndarray, normals: np.ndarray,
                           x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           u: np.ndarray, v: np.ndarray, w: np.ndarray,
                           mu: float, offset: float) -> tuple:
    """Estimate wall/interface shear stress from the near-surface velocity.

    Thin-layer approximation: sample velocity at `points - offset*normals`,
    project onto the local tangent plane, and take
        tau = mu * v_tangential / offset.

    Returns (tau_vec, tau_mag), shaped (n_points, 3) and (n_points,).
    NaN where the sample point falls outside the velocity field's domain.
    """
    sample_pts = points - offset * normals
    interp = RegularGridInterpolator(
        (x, y, z), np.stack([u, v, w], axis=-1),
        method="linear", bounds_error=False, fill_value=np.nan)
    vel = interp(sample_pts)

    v_normal = np.sum(vel * normals, axis=1, keepdims=True) * normals
    v_tangent = vel - v_normal
    tau_vec = (mu / offset) * v_tangent
    tau_mag = np.linalg.norm(tau_vec, axis=1)
    return tau_vec, tau_mag


def tawss_osi_rrt(tau_vec: np.ndarray, tau_mag: np.ndarray, eps: float = 1e-12) -> tuple:
    """Time-averaged WSS, oscillatory shear index, and relative residence
    time over a full cycle.

    tau_vec: (n_points, n_phase, 3), tau_mag: (n_points, n_phase).
    Returns (TAWSS, OSI, RRT), each (n_points,).
    """
    tawss = np.nanmean(tau_mag, axis=1)
    sum_vec = np.nansum(tau_vec, axis=1)
    num = np.linalg.norm(sum_vec, axis=1)
    den = np.nansum(tau_mag, axis=1) + eps
    osi = 0.5 * (1 - num / den)
    rrt = 1.0 / ((1 - 2 * osi) * tawss + eps)
    return tawss, osi, rrt


def running_tawss_osi_rrt(tau_vec: np.ndarray, tau_mag: np.ndarray, eps: float = 1e-12) -> tuple:
    """Cumulative-in-phase TAWSS/OSI/RRT: value at phase k uses only phases 1..k.

    Same inputs/shapes as tawss_osi_rrt; returns arrays shaped (n_points, n_phase).
    """
    valid = ~np.isnan(tau_mag)
    count = np.cumsum(valid, axis=1)
    mag_filled = np.where(valid, tau_mag, 0.0)
    vec_filled = np.where(valid[..., None], tau_vec, 0.0)

    sum_mag = np.cumsum(mag_filled, axis=1)
    sum_vec = np.cumsum(vec_filled, axis=1)

    tawss = sum_mag / np.maximum(count, 1)
    num = np.linalg.norm(sum_vec, axis=2)
    den = sum_mag + eps
    osi = 0.5 * (1 - num / den)
    rrt = 1.0 / ((1 - 2 * osi) * tawss + eps)
    return tawss, osi, rrt


if __name__ == "__main__":
    # Cube wall (12 triangles), constant shear flow u=(z, 0, 0): du/dz=1 is
    # the only nonzero gradient, so on the z=0 face (normal -z, offset inward
    # along +z) the tangential velocity at the sample point is offset*1 in x,
    # giving tau_x = mu*offset/offset = mu exactly, independent of offset.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # bottom face z=0 only, needed part
    normals = np.tile([0, 0, -1.0], (4, 1))  # outward normal of bottom face

    n = 5
    x = y = z = np.linspace(-0.5, 1.5, n)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    mu = 0.5
    u, v, w = zz.copy(), np.zeros_like(zz), np.zeros_like(zz)

    points = vertices[:4]
    tau_vec, tau_mag = near_wall_shear_stress(points, normals, x, y, z, u, v, w, mu, offset=0.1)
    assert np.allclose(tau_vec[:, 0], mu, atol=1e-9), tau_vec
    assert np.allclose(tau_mag, mu, atol=1e-9)

    n_phase = 6
    tau_series = np.tile(tau_vec[:, None, :], (1, n_phase, 1))
    mag_series = np.tile(tau_mag[:, None], (1, n_phase))
    tawss, osi, rrt = tawss_osi_rrt(tau_series, mag_series)
    assert np.allclose(tawss, mu)
    assert np.allclose(osi, 0.0, atol=1e-9)  # steady shear -> no oscillation

    run_tawss, run_osi, run_rrt = running_tawss_osi_rrt(tau_series, mag_series)
    assert np.allclose(run_tawss[:, -1], tawss)
    assert np.allclose(run_osi[:, -1], osi, atol=1e-9)

    print("ok")
