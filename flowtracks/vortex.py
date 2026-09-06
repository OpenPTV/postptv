"""Vortex-identification scalars from a 3-D velocity field.

Q-criterion, lambda2 (Jeong & Hussain 1995), and enstrophy -- standard,
non-proprietary turbulence-analysis quantities. Vectorized over the whole
grid (and any extra dims, e.g. phase) via batched eigenvalue decomposition,
rather than a per-voxel loop.
"""
import numpy as np


def vortex_identification(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                          dx: float, dy: float, dz: float) -> tuple:
    """Q-criterion, lambda2, and enstrophy for a velocity field on a regular grid.

    u, v, w: arrays shaped (nx, ny, nz, ...) -- any trailing dims (e.g. phase)
    are treated independently, batched through the same eigendecomposition.

    Returns (Q, lambda2, enstrophy), each shaped like u.
    """
    ux, uy, uz = np.gradient(u, dx, dy, dz, axis=(0, 1, 2))
    vx, vy, vz = np.gradient(v, dx, dy, dz, axis=(0, 1, 2))
    wx, wy, wz = np.gradient(w, dx, dy, dz, axis=(0, 1, 2))

    # velocity-gradient tensor grad[..., i, j] = d(u_i)/dx_j
    grad = np.stack([
        np.stack([ux, uy, uz], axis=-1),
        np.stack([vx, vy, vz], axis=-1),
        np.stack([wx, wy, wz], axis=-1),
    ], axis=-2)

    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    spin = 0.5 * (grad - np.swapaxes(grad, -1, -2))

    norm_s2 = np.sum(strain**2, axis=(-2, -1))
    norm_om2 = np.sum(spin**2, axis=(-2, -1))
    q_criterion = 0.5 * (norm_om2 - norm_s2)

    a = strain @ strain + spin @ spin
    eigvals = np.linalg.eigvalsh(a)  # ascending
    lambda2 = eigvals[..., 1]

    vort_x, vort_y, vort_z = wy - vz, uz - wx, vx - uy
    enstrophy = 0.5 * (vort_x**2 + vort_y**2 + vort_z**2)

    return q_criterion, lambda2, enstrophy


if __name__ == "__main__":
    # Solid-body rotation about z: u=-omega*y, v=omega*x, w=0 -> pure rotation,
    # vorticity magnitude 2*omega everywhere, no strain, Q = omega^2 > 0.
    omega = 2.0
    n = 8
    x = y = z = np.linspace(-1, 1, n)
    dx = dy = dz = x[1] - x[0]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u, v, w = -omega * yy, omega * xx, np.zeros_like(xx)

    Q, lambda2, enstrophy = vortex_identification(u, v, w, dx, dy, dz)
    interior = slice(2, -2)
    assert np.allclose(enstrophy[interior, interior, interior], 2 * omega**2, atol=1e-6)
    assert np.all(Q[interior, interior, interior] > 0)
    assert np.all(lambda2[interior, interior, interior] < 0)  # rotation -> negative lambda2
    print("ok")
