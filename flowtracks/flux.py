"""Volume flow rate through cross-sections of an internal flow, from particles.

For Lagrangian (PTV) data in a pipe, channel or vessel: in an incompressible
flow the rate through every cross-section of a rigid conduit is the same at
any instant, so the flow rate along the conduit is a truth-free check of the
measured velocities (in a compliant conduit the differences also contain the
volume stored by wall motion).

Pipeline, each step usable on its own:

1. :func:`fluid_domain` -- where the fluid is: voxel occupancy of the
   particle positions, smoothed, closed, largest connected component.
2. :func:`centerline` -- the conduit axis: the longest path through the 3D
   skeleton of the domain, smoothed, with stations every ``spacing`` along
   it and their unit tangents (the section normals). Needs scikit-image
   (``pip install flowtracks[flux]``).
3. :func:`section_lumen` -- one cross-section's lumen: the convex hull of
   where particles passed through a slab around the plane over the whole
   record, clipped to the part of the domain the plane cuts around the
   station (so a plane crossing two branches keeps only its own).
4. :func:`section_flux` -- the flow rate through that lumen per time window,
   sum over cells of mean(u . n) x lumen area inside the cell, from the
   samples in the slab.
5. :func:`field_flux` -- the exact flow rate of a KNOWN velocity field
   through a section, to test 1-4 against ground truth.

Units are the caller's: positions in any length unit L (e.g. mm), velocities
in any unit V; areas come out in L^2 and flow rates in V * L^2 (positions in
mm and velocities in m/s give mL/s).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = ["Domain", "Centerline", "Section", "fluid_domain", "centerline",
           "section_lumen", "section_flux", "field_flux"]


@dataclass
class Domain:
    """Boolean voxel mask of the fluid, with its grid origin and voxel size."""

    mask: np.ndarray
    origin: np.ndarray
    voxel: float

    def inside(self, points):
        """(N,) bool: which points (N, 3) fall in a fluid voxel."""
        idx = np.floor((np.asarray(points, float) - self.origin) / self.voxel).astype(int)
        ok = np.all((idx >= 0) & (idx < self.mask.shape), axis=1)
        out = np.zeros(len(idx), bool)
        out[ok] = self.mask[tuple(idx[ok].T)]
        return out

    @property
    def volume(self):
        return float(self.mask.sum() * self.voxel**3)


@dataclass
class Centerline:
    """The conduit axis (``path``) and the section stations along it."""

    path: np.ndarray       # (P, 3) smoothed axis points
    stations: np.ndarray   # (K, 3) section centres
    normals: np.ndarray    # (K, 3) unit tangents = section normals
    arc: np.ndarray        # (K,) distance of each station along the path


@dataclass
class Section:
    """One cross-section: plane, in-plane grid and lumen (see section_lumen)."""

    center: np.ndarray
    normal: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    rmax: float
    cell: float            # flux cell size
    refine: int            # lumen cells are cell / refine
    slab: float
    lumen: np.ndarray      # (nf, nf) bool, fine grid
    area_in_cell: np.ndarray  # (nc, nc) lumen area inside each flux cell

    @property
    def area(self):
        return float(self.area_in_cell.sum())

    def lumen_points(self):
        """3D centres of the fine lumen cells (for plotting)."""
        h = self.cell / self.refine
        g = -self.rmax + h * (np.arange(self.lumen.shape[0]) + 0.5)
        A, B = np.meshgrid(g, g, indexing="ij")
        pts = self.center + A[..., None] * self.e1 + B[..., None] * self.e2
        return pts[self.lumen]

    def _fine_index(self, points):
        """Fine-cell indices of the points inside the slab, and that mask."""
        rel = np.asarray(points, float) - self.center
        near = np.abs(rel @ self.normal) < self.slab / 2
        h = self.cell / self.refine
        nf = self.lumen.shape[0]
        fa = np.floor((rel[near] @ self.e1 + self.rmax) / h).astype(int)
        fb = np.floor((rel[near] @ self.e2 + self.rmax) / h).astype(int)
        ok = (fa >= 0) & (fa < nf) & (fb >= 0) & (fb < nf)
        sel = np.flatnonzero(near)[ok]
        return fa[ok], fb[ok], sel


def _plane_basis(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return n, e1, np.cross(n, e1)


def _largest_component(mask):
    from scipy.ndimage import label

    lab, _ = label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def fluid_domain(points, voxel, min_density=3.0, smooth=1.0, margin=3):
    """Fluid domain from particle positions (N, 3): voxels whose smoothed
    occupancy (Gaussian, ``smooth`` voxels) reaches ``min_density`` points,
    closed and hole-filled, largest connected component.

    Pass positions of MOVING particles only: points that never move (dirt,
    reflections on a wall) are not fluid. Returns a :class:`Domain`.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

    points = np.asarray(points, float)
    lo = np.percentile(points, 0.2, axis=0) - margin * voxel
    hi = np.percentile(points, 99.8, axis=0) + margin * voxel
    shape = tuple(int(s) for s in np.ceil((hi - lo) / voxel))
    idx = np.floor((points - lo) / voxel).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < shape), axis=1)]
    count = np.zeros(shape)
    np.add.at(count, tuple(idx.T), 1)
    mask = binary_fill_holes(binary_closing(gaussian_filter(count, smooth) >= min_density, iterations=2))
    return Domain(_largest_component(mask), lo, float(voxel))


def centerline(domain, spacing, smooth=7, tangent_span=None):
    """Conduit axis: the longest path through the 3D skeleton of the domain
    (two breadth-first sweeps), smoothed over ``smooth`` voxels, with
    stations every ``spacing`` along it (starting spacing/2 in) and unit
    tangents from a central difference over ``tangent_span`` (default: two
    voxels each side). The orientation along the path is arbitrary; flip the
    normals if a positive flux should mean the other direction.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("flowtracks.flux.centerline needs scikit-image: "
                          "pip install 'flowtracks[flux]'") from e
    pts = np.argwhere(skeletonize(domain.mask))
    if len(pts) < 2:
        raise ValueError("the domain skeleton has fewer than 2 voxels")
    index = {tuple(p): i for i, p in enumerate(pts)}
    offs = [np.array(d) - 1 for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)]
    nbr = [[index[tuple(p + o)] for o in offs if tuple(p + o) in index] for p in pts]

    def sweep(start):
        dist = np.full(len(pts), -1)
        prev = np.full(len(pts), -1)
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for w in nbr[u]:
                if dist[w] < 0:
                    dist[w], prev[w] = dist[u] + 1, u
                    q.append(w)
        return dist, prev

    d0, _ = sweep(int(np.argmax([len(x) for x in nbr])))
    a = int(d0.argmax())
    da, prev = sweep(a)
    order = [int(da.argmax())]
    while order[-1] != a:
        order.append(int(prev[order[-1]]))
    path = domain.origin + domain.voxel * (pts[order] + 0.5)
    k = max(1, int(smooth))
    kernel = np.ones(k) / k
    path = np.column_stack([np.convolve(np.pad(path[:, j], k // 2, mode="edge"), kernel, "valid")
                            for j in range(3)])[: len(path)]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    at = np.arange(spacing / 2, s[-1] - spacing / 4, spacing)
    interp = lambda x: np.column_stack([np.interp(x, s, path[:, j]) for j in range(3)])  # noqa: E731
    ds = tangent_span if tangent_span is not None else 2 * domain.voxel
    t = interp(np.minimum(at + ds, s[-1])) - interp(np.maximum(at - ds, 0.0))
    return Centerline(path, interp(at), t / np.linalg.norm(t, axis=1, keepdims=True), at)


def section_lumen(points, domain, center, normal, slab, cell, rmax, refine=3):
    """The lumen of the section (center, normal): the convex hull of the fine
    cells (cell / refine) that particles passed through within ``slab`` of
    the plane over the whole record, clipped to the connected part of the
    domain around ``center`` (dilated by one flux cell). Using all times
    keeps sparse near-wall sampling from shrinking the section; the domain
    clip keeps a non-convex conduit from spilling outside the fluid.
    Returns a :class:`Section`, or None if the plane misses the domain.
    """
    from matplotlib.path import Path as PolygonPath
    from scipy.ndimage import binary_dilation, label
    from scipy.spatial import ConvexHull, QhullError

    normal, e1, e2 = _plane_basis(normal)
    center = np.asarray(center, float)
    h = cell / refine
    nf = refine * int(np.ceil(2 * rmax / cell))
    g = -rmax + h * (np.arange(nf) + 0.5)
    A, B = np.meshgrid(g, g, indexing="ij")
    grid = center + A[..., None] * e1 + B[..., None] * e2
    inside = domain.inside(grid.reshape(-1, 3)).reshape(A.shape)
    lab, _ = label(inside)
    mid = nf // 2
    seed = lab[mid, mid]
    if seed == 0:
        cand = np.argwhere(inside)
        if len(cand) == 0:
            return None
        seed = lab[tuple(cand[np.argmin(np.sum((cand - mid) ** 2, axis=1))])]
    region = binary_dilation(lab == seed, iterations=refine)
    sec = Section(center, normal, e1, e2, float(rmax), float(cell), int(refine), float(slab),
                  np.zeros(A.shape, bool), np.zeros((nf // refine, nf // refine)))
    fa, fb, _ = sec._fine_index(points)
    seen = np.zeros(A.shape, bool)
    seen[fa, fb] = True
    seen &= region
    cells = np.argwhere(seen)
    if len(cells) < 3:
        return None
    try:
        hull = ConvexHull(cells + 0.5)
    except QhullError:  # all visited cells on a line
        return None
    poly = PolygonPath(hull.points[hull.vertices])
    idx = np.argwhere(np.ones(A.shape, bool)) + 0.5
    lumen = poly.contains_points(idx, radius=1e-9).reshape(A.shape) & region
    sec.lumen = lumen
    nc = nf // refine
    sec.area_in_cell = lumen.reshape(nc, refine, nc, refine).sum(axis=(1, 3)) * h * h
    return sec


def section_flux(section, points, velocities, times, edges, min_count=3):
    """Flow rate through ``section`` in each time window [edges[w], edges[w+1]).

    Samples: the points (N, 3) within the section's slab, with velocities
    (N, 3) and times (N,). A flux cell counts in a window if it has at least
    ``min_count`` samples; it contributes mean(u . n) x the lumen area inside
    it. Returns a dict of per-window arrays:

    measured       -- cells with data only (a lower bound where the missing
                      area carries flow in the same direction)
    area_corrected -- the measured mean flux scaled to the whole lumen
    coverage       -- lumen area with data / lumen area
    samples        -- samples in the lumen, per window
    """
    fa, fb, sel = section._fine_index(points)
    keep = section.lumen[fa, fb]
    fa, fb, sel = fa[keep], fb[keep], sel[keep]
    un = np.asarray(velocities, float)[sel] @ section.normal
    tt = np.asarray(times, float)[sel]
    r = section.refine
    ca, cb = fa // r, fb // r
    nc = section.area_in_cell.shape[0]
    out = {k: [] for k in ("measured", "area_corrected", "coverage", "samples")}
    for w in range(len(edges) - 1):
        m = (tt >= edges[w]) & (tt < edges[w + 1])
        cnt = np.zeros((nc, nc))
        sm = np.zeros((nc, nc))
        np.add.at(cnt, (ca[m], cb[m]), 1)
        np.add.at(sm, (ca[m], cb[m]), un[m])
        good = (cnt >= min_count) & (section.area_in_cell > 0)
        q = float(np.sum(sm[good] / cnt[good] * section.area_in_cell[good]))
        cov = float(section.area_in_cell[good].sum() / max(section.area, 1e-300))
        out["measured"].append(q)
        out["area_corrected"].append(q / cov if cov > 0 else np.nan)
        out["coverage"].append(cov)
        out["samples"].append(int(m.sum()))
    return {k: np.array(v) for k, v in out.items()}


def field_flux(velocity, inside, center, normal, times, rmax, h):
    """Exact flow rate of a KNOWN field through the section (center, normal):
    velocity(points, t) -> (N, 3) and inside(points) -> (N,) bool (the fluid)
    are evaluated on h-sized cells of the plane, over the connected part of
    the fluid around ``center``. Returns (flow rate per time, section area);
    units as the caller's (velocity unit x length unit^2).
    """
    from scipy.ndimage import label

    normal, e1, e2 = _plane_basis(normal)
    k = int(np.ceil(rmax / h))
    g = h * np.arange(-k, k + 1)
    A, B = np.meshgrid(g, g, indexing="ij")
    pts = (np.asarray(center, float) + A[..., None] * e1 + B[..., None] * e2).reshape(-1, 3)
    ins = np.asarray(inside(pts)).reshape(A.shape)
    lab, _ = label(ins)
    seed = lab[k, k]
    if seed == 0:
        cand = np.argwhere(ins)
        if len(cand) == 0:
            return np.full(len(times), np.nan), 0.0
        seed = lab[tuple(cand[np.argmin(np.sum((cand - k) ** 2, axis=1))])]
    sec = (lab == seed).ravel()
    q = np.array([float(np.sum(np.asarray(velocity(pts[sec], t)) @ normal) * h * h) for t in times])
    return q, float(sec.sum() * h * h)
