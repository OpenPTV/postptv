# -*- coding: utf-8 -*-
# Created on Tue May 28 10:27:15 2013

"""
Interpolation routines.

.. rubric:: References

.. [#IDW] http://en.wikipedia.org/wiki/Inverse_distance_weighting

.. [#BL] Lüthi, Beat. Some Aspects of Strain, Vorticity and Material Element \
   Dynamics as Measured with 3D Particle Tracking Velocimetry in a \
   Turbulent Flow. PhD Thesis, ETH-Zürich (2002).

.. [#RBF] http://en.wikipedia.org/wiki/Radial_basis_function

.. rubric:: Documentation
"""

import numpy as np
import warnings
from scipy.spatial import cKDTree
from configparser import ConfigParser


def select_neighbs(tracer_pos, interp_points, radius=None, num_neighbs=None,
                   companionship=None):
    """
    For each of m interpolation points, find its distance to all tracers. Use
    result to decide which tracers are the neighbours of each interpolation
    point, based on either a fixed radius or the closest num_neighbs.

    Arguments:
    tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row, [m]
    interp_points - (m,3) array, coordinates of points where interpolation will
        be done.
    radius - of the search area for neighbours, [m]. If None, select closest
        num_neighbs.
    num_neighbs - number of closest neighbours to interpolate from. If None.
        uses all neighbours in a given radius. ``radius`` has precedence.
    companionship - an optional array denoting for each interpolation point the
        index of a tracer that should be excluded from it ("companion tracer"),
        useful esp. for interpolating tracers unto themselves and for analysing
        a simulated particle that started from a true tracer.

    Returns:
    dists - (m,n) array, the distance from each interpolation point to each
        tracer.
    use_parts - (m,n) boolean array, True where tracer :math:`j=1...n` is a
        neighbour of interpolation point :math:`i=1...m`.
    """
    n = tracer_pos.shape[0]
    m = interp_points.shape[0]

    # --- Path selection ---
    # KD-tree path: only when num_neighbs is given, no radius, and n is large
    # enough that the companion-padding edge case (n <= num_neighbs) is avoided.
    if radius is None and num_neighbs is not None and n > num_neighbs:
        return _select_neighbs_kdtree(tracer_pos, interp_points,
                                      num_neighbs, companionship)

    # Dense fallback: radius mode, n <= num_neighbs, or no num_neighbs.
    return _select_neighbs_dense(tracer_pos, interp_points,
                                 radius, num_neighbs, companionship)


def _select_neighbs_dense(tracer_pos, interp_points, radius=None,
                          num_neighbs=None, companionship=None):
    """
    Original O(m·n) dense selection.  Always produces the exact return that the
    public ``select_neighbs`` documents, and is kept as the fallback for radius
    mode, the n ≤ num_neighbs companion-padding case, and whenever a boundary
    distance tie makes the KD-tree result ambiguous.
    """
    dists = np.linalg.norm(tracer_pos[None, :, :] - interp_points[:, None, :],
                            axis=2)

    # Only for selection phase, later changed back.
    dists[dists <= 0] = np.inf
    if companionship is not None:
        cif = companionship >= 0.  # companion in frame
        dists[np.nonzero(cif)[0], companionship[cif]] = np.inf

    if radius is None:
        if num_neighbs is None:
            raise ValueError("Either radius or num_neighbs must be given.")

        dist_sort = np.argsort(dists, axis=1)
        use_parts = np.zeros(dists.shape, dtype=np.bool_)

        eff_num_neighbs = min(num_neighbs, tracer_pos.shape[0])
        use_parts[
            np.repeat(np.arange(interp_points.shape[0]), eff_num_neighbs),
            dist_sort[:, :num_neighbs].flatten()] = True

    else:
        use_parts = dists < radius

    dists[np.isinf(dists)] = 0.
    return dists, use_parts


def _select_neighbs_kdtree(tracer_pos, interp_points, num_neighbs,
                           companionship=None):
    """
    KD-tree accelerated neighbour selection for the ``num_neighbs`` mode when
    ``n > num_neighbs``.

    1. Query the KD-tree for ``num_neighbs + 1`` candidates.
    2. Re-compute exact pairwise distances for those candidates only (matching
       the formula used by the dense path – cheap at O(m·k)).
    3. If a tie exists at the selection boundary (the ``num_neighbs``-th and
       ``num_neighbs+1``-th exact distances are equal), fall back to the dense
       path because the neighbour set becomes tie-order-dependent.
    4. Otherwise scatter the exact distances into the dense (m,n) output arrays.
    """
    n = tracer_pos.shape[0]
    m = interp_points.shape[0]

    tree = cKDTree(tracer_pos)
    _, qidx = tree.query(interp_points, num_neighbs + 1)  # (m, k) indices

    # Exact distances for the k candidates (same formula as the dense path).
    delta = tracer_pos[qidx] - interp_points[:, None, :]  # (m, k, 3)
    exact_dists = np.sqrt(np.sum(delta * delta, axis=-1))  # (m, k)

    # If any row has a tie at the selection boundary, the neighbour set is
    # tie-order-dependent – fall back to dense where np.argsort determines it.
    if np.any(exact_dists[:, num_neighbs - 1] == exact_dists[:, num_neighbs]):
        return _select_neighbs_dense(tracer_pos, interp_points,
                                     radius=None, num_neighbs=num_neighbs,
                                     companionship=companionship)

    # No boundary tie → the set of num_neighbs nearest neighbours is
    # uniquely determined.  Scatter into the public return format.
    forbidden = exact_dists <= 0.
    if companionship is not None:
        comp = np.atleast_1d(companionship)
        forbidden |= (qidx == comp[:, None])

    # Stable argsort: non-forbidden entries sort by distance (ascending),
    # then forbidden entries sort to the end (they become distance-0
    # padding, exactly matching the dense inf → 0 round-trip).
    order = np.argsort(np.where(forbidden, np.inf, exact_dists), axis=1,
                       kind='stable')
    sel = order[:, :num_neighbs]          # (m, num_neighbs)
    rows = np.arange(m)[:, None]
    chosen = qidx[rows, sel]
    chosen_dists = np.where(forbidden[rows, sel], 0.,
                            exact_dists[rows, sel])

    use_parts = np.zeros((m, n), dtype=bool)
    use_parts[rows, chosen] = True
    dists = np.zeros((m, n))
    dists[rows, chosen] = chosen_dists
    return dists, use_parts


def corrfun_interp(dists, use_parts, data, corrs_hist, corrs_bins):
    """
    For each of n particle, generate the velocity interpolated to its
    position from all neighbours as selected by caller. The weighting of
    neighbours is by the correlation function, e.g. if the distance at
    neighbor i is :math:`r_i`, then it adds :math:`\\rho(r_i)*v_i` to the
    interpolated velocity. This is done for each component separately.

    Arguments:
    dists - (m,n) array, the distance of interpolation_point :math:`i=1...m`
        from tracer :math:`j=1...n`, for (row,col) (i,j) [m]
    use_parts - (m,n) boolean array, whether tracer j is a neighbor of
        particle i, same indexing as ``dists``.
    data - (n,d) array, the d components of the data that is interpolated from,
        for each of n tracers.
    corrs_hist - the correlation function histogram, an array of b bins.
    corrs_bins - same size array, the bin start point for each bin.

    Returns:
    vel_avg - an (m,3) array with the interpolated velocity at each
        interpolation point, [units of ``data``].
    """
    weights = np.zeros(dists.shape + (data.shape[-1],))
    weights[use_parts] = corrs_hist[
        np.digitize(dists[use_parts].flatten(), corrs_bins) - 1]

    vel_avg = (weights * data[None, ...]).sum(axis=1) / \
        weights.sum(axis=1)

    return vel_avg


def rbf_interp(tracer_dists, dists, use_parts, data, epsilon=1e-2):
    """
    Radial-basis interpolation [3] for each particle, from all neighbours
    selected by caller. The difference from inv_dist_interp is that the
    weights are independent of interpolation point, among other differences.

    Arguments:
    tracer_dists - (n,n) array, the distance of tracer :math:`i=1...n` from
        tracer :math:`j=1...n`, for (row,col) (i,j) [m]
    dists - (m,n) array, the distance from interpolation point
        :math:`i=1...m` to tracer j. [m]
    use_parts - (m,n) boolean array, True where tracer :math:`j=1...n` is a
        neighbour of interpolation point :math:`i=1...m`.
    data - (n,d) array, the d components of the data for each of n tracers.

    Returns:
    vel_interp - an (m,3) array with the interpolated velocity at the position
        of each particle, [m/s].
    """
    kernel = np.exp(-tracer_dists**2 * epsilon)

    k_per_row = use_parts.sum(axis=1)

    # When ``use_parts`` is a boolean mask with the same fixed number of
    # neighbours per point, solve all neighbour systems at once (numpy
    # broadcasts the leading dimensions). If it is an index array (as passed by
    # the lazy scene machinery) we keep the original per-point loop, which
    # relies on np.nonzero() of the indices.
    is_mask = (use_parts.dtype == np.bool_)

    if is_mask and k_per_row.shape[0] > 0 \
            and np.all(k_per_row == k_per_row[0]) and k_per_row[0] > 0:
        k = int(k_per_row[0])
        rows = np.arange(dists.shape[0])
        nbrs = np.where(use_parts)[1].reshape(dists.shape[0], k)
        K_stack = kernel[nbrs[:, :, None], nbrs[:, None, :]]   # (m,k,k)
        data_stack = data[nbrs]                                # (m,k,d)
        coeffs_stack = np.linalg.solve(K_stack, data_stack)    # (m,k,d)

        chosen_dists = dists[rows[:, None], nbrs]
        rbf_chosen = np.exp(-chosen_dists**2 * epsilon)
        return np.sum(rbf_chosen[..., None] * coeffs_stack, axis=1)
    else:
        coeffs = np.zeros(dists.shape + (data.shape[-1],))
        for pix in range(dists.shape[0]):
            neighbs = np.nonzero(use_parts[pix])[0]
            K = kernel[np.ix_(neighbs, neighbs)]
            coeffs[pix, neighbs] = np.linalg.solve(K, data[neighbs])

        rbf = np.exp(-dists**2 * epsilon)
        vel_interp = np.sum(rbf[..., None] * coeffs, axis=1)
        return vel_interp


def interpolant(method, num_neighbs=None, radius=None, param=None):
    """
    Factory function. Returns an object of the interpolant class that matches
    the given method. All classes are subclassed from GeneralInterpolant.

    Arguments:
    method - interpolation method. Either 'inv' for inverse-distance
        weighting, 'rbf' for gaussian-kernel Radial Basis Function
        method, or 'corrfun' for using a correlation function.
    radius - of the search area for neighbours, [m]. If None, select
        closest ``neighbs``.
    neighbs - number of closest neighbours to interpolate from. If None.
        uses 4 neighbours for 'inv' method, and 7 for 'rbf', unless
        ``radius`` is not None, then ``neighbs`` is ignored.
    param - the parameter adjusting the interpolation method. For IDW it is
        the inverse power (default 1), for rbf it is epsilon (default 1e5).
    """
    if method == 'inv':
        return InverseDistanceWeighter(num_neighbs, radius, param)
    else:
        return GeneralInterpolant(method, num_neighbs, radius, param)


Interpolant = interpolant  # B.C.


class GeneralInterpolant(object):
    """
    Holds all parameters necessary for performing an interpolation. Use is as
    a callable object after initialization, see :meth:`__call__`.
    """
    def __init__(self, method, num_neighbs=None, radius=None, param=None):
        """
        Arguments:
        method - interpolation method. Either 'inv' for inverse-distance
            weighting, 'rbf' for gaussian-kernel Radial Basis Function
            method, or 'corrfun' for using a correlation function.
        radius - of the search area for neighbours, [m]. If None, select
            closest ``neighbs``.
        neighbs - number of closest neighbours to interpolate from. If None.
            uses 4 neighbours for 'inv' method, and 7 for 'rbf', unless
            ``radius`` is not None, then ``neighbs`` is ignored.
        param - the parameter adjusting the interpolation method. For IDW it is
            the inverse power (default 1), for rbf it is epsilon (default 1e5).
        """
        if method == 'subclass':
            pass

        elif method == 'rbf':
            if num_neighbs is None:
                num_neighbs = 7
            if param is None:
                param = 1e5

        elif method == 'corrfun':
            if num_neighbs is None:
                num_neighbs = 4
            if param is None:
                raise ValueError("'corrfun' method requires param to be "
                                 "an NPZ file name containing the corrs and"
                                 "bins arrays.")
            c = np.load(param)
            self._corrs = c['corrs']
            self._bins = c['bins']

        else:
            raise NotImplementedError("Interpolation method %s not supported"
                                      % method)

        self._method = method
        self._neighbs = num_neighbs
        self._par = param

        # What's actually used is the upper bound distance (upb) rather than
        # _radius.
        self._radius = radius
        if self._radius is None:
            self._upb = np.inf
        else:
            self._upb = self._radius

    def num_neighbs(self):
        return self._neighbs

    def radius(self):
        return self._radius

    def set_scene(self, tracer_pos, interp_points,
                  data=None, companionship=None):
        """
        Records scene data for future interpolation using the same scene.

        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row,
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation
            will be done.
        data - (n,d) array, the for the d-dimensional data for tracer n. For
            example, in velocity interpolation this would be (n,3), each tracer
            having 3 components of velocity.
        companionship - an optional array denoting for each interpolation point
            the index of a tracer that should be excluded from it ("companion
            tracer"), useful esp. for analysing a simulated particle that
            started from a true tracer.
        """
        self.set_field_positions(tracer_pos)
        self.set_interp_points(interp_points, companionship)

        if data is not None:
            self.set_data_on_current_field(data)

    def _drop_neighbours_cache(self):
        """
        Clear cached results of nearest-neighbours calculations.
        """
        self.__rel_pos = None
        self.__dists = None
        self.__active_neighbs = None
        self.__matched_data = None

    def set_field_positions(self, positions):
        """
        Sets the positions of points where there is data for interpolation.
        This sets up the structures for efficiently finding distances etc.

        Arguments:
        positions - (n,3) array, for position of n points in 3D space.
        """
        # Keep the original data, for B.C purposes mostly.
        self.__tracers = np.atleast_2d(positions)
        if len(self.__tracers) == 0:
            return
        self.__field_tree = cKDTree(positions)
        self._drop_neighbours_cache()

    def field_positions(self):
        return self.__tracers

    def set_interp_points(self, points, companionship=None):
        """
        Sets the points into which interpolation will be done. It is possible
        to set this once and then do interpolation of several datasets into
        these points.

        Arguments:
        positions - (m,3) array, for position of m target points in 3D space.
        companionship - an optional array denoting for each interpolation point
            the index of a tracer that should be excluded from it ("companion
            tracer"), useful esp. for analysing a simulated particle that
            started from a true tracer.
        """
        self.__interp_pts = np.atleast_2d(points)
        if companionship is None:
            self.__comp = None
        else:
            self.__comp = np.atleast_1d(companionship)
        self._drop_neighbours_cache()

    def set_data_on_current_field(self, data):
        """
        Change the data on the existing interpolation points. This enables
        redoing an interpolation on a scene without recalculating weights,
        when weights are only dependent on position, as they are for most
        interpolation methods.

        Arguments:
        data - (n,d) array, the for the d-dimensional data for n tracers. For
            example, in velocity interpolation this would be (n,3), each tracer
            having 3 components of velocity.
        """
        # Data can be 1d because it needs to be (1,d) or because it's (n,1),
        if data.ndim < 2:
            if data.shape[0] == self.__tracers.shape[0]:
                data = data[:, None]
            else:
                data = data[None, :]
        self.__data = data

    def trim_points(self, which):
        """
        Remove interpolation points from the scene.

        Arguments:
        which - a boolean array, length is number of current particle list
            (as given in set_scene), True to trim a point, False to keep.
        """
        keep = ~which
        self.__interp_pts = self.__interp_pts[keep]
        if self.__dists is not None:
            self.__dists = self.__dists[keep]
            self.__active_neighbs = self.__active_neighbs[keep]

    def _select_neighbs(self, interp_pts, comp=None):
        """
        Find the respective nearest neighbours for each interpolation point,
        and the respective distances to them.

        Arguments:
        interp_pts - (m,d) array for m interpolation points of dimension d.
        comp - optional (m,) array, the companion trajectory id.

        Returns:
        dists - (m,k) array, the distance to each of k nearest neighbours.
            missing neighbours (not enough of them in radius) have infinite
            distance.
        use_part - (m,k) array, the respective indices. Missing neighbours are
            represented by the total number of data points.
        """
        dists, active_neighbs = self.__field_tree.query(
            interp_pts, self._neighbs + 1, distance_upper_bound=self._upb)
        keep = dists > 0.

        if comp is not None:
            keep &= active_neighbs != comp[:, None]

        keep[np.all(keep, axis=1), -1] = False
        dists = dists[keep].reshape(-1, self._neighbs)
        active_neighbs = active_neighbs[keep].reshape(-1, self._neighbs)

        return dists, active_neighbs

    def _forego_laziness(self):
        """
        Populate the neighbours cache.
        """
        # Take one more neighbour because one will be removed, either for
        # being in the same position as the interp points or being its
        # companion (worst case, the extra farthest neighbour is removed).
        self.__dists, self.__active_neighbs = self._select_neighbs(
            self.__interp_pts, self.__comp)

        self.__has_data = self.__active_neighbs < self.__tracers.shape[0]
        matched_pos = np.empty(
            self.__active_neighbs.shape + (self.__tracers.shape[1],))
        matched_pos[self.__has_data] = \
            self.__tracers[self.__active_neighbs[self.__has_data]]
        self.__rel_pos = matched_pos - self.__interp_pts[:, None, :]

        if self._method == 'rbf':
            self.__tracer_dists, _ = _select_neighbs_dense(
                self.__tracers, self.__tracers, self._radius, self._neighbs,
                self.__comp)

    def current_relative_positions(self):
        """
        Returns an (m,k,3) array, the distance between interpolation point m
        and each of k nearest neighbours on each axis.
        """
        return self.__rel_pos

    def current_dists(self):
        if self.__active_neighbs is None:
            self._forego_laziness()

        return self.__dists

    def current_active_neighbs(self):
        if self.__active_neighbs is None:
            self._forego_laziness()

        return self.__active_neighbs

    def current_data(self):
        return self.__data

    def _ensure_matched_data(self):
        """
        Calculate or retrieve cache of data in (m,k,d) structure.
        """
        if self.__matched_data is None:
            self.__matched_data = np.empty(
                self.__active_neighbs.shape + (self.__tracers.shape[1],))
            self.__matched_data[self.__has_data] = \
                self.__data[self.__active_neighbs[self.__has_data]]
        return self.__matched_data

    def interpolate(self, subset=None):
        """
        Performs an interpolation over the recorded scene.

        Arguments:
        subset - a neighbours selection array, such as returned from
            :meth:`which_neighbours`, to replace the recorded selection.
            Default value (None) uses the recorded selection. The recorded
            selection is not changed, so ``subset`` is forgotten after the
            call.

        Returns:
        an (m,3) array with the interpolated value at the position of each
        of m particles.
        """
        # If for some reason tracking failed for a whole frame,
        # interpolation is impossible at that frame. This checks for frame
        # tracking failure.
        if len(self.__tracers) == 0:
            # Temporary measure until I can safely discard frames.
            warnings.warn("No tracers in frame, interpolation returned zeros.")
            ret_shape = self.__data.shape[-1] if self.__data.ndim > 1 else 1
            return np.zeros((self.__interp_pts.shape[0], ret_shape))

        # Check that the cache is populated:
        if self.__active_neighbs is None:
            self._forego_laziness()

        act_neighbs = self.__active_neighbs if subset is None else subset

        return self._meth_interp(act_neighbs)

    def _meth_interp(self, act_neighbs):
        """
        Implement the actual interpolation. Subclass this, not
        :meth:`interpolate`.

        Arguments:
        act_neighbs - a neighbours selection array, such as returned from
            :meth:`which_neighbours`, to replace the recorded selection.
            Default value (None) uses the recorded selection. The recorded
            selection is not changed, so ``subset`` is forgotten after the
            call.
        """
        if self._method == 'rbf':
            return rbf_interp(self.__tracer_dists, self.__dists, act_neighbs,
                              self.__data, self._par)

        if self._method == 'corrfun':
            return corrfun_interp(self.__dists, act_neighbs, self.__data,
                                  self._corrs, self._bins)

        # This isn't supposed to ever happen. The constructor should fail.
        raise NotImplementedError("Interpolation method %s not supported"
                                  % self._method)

    def __call__(self, tracer_pos, interp_points, data, companionship=None):
        """
        Sets up the necessary parameters, and performs the interpolation.
        Does not change the scene set by set_scene if any, so may be used
        for any off-scene interpolation.

        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row,
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation
            will be done.
        data - (n,d) array, the for the d-dimensional data for tracer n. For
            example, in velocity interpolation this would be (n,3), each tracer
            having 3 components of velocity.
        companionship - an optional array denoting for each interpolation point
            the index of a tracer that should be excluded from it ("companion
            tracer"), useful esp. for analysing a simulated particle that
            started from a true tracer.

        Returns:
        vel_interp - an (m,3) array with the interpolated value at the position
            of each particle, [m/s].
        """
        # If for some reason tracking failed for a whole frame, interpolation
        # is impossible at that frame. This checks for frame tracking failure.
        if len(tracer_pos) == 0:
            # Temporary measure until I can safely discard frames.
            warnings.warn("No tracers in frame, interpolation returned zeros.")
            ret_shape = data.shape[-1] if data.ndim > 1 else 1
            return np.zeros((interp_points.shape[0], ret_shape))

        dists, use_parts = select_neighbs(tracer_pos, interp_points,
                                          self._radius, self._neighbs,
                                          companionship)

        if self._method == 'rbf':
            is_mask = (use_parts.dtype == np.bool_)
            k_per_row = use_parts.sum(axis=1) if is_mask else np.array([])
            if is_mask and k_per_row.shape[0] > 0 \
                    and np.all(k_per_row == k_per_row[0]) and k_per_row[0] > 0:
                k = int(k_per_row[0])
                rows = np.arange(dists.shape[0])
                nbrs = np.where(use_parts)[1].reshape(dists.shape[0], k)
                p_nbrs = tracer_pos[nbrs]
                delta = p_nbrs[:, :, None, :] - p_nbrs[:, None, :, :]
                dists_sq = np.sum(delta * delta, axis=-1)
                K_stack = np.exp(-dists_sq * self._par)
                data_stack = data[nbrs]
                coeffs_stack = np.linalg.solve(K_stack, data_stack)

                chosen_dists = dists[rows[:, None], nbrs]
                rbf_chosen = np.exp(-chosen_dists**2 * self._par)
                return np.sum(rbf_chosen[..., None] * coeffs_stack, axis=1)

            tracer_dists = _select_neighbs_dense(tracer_pos, tracer_pos,
                self._radius, self._neighbs, companionship)[0]
            return rbf_interp(tracer_dists, dists, use_parts, data, self._par)

        elif self._method == 'corrfun':
            return corrfun_interp(dists, use_parts, data,
                self._corrs, self._bins)

        else:
            # This isn't supposed to ever happen. The constructor should fail.
            raise NotImplementedError("Interpolation method %s not supported" \
                % self._method)

    def eulerian_jacobian(self, local_interp=None, eps=100e-6):
        """
        A general way to calculate the velocity derivatives. It could be
        enhanced in the future by specific analytical derivatives of the
        different interpolation methods. The Jacobian is calculated for the
        current scene, as recorded with ``set_scene()``

        Arguments:
        local_interp - results of interpolation already performed at the
            position where derivatives are wanted. If not given, an
            interpolation of recorded scene data is automatically performed.
        eps - the dx in each direction.

        Returns: (m,3,3) array, for m interpolation points, [i,j] = du_i/dx_j
        """
        if local_interp is None:
            local_interp = self.interpolate()

        ret = np.empty((self.__interp_pts.shape[0], 3, 3))
        ret[:,:,0] = self(self.__tracers,
            self.__interp_pts + np.r_[eps,0,0], self.__data)
        ret[:,:,1] = self(self.__tracers,
            self.__interp_pts + np.r_[0,eps,0], self.__data)
        ret[:,:,2] = self(self.__tracers,
            self.__interp_pts + np.r_[0,0,eps], self.__data)
        ret = (ret - local_interp[:,:,None]) / eps
        return ret

    def neighb_dists(self, tracer_pos, interp_points, companionship=None):
        """
        The distance from each interpolation point to each data point of those
        used for interpolation. Assumes, for now, a constant number of
        neighbours.

        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row,
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation
            will be done.
        companionship - an optional array denoting for each interpolation point
            the index of a tracer that should be excluded from it ("companion
            tracer"), useful esp. for analysing a simulated particle that
            started from a true tracer.

        Returns:
        ndists - an (m,c) array, for c closest neighbours as defined during
            object construction.
        """
        dists, use_parts = select_neighbs(tracer_pos, interp_points,
            None, self._neighbs, companionship)

        nearest_tracers_count = min(tracer_pos.shape[0], self._neighbs)
        # ``use_parts`` has exactly ``nearest_tracers_count`` True entries per
        # row, so the flattened selection reshapes cleanly. The KD-tree path in
        # ``select_neighbs`` fills only the active neighbour columns, leaving
        # the rest at 0 -- which ``use_parts`` masks out, so the values read
        # back here are identical to the dense computation.
        return dists[use_parts].reshape(
            interp_points.shape[0], nearest_tracers_count)

    def save_config(self, cfg):
        """
        Adds the keys necessary for recreating this interpolant into a
        configuration object. It is the caller's responsibility to do a
        writeback to file.

        Arguments:
        cfg - a ConfigParser object.
        """
        if not cfg.has_section("Interpolant"):
            cfg.add_section("Interpolant")
        cfg.set('Interpolant', 'radius', str(self.radius()))
        cfg.set('Interpolant', 'num_neighbs', str(self.num_neighbs()))
        cfg.set('Interpolant', 'param', str(self._par))
        cfg.set('Interpolant', 'method', self._method)

class InverseDistanceWeighter(GeneralInterpolant):
    """
    Holds all parameters necessary for performing an inverse-distance
    interpolation [#IDW]_. Use is either as a callable object after
    initialization, see :meth:`__call__`, or by setting a scene for repeated
    interpolation, see :meth:`set_scene` and :meth:`interpolate`
    """
    def __init__(self, num_neighbs=None, radius=None, param=None):
        """
        Arguments:
        num_neighbs - number of closest neighbours to interpolate from. If None
            uses 4 neighbours, unless ``radius`` is not None, then ``neighbs``
            is ignored.
        radius - of the search area for neighbours, [m]. If None, select
            closest ``neighbs``.
        param - the inverse power of distance to use (default 1).
        """
        # Defaults:
        if num_neighbs is None:
            num_neighbs = 4
        if param is None:
            param = 1

        GeneralInterpolant.__init__(
            self, 'subclass', num_neighbs, radius, param)
        self._method = 'inv'

    def weights(self, dists, use_parts, unused_marker=None):
        """
        Calculate the respective weight of each tracer j=1..n in the
        interpolation point i=1..m. The actual weight is normalized to the sum
        of weights in the interpolation, not here.

        Arguments:
        dists - an (m,k) array, the respective distance to each
            of k nearest neighbours of each of m interpolation points.
        use_parts - (m,k) boolean array, the index of neighbour 1..k to
            interpolation point 1..m.

        Returns:
        weights - an (m,k) array.
        """
        weights = dists**-self._par
        # If use_parts is boolean, just mask out non-neighbors
        if use_parts.dtype == bool:
            weights[~use_parts] = 0.
        else:
            # If use_parts is indices, mask out invalid indices
            if unused_marker is None:
                unused_marker = dists.shape[1]
            weights[use_parts == unused_marker] = 0.
        return weights

    def set_scene(self, tracer_pos, interp_points,
        data=None, companionship=None):
        """
        Adds to the base class only a precalculation of weights.
        """
        GeneralInterpolant.set_scene(self, tracer_pos, interp_points, data,
            companionship)

    def set_interp_points(self, points, companionship=None):
        GeneralInterpolant.set_interp_points(self, points, companionship)
        if len(self.field_positions()) == 0:
            return

        self._forego_laziness()
        self.__weights = self.weights(self.current_dists(),
            self.current_active_neighbs())

    def trim_points(self, which):
        """
        Remove interpolation points from the scene.

        Arguments:
        which - a boolean array, length is number of current particle list
            (as given in set_scene), True to trim a point, False to keep.
        """
        GeneralInterpolant.trim_points(self, which)
        self.__weights = self.__weights[~which]

    def _apply_weights(self, weights, data):
        """
        Do the actual interpolation after weights have been determined.

        Arguments:
        weights - an (m,k) array, the respective non-normalized weight of each
            of k nearest neighbours of each of m interpolation points.
        data - an (m,k,d) array, for n data points to interpolate from.
        """
        return (weights[...,None] * data).sum(axis=1) \
            / weights.sum(axis=1)[:,None]

    def __call__(self, tracer_pos, interp_points, data, companionship=None):
        """
        Sets up the necessary parameters, and performs the interpolation.
        Does not change the scene set by set_scene if any, so may be used
        for any off-scene interpolation.

        Arguments:
        tracer_pos - (n,3) array, the x,y,z coordinates of one tracer per row,
            in [m]
        interp_points - (m,3) array, coordinates of points where interpolation
            will be done.
        data - (n,d) array, the for the d-dimensional data for tracer n. For
            example, in velocity interpolation this would be (n,3), each tracer
            having 3 components of velocity.
        companionship - an optional array denoting for each interpolation point
            the index of a tracer that should be excluded from it ("companion
            tracer"), useful esp. for analysing a simulated particle that
            started from a true tracer.

        Returns:
        vel_interp - an (m,3) array with the interpolated value at the position
            of each particle, [m/s].
        """
        if len(tracer_pos) == 0:
            warnings.warn("No tracers in frame, interpolation returned zeros.")
            ret_shape = data.shape[-1] if data.ndim > 1 else 1
            return np.zeros((interp_points.shape[0], ret_shape))

        dists, use_parts = select_neighbs(tracer_pos, interp_points,
                                          self._radius, self._neighbs,
                                          companionship)

        m, n = dists.shape
        if data.ndim == 1:
            data = data[:, None]
            
        exact_match = (dists == 0)
        has_exact = exact_match.any(axis=1)
        vel_interp = np.zeros((m, data.shape[1]), dtype=data.dtype)
        weights = self.weights(dists, use_parts)
        
        sum_weights = weights.sum(axis=1)
        valid = (sum_weights != 0) & (~has_exact)
        
        if valid.any():
            vel_interp[valid] = (weights[valid] @ data) / sum_weights[valid, None]
            
        if has_exact.any():
            row_idx = np.where(has_exact)[0]
            col_idx = np.argmax(exact_match[has_exact], axis=1)
            vel_interp[row_idx] = data[col_idx]
            
        # Always return a 2D array (m, d) even for single-point or 1D data
        if vel_interp.ndim == 1:
            vel_interp = vel_interp[:, None]
        return vel_interp

    def _meth_interp(self, act_neighbs=None):
        """
        Implement the actual interpolation. Subclass this, not
        :meth:`interpolate`.

        Arguments:
        act_neighbs - a neighbours selection array, such as returned from
            :meth:`current_active_neighbs`, to replace the recorded selection.
            Default value (None) uses the recorded selection. The recorded
            selection is not changed, so ``subset`` is forgotten after the call.
        """
        if act_neighbs is None:
            act_neighbs = self.current_active_neighbs()
            matched_data = self._ensure_matched_data()
        else:
            data = self.current_data()
            matched_data = np.empty(act_neighbs.shape + (data.shape[1],))
            has_data = act_neighbs < data.shape[0]
            matched_data[has_data] = data[act_neighbs[has_data]]

        if act_neighbs is not self.current_active_neighbs():
            weights = self.weights(self.current_dists(), act_neighbs)
        else:
            weights = self.__weights
        return self._apply_weights(weights, matched_data)

    def eulerian_jacobian(self, local_interp=None, eps=None):
        """
        Velocity derivatives. The Jacobian is calculated for the
        current scene, as recorded with ``set_scene()``

        Arguments:
        local_interp - results of interpolation already performed at the
            position where derivatives are wanted. If not given, an
            interpolation of recorded scene data is automatically performed.
        eps - unused, here for compatibility with base class.

        Returns:
        (m,d,3) array, for m interpolation points and d interpolation
        dimentions. For each point, [i,j] = du_i/dx_j
        """
        if local_interp is None:
            local_interp = self.interpolate()

        dists = self.current_dists()
        use_parts = self.current_active_neighbs()
        rel_pos = self.current_relative_positions()
        matched_data = self._ensure_matched_data()

        der_inv_dists = dists**-(self._par + 2) # m x k
        der_inv_dists[use_parts == self.field_positions().shape[0]] = 0.

        vel_diffs = (matched_data - local_interp[:,None,:]) # m x k x d
        jac = self._par/self.__weights.sum(
                axis=1, keepdims=True)[:,None,None, 0] \
            * np.sum(der_inv_dists[...,None,None]*rel_pos[:,:,None,:]*\
                   vel_diffs[...,None], axis=1)

        return jac

def read_interpolant(conf_fname):
    """
    Builds an Interpolant object based on values in an INI-formatted file.

    Arguments:
    conf_fname - path to configuration file.

    Returns:
    an Interpolant object constructed from values in the configuration file.
    """
    parser = ConfigParser()
    parser.read(conf_fname)

    # Optional arguments:
    kwds = {}
    if parser.has_option('Interpolant', 'num_neighbs'):
        kwds['num_neighbs'] = parser.getint('Interpolant', 'num_neighbs')
    if parser.has_option('Interpolant', 'radius'):
        kwds['radius'] = parser.getfloat('Interpolant', 'radius')
    if parser.has_option('Interpolant', 'param'):
        kwds['param'] = parser.getfloat('Interpolant', 'param')

    return interpolant(parser.get('Interpolant', 'method'), **kwds)
