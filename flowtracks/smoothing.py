"""
Trajectory smoothing routines. These are routines that are out of the
Trajectory object because they precompute values that are dependent only on the
smoothing method, and not on the trajectory itself, so they may be shared for
processing a whole list of trajectories.
"""
import numpy as np

from flowtracks.trajectory import Trajectory


def savitzky_golay(trajs, fps, window_size, order, min_window=None):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters:
    trajs - a list of Trajectory objects
    window_size - int,
        the length of the window. Must be an odd integer number.
    fps - frames per second, used for calculating velocity and acceleration.
    order - int,
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    min_window - int or None. None (default): trajectories shorter than
        `window_size` are discarded. Otherwise a shorter trajectory is
        smoothed with the largest odd window it fills (down to
        `min_window`) instead of being dropped -- dropping them removes the
        short, often fast, tracks and biases flow statistics.

    Returns:
    new_trajs - a list of Trajectory objects representing the smoothed
        trajectories. Trajectories shorter than the window size (or than
        `min_window` when given) are discarded.

    Notes:
    The Savitzky-Golay [1][3] is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point [2].

    References:

    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of \
       Data by Simplified Least Squares Procedures. Analytical \
       Chemistry, 1964, 36 (8), pp 1627-1639.

    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing \
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery \
       Cambridge University Press ISBN-13: 9780521880688

    .. [3] http://wiki.scipy.org/Cookbook/SavitzkyGolay
    """
    try:
        window_size = np.abs(np.int_(window_size))
        order = np.abs(np.int_(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    if min_window is None:
        min_window = window_size
    min_window = int(min_window)
    if min_window % 2 != 1 or min_window < order + 2 or min_window > window_size:
        raise TypeError("min_window must be odd, >= order + 2 and <= window_size")
    order_range = range(order+1)

    # Properties that should not be copied from the old trajectory because
    # they are obtained otherwise (or copied elsewhere).
    smoothed_keys = ['pos', 'velocity', 'accel', 'acc_pp', 'time',
        'trajid']

    def coeffs(window):
        """(M, M_head, M_tail) for one window size. Least-squares polynomial
        over the window: c = m @ y for y(k) = sum_p c_p k^p; each (4, window)
        row block gives pos/vel/acc/jerk of that polynomial at an offset s
        from the window centre (s=0: classic SG). The first/last half_window
        points are NOT padded: they are evaluated on the polynomial fitted to
        the first/last full window (scipy savgol_filter mode="interp"). The
        previous abs()-mirror padding was only right for increasing
        coordinates; on decreasing ones it folded the track back and pulled
        the end velocities toward zero (~20 % of the speed on a fast jet)."""
        half = (window - 1) // 2
        b = np.array([[k**i for i in order_range] for k in range(-half, half + 1)])
        m = np.linalg.pinv(b)

        def at(s):
            rows = np.zeros((4, window))
            for p in order_range:
                for d in range(min(p, 3) + 1):
                    fall = np.prod(np.arange(p, p - d, -1)) if d else 1.0
                    rows[d] += fall * float(s) ** (p - d) * m[p] * fps**d
            return rows

        return at(0), [at(i - half) for i in range(half)], [at(i + 1) for i in range(half)]

    cache = {}

    new_trajs = []
    for traj in trajs:
        time = np.asarray(traj.time())
        pos = traj.pos()
        # Frame gaps (e.g. a repair join across a missed detection): the
        # filter assumes uniform spacing, so fill the missing frames by
        # linear interpolation, smooth on the full grid, and return values
        # only at the frames that were actually observed.
        steps = np.diff(time)
        observed = slice(None)
        if len(steps) and np.any(steps != steps.min()) and steps.min() > 0:
            grid = np.arange(time[0], time[-1] + steps.min() / 2, steps.min())
            pos = np.column_stack([np.interp(grid, time, pos[:, k]) for k in range(pos.shape[1])])
            observed = np.searchsorted(grid, time)
        if len(pos) < min_window:
            continue
        window = min(window_size, len(pos) if len(pos) % 2 else len(pos) - 1)
        if window not in cache:
            cache[window] = coeffs(window)
        M, M_head, M_tail = cache[window]
        half_window = (window - 1) // 2

        windows = np.lib.stride_tricks.sliding_window_view(
            pos, window, axis=0)                         # (L-w+1, 3, window)
        out = np.concatenate([
            np.stack([windows[0] @ Mh.T for Mh in M_head]),
            windows @ M.T,
            np.stack([windows[-1] @ Mt.T for Mt in M_tail]),
        ]) if half_window else windows @ M.T              # (L, 3, 4)
        out = out[observed]

        newpos = out[:, :, 0]
        newvel = out[:, :, 1]
        newacc = out[:, :, 2]
        jerk = out[:, :, 3]

        # Velocity and acceleration evaluated at i = 1 rather than i = 0,
        # for comparison with the i = 0 values from next polynomial.
        # Delta t treatment is in m_*.
        # Assumed that the first point is trimmed, the zeros are  just for
        # alignment.
        nextvel = np.vstack((np.zeros(3), newvel + newacc/fps + jerk/2./fps**2))[:-1]
        nextacc = np.vstack((np.zeros(3), newacc + jerk/fps))[:-1]

        newtraj = Trajectory(newpos, newvel, traj.time(), traj.trajid(),
            accel=newacc, vel_pp=nextvel, acc_pp = nextacc)

        # Copy unsmoothed properties from old trajectory:
        for k, v in traj.as_dict().items():
            if k not in smoothed_keys:
                newtraj.create_property(k, v)

        new_trajs.append(newtraj)

    return new_trajs
