"""
Trajectory smoothing routines. These are routines that are out of the
Trajectory object because they precompute values that are dependent only on the
smoothing method, and not on the trajectory itself, so they may be shared for
processing a whole list of trajectories.
"""
from flowtracks.trajectory import Trajectory
import numpy as np

def savitzky_golay(trajs, fps, window_size, order):
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

    Returns:
    new_trajs - a list of Trajectory objects representing the smoothed
        trajectories. Trajectories shorter than the window size are discarded.

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
    order_range = range(order+1)
    half_window = (window_size -1) // 2

    # Properties that should not be copied from the old trajectory because
    # they are obtained otherwise (or copied elsewhere).
    smoothed_keys = ['pos', 'velocity', 'accel', 'acc_pp', 'time',
        'trajid']

    # precompute coefficients
    b = np.array([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b)
    m_pos = m[0]
    m_vel = m[1] * fps
    m_acc = m[2] * (fps**2 * 2) if order >= 2 else np.zeros(window_size)
    m_jerk = m[3] * (fps**3 * 6) if order >= 3 else np.zeros(window_size)

    # Coefficient rows, reused for every component and every output derivative
    # (pos/vel/acc/jerk). Multiplying a windowed view of the padded signal by
    # this matrix is mathematically identical to the previous per-component
    # ``np.convolve(m_*[::-1], y, mode='valid')`` loop, because
    # ``convolve(m[::-1], y, 'valid')`` evaluates to ``window . m`` (a forward
    # dot product of the window with the coefficients).
    M = np.stack([m_pos, m_vel, m_acc, m_jerk])   # (4, window_size)

    new_trajs = []
    for traj in trajs:
        if len(traj) < window_size:
            continue

        pos = traj.pos()
        # pad the signal at the extremes with values mirrored from the signal
        # itself (the abs makes it a reflection, not a plain reflect).
        firstvals = pos[0] - np.abs(pos[1:half_window+1][::-1] - pos[0])
        lastvals = pos[-1] + np.abs(pos[-half_window-1:-1][::-1] - pos[-1])
        padded = np.concatenate((firstvals, pos, lastvals), axis=0)

        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_size, axis=0)                 # (L, 3, window_size)
        out = windows @ M.T                              # (L, 3, 4)

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
