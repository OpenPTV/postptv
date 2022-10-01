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
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
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
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A
    m_pos = m[0]
    m_vel = m[1] * fps
    m_acc = m[2] * (fps**2 * 2)
    m_jerk = m[3] * (fps**3 * 6)
    
    new_trajs = []
    for traj in trajs:
        if len(traj) < window_size:
            continue
        
        newpos = []
        newvel = []
        newacc = []
        jerk = []
        
        nextacc = []
        nextvel = []
        for y in traj.pos().T: # For each component of pos
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            
            newpos.append(np.convolve( m_pos[::-1], y, mode='valid'))
            newvel.append(np.convolve( m_vel[::-1], y, mode='valid'))
            newacc.append(np.convolve( m_acc[::-1], y, mode='valid'))
            jerk.append(np.convolve( m_jerk[::-1], y, mode='valid'))
            
        newpos = np.r_[newpos].T
        newvel = np.r_[newvel].T
        newacc = np.r_[newacc].T
        jerk = np.r_[jerk].T
        
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
