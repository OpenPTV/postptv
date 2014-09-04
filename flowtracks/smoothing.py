# -*- coding: utf-8 -*-
"""
Trajectory smoothing routines. These are routines that are out of the 
Trajectory object because they precompute values that are dependent only on the
smoothing method, and not on the trajectory itself, so they may be shared for
processing a whole list of trajectories.

Created on Thu Nov 14 11:41:53 2013

@author: yosef
"""

from flowtracks.trajectory import Trajectory
import numpy as np

def savitzky_golay(trajs, fps, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Parameters
    ----------
    trajs : a list of Trajectory objects
    window_size : int
        the length of the window. Must be an odd integer number.
    fps : frames per second, used for calculating velocity and acceleration.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    
    Returns
    -------
    new_trajs : a list of Trajectory objects representing the smoothed 
        trajectories. Trajectories shorter than the window size are discarded.
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    .. [3] http://wiki.scipy.org/Cookbook/SavitzkyGolay
    """
    from math import factorial

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
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    new_trajs = []
    for traj in trajs:
        if len(traj) < window_size:
            continue
        
        newpos = []
        for y in traj.pos().T:
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            newpos.append(np.convolve( m[::-1], y, mode='valid'))
            
        newpos = np.r_[newpos].T
        newvel = np.vstack(( np.diff(newpos, axis=0)*fps, np.zeros((1,3)) ))
        newacc = np.vstack(( np.diff(newvel[:-1], axis=0)*fps, np.zeros((2,3)) ))
        
        new_trajs.append(Trajectory(newpos, newvel, traj.time(), traj.trajid(),
            accel=newacc))
    
    return new_trajs
