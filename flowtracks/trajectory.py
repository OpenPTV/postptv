# -*- coding: utf-8 -*-

import types, numpy as np
import scipy.interpolate as interp
from builtins import object

class Frame(object):
    """
    This is basically a structure with no fancy behaviour. When it is 
    returned from a Flowtracks function, it has two attributes, ``particles``
    and ``tracers`` - each pointing to a :class:`ParticleSnapshot` object 
    holding data for particles of the respective type.
    """
    pass

class ParticleSet(object):
    """
    A base class for manipulting particle data. Knows how many particles it
    has, and holds a varying number of particle properties, each given for
    the entire set. Properties may be created at construction time or later.
    
    When a property is created, it gets a setter method of the same name and
    a getter method prefixed with ``set_``. This applies also for mandatory
    properties.
    """
    def __init__(self, pos, velocity, **kwds):
        """
        Arguments:
        pos - a (t,3) array, the position of one particle in t time-points,
            [m].
        velocity - (t,3) array, corresponding velocity, [m/s].
        kwds - keyword arguments should be arrays whose first dimension == t.
            these are treated as extra attributes to be sliced when creating
            segments.
        """
        base_vals = {
            'pos': pos,
            'velocity': velocity,
        }
        base_vals.update(kwds)
        
        self._check_attr = [] # Attrs to look for when concatenating bundles
        for n, v in base_vals.items():
            self.create_property(n, v)
    
    def create_property(self, propname, init_val):
        """
        Add a property of the set, expected to be an array whose 
        shape[0] == len(self).
        
        Creates the method <propname>(self, selector=None). If selector is
        given, it will return only the selected time-points. Also creates
        set_<propname>(self, value, selector=None) which sets either
        the value over the entire trajectory or just for the selected time 
        points (this requires the property to already exist for the full
        trajectory).
        
        Arguments:
        propname - a string, should be a valid Python identifier.
        init_val - the initial value for the property.
        """
        attr = '_' + propname
        self._check_attr.append(propname)
        
        def getter(self, selector=None):
            if selector is None:
                return self.__dict__[attr]
            else:
                return self.__dict__[attr][selector]
        
        def setter(self, new_val, selector=None):
            if selector is None:
                self.__dict__[attr] = new_val
            else:
                self.__dict__[attr][selector] = new_val
        
        self.__dict__[propname] = \
            types.MethodType(getter, self)
        self.__dict__['set_' + propname] = \
            types.MethodType(setter, self)
        
        if init_val is not None:
            self.__dict__['set_' + propname](init_val)
    
    def has_property(self, propname):
        """
        Checks whether the looked-after property ``propname`` exists for this
        particle set.
        """
        return (propname in self._check_attr)
    
    def schema(self):
        """
        Creates a dictionary keyed by property name whose values are the shape
        of one particle's value for that property. Example: {'pos': (3,),
        'velocity': (3,)}
        """
        return dict((propname, self.__dict__['_' + propname].shape[1:]) \
            for propname in self._check_attr)
    
    def ext_schema(self):
        """
        Extended schema. Like :meth:`schema` but the values of the returned 
        dictionary are a tuple (type, shape). The shape is scalar, so it only 
        supports 1D or 0D items.
        """
        schm = {}
        for propname in self._check_attr:
            prop = self.__dict__['_' + propname]
            shape = prop.shape[-1] if prop.ndim > 1 else 1
            schm[propname] = (prop.dtype.type, shape)
            
        return schm
        
    def as_dict(self):
        """
        Returns a dictionary with the "business" properties only, without all
        the Python bookkeeping and other stuff in the __dict__.
        """
        return dict((propname, self.__dict__['_' + propname]) \
            for propname in self._check_attr)
    
    def __len__(self):
        """Return the number of particles in the set."""
        return self._pos.shape[0]
    
class Trajectory(ParticleSet):
    """
    This is one of the two main classes used for iteration over a scene. It
    inherits from :class:`ParticleSet` with the added demand that a scalar
    trajectory ID (an integer unique amond the scene's trajectories) and a
    ``time`` property.
    """
    def __init__(self, pos, velocity, time, trajid, **kwds):
        """
        Arguments:
        pos - a (t,3) array, the position of one particle in t time-points,
            [m].
        velocity - (t,3) array, corresponding velocity, [m/s].
        time - (t,) array, the clock ticks. No specific units needed.
        trajid - the unique identifier of this trajectory in the set of
            trajectories that belong to the same sequence.
        kwds - keyword arguments should be arrays whose first dimension == t.
            these are treated as extra attributes to be sliced when creating
            segments.
        """
        self._id = trajid
        kwds['time'] = time
        ParticleSet.__init__(self, pos, velocity, **kwds)
    
    def trajid(self):
        return self._id    
    
    def __getitem__(self, selector):
        """
        Gets the data for time points selected as a table of shape (n,8),
        concatenating position, velocity, time, broadcasted trajid.
        
        Arguments:
        selector - any 1d indexing expression known to numpy.
        """
        return np.hstack((self._pos[selector], self._velocity[selector],
            self._time[selector][...,None], 
            np.ones(self._pos[selector].shape[:-1] + (1,))*self._id))
            
    def smoothed(self, err_bound, order):
        """
        Creates a trajectory generated from this trajectory using cubic 
        B-spline interpolation.
        
        Arguments:
        err_bound - amount of deviation a particle is expeted to have around
            its observed place. Determines strength of smoothing.
        order - of the spline (odd, up to 5).
        
        Returns:
        a new :class:`Trajectory` object with the interpolated positions and 
        velocities. If the length of the trajectory < 4, returns self.
        """
        if len(self.time()) < order + 1: return self
        s = (len(self.time()) * err_bound)**2
        
        spline, eval_prms = interp.splprep(list(self.pos().T), k=order, s=s)
        
        new_pos = np.array(interp.splev(eval_prms, spline)).T
        new_vel = np.array(interp.splev(eval_prms, spline, der=1)).T
        new_accel = np.array(interp.splev(eval_prms, spline, der=2)).T
        
        return Trajectory(new_pos, new_vel, self.time(), self.trajid(),
            accel=new_accel)

class ParticleSnapshot(ParticleSet):
    """
    This is one of the two main classes used for iteration over a scene. It
    inherits from :class:`ParticleSet` with the added demand for a scalar
    time and a ``trajid`` property for trajectory ID (an integer unique 
    among the scene's trajectories).
    """
    def __init__(self, pos, velocity, time, trajid, **kwds):
        """
        Arguments:
        pos - a (p,3) array, the position of one particle of p, [m].
        velocity - (p,3) array, corresponding velocity, [m/s].
        trajid - (p,3) array, for each particle in the snapshot, the unique 
            identifier of the trajectory it belongs to.
        time - scalar, the identifier of the frame from which this snapshot
            is taken.
        kwds - keyword arguments should be arrays whose first dimension == p.
            these are treated as extra attributes to be sliced when creating
            segments.
        """
        self._t = time
        kwds['trajid'] = trajid
        ParticleSet.__init__(self, pos, velocity, **kwds)
    
    def time(self):
        return self._t


def mark_unique_rows(all_rows):
    """
    Filter out rows whose position columns represent a particle that already
    appears, so that each particle position appears only once.
    
    Arguments:
    all_rows - an array with n rows and at least 3 columns for position.
    
    Returns:
    an array with the indices of rows to take from the input such that in the
    result, the first 3 columns form a unique combination.
    """
    # Remove duplicates (particles occupying same position):
    srt = np.lexsort(all_rows[:,:3].T)
    diff = np.diff(all_rows[srt,:3], axis=0).any(axis=1)
    uniq = np.r_[srt[0], srt[1:][diff]]
    uniq.sort()
        
    return uniq

def trajectories_in_frame(trajects, frame_num,
    start_times=None, end_times=None, segs=False):
    """
    Notes the indices of trajectories participating in the frame for later 
    extraction.
    
    Arguments:
    trajects - a list of :class:Trajectory objects to filter.
    frame_num - the time value (as found in trajectory.time()) at which the
        trajectory should be active.
    start_times, end_times - each a ``len(trajects)`` array containing the 
        corresponding start/end frame number of each trajectory, respectively.
    segs - true if the trajectory should be active also in the following frame.
    
    Returns:
    traj_nums = the indices of active trajectories in ``trajects``.
    """
    if start_times is None or end_times is None:
        start_end = [(tr.time()[0], tr.time()[-1]) for tr in trajects]
        start_times, end_times = map(np.array, zip(*start_end))
    
    end_frm = (frame_num + 1) if segs else frame_num
    cands = (frame_num >= start_times) & (end_frm <= end_times)
    cand_nums = np.nonzero(cands)[0]
    
    if len(cand_nums) > 0:
        # Filter candidates with overlapping particles.
        frm_ixs = frame_num - start_times[cands]
        pos = np.array([trajects[trix].pos()[frm] \
            for trix, frm in zip(cand_nums, frm_ixs)])
        
        update_cands = np.zeros(pos.shape[0], dtype=np.bool)
        update_cands[mark_unique_rows(pos)] = True
        cands[cands] = update_cands
        cand_nums = np.nonzero(cands)[0]
    
    return cand_nums

def take_snapshot(trajects, frame, schema):
    """
    Goes over a list of trajectories and extracts the particle data at a given
    time point. If the trajectory list is empty, creates an empty snapshot.
    
    Arguments:
    trajects - a list of :class:Trajectory objects to query.
    frame - the frame number to which snapshot data belongs.
    schema - a dict, ``{propname: shape tuple}``, as given by the trajectory's
        :meth:`~.ParticleSet.schema`. This is only needed for consistency in 
        the case of an empty trajectory list resulting in an empty snapshot.
    
    Returns:
    a :class:`ParticleSnapshot` object with all the particles in the given frame.
    """
    if len(trajects) == 0:
        kwds = dict((k, np.empty((0,) + v)) for k, v in schema.items())
        kwds['time'] = frame
        return ParticleSnapshot(trajid=np.empty(0), **kwds)
    
    kwds = dict((k, np.empty(
        (len(trajects),) + v, 
        dtype=trajects[0].__dict__['_' + k].dtype)) \
        for k, v in schema.items())
    copy_keys = kwds.keys()
    kwds['trajid'] = np.empty(len(trajects), dtype=np.int_)
    
    for trix, traj in enumerate(trajects):
        first_frame = traj.time()[0]
        for prop in copy_keys:
            kwds[prop][trix] = traj.__dict__[prop](frame - first_frame)
        kwds['trajid'][trix] = traj.trajid()
    
    kwds['time'] = frame
    return ParticleSnapshot(**kwds)
