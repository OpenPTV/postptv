# -*- coding: utf-8 -*-
"""
A class for accessing data and analyses of a scene analysed and saved in the 
format used by flowtracks.analysis.analyse().

Created on Sun Aug 31 11:18:54 2014

@author: yosef
"""

import tables, itertools as it, numpy as np
from .scene import read_dual_scene

class AnalysedScene(object):
    def __init__(self, analysis_file):
        """
        Initializes the objects according to config and data-source metadata
        saved in the analysis file.
        
        Arguments:
        analysis_file - path to the HDF file containing analysis results.
        """
        self._file = tables.open_file(analysis_file, "r")
        self._table = self._file.get_node('/analysis')
        
        config = self._table.attrs['config']
        self._scene = read_dual_scene(config)
        
        # Cache data on user-visible columsn:
        filt = ('trajid', 'time')
        self._keys = []
        self._shapes = []
        desc = self._table.coldescrs
        for name in self._table.colnames:
            if name in filt:
                continue
            self._keys.append(name)
            shape = desc[name].shape
            self._shapes.append(1 if len(shape) == 0 else shape[0])
    
    def __del__(self):
        self._file.close()
    
    def keys(self):
        """
        Return names that may be used to access data in any of the data sources
        available, whether analyses or inertial particles.
        """
        return self._scene.get_particles().keys() + self._keys
    
    def shapes(self):
        """
        Return the number of components per item of each key in the order 
        returned by ``keys()``.
        """
        return self._scene.get_particles().shapes() + self._shapes
        
    def _iter_frame_arrays(self, cond=None):
        """
        Private. Breaks the file down to its frames, and makes arrays of 
        them, iteratively. Also allows filtering the frames.
        
        Arguments:
        cond - an optional PyTables condition string to apply to each frame.
        """
        query_string = '(time == t)'
        if cond is not None:
            query_string = '&'.join([query_string, cond])
            
        for t in xrange(*self._scene.get_range()):
            yield t, self._table.read_where(query_string)
    
    def collect(self, keys, where=None):
        """
        Get values of a given key, either some of them or the ones 
        corresponding to a selection given by 'where'
        
        Arguments:
        keys - a list of keys to take from the data
        where - a dictionary of derived-results keys, with a tuple 
            (min,max,invert) as values. If ``invert`` is false, the search 
            range is between min and max. Otherwise it is anywhere except that.
        
        Returns:
        a list of arrays, in the order of 'keys'.
        """
        # Divide the where condition into the trajectory conditions and 
        # analysis conditions.
        part_cond = None
        an_cond = None
        
        pkeys = self._scene.get_particles().keys()
        if where is not None:
            pc_add = []
            an_cond_add = []
            
            for key, rng in where.iteritems():
                cop1, cop2, lop = ('<','>=','|') if rng[2] else ('>=','<','&')
                cond_string = "((%s %s %g) %s (%s %s %g))" % \
                    (key, cop1, rng[0], lop, key, cop2, rng[1])
                
                if key in pkeys:
                    pc_add.append(cond_string)
                else:
                    an_cond_add.append(cond_string)
            
            if len(pc_add) != 0:
                part_cond = ' & '.join(pc_add)
            if len(an_cond_add) != 0:
                an_cond = ' & '.join(an_cond_add)
                
        # Iterate over dual frame, each from the right source using the 
        # divided conditions.
        res = dict((k, []) for k in keys)
        
        dframe_it = it.izip(
            self._scene.get_particles()._iter_frame_arrays(part_cond),
            self._iter_frame_arrays(an_cond))
            
        for tr_frm, an_frm in dframe_it:
            # Cross reference matching rows.
            tp, p_arr = tr_frm
            ta, a_arr = an_frm
            
            p_trids = p_arr['trajid']
            a_trids = a_arr['trajid']
            trajids = set(p_trids) & set(a_trids)
            
            # select only those from the two frames:
            in_p = np.array([True if tr in trajids else False \
                for tr in p_trids])
            in_a = np.array([True if tr in trajids else False \
                for tr in a_trids])
            
            if len(in_p) > 0:
                p_arr = p_arr[in_p]
            if len(a_arr) > 0:
                a_arr = a_arr[in_a]
            
            # For each dual frame, add the columns of specified keys to the list
            # of that key's results
            for k in keys:
                if k in pkeys:
                    res[k].append(p_arr[k])
                else:
                    res[k].append(a_arr[k])
        
        # stack and return.
        return [np.concatenate(res[k], axis=0) for k in keys]
