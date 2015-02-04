# -*- coding: utf-8 -*-

"""
Contains functions for reading frame-by-frame flow data and trajectories in 
various formats.
"""
import os, os.path, re
from ConfigParser import SafeConfigParser
from StringIO import StringIO

import numpy as np
from scipy import io
import tables

from .particle import Particle
from .trajectory import Trajectory, mark_unique_rows, \
    Frame, take_snapshot, trajectories_in_frame

class FramesIterator(object):
    def __init__(self, fname_tmpl, fmt, skip, first=None, last=None):
        """
        Arguments:
        fname_tmpl - a template file name representing all ptv_is/xuap files in
            the directory, with exactly one '%d' representing the frame number.
        fmt - a dtype object describing the table structure to be read.
        skip - number of header lines to skip in each file.
        first, last - inclusive range of frames to read, rel. filename 
            numbering.
        """
        self._frmix = 0
        self._read_frame = lambda fix: np.atleast_1d(
            np.loadtxt(fname_tmpl % fix, dtype=fmt, skiprows=skip))
        
        dirname, basename = os.path.split(fname_tmpl)
        is_data_file = re.compile(basename.replace('%d', '(\d+)', 1))

        # Collect existing frames. This is necessary to ensure that frames are
        # processed in the correct order.
        self._frame_nums = []
        for name in os.listdir(dirname):
            match = is_data_file.match(name)
            if match is None: continue
            frame = int(match.group(1))

            if first is not None and frame < first: continue
            if last is not None and frame > last: continue
            # Note that we're reading one extra frame, otherwise the last frame
            # has 0 path segments.

            self._frame_nums.append(frame)

        # Process frames in order.
        self._frame_nums.sort()
    
    def __iter__(self):
        return self
    
    def next(self):
        """
        Returns:
        frm_num - frame number as recorded in the file names.
        frame - a table corresponding to the format (``fmt``) given to 
            __init__().
        """
        curframenum = self._frmix
        if len(self._frame_nums) <= curframenum:
            raise StopIteration
        
        frame = self._read_frame(self._frame_nums[curframenum])
        self._frmix += 1
        return self._frame_nums[curframenum], frame
        
class SingleFileIterator(object):
    def __init__(self, fname, fmt):
        """
        Arguments:
        fname - file name containing concatenated ptv_is frames, separated by
            empty line.
        fmt - a dtype object describing the table structure to be read.
        skip - number of header lines to skip in each file.
        """
        self._frmix = 0
        self._f = open(fname, 'r')
        self._read_frame = lambda str_tbl: np.loadtxt(str_tbl, dtype=fmt)
    
    def __iter__(self):
        return self
    
    def next(self):
        """
        Returns:
        frm_num - frame number as recorded in the file names.
        frame - a table corresponding to the format (``fmt``) given to 
            __init__().
        """
        curframenum = self._frmix
        
        # Make the stringio object to be used by read_frame
        lines = []
        for line in self._f:
            if re.match("^\s*$", line):
                break
            lines.append(line)
        
        if len(lines) == 0:
            raise StopIteration # EOF
        
        str_tbl = StringIO("".join(lines))
        frame = self._read_frame(str_tbl)
        self._frmix += 1
        return curframenum, frame
    
    def __def__(self):
        self._f.close()
    
def collect_particles(fname_tmpl, frame, path_seg=False):
    """
    Going backwards over trajAcc files [2], starting from a given frame,
    collect the data for all particles whose path begins in earlier frames and
    go as far as the given frame.
    
    Arguments:
    fname_tmpl - a format-string with one %d where the frame number should be
        inserted.
    frame - the frame number.
    path_seg - if True, find for each particle also the particle matching it in
        the next time step, so that acceleration can be calculated. Discarts 
        unmatched particles.
    
    Returns:
    a table with columns 0-5,33 from the files, combined from all lines in all
    files that belong to particles in frame ``frame``. If path_seg is True, the
    table has two layers (a 2,n,7 array), the first is the particles in the 
    given frame, the second is their matches in the next time step.
    """
    selected = []
    cur_frame = frame
    fname_tmpl = os.path.expanduser(fname_tmpl)
    
    while os.path.exists(fname_tmpl % cur_frame):
        table = np.loadtxt(fname_tmpl % cur_frame, usecols=(0,1,2,3,4,5,33))
        path_age = frame - cur_frame
        
        if path_seg is True:
            segs = np.nonzero((table[:,-1] == path_age) & \
                (np.roll(table[:,-1], -1) == path_age + 1))[0]
            in_frame = np.concatenate(
                (table[segs,:][None,...], table[segs + 1,:][None,...]), axis=0)
        else:
            in_frame = table[table[:,-1] == path_age]
        
        # When no previous path is long enough to reach ``frame``:
        if in_frame.shape[0] == 0:
            break
        
        selected.append(in_frame)
        cur_frame -= 1
    
    if path_seg is True:
        all_rows = np.concatenate(selected, axis=1)
        return all_rows[:,mark_unique_rows(all_rows[0])]
    else:
        all_rows = np.vstack(selected)
        return all_rows[mark_unique_rows(all_rows)]

def trajectories_mat(fname):
    data = io.loadmat(os.path.expanduser(fname))
    # Get the workspace variable holding the trajectories:
    data_name = [s for s in data.keys() \
        if (not s.startswith('__')) and (not s == 'directory')][0]
    raw = data[data_name][:,0]
    
    trajects = []
    for traj in raw:
        # also convert data from mm to m.
        pos = np.hstack((traj['xf'], traj['yf'], traj['zf']))/1000.
        vel = np.hstack((traj['uf'], traj['vf'], traj['wf']))/1000.
        accel = np.hstack((traj['axf'], traj['ayf'], traj['azf']))/1000.
        t = traj['t'].squeeze()
        trajid = traj['trajid'][0,0]
        trajects.append(Trajectory(pos, vel, t, trajid, accel=accel))
    
    return trajects

def trajectories_acc(fname, first=None, last=None):
    """
    Extract all trajectories in a directory of trajAcc files.
    
    Arguments:
    fname - a template file name representing all trajAcc files in the
        directory, with exactly one '%d' representing the frame number.
    first, last - inclusive range of frames to read, rel. filename numbering.
    
    Returns:
    a list of Trajectory objects.
    """
    trajects = []
    dirname, basename = os.path.split(os.path.expanduser(fname))
    is_data_file = re.compile(basename.replace('%d', '(\d+)', 1))
    
    for fname in os.listdir(dirname):
        match = is_data_file.match(fname)
        if match is None: continue
        frame = int(match.group(1))
        
        if first is not None and frame < first: continue
        if last is not None and frame >= last: break
        
        table = np.loadtxt(os.path.join(dirname, fname),
            usecols=(0,1,2,3,4,5,6,7,8,33))
        traj_starts = np.nonzero(table[:,-1] == 0)[0]
        traj_ends = np.r_[traj_starts[1:], table.shape[0]]
        
        for s, e in zip(traj_starts, traj_ends):
            trajects.append(Trajectory(
                table[s:e,0:3], table[s:e,3:6], table[s:e,-1] + frame,
                len(trajects), accel= table[s:e,6:9]))
    
    return trajects

def iter_trajectories_ptvis(fname, first=None, last=None, frate=1., xuap=False,
    traj_min_len=None):
    """
    Extract all trajectories in a directory of ptv_is files, as generated by
    programs in the 3d-ptv/pyptv family.
    
    Arguments:
    fname - a template file name representing all ptv_is/xuap files in the
        directory, with exactly one '%d' representing the frame number. If
        no '%d' is found, the input is assumed to be in the Ron format - single
        file of concatenated ptv_is files, each stripped of the particle count
        line (first line) and separated from the next by an empty line.
    first, last - inclusive range of frames to read, rel. filename numbering.
    frate - frame rate, used for calculating velocities by backward 
        derivative.
    xuap - The format is extended with colums for velocity and acceleration.
    traj_min_len - do not include trajectories shorter than this many frames.
    
    Yields:
    each of the trajectories in the ptv_is data in order, as a Trajectory 
    instance with velocity and acceleration.
    """
    fname = os.path.expanduser(fname)
    
    if xuap:
        fmt = np.dtype([('prev', 'i4'), ('next', 'i4'), ('pos', '3f8'),
                        ('pos_int', '3f8'), ('vel', '3f8'), ('acc', '3f8')])
        skip = 0
        count_base = 1
        def_tr_len = 2
    else:
        fmt = np.dtype([('prev', 'i4'), ('next', 'i4'), ('pos', '3f8')])
        skip = 1
        count_base = 0
        def_tr_len = 2
    
    if traj_min_len is None:
        traj_min_len = def_tr_len

    frames = []
    if '%d' in fname:
        frm_iter = FramesIterator(fname, fmt, skip, first, last)
    else:
        frm_iter = SingleFileIterator(fname, fmt)
    
    # In the first frame, every particle starts a trajectory.
    frame_num, table = frm_iter.next()    
    
    pos = table['pos']
    if not xuap: pos /=1000.
    
    if 'vel' in fmt.fields:
        vel = table['vel']
    else:
        vel = np.zeros_like(pos)
    
    max_traj = table.shape[0]
    trids = np.arange(max_traj)
    frame = np.hstack((pos, vel, np.ones((max_traj, 1))*frame_num,
        trids[:,None]))
    frames.append(frame)
    
    traj_starts = {} # what is the starting frame for each trajectory.
    for trid in trids:
        traj_starts[trid] = 0
    
    trajects = {}
    
    # Single-frame trajectories in first frame:
    ending = table['next'] - count_base == -2
    if ending.any():
        ending_trids = np.atleast_1d(np.int_(frame[ending,-1]))
        for trid in ending_trids:
            trajects[trid] = frame[frame[:,-1] == trid]
    
    frame_buffer_start = 0
    
    # Main loop, sequentially read each frame and process:
    for fix, frm in enumerate(frm_iter):
        frame_num, table = frm
        
        if table.ndim == 0:
            frames.append(None)
            continue
            # We assume that the next frame will have no continuing particles,
            # and this case is caused by detection failure. Otherwise the code
            # that generated the data has a bug that can't be dealt with here.
        
        # Continue existing trajectories into this frame:
        cont = table['prev'] - count_base > -1
        traj = np.empty(table['prev'].shape)
        
        if  frames[-1] is not None:
            prev_ix = table['prev'][cont] - count_base
            traj[cont] = frames[-1][:,-1][prev_ix]
        
        # Start new trajectories:
        num_new_traj = np.sum(~cont)
        traj[~cont] = np.arange(max_traj, max_traj + num_new_traj)
        for trid in traj[~cont]:
            traj_starts[trid] = fix + 1
        max_traj += num_new_traj
        
        # Consolidate into frame table.
        pos = table['pos']
        if not xuap: pos /= 1000.
        t = np.ones((table.shape[0], 1))*frame_num
        
        if 'vel' in fmt.fields:
            vel = table['vel']
        else:
            vel = np.zeros_like(pos)
        
        frame = np.hstack((pos, vel, t, traj[:,None]))
        if 'vel' not in fmt.fields and frames[-1] is not None:
            # Update velocity of previous frame's continuing particles
            frames[-1][prev_ix,3:6] = \
                (pos[cont] - frames[-1][prev_ix,:3]) * frate
        frames.append(frame)
        
        # Make Trajectory objects from fully-read trajectories, so we can 
        # discard early frames they're in.
        ending = table['next'] - count_base == -2
        if not ending.any():
            continue
        
        ending_trids = np.atleast_1d(np.int_(frame[ending,-1]))
        ending_starts = np.r_[[traj_starts[trid] for trid in ending_trids]]
        
        # Filter short trajectories:
        traj_lens = fix - ending_starts + 2
        long_trjs = traj_lens >= traj_min_len
        traj_lens = traj_lens[long_trjs]
        ending_trids = ending_trids[long_trjs]
        ending_starts = ending_starts[long_trjs]
        
        # Preallocate memory for speed.
        for trid, trlen in zip(ending_trids, traj_lens):
            trajects[trid] = np.empty((trlen, frames[-1].shape[-1]))
            
        for scanix, past_frame in enumerate(frames):
            if past_frame is None: continue
            
            in_frame = ending_starts <= scanix + frame_buffer_start
            for trid in ending_trids[in_frame]:
                traj_rel_ix = scanix + frame_buffer_start - traj_starts[trid]
                traj_locator = past_frame[:,-1] == trid
                trajects[trid][traj_rel_ix] = past_frame[traj_locator][0]
        
        # Discard frames that only have trajectories that ended.
        cont_trids = frame[~ending,-1]
        if not cont_trids.any():
            new_start = fix
        else:
            new_start = min([traj_starts[trid] for trid in cont_trids])
        
        frames = frames[new_start - frame_buffer_start : ]
        frame_buffer_start = new_start
        
        # Convert the dictionary of trajectory arrays to list of Trajectory 
        # objects and give them back.
        for trid in ending_trids:
            traj = trajects[trid]
            traj = Trajectory(traj[:,:3], traj[:,3:6], traj[:,6],
                np.int(traj[0,7]))
            
            # Add forward-difference acceleration:
            accel = np.empty_like(traj.velocity())
            accel[[-2,-1],:] = 0.
            accel[:-2] = (traj.velocity()[1:-1] - traj.velocity()[:-2]) * frate
            traj.create_property('accel', accel)
            
            # Get rid of the working memory.
            del trajects[trid]
            del traj_starts[trid]
            yield traj
                
def trajectories_ptvis(fname, first=None, last=None, frate=1., xuap=False,
    traj_min_len=None):
    """
    Extract all trajectories in a directory of ptv_is files, as generated by
    programs in the 3d-ptv/pyptv family.
    
    Arguments:
    fname - a template file name representing all ptv_is/xuap files in the
        directory, with exactly one '%d' representing the frame number. If
        no '%d' is found, the input is assumed to be in the Ron format - single
        file of concatenated ptv_is files, each stripped of the particle count
        line (first line) and separated from the next by an empty line.
    first, last - inclusive range of frames to read, rel. filename numbering.
    frate - frame rate, used for calculating velocities by backward 
        derivative.
    xuap - The format is extended with colums for velocity and acceleration.
    traj_min_len - do not include trajectories shorter than this many frames.
    
    Returns:
    a list of Trajectory objects.
    """
    return [t for t in iter_trajectories_ptvis(fname, first, last, frate, 
        xuap, traj_min_len)]

def trajectories(fname, first, last, frate, fmt=None, traj_min_len=None,
    iter_allowed=False):
    """
    Extract all trajectories in a given target location. The location format
    is interpreted based on the format of the data files, in the respective 
    trajectories_* functions.
    
    Trajectories of one frame are filtered out.
    
    Arguments:
    fname - a template file name, as needed by the appropriate suboridinate
        function.
    first, last - inclusive range of frames to read, rel. filename numbering.
    frate - frame rate under which the film was shot - needed for ptvis 
        trajectories.
    traj_min_len - on some formats, (currently ptv_is and xuap) it is possible
        to filter trajectories with less frames than this, saving memory.
    iter_allowed - may return an iterator instead of a list.
    
    Returns:
    a list (or iterator) of Trajectory objects.
    """
    # Infer format:
    if fmt is None:
        fmt = infer_format(fname)
    
    filter_needed = True
    
    if fmt == 'mat':
        traj = trajectories_mat(fname)
    
    elif fmt == 'npz':
        traj, _ = load_trajectories(fname, first, last)
    
    elif fmt == 'acc':
        traj = trajectories_acc(fname, first, last)
    
    elif fmt == 'ptvis':
        filter_needed = False
        if iter_allowed:
            traj = iter_trajectories_ptvis(fname, first, last, frate,
                traj_min_len=traj_min_len)
        else:
            traj = trajectories_ptvis(fname, first, last, frate,
                traj_min_len=traj_min_len)
    
    elif fmt == 'xuap':
        traj = trajectories_ptvis(fname, first, last, frate, xuap=True,
            traj_min_len=traj_min_len)
    
    elif fmt == 'hdf':
        traj = trajectories_table(fname, first, last)
    
    if filter_needed:
        if traj_min_len is None:
            traj_min_len = 2
        traj = [tr for tr in traj if len(tr) >= traj_min_len]
    
    return traj
        
def infer_format(fname):
    """
    Try to guess the format of a particles data file by its name.
    
    Arguments:
    fname - the file name from which to guess the format.
    
    Returns:
    A string marking the format. Currently one of 'acc', 'mat' or 'ptvis'.
    """
    if fname.endswith('mat'):
        return 'mat'
    elif fname.endswith('/'):
        return 'npz'
    elif 'ptv_is' in fname or fname.endswith('.txt'):
        return 'ptvis'
    elif 'xuap' in fname:
        return 'xuap'
    elif fname.endswith('h5') or fname.endswith('hdf'):
        return 'hdf'
    else:
        return 'acc'

def collect_particles_mat(fname, frame, path_seg=False):
    """
    The same as collect_particles, but uses mat files as generated by the PTV
    post-processing code.
    """
    trajects = trajectories_mat(fname)
    return collect_particles_generic(trajects, frame, path_seg)
    
def collect_particles_generic(trajects, frame, path_seg=False):
    """
    Collect from a list of trajectories the particles appearing in a given
    frame.
    
    Arguments:
    trajects - a list of Trajectory objects.
    frame - the frame number.
    path_seg - if True, find for each particle also the particle matching it in
        the next time step, so that acceleration can be calculated. Discarts 
        unmatched particles.
    
    Returns:
    a table with columns 0-2 for position, 3-5 for velocity, 6 for frame
    number and 7 for trajectory id. If path_seg is True, the table has two
    layers (a 2,n,7 array), the first is the particles in the given frame,
    the second is their matches in the next time step.
    """
    selected = []
    for traj in trajects:
        if path_seg is True:
            t = np.nonzero((traj.time()[:-1] == frame) & \
                (traj.time()[1:] == frame + 1))[0]
            if len(t) == 0: continue
            
            t = t[0]
            sel = traj[t : t + 2].reshape(2, 1, -1)
            
        else:
            t = np.nonzero(traj.time() == frame)[0]
            if len(t) == 0: continue
            sel = traj[t[0]]
        
        selected.append(sel)
    
    if len(selected) == 0:
        return np.empty((2,0,7))
    
    if path_seg is True:
        all_rows = np.concatenate(selected, axis=1)
        return all_rows[:,mark_unique_rows(all_rows[0])]
    else:
        all_rows = np.vstack(selected)
        return all_rows[mark_unique_rows(all_rows)]


def read_frame_data(conf_fname):
    """
    Read a configuration file in INI format, which specifies the locations 
    where particle positions and velocities should be read from, and directly
    stores some scalar frame values, like particle densidy etc.
    
    Arguments:
    conf_fname - name of the config file
    
    Returns:
    particle - a Particle object holding particle properties.
    frate - the frame rate at which the scene was shot.
    frame, next_frame - Frame objects holding the tracers and particles data
        for the time points indicated in config, and the one immediately 
        following it.
    """
    parser = SafeConfigParser()
    parser.read(conf_fname)
    
    particle = Particle(
        parser.getfloat("Particle", "diameter"),
        parser.getfloat("Particle", "density"))
    
    first_frame = parser.getint("Scene", "frame")
    frate = parser.getfloat("Scene", "frame rate")
    
    fname = parser.get("Scene", "tracer_file")
    tracer_trjs = trajectories(fname, first_frame, first_frame + 2, 
        frate, None)
    tracer_ixs = trajectories_in_frame(tracer_trjs, first_frame, segs=True)
    
    fname = parser.get("Scene", "part_file")
    part_trjs = trajectories(fname, first_frame, first_frame + 2, frate, None)
    part_ixs = trajectories_in_frame(part_trjs, first_frame, segs=True)

    data = []
    for frame_num in [first_frame, first_frame + 1]:
        frame = Frame()
        frame.tracers = take_snapshot([tracer_trjs[t] for t in tracer_ixs], 
            frame_num, tracer_trjs[0].schema())
    
        frame.particles = take_snapshot([part_trjs[t] for t in part_ixs],
            frame_num, part_trjs[0].schema())
        
        data.append(frame)

    return particle, frate, data[0], data[1]

def save_trajectories(output_dir, trajects, per_traject_adds, **kwds):
    """
    Save for each trajectory the data for this trajectory, as well as 
    additional data attached to each trajectory, such as trajectory 
    reconstructions. Creates in the output directory one npz file per
    trajectory, containing the arrays of the trajectory as well as the added 
    arrays.
    
    Arguments:
    output_dir - name of the directory where output should be placed. Will be 
        created if it does not exist.
    trajects - a list of Trajectory objects.
    per_traject_adds - a dictionary, whose keys are the array names to use when
        saving, and vaslues are trajid-keyed dictionaries with the actual 
        arrays to save for each trajectory.
    kwds - free arrays to save in the output dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for traj in trajects:
        save_data = dict(('traj:' + k, v) \
            for k, v in traj.as_dict().iteritems())
        for k, v in per_traject_adds.iteritems():
            save_data[k] = v[traj.trajid()]
        
        np.savez(os.path.join(output_dir, 'traj_%d' % traj.trajid()),
            **save_data)
    
    # Save non-trajectory arrays:
    for k, v in kwds.iteritems():
        np.save(os.path.join(output_dir, k), v)
    
def save_particles_table(filename, trajects, trim=None):
    """
    Save trajectory data as a table of particles, with added columns for time
    (frame number) and trajid - the last one may be indexed. Note that no extra
    (per-trajectory or meta) data is allowed here, unlike the npz save format.
    
    Arguments:
    filename - name of output PyTables HDF5 file to create. The 'h5' extension
        is recommended so that infer_format() knows what to do with it.
    trajects - a list of Trajectory objects to save.
    trim - if None, remove this many time points from each end of each 
        trajectory before saving.
    """
    table = None
    trim_len = 0 if trim is None else trim * 2
    
    outfile = tables.openFile(filename, mode='w')
    bounds_tab = outfile.createTable('/', 'bounds', 
        np.dtype([('trajid', int, 1), ('first', int, 1), ('last', int, 1)]))
    
    for traj in trajects:
        if len(traj) - trim_len <= 0:
            continue
        
        # First trajectory creates the table:
        if table is None:
            # Format of records in a trajectory array :
            fields = [('trajid', int, 1)] + [(field,) + desc \
                for field, desc in traj.ext_schema().iteritems()]
            dtype = np.dtype(fields)
            table = outfile.createTable('/', 'particles', dtype)

        arr = np.empty(len(traj) - trim_len, dtype=dtype)
        arr['trajid'] = traj.trajid()
        
        for k, v in traj.as_dict().iteritems():
            if trim is None:
                arr[k] = v
            else:
                arr[k] = v[trim:-trim]
        
        table.append(arr)
        bounds_tab.append([
            (traj.trajid(), traj.time()[trim_len], traj.time()[-trim_len])])
    
    table.cols.trajid.createIndex()
    table.cols.time.createIndex()
    bounds_tab.cols.trajid.create_index()
    
    outfile.flush()
    outfile.close()

def trajectories_table(fname, first=None, last=None):
    """
    Reads trajectories from a PyTables HDF5 file, as saved by
    save_particles_table().
    
    Arguments:
    fname - path to file to read.
    first, last - inclusive range of frames to read.
    
    Returns:
    trajects - a list of Trajectory objects, each trimmed to the frame range.
    """
    outfile = tables.openFile(fname, mode='r')
    table = outfile.getNode('/particles')
    
    query_string = ('(trajid == trid)')
    if first is not None:
        query_string += " & (time >= %d)" % first
    if last is not None:
        query_string += " & (time <= %d)" % last
                
    trajects = []
    for trid in np.unique(table.col('trajid')):
        arr = table.read_where(query_string)
        kwds = dict((field, arr[field]) for field in arr.dtype.fields \
            if field != 'trajid')
        kwds['trajid'] = trid
        trajects.append(Trajectory(**kwds))
    
    outfile.close()
    return trajects

def load_trajectories(res_dir, first=None, last=None):
    """
    Load a series of trajectories and associated data from a directory 
    containing npz trajectory files, as created by save_trajectories().
    
    Arguments:
    res_dir - path to the directory holding the trajectory files.
    
    Returns:
    trajects - a list of Trajectory objects created from the files is res_dir
    per_traject_adds - a dictionary of named added date. Each value is a 
        dictionary keyed by trajid.
    """
    res_dir = os.path.expanduser(res_dir)
    
    trajects = []
    per_traject_adds = {}
    
    for tr_file in os.listdir(res_dir):
        if not tr_file.endswith('.npz'): continue
        
        data = np.load(os.path.join(res_dir, tr_file))
        trajid = int(tr_file.split('.')[0][5:]) # traj_*.pyz
        
        kwds = {}
        for k in data.files:
            if k.startswith('traj:'):
                kwds[k[5:]] = data[k]
            else:
                per_traject_adds.setdefault(k, {})
                per_traject_adds[k][trajid] = data[k]
        
        if (first is not None) or (last is not None):
            in_range = np.ones(kwds['time'].shape, dtype=np.bool)
            if first is not None:
                in_range[kwds['time'] < first] = False
            if last is not None:
                in_range[kwds['time'] > last] = False
            # Note that we're reading one extra frame, otherwise the last frame
            # has 0 path segments.
            
            if in_range.sum() < 1:
                continue # Filter out empty trajectories (that are really empty
                         # or completely out of range)
            
            for k in kwds.keys():
                kwds[k] = kwds[k][in_range]
            # per_traject_adds do not get the same treatment as it is 
            # impossible to know their size. It is therefore up to the user to
            # create only per_traject_adds in the range matching the processed
            # range.
        
        kwds['trajid'] = trajid
        trajects.append(Trajectory(**kwds))
    
    return trajects, per_traject_adds
