
# -*- coding: utf-8 -*-

"""
The main entry points for using the module are the :func:`trajectories`
function (or its counterpart :func:`iter_trajectories`) for reading the data
for a scene; and either :func:`save_trajectories` or
:func:`save_particles_table` for saving scene data in, respectively, an obsolete
format based on a directory of NPZ files, or in the newer, recommended, HDF5
format.

The trajectory reader, unless otherwise noted, will try to infer the format
from the file name (see :func:`infer_format`).

The rest of the content of this module is composed of readers and writers for
the various formats. They are documented here alongside the main entry points,
so that users may access them directly if needed.
"""
import os, os.path, re, itertools as itr
from configparser import ConfigParser
from io import StringIO

import numpy as np
from scipy import io
import tables

from .scene import Scene
from .particle import Particle
from .trajectory import Trajectory, mark_unique_rows, \
    Frame, take_snapshot, trajectories_in_frame


class FramesIterator(object):
    def __init__(self, fname_tmpl, fmt, skip, first=None, last=None, usecols=None):
        """
        Arguments:
        fname_tmpl - a template file name representing all ptv_is/xuap files in
            the directory, with exactly one '%d' representing the frame number.
        fmt - a dtype object describing the table structure to be read.
        skip - number of header lines to skip in each file.
        first, last - inclusive range of frames to read, rel. filename
            numbering.
        usecols - columns to use from the file (for xuap format with extra columns).
        """
        self._frmix = 0
        # Use converters to handle integer parsing properly
        converters = {0: float, 1: float}  # Convert columns 0 and 1 (prev and next) to float first
        self._read_frame = lambda fix: np.atleast_1d(
            np.loadtxt(fname_tmpl % fix, dtype=fmt, skiprows=skip, usecols=usecols, converters=converters))

        dirname, basename = os.path.split(fname_tmpl)
        is_data_file = re.compile(basename.replace('%d', r'(\d+)', 1))

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

    def __next__(self):
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
        """
        self._frmix = 0
        self._f = open(fname, 'r')
        self._read_frame = lambda str_tbl: np.loadtxt(str_tbl, dtype=fmt)

    def __iter__(self):
        return self

    def __next__(self):
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
            if re.match(r'^\s*$', line):
                break
            lines.append(line)

        if len(lines) == 0:
            raise StopIteration # EOF

        str_tbl = StringIO("".join(lines))
        frame = self._read_frame(str_tbl)
        self._frmix += 1
        return curframenum, frame

    def __del__(self):
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
    """
    Extracts all trajectories from a Matlab file. the file is formated as a
    list of trajectory record arrays, containing attributes 'xf', 'yf', 'zf'
    for position, 'uf', 'vf', 'wf' for velocity, and 'axf', 'ayf', 'azf' for
    acceleration.

    Arguments:
    fname - path to the Matlab file.

    Returns:
    trajects - a list of :class:`~flowtracks.trajectory.Trajectory` objects,
        one for each trajectory contained in the mat file.
    """
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
    trajects - a list of :class:`~flowtracks.trajectory.Trajectory` objects,
        one for each trajectory contained in the mat file.
    """
    trajects = []
    dirname, basename = os.path.split(os.path.expanduser(fname))
    is_data_file = re.compile(basename.replace('%d', r'(\d+)', 1))

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
    Extract all trajectories in a directory of ptv_is/xuap files, as
    generated by programs in the 3d-ptv/pyptv family.

    Arguments:
    fname - a template file name representing all ptv_is/xuap files in the
        directory, with exactly one '%d' representing the frame number. If
        no '%d' is found, the input is assumed to be in the Ron Shnapp
        format- single file of concatenated ptv_is files, each stripped of
        the particle count line (first line) and separated from the next by
        an empty line.
    first, last - inclusive range of frames to read, rel. filename numbering.
    frate - frame rate, used for calculating velocities by backward
        derivative.
    xuap - The format is extended with colums for velocity and acceleration.
    traj_min_len - do not include trajectories shorter than this many frames.

    Yields:
    each of the trajectories in the ptv_is data in order, as a
    :class:`~flowtracks.trajectory.Trajectory` instance with velocity and
    acceleration.
    """
    fname = os.path.expanduser(fname)

    if xuap:
        # The xuap format has 15 columns: 2 integers + 3 arrays of 3 floats + 1 array of 3 floats + 1 extra float
        # We'll use usecols to select only the columns we need
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
    if re.search(r'%\d*?d', fname) is not None:
        # For xuap format with 15 columns, use only the first 14 columns
        usecols = range(14) if xuap else None
        frm_iter = FramesIterator(fname, fmt, skip, first, last, usecols=usecols)
    else:
        frm_iter = SingleFileIterator(fname, fmt)

    # In the first frame, every particle starts a trajectory.
    frame_num, table = next(frm_iter)

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
    frames_sort_cache = [(trids, trids)]

    traj_starts = dict.fromkeys(trids, 0)

    trajects = {}

    # Single-frame trajectories in first frame:
    ending = table['next'] - count_base == -2
    if ending.any():
        ending_trids = np.atleast_1d(frame[ending,-1].astype(np.int_))
        for trid in ending_trids:
            trajects[trid] = frame[trid:trid+1]

    frame_buffer_start = 0

    # Main loop, sequentially read each frame and process:
    for fix, frm in enumerate(frm_iter):
        frame_num, table = frm

        if table.ndim == 0:
            frames.append(None)
            frames_sort_cache.append(None)
            continue
            # We assume that the next frame will have no continuing particles,
            # and this case is caused by detection failure. Otherwise the code
            # that generated the data has a bug that can't be dealt with here.

        # Continue existing trajectories into this frame:
        cont = table['prev'] - count_base > -1
        traj = np.empty(table['prev'].shape)

        has_history = (len(frames) > 0) and (frames[-1] is not None)
        if has_history:
            prev_ix = table['prev'][cont] - count_base
            traj[cont] = frames[-1][:,-1][prev_ix]

        # Start new trajectories:
        num_new_traj = np.sum(~cont)
        new_trids = np.arange(max_traj, max_traj + num_new_traj)
        traj[~cont] = new_trids
        traj_starts.update(dict.fromkeys(new_trids, fix + 1))
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
        if 'vel' not in fmt.fields and has_history:
            # Update velocity of previous frame's continuing particles
            frames[-1][prev_ix,3:6] = \
                (pos[cont] - frames[-1][prev_ix,:3]) * frate
        frames.append(frame)
        trids_in_frame = frame[:, -1].astype(np.int64)
        sort_ix = np.argsort(trids_in_frame)
        frames_sort_cache.append((trids_in_frame[sort_ix], sort_ix))

        # Make Trajectory objects from fully-read trajectories, so we can
        # discard early frames they're in.
        ending = table['next'] - count_base == -2
        if not ending.any():
            continue

        ending_trids = np.atleast_1d(frame[ending,-1].astype(np.int_))
        ending_starts = np.fromiter((traj_starts[trid] for trid in ending_trids), dtype=np.int64, count=len(ending_trids))

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
            active_ending_trids = ending_trids[in_frame]
            if len(active_ending_trids) == 0: continue

            sorted_trids, sort_ix = frames_sort_cache[scanix]

            locs = np.searchsorted(sorted_trids, active_ending_trids)
            valid = locs < len(sorted_trids)
            matched = valid.copy()
            if matched.any():
                matched[valid] = sorted_trids[locs[valid]] == active_ending_trids[valid]

            valid_active_trids = active_ending_trids[matched]
            valid_row_ixs = sort_ix[locs[matched]]

            for trid, row_ix in zip(valid_active_trids, valid_row_ixs):
                traj_rel_ix = scanix + frame_buffer_start - traj_starts[trid]
                trajects[trid][traj_rel_ix] = past_frame[row_ix]

        # Discard frames that only have trajectories that ended.
        cont_trids = frame[~ending,-1]
        if len(cont_trids) == 0:
            new_start = fix
        else:
            new_start = min(traj_starts[int(trid)] for trid in cont_trids)

        frames = frames[new_start - frame_buffer_start : ]
        frames_sort_cache = frames_sort_cache[new_start - frame_buffer_start : ]
        frame_buffer_start = new_start

        # Convert the dictionary of trajectory arrays to list of Trajectory
        # objects and give them back.
        for trid in ending_trids:
            traj = trajects[trid]
            traj = Trajectory(traj[:,:3], traj[:,3:6], traj[:,6],
                np.int_(traj[0,7]))

            # Add forward-difference acceleration:
            accel = np.empty_like(traj.velocity())
            accel[:-3:-1,:] = 0.
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
    programs in the 3d-ptv/pyptv family. supports xuap files as well.

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
    each of the trajectories in the ptv_is/xuap data in order, as a
    :class:`~flowtracks.trajectory.Trajectory` instance with velocity and
    acceleration.
    """
    return [t for t in iter_trajectories_ptvis(fname, first, last, frate,
        xuap, traj_min_len)]

def trajectories(fname, first=None, last=None, frate=1.0, fmt=None, traj_min_len=None,
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
            traj = iter_trajectories_ptvis(fname, first, last, frate, xuap=False,
                traj_min_len=traj_min_len)
        else:
            traj = trajectories_ptvis(fname, first, last, frate, xuap=False,
                traj_min_len=traj_min_len)

    elif fmt == 'xuap':
        traj = trajectories_ptvis(fname, first, last, frate, xuap=True,
            traj_min_len=traj_min_len)

    elif fmt == 'hdf':
        scene = Scene(fname, (first, last))
        it = scene.iter_trajectories()
        if iter_allowed:
            traj = it
        else:
            traj = [t for t in it]

    elif fmt == 'zarr':
        traj = read_zarr_trajectories(fname, first, last)

    if filter_needed:
        if traj_min_len is None:
            traj_min_len = 2

        if iter_allowed:
            traj = itr.ifilter(lambda tr: len(tr) >= traj_min_len, traj)
        else:
            traj = [tr for tr in traj if len(tr) >= traj_min_len]

    return traj

def infer_format(fname):
    """
    Try to guess the format of a particles data file by its name.

    Arguments:
    fname - the file name from which to guess the format.

    Returns:
    A string marking the format. Currently one of 'acc', 'mat', 'xuap',
    'npz', 'hdf', 'zarr', or 'ptvis'.
    """
    if fname.endswith('zarr') or fname.endswith('zarr/') or '.zarr' in fname:
        return 'zarr'
    elif fname.endswith('mat'):
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
    parser = ConfigParser()
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
            for k, v in traj.as_dict().items())
        for k, v in per_traject_adds.items():
            save_data[k] = v[traj.trajid()]

        np.savez(os.path.join(output_dir, 'traj_%d' % traj.trajid()),
            **save_data)

    # Save non-trajectory arrays:
    for k, v in kwds.items():
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

    outfile = tables.open_file(filename, mode='w')
    # Use scalar shape for each field
    bounds_tab = outfile.create_table('/', 'bounds',
        np.dtype([('trajid', int), ('first', int), ('last', int)]))

    for traj in trajects:
        if len(traj) - trim_len <= 0:
            continue

        # First trajectory creates the table:
        if table is None:
            # Format of records in a trajectory array :
            # Use np.int_ instead of int to avoid NumPy deprecation warning
            fields = [('trajid', np.int_)]

            # Handle the shape properly to avoid NumPy deprecation warning
            for field, (dtype_type, shape) in traj.ext_schema().items():
                if shape == 1:
                    fields.append((field, dtype_type))
                else:
                    fields.append((field, dtype_type, (shape,)))

            dtype = np.dtype(fields)
            table = outfile.create_table('/', 'particles', dtype)

        arr = np.empty(len(traj) - trim_len, dtype=dtype)
        arr['trajid'] = traj.trajid()

        for k, v in traj.as_dict().items():
            if trim is None:
                arr[k] = v
            else:
                arr[k] = v[trim:-trim]

        table.append(arr)
        bounds_tab.append([
            (traj.trajid(), arr['time'][0], arr['time'][-1])])

    table.cols.trajid.create_index()
    table.cols.time.create_index()
    bounds_tab.cols.trajid.create_index()

    outfile.flush()
    outfile.close()

def save_frames_hdf(filename, frames):
    """
    Save scene data as a table of particle properties, with added columns for
    time (frame number) and trajid - the last one may be indexed. Any extra
    per-frame data should be added to the frames beforehand.

    Arguments:
    filename - name of output PyTables HDF5 file to create. The 'h5' extension
        is recommended so that ``infer_format()`` knows what to do with it.
    frames - an iterable sequence of ParticleSnapshot objects to save.
    """
    table = None
    outfile = tables.open_file(filename, mode='w')
    min_pos = np.full(3, np.inf)
    max_pos = np.full(3, -np.inf)

    ongoing_trajects = {}
    for frame in frames:
        if len(frame) <= 0:
            continue

        # First frame creates the table:
        if table is None:
            # Format of records in a frame array:
            # Use np.int_ instead of int to avoid NumPy deprecation warning
            fields = [('time', np.int_)]

            # Handle the shape properly to avoid NumPy deprecation warning
            for field, (dtype_type, shape) in frame.ext_schema().items():
                if shape == 1:
                    fields.append((field, dtype_type))
                else:
                    fields.append((field, dtype_type, (shape,)))

            dtype = np.dtype(fields)
            table = outfile.create_table('/', 'particles', dtype)

        min_pos = np.min(np.vstack((min_pos, frame.pos())), axis=0)
        max_pos = np.max(np.vstack((max_pos, frame.pos())), axis=0)

        arr = np.empty(len(frame), dtype=dtype)
        arr['time'] = frame.time()

        for k, v in frame.as_dict().items():
            arr[k] = v
        table.append(arr)

        # Keep track of trajectory starts/ends for the bounds table.
        # Allowing for holes forces us to keep the entire bounds table in
        # memory, but it's not a such big chunk, so I don't worry too much.
        for trid in frame.trajid():
            if trid in ongoing_trajects:
                # This allows for "hole frames". Otherwise we'd just wait for
                # the trajectory to disappear then record its end in time - 1.
                ongoing_trajects[trid][1] = frame.time()
            else:
                ongoing_trajects[trid] = np.r_[frame.time(), frame.time()]

    table._v_attrs.min_pos = min_pos
    table._v_attrs.max_pos = max_pos

    table.cols.trajid.create_index()
    table.cols.time.create_index()

    # Use scalar shape for each field
    bounds_tab = outfile.create_table('/', 'bounds',
        np.dtype([('trajid', int), ('first', int), ('last', int)]))
    for trid, bounds in ongoing_trajects.items():
        bounds_tab.append([(trid, bounds[0], bounds[1])])
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
    outfile = tables.open_file(fname, mode='r')
    table = outfile.get_node('/particles')

    conds = []
    if first is not None:
        conds.append("(time >= %d)" % first)
    if last is not None:
        conds.append("(time <= %d)" % last)
    read_cond = ' & '.join(conds)

    # Read the whole (possibly time-filtered) table once, then group by trajid
    # in memory, instead of one ``read_where`` query per trajectory id.
    arr = table.read_where(read_cond) if read_cond else table.read()

    all_trids = np.unique(table.col('trajid'))
    order = np.argsort(arr['trajid'], kind='stable')
    arr = arr[order]
    bounds = np.flatnonzero(np.diff(arr['trajid'])) + 1
    groups = np.split(arr, bounds)
    by_trid = {int(g['trajid'][0]): g for g in groups}

    trajects = []
    # Iterate over every trajid present in the file (matching the old per-id
    # query loop, which also appended an empty Trajectory for ids that fall
    # outside the time range).
    for trid in all_trids:
        chunk = by_trid.get(int(trid), arr[0:0])
        kwds = dict((field, chunk[field]) for field in chunk.dtype.fields \
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


def read_zarr_trajectories(zarr_path, first=None, last=None, group="trajectories"):
    """
    Extract all trajectories from a Zarr store directory.

    Arguments:
    zarr_path - path to the .zarr directory or Zarr group.
    first, last - inclusive range of frame numbers to read.
    group - sub-group name inside the Zarr store ('trajectories' or
            'trajectories/smoothed' or 'correspondences').

    Returns:
    trajects - a list of Trajectory objects.
    """
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    target_group = root[group] if group in root else root

    # The tracker's linkage is the source of truth; /trajectories is a derived
    # cache written by post-processing (openptv_cloud.post.convert). Re-tracking
    # does not refresh it, so preferring it silently replays a stale result --
    # observed as trajectories jumping tens of mm/frame, far past the tracker's
    # own velocity bound. Read it only when there is no linkage to walk.
    has_linkage = "linkage" in root and len(list(root["linkage"].keys())) > 0

    # Case 1: Structured dataset in /trajectories (arrays: pos, vel, accel, time, trajid)
    if "trajid" in target_group and not has_linkage:
        trajid_arr = np.asarray(target_group["trajid"])
        time_arr = np.asarray(target_group["time"])
        pos_arr = np.asarray(target_group["pos"])

        vel_arr = np.asarray(target_group["vel"]) if "vel" in target_group else None
        accel_arr = np.asarray(target_group["accel"]) if "accel" in target_group else None

        # Filter frame range
        mask = np.ones(len(time_arr), dtype=bool)
        if first is not None:
            mask &= time_arr >= first
        if last is not None:
            mask &= time_arr <= last

        trajid_arr = trajid_arr[mask]
        time_arr = time_arr[mask]
        pos_arr = pos_arr[mask]
        if vel_arr is not None:
            vel_arr = vel_arr[mask]
        if accel_arr is not None:
            accel_arr = accel_arr[mask]

        # Group by trajid
        if len(trajid_arr) == 0:
            return []

        order = np.lexsort((time_arr, trajid_arr))
        trajid_sorted = trajid_arr[order]
        time_sorted = time_arr[order]
        pos_sorted = pos_arr[order]
        vel_sorted = vel_arr[order] if vel_arr is not None else None
        accel_sorted = accel_arr[order] if accel_arr is not None else None

        bounds = np.flatnonzero(np.diff(trajid_sorted)) + 1
        id_groups = np.split(trajid_sorted, bounds)
        pos_groups = np.split(pos_sorted, bounds)
        time_groups = np.split(time_sorted, bounds)
        vel_groups = np.split(vel_sorted, bounds) if vel_sorted is not None else None
        accel_groups = np.split(accel_sorted, bounds) if accel_sorted is not None else None

        trajects = []
        for i, (g_id, g_pos, g_time) in enumerate(zip(id_groups, pos_groups, time_groups)):
            if len(g_id) == 0:
                continue
            trid = int(g_id[0])
            vel_val = vel_groups[i] if vel_groups is not None else np.zeros_like(g_pos)
            kws = {}
            if accel_groups is not None:
                kws["accel"] = accel_groups[i]
            trajects.append(Trajectory(g_pos, vel_val, g_time, trid, **kws))

        return trajects

    # Case 2: Walk the tracker's own prev/next linkage (openptv2's
    # linkage/<name>/frame_NNNNN groups, written by ZarrStore.write_linkage
    # in the same prev/next/pos shape as a classic ptv_is.# file). This is
    # the real particle identity across frames. The correspondences group
    # (case 3 below) has no such column: its 4th+ fields are per-camera 2D
    # target array indices, which reset every frame and are NOT a trajectory
    # id, so grouping by them stitches together unrelated particles.
    elif "linkage" in root:
        link_root = root["linkage"]
        linkage_name = "ptv_is" if "ptv_is" in link_root else next(iter(link_root.keys()), None)
        link_group = link_root[linkage_name] if linkage_name else None
        frame_keys = sorted(
            [k for k in link_group.keys() if k.startswith("frame_")],
            key=lambda k: int(k.split("_")[1]),
        ) if link_group is not None else []
        if first is not None:
            frame_keys = [k for k in frame_keys if int(k.split("_")[1]) >= first]
        if last is not None:
            frame_keys = [k for k in frame_keys if int(k.split("_")[1]) <= last]

        pos_l, time_l, trajid_l = [], [], []
        prev_trajids = None
        prev_frame_num = None
        next_trajid = 0
        for fkey in frame_keys:
            frame_num = int(fkey.split("_")[1])
            fg = link_group[fkey]
            prev_ids = np.asarray(fg["prev"])
            pos = np.asarray(fg["pos"])
            n = len(pos)
            trajids = np.empty(n, dtype=np.int64)
            if prev_trajids is not None and (
                prev_frame_num != frame_num - 1
                or (prev_ids.size and prev_ids.max() >= len(prev_trajids))
            ):
                # `prev` indexes the frame immediately before this one. A gap
                # in the stored frames -- or a row count that disagrees with
                # the linkage (frames left over from an earlier run in an
                # appended store) -- means those indices address unrelated
                # particles. Break every chain rather than invent a link.
                prev_trajids = None
            if prev_trajids is None:
                # First frame in range: nothing to inherit from, every
                # particle starts a new trajectory (mirrors
                # iter_trajectories_ptvis' treatment of its first frame).
                trajids[:] = np.arange(next_trajid, next_trajid + n)
                next_trajid += n
            else:
                # A `prev` claim must be unique to count as a real link: if
                # two particles in this frame both claim the same
                # predecessor, the tracker's linkage is ambiguous for that
                # predecessor and neither claim can be trusted. Merging them
                # under one trajid was observed to snowball into a single
                # "trajectory" absorbing hundreds of unrelated particles
                # over a 100-frame run - worse than the bug this replaced.
                claimed, claim_counts = np.unique(
                    prev_ids[prev_ids >= 0], return_counts=True
                )
                ambiguous = set(claimed[claim_counts > 1].tolist())
                linked = np.array(
                    [p >= 0 and p not in ambiguous for p in prev_ids]
                )
                trajids[linked] = prev_trajids[prev_ids[linked]]
                n_new = int((~linked).sum())
                trajids[~linked] = np.arange(next_trajid, next_trajid + n_new)
                next_trajid += n_new
            pos_l.append(pos)
            time_l.append(np.full(n, frame_num, dtype=np.int64))
            trajid_l.append(trajids)
            prev_trajids = trajids
            prev_frame_num = frame_num

        if not pos_l:
            return []

        pos_all = np.concatenate(pos_l)
        time_all = np.concatenate(time_l)
        trajid_all = np.concatenate(trajid_l)

        order = np.lexsort((time_all, trajid_all))
        trajid_sorted = trajid_all[order]
        pos_sorted = pos_all[order]
        time_sorted = time_all[order]

        bounds = np.flatnonzero(np.diff(trajid_sorted)) + 1
        id_groups = np.split(trajid_sorted, bounds)
        pos_groups = np.split(pos_sorted, bounds)
        time_groups = np.split(time_sorted, bounds)

        trajects = []
        for g_id, g_pos, g_time in zip(id_groups, pos_groups, time_groups):
            if len(g_id) < 2:
                continue
            trid = int(g_id[0])
            p_vel = np.zeros_like(g_pos)
            trajects.append(Trajectory(g_pos, p_vel, g_time, trid))

        return trajects

    # Case 3: Reading frame-by-frame 3D correspondences from openptv2 (e.g., correspondences/frame_10000)
    elif "correspondences" in root or group == "correspondences":
        corr_group = root["correspondences"] if "correspondences" in root else root
        frame_keys = sorted(
            [k for k in corr_group.keys() if k.startswith("frame_")],
            key=lambda k: int(k.split("_")[1])
        )

        all_points = []
        for fkey in frame_keys:
            frame_num = int(fkey.split("_")[1])
            if first is not None and frame_num < first:
                continue
            if last is not None and frame_num > last:
                continue

            arr = np.asarray(corr_group[fkey])
            if len(arr) == 0:
                continue
            for row in arr:
                pt_x, pt_y, pt_z = row[0], row[1], row[2]
                pt_id = int(row[3]) if len(row) > 3 else 0
                all_points.append((pt_x, pt_y, pt_z, frame_num, pt_id if pt_id != -1 else 0))

        if len(all_points) == 0:
            return []

        pts = np.array(all_points)
        time_pts = pts[:, 3].astype(int)
        trid_pts = pts[:, 4].astype(int)
        pos_pts = pts[:, :3]

        order = np.lexsort((time_pts, trid_pts))
        trid_sorted = trid_pts[order]
        pos_sorted = pos_pts[order]
        time_sorted = time_pts[order]

        bounds = np.flatnonzero(np.diff(trid_sorted)) + 1
        id_groups = np.split(trid_sorted, bounds)
        pos_groups = np.split(pos_sorted, bounds)
        time_groups = np.split(time_sorted, bounds)

        trajects = []
        for g_id, g_pos, g_time in zip(id_groups, pos_groups, time_groups):
            if len(g_id) < 2:
                continue
            trid = int(g_id[0])
            p_vel = np.zeros_like(g_pos)
            trajects.append(Trajectory(g_pos, p_vel, g_time, trid))

        return trajects

    return []


def save_zarr_trajectories(trajects, zarr_path, group="trajectories", overwrite=True):
    """
    Save a list of Trajectory objects into a Zarr directory store.

    Arguments:
    trajects - list of Trajectory objects.
    zarr_path - path to the target .zarr directory.
    group - sub-group name inside the Zarr store.
    overwrite - if True, overwrite existing arrays in the group.
    """
    import zarr

    root = zarr.open_group(str(zarr_path), mode="a")
    target_group = root.require_group(group)

    if len(trajects) == 0:
        return

    lens = np.fromiter((len(tr) for tr in trajects), dtype=np.int64, count=len(trajects))
    trids = np.fromiter((tr.trajid() for tr in trajects), dtype=np.int64, count=len(trajects))
    trajid_arr = np.repeat(trids, lens)

    all_pos = [tr.pos() for tr in trajects]
    all_time = [tr.time() for tr in trajects]
    pos_arr = np.concatenate(all_pos, axis=0)
    time_arr = np.concatenate(all_time, axis=0)

    target_group.create_array("pos", data=pos_arr, overwrite=overwrite)
    target_group.create_array("time", data=time_arr, overwrite=overwrite)
    target_group.create_array("trajid", data=trajid_arr, overwrite=overwrite)

    first_tr = trajects[0]
    has_vel = (hasattr(first_tr, "velocity") and first_tr.velocity() is not None) or (
        hasattr(first_tr, "vel") and first_tr.vel() is not None
    )
    if has_vel:
        all_vel = [
            tr.velocity() if (hasattr(tr, "velocity") and tr.velocity() is not None) else tr.vel()
            for tr in trajects
        ]
        vel_arr = np.concatenate(all_vel, axis=0)
        target_group.create_array("vel", data=vel_arr, overwrite=overwrite)

    has_accel = hasattr(first_tr, "accel") and first_tr.accel() is not None
    if has_accel:
        all_accel = [tr.accel() for tr in trajects]
        accel_arr = np.concatenate(all_accel, axis=0)
        target_group.create_array("accel", data=accel_arr, overwrite=overwrite)


