"""A Zarr-backed counterpart to :class:`flowtracks.scene.Scene`.

``Scene`` reads a PyTables HDF5 ``/particles`` table via ``read_where()``
query strings. ``ZarrScene`` reads the same particle-observation table shape
from a Zarr store's ``trajectories/`` group instead -- either one written by
:func:`flowtracks.io.save_zarr_trajectories` (a standalone flowtracks Zarr
export) or one written by ``openptv2.storage.seal.seal()`` (the run store
that openptv2's tracking pipeline produces directly, see
openptv2's docs/plans/2026-08-14-storage-formats-as-built.md and its
follow-on Phase A-D plan). Both write the same five arrays
(``pos``, ``vel``, ``time``, ``trajid``, optionally ``accel``), so one reader
handles both without a conversion step.

Implements the same public surface as ``Scene`` (duck-typed, not a subclass,
since the two hold fundamentally different backing stores) so existing
callers -- ``eulerian.py``'s ``scene.collect(...)`` fast path chief among
them -- work unmodified against either.

``collect()``'s ``where`` filtering is reimplemented directly in numpy
instead of composing a PyTables ``read_where()`` query string (the
``gen_query_string`` mini-language in ``scene.py``); there is no query
engine to target, so this is strictly simpler, not a re-implementation of
the string DSL itself.
"""

from __future__ import annotations

import itertools as it
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np

from .trajectory import ParticleSnapshot, Trajectory


def _pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


class ZarrScene:
    """Programmer's interface to a Zarr-stored particle trajectory table,
    mirroring :class:`flowtracks.scene.Scene`'s public methods."""

    def __init__(self, store_path: Union[str, Path], frame_range=None):
        import zarr

        self.store_path = Path(store_path)
        root = zarr.open_group(str(self.store_path), mode="r")
        target_group = root["trajectories"] if "trajectories" in root else root
        if "trajid" not in target_group:
            raise ValueError(
                f"{self.store_path}: no trajectories/{{pos,vel,time,trajid}} "
                "arrays found -- not a flowtracks or openptv2 RunStore Zarr layout."
            )

        self._pos = np.asarray(target_group["pos"], dtype=np.float64)
        self._velocity = np.asarray(target_group["vel"], dtype=np.float64)
        self._time = np.asarray(target_group["time"])
        self._trajid = np.asarray(target_group["trajid"])
        self._accel = (
            np.asarray(target_group["accel"], dtype=np.float64)
            if "accel" in target_group
            else None
        )

        # openptv2's RunStore additionally writes a traj/ index (trajid,
        # first, last, length) built by seal() -- the same triple as
        # flowtracks' own /bounds table. Use it when present; otherwise
        # derive trajectory ids from the data itself, same as Scene's own
        # fallback when an HDF5 file has no /bounds node.
        if "traj" in root and "trajid" in root["traj"]:
            traj_grp = root["traj"]
            self._trids = np.asarray(traj_grp["trajid"])
            if "first" in traj_grp and "last" in traj_grp:
                self._tags = np.column_stack(
                    [
                        self._trids,
                        np.asarray(traj_grp["first"]),
                        np.asarray(traj_grp["last"]),
                    ]
                )
            else:
                self._tags = None
        else:
            self._trids = np.unique(self._trajid)
            self._tags = None

        self.set_frame_range(frame_range)

        self._keys = ["pos", "velocity"] + (["accel"] if self._accel is not None else [])
        self._shapes = [3, 3] + ([3] if self._accel is not None else [])

    # -- construction helpers --------------------------------------------

    def _key_array(self, key: str) -> np.ndarray:
        if key == "pos":
            return self._pos
        if key == "velocity":
            return self._velocity
        if key == "accel":
            if self._accel is None:
                raise KeyError("accel")
            return self._accel
        if key == "time":
            return self._time
        if key == "trajid":
            return self._trajid
        raise KeyError(key)

    def _range_mask(self) -> np.ndarray:
        return (self._time >= self._first) & (self._time < self._last)

    # -- Scene-compatible surface -----------------------------------------

    def trajectory_tags(self) -> np.ndarray:
        if self._tags is not None:
            return self._tags
        if len(self._trajid) == 0:
            return np.empty((0, 3), dtype=int)
        # No traj/ index: derive (trajid, first, last) from the data.
        order = np.argsort(self._trajid, kind="stable")
        trajid_sorted = self._trajid[order]
        time_sorted = self._time[order]
        bounds = np.flatnonzero(np.diff(trajid_sorted)) + 1
        starts = np.r_[0, bounds]
        unique_ids = trajid_sorted[starts]
        min_times = np.minimum.reduceat(time_sorted, starts)
        max_times = np.maximum.reduceat(time_sorted, starts)
        return np.column_stack([unique_ids, min_times, max_times]).astype(int)

    def set_frame_range(self, frame_range) -> None:
        if frame_range is None:
            self._first = int(self._time.min()) if len(self._time) else 0
            self._last = (int(self._time.max()) + 1) if len(self._time) else 0
            return
        first, last = frame_range
        self._first = int(first) if first is not None else (
            int(self._time.min()) if len(self._time) else 0
        )
        self._last = (
            int(last) if last is not None else
            ((int(self._time.max()) + 1) if len(self._time) else 0)
        )

    def frame_range(self):
        return self._first, self._last

    def keys(self) -> list:
        return self._keys

    def shapes(self) -> list:
        return self._shapes

    def trajectory_ids(self) -> np.ndarray:
        return self._trids

    def trajectory_by_id(self, trid) -> Trajectory:
        mask = (self._trajid == trid) & self._range_mask()
        idx = np.flatnonzero(mask)
        order = np.argsort(self._time[idx], kind="stable")
        idx = idx[order]
        kwds: dict[str, Any] = {}
        if self._accel is not None:
            kwds["accel"] = self._accel[idx]
        return Trajectory(self._pos[idx], self._velocity[idx], self._time[idx], trid, **kwds)

    def iter_trajectories(self) -> Iterator[Trajectory]:
        mask = self._range_mask()
        idx_all = np.flatnonzero(mask)
        order = np.lexsort((self._time[idx_all], self._trajid[idx_all]))
        idx_all = idx_all[order]
        trajid_sorted = self._trajid[idx_all]
        bounds = np.flatnonzero(np.diff(trajid_sorted)) + 1
        groups = np.split(idx_all, bounds)
        by_trid = {int(self._trajid[g[0]]): g for g in groups if len(g)}
        empty = np.array([], dtype=np.int64)

        for trid in self._trids:
            idx = by_trid.get(int(trid), empty)
            kwds: dict[str, Any] = {}
            if self._accel is not None:
                kwds["accel"] = self._accel[idx]
            yield Trajectory(self._pos[idx], self._velocity[idx], self._time[idx], trid, **kwds)

    def _iter_frame_arrays(self, cond=None):
        """Like ``iter_frames`` but yields ``(t, index_array)`` pairs instead
        of building ``ParticleSnapshot`` objects. Unlike ``Scene``'s version,
        ``cond`` is not a PyTables query string (there is no query engine
        here); pass a boolean mask over the full table instead, or leave it
        None. No current caller needs the string form."""
        mask = self._range_mask()
        if cond is not None:
            mask = mask & cond
        idx_all = np.flatnonzero(mask)
        order = np.argsort(self._time[idx_all], kind="stable")
        idx_all = idx_all[order]
        time_sorted = self._time[idx_all]
        bounds = np.flatnonzero(np.diff(time_sorted)) + 1
        groups = np.split(idx_all, bounds)
        by_time = {int(self._time[g[0]]): g for g in groups if len(g)}
        empty = np.array([], dtype=np.int64)

        for t in range(self._first, self._last):
            yield t, by_time.get(t, empty)

    def _snapshot_from_idx(self, t, idx) -> ParticleSnapshot:
        kwds: dict[str, Any] = {}
        if self._accel is not None:
            kwds["accel"] = self._accel[idx]
        return ParticleSnapshot(
            self._pos[idx], self._velocity[idx], t, self._trajid[idx], **kwds
        )

    def iter_frames(self) -> Iterator[ParticleSnapshot]:
        for t, idx in self._iter_frame_arrays():
            yield self._snapshot_from_idx(t, idx)

    def frame_by_time(self, t) -> ParticleSnapshot:
        idx = np.flatnonzero(self._time == t)
        return self._snapshot_from_idx(t, idx)

    def iter_segments(self):
        for (t, idx), (tn, next_idx) in _pairwise(self._iter_frame_arrays()):
            trids = self._trajid[idx]
            next_trids = self._trajid[next_idx]
            common = np.intersect1d(trids, next_trids, assume_unique=True)

            in_idx = idx[np.isin(trids, common, assume_unique=True)]
            in_next = next_idx[np.isin(next_trids, common, assume_unique=True)]

            # Match ordering so corresponding rows describe the same particle.
            order = np.argsort(self._trajid[in_idx])
            in_idx = in_idx[order]
            order_n = np.argsort(self._trajid[in_next])
            in_next = in_next[order_n]

            yield self._snapshot_from_idx(t, in_idx), self._snapshot_from_idx(tn, in_next)

    def collect(
        self, keys: Sequence[str], where: Optional[dict] = None
    ) -> list:
        mask = self._range_mask()
        if where is not None:
            for key, (smin, smax, invert) in where.items():
                col = self._key_array(key)
                in_range = (col >= smin) & (col < smax)
                mask = mask & (~in_range if invert else in_range)

        idx = np.flatnonzero(mask)
        return [self._key_array(k)[idx] for k in keys]

    def bounding_box(self):
        pos = self.collect(["pos"])[0]
        return pos.min(axis=0), pos.max(axis=0)
