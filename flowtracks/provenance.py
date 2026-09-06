"""Run provenance: what was run, with which parameters, and what it produced.

Appends a small JSON-serializable record to run.zarr/meta.attrs["runs"] --
the same `meta` group openptv2's RunStore already writes (schema_version,
sealed, source_hash). One place per store, no separate file to lose track of.

This exists because a run.zarr can otherwise go stale silently: a 5-frame
smoke-test run and a full run look identical from the array shapes alone,
and nothing on disk says which parameters, frame range, or pipeline stage
produced what's there. Seen in practice: one run's run.zarr held a 5-frame
test-tracking run with no record of that fact anywhere, and it looked
identical to a real run until someone opened it.
"""

from datetime import datetime, timezone
from pathlib import Path


def record_run(zarr_path, stage: str, **info) -> dict:
    """Append one provenance record for `stage` (e.g. "tracking", "lagrangian",
    "eulerian") to run.zarr/meta.attrs["runs"]. `info` is any JSON-serializable
    metadata worth keeping -- params_file, first/last frame, frame_rate,
    min_length, stitching config, grid config, counts produced, etc.

    Returns the record that was written.
    """
    import zarr

    root = zarr.open_group(str(zarr_path), mode="a")
    meta = root.require_group("meta")
    runs = list(meta.attrs.get("runs", []))
    record = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "flowtracks_version": _flowtracks_version(),
        **info,
    }
    runs.append(record)
    meta.attrs["runs"] = runs
    return record


def read_runs(zarr_path) -> list[dict]:
    """Return every provenance record stored at run.zarr/meta.attrs["runs"],
    oldest first. Empty list if the store has none (or predates this)."""
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r")
    if "meta" not in root:
        return []
    return list(root["meta"].attrs.get("runs", []))


def _flowtracks_version() -> str:
    from flowtracks import __version__

    return __version__
