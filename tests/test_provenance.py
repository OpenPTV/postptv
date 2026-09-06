import zarr

from flowtracks.provenance import read_runs, record_run


def test_record_and_read_runs(tmp_path):
    store = tmp_path / "run.zarr"
    zarr.open_group(str(store), mode="a")  # empty store, no meta yet

    record_run(store, "tracking", params_file="parameters_wp1.yaml", first=1, last=5)
    record_run(store, "lagrangian", frame_rate=5000.0, min_length=5, n_trajectories=1288)

    runs = read_runs(store)
    assert len(runs) == 2
    assert runs[0]["stage"] == "tracking"
    assert runs[0]["first"] == 1
    assert runs[1]["stage"] == "lagrangian"
    assert runs[1]["n_trajectories"] == 1288
    assert all("timestamp" in r and "flowtracks_version" in r for r in runs)


def test_read_runs_on_store_with_no_meta(tmp_path):
    store = tmp_path / "empty.zarr"
    zarr.open_group(str(store), mode="a")
    assert read_runs(store) == []
