"""Synthetic two-set periodic-flow scenario: does shift_phase + phase_average
generalize to any periodic flow (cardiac/LV, aorta, piston, ...) where each
acquisition set's cycle happens to start at a different frame?

Two 20-frame sets, uniform-direction flow whose magnitude rises and falls
once per cycle (a raised-cosine stand-in for a pulsatile velocity trace):
set "a" peaks at frame 3, set "b" peaks at frame 6. Naively averaging them
smears the peak; shifting each set to a common reference phase first (with
the existing shift_phase) recovers it exactly. Uses only existing
flowtracks code — no new pipeline logic.
"""
import numpy as np
import pytest
import xarray as xr

from flowtracks.eulerian import shift_phase
from flowtracks.phase_average import phase_average

N = 20  # frames per cycle


def _bump(peak, n=N, amplitude=1.0):
    """Raised-cosine pulse peaking at frame `peak`, period n."""
    phase = np.arange(n)
    return amplitude * 0.5 * (1 + np.cos(2 * np.pi * (phase - peak) / n))


def _set_dataset(peak, amplitude=1.0, n=N):
    u = _bump(peak, n, amplitude)
    zeros = np.zeros(n)
    return xr.Dataset(
        {"u": (("phase",), u), "v": (("phase",), zeros), "w": (("phase",), zeros)},
        coords={"phase": np.arange(n)},
    )


def _stack(sets_by_name):
    names = list(sets_by_name)
    return xr.concat([sets_by_name[n] for n in names],
                     dim=xr.DataArray(names, dims="set", name="set"), join="exact")


def test_naive_average_of_misaligned_sets_blurs_and_shrinks_peak():
    stacked = _stack({"a": _set_dataset(peak=3), "b": _set_dataset(peak=6)})

    naive_avg = phase_average(stacked)

    assert float(naive_avg["u"].max()) < 1.0  # below either set's own peak amplitude
    peak_idx = int(np.argmax(naive_avg["u"].values))
    assert peak_idx not in (3, 6)  # smeared to somewhere between the two


def test_shift_phase_then_average_recovers_the_true_peak():
    aligned_a = shift_phase(_set_dataset(peak=3), -3)  # peak frame 3 -> 0
    aligned_b = shift_phase(_set_dataset(peak=6), -6)  # peak frame 6 -> 0

    avg = phase_average(_stack({"a": aligned_a, "b": aligned_b}))

    assert int(np.argmax(avg["u"].values)) == 0
    assert float(avg["u"].max()) == pytest.approx(1.0)
    np.testing.assert_allclose(avg["u"].values, _bump(peak=0))
    np.testing.assert_allclose(avg["v"].values, 0.0)
    np.testing.assert_allclose(avg["w"].values, 0.0)
