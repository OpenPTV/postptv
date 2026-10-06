# SPDX-License-Identifier: GPL-3.0-only
"""One shared phase-bin definition for periodic PTV post-processing.

Every package in the OpenPTV family (flowtracks itself, openptv-cloud,
openptv-analysis — same users) uses these names and these rules, so a
``post_analysis.binning`` block means the same thing everywhere:

Canonical names (all self-explainable):

  n_frames_in_period — period length, in frames
  n_phases           — number of phase bins per cycle (the output count)
  phase_width        — bin width in cycle units [0, 1); default 1/n_phases
  phase_zero_frame   — frame anchoring phase 0

Rules enforced by :func:`validate_bins` (fail loudly, never ship NaNs):

  1 <= n_phases <= n_frames_in_period (more bins than frames would stay
  empty by construction — with whole-frame counting there are only
  n_frames_in_period distinct phases per cycle);
  0 < phase_width <= 1/n_phases (bins never overlap in time; narrower bins
  leave gaps — some frames go unused, which is intentional).

Bins are uniform and start-anchored: bin ``i`` covers
``[i/n_phases, i/n_phases + phase_width)``. :func:`assign_bins` returns -1
for samples falling in gaps.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "validate_bins",
    "assign_bins",
    "phase_of_frame",
    "bin_of_frame",
]


def validate_bins(n_frames_in_period, n_phases, phase_width=None) -> float:
    """Check one bin definition; return the effective width (cycle units).

    Raises SystemExit on violation — a bad binning must fail loudly before
    any averaging, never ship silent NaN phases.
    """
    try:
        period = int(n_frames_in_period)
    except (TypeError, ValueError):
        raise SystemExit(
            "[phasing] n_frames_in_period must be a positive integer "
            f"(got {n_frames_in_period!r})."
        )
    try:
        n = int(n_phases)
    except (TypeError, ValueError):
        raise SystemExit(
            f"[phasing] n_phases must be a positive integer (got {n_phases!r})."
        )
    if period < 1:
        raise SystemExit("[phasing] n_frames_in_period must be >= 1.")
    if n < 1:
        raise SystemExit("[phasing] n_phases must be >= 1.")
    if n > period:
        raise SystemExit(
            f"[phasing] n_phases ({n}) cannot exceed n_frames_in_period "
            f"({period}): with whole-frame counting there are only {period} "
            "distinct phases per cycle, so the extra bins would stay empty. "
            f"Use n_phases <= {period}."
        )
    spacing = 1.0 / n
    width = float(phase_width) if phase_width is not None else spacing
    if not width > 0:
        raise SystemExit("[phasing] phase_width must be positive.")
    if width > spacing + 1e-12:
        raise SystemExit(
            f"[phasing] phase_width ({width:.4g}) exceeds the phase spacing "
            f"(1/n_phases = {spacing:.4g}): bins would overlap in time. "
            "Maximum is 1/n_phases (contiguous bins, no gaps)."
        )
    return width


def phase_of_frame(frame, n_frames_in_period, phase_zero_frame=0) -> float:
    """Phase in [0, 1) of one frame number."""
    period = int(n_frames_in_period)
    return ((int(frame) - int(phase_zero_frame)) % period) / period


def bin_of_frame(frame, n_phases, n_frames_in_period,
                 phase_width=None, phase_zero_frame=0) -> int:
    """Bin index of one frame number, or -1 when it falls in a gap."""
    n = int(n_phases)
    width = float(phase_width) if phase_width is not None else 1.0 / n
    phase = phase_of_frame(frame, n_frames_in_period, phase_zero_frame)
    idx = min(int(phase * n), n - 1)
    return idx if (phase * n - idx) < width * n - 1e-12 else -1


def assign_bins(phase, n_phases, phase_width=None):
    """Bin index per sample phase, or -1 for samples in gaps.

    phase: array-like floats in [0, 1). Default width (None) = 1/n_phases,
    i.e. contiguous tiling with no gaps.
    """
    phase = np.asarray(phase, dtype=float)
    n = int(n_phases)
    width = float(phase_width) if phase_width is not None else 1.0 / n
    idx = np.minimum((phase * n).astype(int), n - 1)
    intra = phase * n - idx  # position inside the bin, 0..1
    keep_fraction = width * n  # how much of each bin is covered
    return np.where(intra < keep_fraction - 1e-12, idx, -1)
