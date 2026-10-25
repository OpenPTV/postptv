"""Synthetic comparison of Savitzky-Golay vs. scipy smoothing-spline smoothing.

Generates a known smooth 3D trajectory, adds position noise, introduces
frame gaps, and (optionally) splits/relinks the track. Runs both
``flowtracks.smoothing.savitzky_golay`` and ``smoothing_spline`` and reports
RMSE of position / velocity / acceleration against the noise-free ground
truth at the observed frames.

Run with:  uv run python examples/compare_smoothing.py
"""

import numpy as np

from flowtracks import smoothing
from flowtracks.trajectory import Trajectory


def ground_truth(t):
    pos = np.column_stack(
        [
            0.05 * t + 0.3 * np.sin(0.4 * t),
            0.2 * np.cos(0.3 * t),
            0.1 * t + 0.05 * np.sin(0.7 * t),
        ]
    )
    vel = np.column_stack(
        [
            0.05 + 0.3 * 0.4 * np.cos(0.4 * t),
            -0.2 * 0.3 * np.sin(0.3 * t),
            0.1 + 0.05 * 0.7 * np.cos(0.7 * t),
        ]
    )
    acc = np.column_stack(
        [
            -0.3 * 0.4 * 0.4 * np.sin(0.4 * t),
            -0.2 * 0.3 * 0.3 * np.cos(0.3 * t),
            -0.05 * 0.7 * 0.7 * np.sin(0.7 * t),
        ]
    )
    return pos, vel, acc


def make_noisy_track(n=120, fps=30.0, noise=0.02, gap_frames=(), seed=0):
    rng = np.random.default_rng(seed)
    t_full = np.arange(n, dtype=float)
    pos0, _, _ = ground_truth(t_full)
    keep = np.setdiff1d(t_full, np.asarray(gap_frames, dtype=float))
    t = keep.astype(float)
    pos = pos0[keep.astype(int)] + rng.normal(0, noise, (len(t), 3))
    return t, pos, t_full, pos0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_case(label, t, pos, t_full, pos0, fps):
    tr = Trajectory(pos, np.zeros_like(pos), t, 0)
    gt_pos = pos0[t.astype(int)]
    gt_vel = ground_truth(t)[1]
    gt_acc = ground_truth(t)[2]

    # Savitzky-Golay (needs uniform-ish; use window 11)
    try:
        sg = smoothing.savitzky_golay(
            [tr], fps=fps, window_size=11, order=3, min_window=7
        )
        sg = sg[0] if sg else None
    except Exception as e:
        sg = None
        print("  SG failed:", e)

    sp = smoothing.smoothing_spline([tr], fps=fps)
    sp = sp[0] if sp else None

    print(f"\n=== {label}  (n={len(t)}, fps={fps}) ===")
    print(f"{'method':<16}{'pos RMSE':>12}{'vel RMSE':>12}{'acc RMSE':>12}")
    for name, sm in (
        ("noisy input", None),
        ("Savitzky-Golay", sg),
        ("smoothing spline", sp),
    ):
        if sm is None and name != "noisy input":
            print(f"{name:<16}{'n/a':>12}{'n/a':>12}{'n/a':>12}")
            continue
        if name == "noisy input":
            p_rmse = rmse(pos, gt_pos)
            print(f"{name:<16}{p_rmse:>12.4e}{'-':>12}{'-':>12}")
            continue
        p = rmse(sm.pos(), gt_pos)
        v = rmse(sm.velocity(), gt_vel)
        a = rmse(sm.accel(), gt_acc)
        print(f"{name:<16}{p:>12.4e}{v:>12.4e}{a:>12.4e}")


def main():
    fps = 30.0

    # 1. Clean-ish, uniform, no gaps
    t, pos, tf, p0 = make_noisy_track(n=120, fps=fps, noise=0.02, seed=1)
    run_case("uniform, no gaps", t, pos, tf, p0, fps)

    # 2. With frame gaps
    t, pos, tf, p0 = make_noisy_track(
        n=120, fps=fps, noise=0.02, gap_frames=[20, 35, 36, 80], seed=2
    )
    run_case("with gaps", t, pos, tf, p0, fps)

    # 3. Relinked: split into two segments then concatenate time (gap in middle)
    t, pos, tf, p0 = make_noisy_track(n=120, fps=fps, noise=0.02, seed=3)
    # simulate a break: remove a chunk, then "relink" by just keeping rest
    mask = (t < 40) | (t > 55)
    t_r = t[mask]
    pos_r = pos[mask]
    run_case("relinked (missing middle)", t_r, pos_r, tf, p0, fps)

    # 4. High noise
    t, pos, tf, p0 = make_noisy_track(n=200, fps=fps, noise=0.1, seed=4)
    run_case("high noise, n=200", t, pos, tf, p0, fps)


if __name__ == "__main__":
    main()
