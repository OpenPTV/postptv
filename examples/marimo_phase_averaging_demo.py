import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔄 Phase Averaging Across Misaligned Periodic-Flow Sets

    Two acquisition sets of a periodic flow (cardiac/LV, aorta, piston, ...)
    rarely start their cycle at the same frame. This notebook shows, step by
    step, the **existing** `flowtracks` machinery
    (`flowtracks.eulerian.shift_phase` + `flowtracks.phase_average.phase_average`)
    that handles this:

    1. **Peak selection** — find each set's own cycle peak (`argmax`).
    2. **Rolling the cycle around the peak** — `shift_phase` cyclically rolls
       each set so its peak lands on a common reference phase.
    3. **Phase averaging** — `phase_average` (the `set`-wise mean) combines the
       aligned sets; without step 2 the same averaging smears the peak.

    Same math as `tests/test_phase_align_synthetic_sets.py`, made interactive.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    from flowtracks.eulerian import shift_phase
    from flowtracks.phase_average import phase_average

    return mo, np, phase_average, plt, shift_phase, xr


@app.cell(hide_code=True)
def _(mo):
    n_frames_ui = mo.ui.slider(start=10, stop=40, step=1, value=20, label="Frames per cycle (N)")
    peak_a_ui = mo.ui.slider(start=0, stop=39, step=1, value=3, label="Set A peak frame")
    peak_b_ui = mo.ui.slider(start=0, stop=39, step=1, value=6, label="Set B peak frame")
    amp_a_ui = mo.ui.slider(start=0.2, stop=2.0, step=0.1, value=1.0, label="Set A amplitude")
    amp_b_ui = mo.ui.slider(start=0.2, stop=2.0, step=0.1, value=1.0, label="Set B amplitude")
    ref_phase_ui = mo.ui.slider(start=0, stop=39, step=1, value=0, label="Reference phase (align peaks here)")
    noise_ui = mo.ui.slider(start=0.0, stop=0.3, step=0.02, value=0.0, label="Noise std dev")

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([n_frames_ui, ref_phase_ui, noise_ui], gap=2),
        mo.hstack([peak_a_ui, amp_a_ui], gap=2),
        mo.hstack([peak_b_ui, amp_b_ui], gap=2),
    ])
    return (
        amp_a_ui,
        amp_b_ui,
        n_frames_ui,
        noise_ui,
        peak_a_ui,
        peak_b_ui,
        ref_phase_ui,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0 — synthetic uniform-direction pulsatile flow
    """)
    return


@app.cell
def _(amp_a_ui, amp_b_ui, n_frames_ui, noise_ui, np, peak_a_ui, peak_b_ui, xr):
    N = n_frames_ui.value
    rng = np.random.default_rng(0)

    def bump(peak, amplitude, n=N):
        """Raised-cosine pulse peaking at frame `peak`, period n — a stand-in
        for a real pulsatile/periodic-flow velocity trace."""
        phase = np.arange(n)
        return amplitude * 0.5 * (1 + np.cos(2 * np.pi * (phase - peak) / n))

    def set_dataset(peak, amplitude, noise_std, n=N, seed_offset=0):
        u = bump(peak, amplitude, n)
        if noise_std:
            u = u + rng.normal(0, noise_std, size=n)
        zeros = np.zeros(n)
        return xr.Dataset(
            {"u": (("phase",), u), "v": (("phase",), zeros), "w": (("phase",), zeros)},
            coords={"phase": np.arange(n)},
        )

    set_a = set_dataset(peak_a_ui.value, amp_a_ui.value, noise_ui.value)
    set_b = set_dataset(peak_b_ui.value, amp_b_ui.value, noise_ui.value)
    return N, set_a, set_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 — peak selection (`argmax`, per set)
    """)
    return


@app.cell
def _(np, set_a, set_b):
    detected_peak_a = int(np.argmax(set_a["u"].values))
    detected_peak_b = int(np.argmax(set_b["u"].values))
    return detected_peak_a, detected_peak_b


@app.cell
def _(detected_peak_a, detected_peak_b, mo, plt, set_a, set_b):
    fig_raw, ax_raw = plt.subplots(figsize=(9, 4), dpi=100)
    ax_raw.plot(set_a["phase"], set_a["u"], "o-", color="tab:blue", label="Set A")
    ax_raw.plot(set_b["phase"], set_b["u"], "s-", color="tab:orange", label="Set B")
    ax_raw.axvline(detected_peak_a, color="tab:blue", linestyle="--", alpha=0.6)
    ax_raw.axvline(detected_peak_b, color="tab:orange", linestyle="--", alpha=0.6)
    ax_raw.scatter([detected_peak_a], [set_a["u"].values[detected_peak_a]],
                   color="tab:blue", marker="*", s=250, zorder=5,
                   label=f"A peak @ {detected_peak_a}")
    ax_raw.scatter([detected_peak_b], [set_b["u"].values[detected_peak_b]],
                   color="tab:orange", marker="*", s=250, zorder=5,
                   label=f"B peak @ {detected_peak_b}")
    ax_raw.set_xlabel("frame (phase)")
    ax_raw.set_ylabel("u [uniform-direction velocity]")
    ax_raw.set_title("Raw sets: same cycle shape, different phase of the peak")
    ax_raw.legend(loc="upper right", fontsize=8)
    fig_raw.tight_layout()
    mo.hstack([fig_raw])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 — rolling the cycle around the peak (`shift_phase`)

    `shift_phase(ds, shift)` does `ds.roll(phase=shift)`: a **cyclic** roll, so
    the tail wraps back to the front — the peak moves to the reference phase
    without discarding any frames. The polar view below makes the "cyclic"
    part visible: each set's cycle is a ring of frames, and rolling just
    rotates the ring so the peak (★) lines up with the reference angle.
    """)
    return


@app.cell
def _(
    detected_peak_a,
    detected_peak_b,
    ref_phase_ui,
    set_a,
    set_b,
    shift_phase,
):
    shift_a = ref_phase_ui.value - detected_peak_a
    shift_b = ref_phase_ui.value - detected_peak_b

    aligned_a = shift_phase(set_a, shift_a)
    aligned_b = shift_phase(set_b, shift_b)
    return aligned_a, aligned_b, shift_a, shift_b


@app.cell
def _(
    N,
    aligned_a,
    aligned_b,
    mo,
    np,
    plt,
    ref_phase_ui,
    set_a,
    set_b,
    shift_a,
    shift_b,
):
    def polar_ax(ax, ds, peak_frame, shift, title, color):
        angles = 2 * np.pi * ds["phase"].values / N
        r = ds["u"].values
        ax.plot(angles, r, "o-", color=color, markersize=4)
        ax.plot([angles[peak_frame % N]], [r[peak_frame % N]], marker="*",
                color="red", markersize=18, zorder=5)
        ax.set_title(f"{title}\n(shift={shift:+d})", fontsize=10)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    fig_polar, axes_polar = plt.subplots(
        1, 4, figsize=(16, 4.5), dpi=100, subplot_kw={"projection": "polar"}
    )
    polar_ax(axes_polar[0], set_a, int(np.argmax(set_a["u"].values)), 0, "A: before roll", "tab:blue")
    polar_ax(axes_polar[1], aligned_a, ref_phase_ui.value, shift_a, "A: after roll", "tab:blue")
    polar_ax(axes_polar[2], set_b, int(np.argmax(set_b["u"].values)), 0, "B: before roll", "tab:orange")
    polar_ax(axes_polar[3], aligned_b, ref_phase_ui.value, shift_b, "B: after roll", "tab:orange")
    fig_polar.suptitle("Each ring is one cycle (frame 0 at top, going clockwise); "
                       "★ = peak. Rolling rotates the ring onto the reference phase.")
    fig_polar.tight_layout()
    mo.hstack([fig_polar])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 — phase averaging (`phase_average`)

    Compare the **naive** average (raw sets, peaks misaligned) against the
    **aligned** average (rolled sets, peaks on the reference phase). Same
    `phase_average` call both times — only its input differs.
    """)
    return


@app.cell
def _(aligned_a, aligned_b, mo, phase_average, plt, set_a, set_b, xr):
    def stack(a, b):
        return xr.concat([a, b], dim=xr.DataArray(["a", "b"], dims="set", name="set"), join="exact")

    naive_avg = phase_average(stack(set_a, set_b))
    aligned_avg = phase_average(stack(aligned_a, aligned_b))

    fig_avg, ax_avg = plt.subplots(figsize=(9, 4), dpi=100)
    ax_avg.plot(set_a["phase"], set_a["u"], ":", color="tab:blue", alpha=0.4, label="Set A (raw)")
    ax_avg.plot(set_b["phase"], set_b["u"], ":", color="tab:orange", alpha=0.4, label="Set B (raw)")
    ax_avg.plot(naive_avg["phase"], naive_avg["u"], "o-", color="gray",
               label="Naive average (no roll) — smeared peak")
    ax_avg.plot(aligned_avg["phase"], aligned_avg["u"], "o-", color="tab:green", linewidth=2.5,
               label="Aligned average (rolled first) — true peak")
    ax_avg.set_xlabel("frame (phase)")
    ax_avg.set_ylabel("u")
    ax_avg.set_title("phase_average(): naive vs. shift-aligned")
    ax_avg.legend(loc="upper right", fontsize=8)
    fig_avg.tight_layout()
    mo.hstack([fig_avg])
    return aligned_avg, naive_avg


@app.cell
def _(aligned_avg, mo, naive_avg, np):
    mo.md(f"""
    | | peak amplitude | peak frame |
    |---|---|---|
    | Naive average | {float(naive_avg["u"].max()):.3f} | {int(np.argmax(naive_avg["u"].values))} |
    | Aligned average | {float(aligned_avg["u"].max()):.3f} | {int(np.argmax(aligned_avg["u"].values))} |

    The aligned average recovers the true peak amplitude that the naive
    average smears away — the same effect `test_phase_align_synthetic_sets.py`
    asserts.
    """)
    return


if __name__ == "__main__":
    app.run()
