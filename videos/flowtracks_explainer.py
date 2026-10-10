"""3Blue1Brown-style explainer: the math inside flowtracks post-processing.

Run:  uv run --with manim manim -pqh videos/flowtracks_explainer.py FlowtracksExplainer
"""

import numpy as np
from manim import *

BLUE = BLUE_C
YELLOW = YELLOW_C
TEAL = TEAL_C
RED = RED_C


class Title(Scene):
    def construct(self):
        t = Text("From particles to fields", font_size=64)
        t.set_color_by_gradient(BLUE, TEAL)
        sub = Text("the math inside flowtracks", font_size=30, color=GREY_B)
        sub.next_to(t, DOWN)
        self.play(Write(t), FadeIn(sub, shift=UP * 0.3))
        self.wait(1.5)


class LagrangianVsEulerian(Scene):
    """Scene 1: show tracer particles advecting (Lagrangian view)."""

    def construct(self):
        title = Text("3D particle tracking: Lagrangian data", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # A box (the measurement volume) and some moving particles.
        box = Rectangle(width=9, height=5, color=GREY).shift(DOWN * 0.3)
        self.play(Create(box))

        rng = np.random.default_rng(3)
        dots = []
        vels = []
        for _ in range(28):
            p = rng.uniform([-4.2, -2.5], [4.2, 2.2])
            d = Dot(np.array([p[0], p[1] - 0.3, 0]), radius=0.06, color=YELLOW)
            ang = rng.uniform(-0.4, 0.4) + 0.2
            v = np.array([np.cos(ang), np.sin(ang), 0]) * rng.uniform(0.8, 1.6)
            dots.append(d)
            vels.append(v)
        self.add(*dots)

        # Particles advect along their velocities; wrap around the box.
        for d, v in zip(dots, vels):
            d.v = v

        def advect(mob, dt):
            mob.shift(mob.v * dt)
            xy = mob.get_center()
            if xy[0] > 4.4:
                mob.shift(LEFT * 8.8)
                xy = mob.get_center()
            if xy[1] > 2.1:
                mob.shift(DOWN * 4.6)
            elif xy[1] < -2.7:
                mob.shift(UP * 4.6)

        for d in dots:
            d.add_updater(advect)
        self.wait(3)
        for d in dots:
            d.clear_updaters()

        label = MarkupText("{ x<sub>i</sub>(t),  v<sub>i</sub>(t) }<sub>i = 1…N</sub>", font_size=40)
        label.to_edge(DOWN)
        self.play(FadeIn(label))
        self.wait(1)


class Binning(Scene):
    """Scene 2: histogram particles into voxels; each voxel accumulates
    sum of velocities and count -> mean velocity field."""

    def construct(self):
        title = Text("Step 1 — bin particles into voxels", font_size=36).to_edge(UP)
        self.play(Write(title))

        grid = NumberPlane(x_range=[-4, 4, 1], y_range=[-2.5, 2.5, 1],
                           background_line_style={"stroke_opacity": 0.25})
        grid.scale(0.95).shift(DOWN * 0.2)
        self.play(Create(grid))

        rng = np.random.default_rng(7)
        pts = [Dot(np.array([rng.uniform(-3.6, 3.6), rng.uniform(-2.2, 2.2) - 0.2, 0]),
                   radius=0.05, color=YELLOW) for _ in range(40)]
        self.add(*pts)

        formula = MarkupText("ū<sub>ijk</sub> = (1/N<sub>ijk</sub>) Σ v<sub>p</sub>   over p ∈ V<sub>ijk</sub>", font_size=40)
        formula.to_edge(DOWN, buff=0.6)
        self.play(Write(formula))
        self.wait(2)

        # Highlight one voxel and count its occupants.
        cell = Square(side_length=1.0, color=BLUE, fill_opacity=0.3)
        # place cell centered at a grid intersection midpoint
        cell.move_to(np.array([1.5, 0.3, 0]))
        self.play(Create(cell))
        inside = [p for p in pts if abs(p.get_center()[0] - 1.5) < 0.5 and abs(p.get_center()[1] - 0.3) < 0.5]
        for p in inside:
            p.set_color(BLUE)
        self.wait(1.5)


class KernelSmoothing(Scene):
    """Scene 3: Shepard/kernel estimate — Gaussian-weighted sums of
    velocities and counts sharing one normalization."""

    def construct(self):
        title = Text("Step 2 — kernel (Shepard) smoothing", font_size=36).to_edge(UP)
        self.play(Write(title))

        axes = Axes(x_range=[-3, 3], y_range=[0, 1.2], x_length=7, y_length=3.2,
                    axis_config={"include_tip": False}).shift(UP * 0.6)
        curve = axes.plot(lambda x: np.exp(-x**2 / (2 * 0.8**2)), color=TEAL, x_range=[-3, 3])
        kernel_label = MarkupText("G<sub>σ</sub>(x − x<sub>p</sub>) = exp(−|x − x<sub>p</sub>|² / 2σ²)", font_size=32).next_to(axes, DOWN, buff=0.4)
        self.play(Create(axes), Create(curve), Write(kernel_label))
        self.wait(1.5)

        formula = MarkupText("ū(x) = Σ<sub>p</sub> G<sub>σ</sub>(x−x<sub>p</sub>)·v<sub>p</sub>   /   Σ<sub>p</sub> G<sub>σ</sub>(x−x<sub>p</sub>)",
                             font_size=32, color=YELLOW)
        formula.to_edge(DOWN, buff=0.7)
        self.play(Write(formula))
        self.wait(2)
        note = Text("same Gaussian on velocities and counts — one shared normalization",
                    font_size=24, color=GREY_B).next_to(formula, DOWN)
        self.play(FadeIn(note))
        self.wait(2)


class PhaseAverage(Scene):
    """Scene 4: periodic flow -> fold time into phase, average over cycles."""

    def construct(self):
        title = Text("Step 3 — phase averaging a periodic flow", font_size=36).to_edge(UP)
        self.play(Write(title))

        axes = Axes(x_range=[0, 4 * np.pi], y_range=[-1.4, 1.4],
                    x_length=10, y_length=3.4).shift(UP * 0.8)
        self.play(Create(axes))

        # systole-diastole-like pulse, periodic with period 2π
        def pulse(x):
            s = x % (2 * np.pi)
            return (1.1 * np.exp(-((s - 1.4) ** 2) / 0.12)
                    + 0.4 * np.exp(-((s - 2.4) ** 2) / 0.06) - 0.25)

        rng = np.random.default_rng(11)
        noisy = []
        for _ in range(5):
            amp = rng.uniform(0.85, 1.15)
            shift = rng.uniform(-0.15, 0.15)
            noise = rng.uniform(-0.06, 0.06)
            w = rng.uniform(0, 6)
            noisy.append(lambda x, a=amp, s=shift, n=noise, w=w: a * pulse(x + s) + n * np.sin(3 * x + w))
        curves = [axes.plot(f, color=GREY_B, stroke_opacity=0.45) for f in noisy]
        cyc_label = Text("individual cycles  u(x, t_k)", font_size=26, color=GREY_B).next_to(axes, UP, buff=0.15)
        self.play(*[Create(c) for c in curves], FadeIn(cyc_label))
        self.wait(2)

        mean_curve = axes.plot(pulse, color=TEAL, stroke_width=6)
        mean_label = MarkupText("⟨u⟩(x) = average over cycles", font_size=30, color=TEAL).next_to(axes, UP, buff=0.15)
        self.play(*[FadeOut(c) for c in curves], FadeOut(cyc_label),
                  Create(mean_curve), FadeIn(mean_label))
        self.wait(2)

        phase = MarkupText("φ = (t mod T)/T,   ⟨u⟩(x, φ) = (1/N<sub>φ</sub>) Σ u(x, t)  over all t with φ(t) = φ", font_size=26).to_edge(DOWN, buff=0.7)
        self.play(Write(phase))
        self.wait(1.5)

        # fluctuations: one noisy cycle minus the mean, plotted in red
        resid = axes.plot(lambda x: noisy[0](x) - pulse(x), color=RED, stroke_width=4)
        resid_label = Text("u′(x) = u(x,t) − ⟨u⟩(x)", font_size=28, color=RED).next_to(mean_label, DOWN, buff=0.1).align_to(mean_label, LEFT)
        self.play(Create(resid), FadeIn(resid_label))
        self.wait(2.5)


class Outro(Scene):
    def construct(self):
        lines = VGroup(
            Text("{ xᵢ(t), vᵢ(t) }  →  bin/kernel  →  ū(x)  →  phase avg  →  ⟨u⟩(x, φ)", font_size=30),
            Text("same math for HDF5 (Scene) and Zarr (ZarrScene) backends", font_size=26, color=GREY_B),
        ).arrange(DOWN, buff=0.8)
        self.play(Write(lines[0]))
        self.wait(1.5)
        self.play(FadeIn(lines[1]))
        self.wait(2)


# Single scene stitching everything (easiest to render one file).
class FlowtracksExplainer(Scene):
    def construct(self):
        for cls in (LagrangianVsEulerian, Binning, KernelSmoothing, PhaseAverage, Outro):
            cls.construct(self)
            self.clear()
            self.wait(0.3)
