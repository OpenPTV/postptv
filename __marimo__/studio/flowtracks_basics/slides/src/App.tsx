/// <reference path="./marimo-studio.d.ts" />

import { Deck, Slide } from "@revealjs/react";
import { useEffect, useRef } from "react";
import type { RevealApi } from "reveal.js";
import "reveal.js/reveal.css";

const VIEW_HEADING = "Slides";
const NOTEBOOK_TITLE = "flowtracks basics — from particles to fields";

const keyboardCondition = (event: KeyboardEvent) =>
  !event.composedPath().some(
    (target) =>
      target instanceof Element && target.matches("marimo-cell, marimo-output"),
  );

/** Center slides again once projected notebook content has its final size. */
const useSettledLayout = () => {
  const deck = useRef<RevealApi | null>(null);
  useEffect(() => {
    const layout = () => deck.current?.layout();
    document.addEventListener("marimo-studio:idle", layout);
    return () => document.removeEventListener("marimo-studio:idle", layout);
  }, []);
  return deck;
};

export const App = () => {
  const deck = useSettledLayout();
  return (
    <Deck
      className="studio-deck"
      deckRef={deck}
      config={{
        controls: true,
        keyboardCondition,
        progress: true,
        scrollActivationWidth: 0,
        transition: "slide",
      }}
    >
      <Slide>
        <p className="deck-kicker">{VIEW_HEADING}</p>
        <h1>{NOTEBOOK_TITLE}</h1>
        <p>Read → clean → grid → average → export: the flowtracks pipeline in six steps.</p>
      </Slide>
      {[
        {
          "target": "cell-2",
          "title": "flowtracks basics — from particles to fields",
          "showTitle": false,
        },
        {
          "target": "cell-3",
          "title": "Try it live",
          "showTitle": true,
        },
        {
          "target": "cell-4",
          "title": "1 · Lagrangian data model",
          "showTitle": false,
        },
        {
          "target": "cell-5",
          "title": "Track population",
          "showTitle": true,
        },
        {
          "target": "cell-6",
          "title": "2 · 3D view — what raw tracks look like",
          "showTitle": false,
        },
        {
          "target": "cell-7",
          "title": "Raw tracks in 3D",
          "showTitle": true,
        },
        {
          "target": "cell-8",
          "title": "3 · Smoothing — flowtracks.smoothing.savitzky_golay",
          "showTitle": false,
        },
        {
          "target": "cell-9",
          "title": "Smoothing in action",
          "showTitle": true,
        },
        {
          "target": "cell-10",
          "title": "4 · Stitching — flowtracks.stitching.stitch_trajectories",
          "showTitle": false,
        },
        {
          "target": "cell-11",
          "title": "Stitching result",
          "showTitle": true,
        },
        {
          "target": "cell-12",
          "title": "5 · Eulerian gridding — flowtracks.eulerian",
          "showTitle": false,
        },
        {
          "target": "cell-13",
          "title": "Eulerian grid",
          "showTitle": true,
        },
        {
          "target": "cell-14",
          "title": "6 · Phase averaging & the pipeline",
          "showTitle": false,
        },
        {
          "target": "cell-15",
          "title": "Phase mean",
          "showTitle": true,
        },
        {
          "target": "cell-16",
          "title": "Takeaway",
          "showTitle": false,
        },
      ].map(({ target, title, showTitle }) => (
        <Slide key={target}>
          {showTitle ? <h2>{title}</h2> : null}
          <marimo-cell name={target} />
        </Slide>
      ))}
      <Slide>
        <h2>Next steps</h2>
        <ul>
          <li>
            Install: <code>pip install flowtracks</code> (ParaView export:{" "}
            <code>pip install "flowtracks[vtk]"</code>)
          </li>
          <li>
            Read real data with <code>io.trajectories()</code>, <code>Scene</code>, or{" "}
            <code>ZarrScene</code>
          </li>
          <li>Cite: Meller &amp; Liberzon 2016, J. Open Research Software, 4:e23</li>
        </ul>
      </Slide>
    </Deck>
  );
};
