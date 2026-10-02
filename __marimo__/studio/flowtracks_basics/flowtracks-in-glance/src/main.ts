import { mount } from "svelte";

import App from "./App.svelte";
import "./style.css";

const target = document.getElementById("app-shell");

if (target === null) {
  throw new Error("Svelte view requires #app-shell");
}

mount(App, { target });
