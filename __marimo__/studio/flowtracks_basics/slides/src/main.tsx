import { createRoot } from "react-dom/client";

import { App } from "./App.tsx";
import "./style.css";

const root = document.getElementById("app-shell");

if (root === null) {
  throw new Error("React view requires #app-shell");
}

createRoot(root).render(<App />);
