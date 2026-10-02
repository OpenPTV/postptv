declare module "svelte/elements" {
  interface HTMLAttributes<T> {
    "data-marimo-allow"?: "*";
    "mo-value"?: string;
  }
}

export {};
