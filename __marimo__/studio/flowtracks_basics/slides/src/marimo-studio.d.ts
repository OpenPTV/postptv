// @deno-types="npm:@types/react@19.2.10"
import type {
  DetailedHTMLProps as _DetailedHTMLProps,
  HTMLAttributes as _HTMLAttributes,
} from "react";

type _MarimoHostAttributes = _DetailedHTMLProps<
  _HTMLAttributes<HTMLElement>,
  HTMLElement
>;

declare module "react" {
  interface HTMLAttributes<T> {
    "data-marimo-allow"?: "*";
    "mo-value"?: string;
  }

  namespace JSX {
    interface IntrinsicElements {
      "marimo-cell": _MarimoHostAttributes & { name: string };
      "marimo-output": _MarimoHostAttributes & { value: string };
    }
  }
}
