export type MarimoJsonValue =
  | null
  | boolean
  | number
  | string
  | MarimoJsonValue[]
  | { [key: string]: MarimoJsonValue };

export const MARIMO_DATA_SOURCE = Symbol.for("marimo-studio.data-source");

export interface MarimoDataSource {
  readonly codec: string;
  readonly fingerprint: string;
  readonly bytes: Uint8Array;
}

export interface MarimoColumn<T = unknown> extends Iterable<T> {
  readonly length: number;
  readonly nullCount: number;
  get(index: number): T | null | undefined;
  toArray(): ArrayLike<T> & Iterable<T>;
}

export interface MarimoTable<
  Row extends object = Record<string, unknown>,
> extends Iterable<Row> {
  readonly [MARIMO_DATA_SOURCE]?: MarimoDataSource;
  readonly numRows: number;
  readonly numCols: number;
  readonly names: readonly (keyof Row & string)[];
  readonly schema: {
    readonly fields: readonly {
      readonly name: string;
      readonly type: {
        readonly typeId: number;
        readonly [key: string]: unknown;
      };
      readonly nullable: boolean;
    }[];
  };
  get(index: number): Row | null;
  getChild<Name extends keyof Row & string>(
    name: Name,
  ): MarimoColumn<Row[Name]> | undefined;
  select<Name extends keyof Row & string>(
    names: readonly Name[],
  ): MarimoTable<Pick<Row, Name>>;
  toColumns(): {
    readonly [Name in keyof Row]: ArrayLike<Row[Name]> & Iterable<Row[Name]>;
  };
  toArray(): Row[];
}

export type MarimoValue = MarimoJsonValue | MarimoTable;

export const isMarimoTable = (value: unknown): value is MarimoTable =>
  Object.prototype.toString.call(value) === "[object Table]";

export const getMarimoDataSource = (
  value: unknown,
): MarimoDataSource | undefined => {
  if (
    (typeof value !== "object" && typeof value !== "function") ||
    value === null
  ) {
    return undefined;
  }
  return (value as Record<symbol, MarimoDataSource | undefined>)[
    MARIMO_DATA_SOURCE
  ];
};

export type MarimoValueElement<T = MarimoValue> = HTMLElement & {
  marimoValue?: T;
};

export type MarimoValueError = {
  readonly selector: string;
  readonly code: string;
  readonly message: string;
  readonly hint?: string;
};

/** Read the error a host recorded before its listeners attached. */
const hostError = (host: HTMLElement): MarimoValueError | undefined =>
  host.dataset.marimoError === undefined ? undefined : {
    selector: host.getAttribute("mo-value")?.trim() ?? "",
    code: host.dataset.marimoErrorCode ?? "",
    message: host.dataset.marimoError,
    hint: host.dataset.marimoDiagnosticHint,
  };

export type MarimoValueOptions<T = MarimoValue> = {
  onValue: (value: T) => void;
  onError?: (error: MarimoValueError) => void;
};

/** Subscribe an explicit `mo-value` host to current values and later updates. */
export const observeMarimoValue = <T = MarimoValue>(
  node: HTMLElement,
  options: MarimoValueOptions<T>,
) => {
  let current = options;
  const host = node as MarimoValueElement<T>;

  const sync = () => {
    if (host.marimoValue !== undefined) {
      current.onValue(host.marimoValue);
    }
  };
  const fail = (event: Event) =>
    current.onError?.((event as CustomEvent<MarimoValueError>).detail);

  host.addEventListener("marimo-value-updated", sync);
  host.addEventListener("marimo-value-error", fail);
  const error = hostError(host);
  if (error === undefined) {
    sync();
  } else {
    current.onError?.(error);
  }

  return {
    update(next: MarimoValueOptions<T>) {
      current = next;
    },
    destroy() {
      host.removeEventListener("marimo-value-updated", sync);
      host.removeEventListener("marimo-value-error", fail);
    },
  };
};
