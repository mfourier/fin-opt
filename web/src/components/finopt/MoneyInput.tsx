import * as React from "react";
import { cn } from "@/lib/utils";

type Props = {
  value: number;
  onChange: (v: number) => void;
  placeholder?: string;
  id?: string;
  "aria-label"?: string;
  "aria-invalid"?: boolean;
  className?: string;
  min?: number;
};

const groupFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

function format(n: number): string {
  if (!Number.isFinite(n) || n === 0) return "";
  return groupFmt.format(Math.round(n));
}

function parse(s: string): number {
  const digits = s.replace(/[^\d]/g, "");
  if (!digits) return 0;
  return Number(digits);
}

/** CLP money input with $ prefix and thousands separators. */
export function MoneyInput({
  value,
  onChange,
  placeholder = "0",
  className,
  ...rest
}: Props) {
  const [text, setText] = React.useState(format(value));

  // Keep text in sync if parent value changes externally.
  React.useEffect(() => {
    const parsed = parse(text);
    if (parsed !== value) setText(format(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <div
      className={cn(
        "flex h-10 items-center rounded-md border border-input bg-transparent text-sm shadow-sm focus-within:ring-1 focus-within:ring-ring",
        className,
      )}
    >
      <span className="select-none pl-3 pr-1 text-muted-foreground">$</span>
      <input
        inputMode="numeric"
        autoComplete="off"
        value={text}
        placeholder={placeholder}
        onChange={(e) => {
          const n = parse(e.target.value);
          setText(format(n));
          onChange(n);
        }}
        className="tabular w-full bg-transparent py-2 pr-3 text-right font-medium tracking-tight outline-none placeholder:text-muted-foreground"
        {...rest}
      />
    </div>
  );
}
