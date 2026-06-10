// FinOpt formatting helpers. Currency = Chilean pesos (CLP), no decimals.

const clpFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

export function formatCLP(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "$0";
  const sign = value < 0 ? "-" : "";
  return `${sign}$${clpFormatter.format(Math.abs(Math.round(value)))}`;
}

/** Compact CLP for chart axes: $1.5M, $850K. */
export function formatCLPCompact(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1_000_000_000) return `${sign}$${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${sign}$${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(value / 1_000).toFixed(0)}K`;
  return `${sign}$${Math.round(value)}`;
}

/** months → "4y 2m" / "10m" / "3y". */
export function formatMonths(totalMonths: number | null | undefined): string {
  if (totalMonths === null || totalMonths === undefined) return "—";
  const m = Math.max(0, Math.round(totalMonths));
  const years = Math.floor(m / 12);
  const months = m % 12;
  if (years === 0) return `${months}m`;
  if (months === 0) return `${years}y`;
  return `${years}y ${months}m`;
}

/** Long form for the hero: "4 years 2 months". */
export function formatMonthsLong(totalMonths: number | null | undefined): string {
  if (totalMonths === null || totalMonths === undefined) return "—";
  const m = Math.max(0, Math.round(totalMonths));
  const years = Math.floor(m / 12);
  const months = m % 12;
  const parts: string[] = [];
  if (years > 0) parts.push(`${years} ${years === 1 ? "year" : "years"}`);
  if (months > 0) parts.push(`${months} ${months === 1 ? "month" : "months"}`);
  if (parts.length === 0) return "0 months";
  return parts.join(" ");
}

export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** "8 out of 10 scenarios meet the goal" for a confidence in [0,1]. */
export function describeConfidence(confidence: number): string {
  const n = Math.round(confidence * 10);
  return `${n} out of 10 scenarios meet the goal`;
}

/** Add `monthsFromNow` months to today, return short label like "Aug 2028". */
export function monthLabel(monthsFromNow: number, start: Date = new Date()): string {
  const d = new Date(start.getFullYear(), start.getMonth() + monthsFromNow, 1);
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}
