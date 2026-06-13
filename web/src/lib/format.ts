// FinOpt formatting helpers. Currency = Chilean pesos (CLP), no decimals.

const clpFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

function coerceDate(value: string | Date | null | undefined): Date | null {
  if (value instanceof Date) {
    return new Date(value.getFullYear(), value.getMonth(), value.getDate());
  }

  if (typeof value === "string") {
    const isoMatch = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (isoMatch) {
      return new Date(Number(isoMatch[1]), Number(isoMatch[2]) - 1, Number(isoMatch[3]));
    }

    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed;
    }
  }

  return null;
}

export function formatCLP(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "$0";
  const sign = value < 0 ? "-" : "";
  return `${sign}$${clpFormatter.format(Math.abs(Math.round(value)))}`;
}

/** Compact CLP for chart axes: $1.5M, $850K, -$1.5M. */
export function formatCLPCompact(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1_000_000_000) return `${sign}$${(abs / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
  return `${sign}$${Math.round(abs)}`;
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

/**
 * Certainty presets shown to the user. Each maps to a hidden CVaR confidence
 * level (1 - eps) passed to the optimizer as the goal's `confidence`. The levels
 * 0.10 / 0.60 / 0.90 were chosen by the P0 calibration sweep (see
 * docs/dev-notes/calibration-table.md): on the real model they yield ~59% / ~83%
 * / ~96% *measured* empirical success. We never show the CVaR level — only the
 * preset label here, and the measured empirical on the results page.
 */
export type ConfidencePreset = {
  id: "reach_sooner" | "balanced" | "play_it_safe";
  /** Hidden CVaR confidence level written to the goal's `confidence`. */
  value: number;
  label: string;
  /** One-line, hedged description of the trade-off (no hard promise). */
  blurb: string;
};

export const CONFIDENCE_PRESETS: ConfidencePreset[] = [
  {
    id: "reach_sooner",
    value: 0.1,
    label: "Reach sooner",
    blurb: "Aim for the earliest date. Less certain, but you get there faster.",
  },
  {
    id: "balanced",
    value: 0.6,
    label: "Balanced",
    blurb: "A sensible middle ground between speed and certainty.",
  },
  {
    id: "play_it_safe",
    value: 0.9,
    label: "Play it safe",
    blurb: "Aim for high certainty. The plan may take a little longer.",
  },
];

export const DEFAULT_CONFIDENCE = CONFIDENCE_PRESETS[1].value; // Balanced (0.60)

/** Nearest preset for a stored confidence value (snaps legacy/arbitrary values). */
export function presetForConfidence(confidence: number): ConfidencePreset {
  return CONFIDENCE_PRESETS.reduce((best, p) =>
    Math.abs(p.value - confidence) < Math.abs(best.value - confidence) ? p : best,
  );
}

/**
 * Plain-language description of a *measured* success probability in [0,1].
 * Feed this the empirical likelihood from the results — never the hidden CVaR
 * level (a 0.10 "reach sooner" goal is not "1 out of 10": its measured empirical
 * is typically far higher).
 */
export function describeConfidence(probability: number): string {
  const n = Math.round(probability * 10);
  return `${n} out of 10 scenarios meet the goal`;
}

/**
 * A goal is in the "floor" / collapse regime when it is reached in virtually
 * every simulated scenario **even though** a relaxed certainty was requested.
 * In that regime the certainty preset can't meaningfully move the outcome — the
 * goal is comfortably within reach at any setting (the Shape-C case from the P0
 * calibration: ~100% empirical at every CVaR level). We can infer this honestly
 * at render time because we know both the measured empirical and the requested
 * level: a near-certain result from a low/moderate request means the goal sits on
 * its convex floor. (A high request, e.g. "Play it safe", legitimately yields a
 * high number and is NOT flagged.)
 *
 * @param empirical measured success probability in [0,1]
 * @param requestedConfidence the goal's required_confidence (the hidden CVaR level)
 */
export function isFloorGoal(empirical: number, requestedConfidence: number): boolean {
  return empirical >= 0.99 && requestedConfidence <= 0.6;
}

/** Add `monthsFromNow` months to today, return short label like "Aug 2028". */
export function monthLabel(monthsFromNow: number, start?: string | Date | null): string {
  const base = coerceDate(start) ?? new Date();
  const d = new Date(base.getFullYear(), base.getMonth() + monthsFromNow, 1);
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

export function formatMonthYear(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const parsed = coerceDate(value);
  if (!parsed) return "—";
  return parsed.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

export function formatDateShort(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const parsed = coerceDate(value);
  if (!parsed) return "—";
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
