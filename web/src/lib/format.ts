// FinOpt formatting helpers. Currency = Chilean pesos (CLP), no decimals.

import i18n from "@/i18n/config";

/** BCP-47 locale for Intl, derived from the active UI language. */
function getLocale(): string {
  return i18n.language === "en" ? "en-US" : "es-CL";
}

// Cache one NumberFormat per locale (constructing them is comparatively costly).
const clpFormatters = new Map<string, Intl.NumberFormat>();
function clpFormatterFor(locale: string): Intl.NumberFormat {
  let fmt = clpFormatters.get(locale);
  if (!fmt) {
    fmt = new Intl.NumberFormat(locale, { maximumFractionDigits: 0 });
    clpFormatters.set(locale, fmt);
  }
  return fmt;
}

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
  return `${sign}$${clpFormatterFor(getLocale()).format(Math.abs(Math.round(value)))}`;
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

/** months → compact, locale-aware: "4y 2m" (en) / "4a 2m" (es). */
export function formatMonths(totalMonths: number | null | undefined): string {
  if (totalMonths === null || totalMonths === undefined) return "—";
  const m = Math.max(0, Math.round(totalMonths));
  const years = Math.floor(m / 12);
  const months = m % 12;
  const y = i18n.t("common:duration.yearShort");
  const mo = i18n.t("common:duration.monthShort");
  if (years === 0) return `${months}${mo}`;
  if (months === 0) return `${years}${y}`;
  return `${years}${y} ${months}${mo}`;
}

/** Long form for the hero: "4 years 2 months" / "4 años 2 meses". */
export function formatMonthsLong(totalMonths: number | null | undefined): string {
  if (totalMonths === null || totalMonths === undefined) return "—";
  const m = Math.max(0, Math.round(totalMonths));
  const years = Math.floor(m / 12);
  const months = m % 12;
  const parts: string[] = [];
  if (years > 0) parts.push(i18n.t("common:duration.year", { count: years }));
  if (months > 0) parts.push(i18n.t("common:duration.month", { count: months }));
  if (parts.length === 0) return i18n.t("common:duration.zero");
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
export type ConfidencePresetId = "reach_sooner" | "balanced" | "play_it_safe";

export type ConfidencePreset = {
  id: ConfidencePresetId;
  /** Hidden CVaR confidence level written to the goal's `confidence`. */
  value: number;
};

// Human-readable label/blurb for each preset id live in the `common:confidence.<id>`
// translation keys (resolved at render time), so only the model value lives here.
export const CONFIDENCE_PRESETS: ConfidencePreset[] = [
  { id: "reach_sooner", value: 0.1 },
  { id: "balanced", value: 0.6 },
  { id: "play_it_safe", value: 0.9 },
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
  return i18n.t("common:confidence.describe", { n });
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
  return d.toLocaleDateString(getLocale(), { month: "short", year: "numeric" });
}

export function formatMonthYear(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const parsed = coerceDate(value);
  if (!parsed) return "—";
  return parsed.toLocaleDateString(getLocale(), { month: "short", year: "numeric" });
}

export function formatDateShort(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const parsed = coerceDate(value);
  if (!parsed) return "—";
  return parsed.toLocaleDateString(getLocale(), {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
