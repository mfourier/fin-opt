export type ExplainerFocus =
  | "wealth"
  | "contribution"
  | "allocation"
  | "withdrawal"
  | "return"
  | "median"
  | "likely-band"
  | "possible-band"
  | "goal-probability"
  | "goal-target"
  | "horizon";

export type ExplainerSection = "wealth" | "bands" | "goals";

export function isFocusActive(active: ExplainerFocus | null, values: ExplainerFocus[]): boolean {
  return active !== null && values.includes(active);
}

export function focusToSection(focus: ExplainerFocus | null): ExplainerSection | null {
  if (!focus) return null;
  if (["wealth", "contribution", "allocation", "withdrawal", "return"].includes(focus)) return "wealth";
  if (["median", "likely-band", "possible-band"].includes(focus)) return "bands";
  return "goals";
}

export function focusToLabel(focus: ExplainerFocus | null): string | null {
  if (!focus) return null;

  const labels: Record<ExplainerFocus, string> = {
    wealth: "Wealth path",
    contribution: "Monthly contributions",
    allocation: "Allocation split",
    withdrawal: "Withdrawals",
    return: "Investment return",
    median: "Median path",
    "likely-band": "Likely range",
    "possible-band": "Possible range",
    "goal-probability": "Goal probability",
    "goal-target": "Goal target",
    horizon: "Minimum horizon",
  };

  return labels[focus];
}
