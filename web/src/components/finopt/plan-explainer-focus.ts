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

// Human-readable label per focus id lives in `plan:explainer.concepts.<focus>`
// (resolved with t() in the component), so there is no label map here.
