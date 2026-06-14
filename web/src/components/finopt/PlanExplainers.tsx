import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import type { Profile, Result, Scenario } from "@/mocks/types";
import { cn } from "@/lib/utils";
import type { ExplainerFocus } from "./plan-explainer-focus";
import { focusToSection, type ExplainerSection } from "./plan-explainer-focus";

type Props = {
  profile: Profile;
  scenario: Scenario;
  result: Result;
  activeFocus: ExplainerFocus | null;
  pinnedFocus: ExplainerFocus | null;
  onFocusChange: (focus: ExplainerFocus | null) => void;
  onTogglePin: (focus: ExplainerFocus) => void;
  onClearPin: () => void;
};

export function PlanExplainers({
  profile,
  scenario,
  result,
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
  onClearPin,
}: Props) {
  const hasWithdrawals = Boolean(
    scenario.withdrawals
    && ((scenario.withdrawals.scheduled?.length ?? 0) > 0 || (scenario.withdrawals.stochastic?.length ?? 0) > 0),
  );
  const { t } = useTranslation("plan");
  const goalsCount = scenario.terminal_goals.length + scenario.intermediate_goals.length;
  const [openItems, setOpenItems] = useState<ExplainerSection[]>([]);
  const activeSection = focusToSection(activeFocus);
  const pinnedSection = focusToSection(pinnedFocus);
  const activeLabel = activeFocus ? t(`explainer.concepts.${activeFocus}`) : null;

  // Only auto-open a section when the user *pins* a concept (a click), not on
  // hover — hovering still highlights the matching header/token, but the panel
  // should not expand on its own as the mouse moves across the charts.
  useEffect(() => {
    if (!pinnedSection) return;
    setOpenItems((current) => (current.includes(pinnedSection) ? current : [...current, pinnedSection]));
  }, [pinnedSection]);

  const spotlightText = useMemo(() => {
    if (!activeFocus || !activeLabel) return null;
    if (activeSection === "wealth") return t("explainer.spotlightWealth", { label: activeLabel });
    if (activeSection === "bands") return t("explainer.spotlightBands", { label: activeLabel });
    return t("explainer.spotlightGoals", { label: activeLabel });
  }, [activeFocus, activeLabel, activeSection, t]);

  return (
    <section className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">{t("explainer.title")}</h2>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            {t("explainer.subtitle")}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            {t("explainer.hint")}
          </p>
          <div
            className={cn(
              "mt-3 inline-flex min-h-9 items-center rounded-full border px-3 py-1.5 text-xs transition-all duration-200",
              activeFocus
                ? "border-primary/30 bg-primary/8 text-foreground shadow-[0_0_0_4px_var(--color-primary)/0.08]"
                : "border-border bg-muted/20 text-muted-foreground",
            )}
          >
            <span className="font-medium">
              {activeLabel ? t("explainer.linkedConcept", { label: activeLabel }) : t("explainer.linkedConceptEmpty")}
            </span>
            {spotlightText ? <span className="ml-2 hidden text-muted-foreground sm:inline">{spotlightText}</span> : null}
            {pinnedFocus ? (
              <button
                type="button"
                onClick={onClearPin}
                className="ml-3 rounded-full border border-primary/25 px-2 py-0.5 text-[11px] font-medium text-foreground transition-colors hover:bg-primary/10"
              >
                {t("explainer.clearPin")}
              </button>
            ) : null}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs sm:min-w-[18rem]">
          <MiniStat label={t("explainer.miniStats.accounts")} value={String(profile.accounts_config.length)} />
          <MiniStat label={t("explainer.miniStats.goals")} value={String(goalsCount)} />
          <MiniStat
            label={t("explainer.miniStats.withdrawals")}
            value={hasWithdrawals ? t("explainer.withdrawalsIncluded") : t("explainer.withdrawalsNone")}
          />
        </div>
      </div>

      <Accordion
        type="multiple"
        value={openItems}
        onValueChange={(value) => setOpenItems(value as ExplainerSection[])}
        className="mt-5 w-full space-y-3"
      >
        <AccordionItem
          value="wealth"
          className={cn(
            "overflow-hidden rounded-xl border px-4 transition-all duration-200",
            activeSection === "wealth" && "border-primary/40 bg-primary/5 shadow-[0_0_0_4px_var(--color-primary)/0.08]",
          )}
        >
          <AccordionTrigger className="py-3 text-sm no-underline hover:no-underline">
            {t("explainer.sections.wealth.title")}
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              {t("explainer.sections.wealth.intro")}
            </p>
            <FormulaBlock>
              <FormulaLine
                left={<FocusToken focus="wealth" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}><FormulaVar base="W" sub="t+1" sup="m" /></FocusToken>}
                right={(
                  <>
                    (<FocusToken focus="wealth" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}><FormulaVar base="W" sub="t" sup="m" /></FocusToken> + <FocusToken focus="contribution" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}><FormulaVar base="A" sub="t" /></FocusToken>{" "}
                    <FocusToken focus="allocation" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}><FormulaVar base="x" sub="t" sup="m" /></FocusToken> - <FocusToken focus="withdrawal" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}><FormulaVar base="D" sub="t" sup="m" /></FocusToken>)
                    (1 + <FocusToken focus="return" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}><FormulaVar base="R" sub="t" sup="m" /></FocusToken>)
                  </>
                )}
              />
            </FormulaBlock>
            <VariableGrid
              activeFocus={activeFocus}
              pinnedFocus={pinnedFocus}
              onFocusChange={onFocusChange}
              onTogglePin={onTogglePin}
              items={[
                {
                  focus: "wealth",
                  symbol: <FormulaVar base="W" sub="t" sup="m" />,
                  description: t("explainer.sections.wealth.var.wealth"),
                },
                {
                  focus: "contribution",
                  symbol: <FormulaVar base="A" sub="t" />,
                  description: t("explainer.sections.wealth.var.contribution"),
                },
                {
                  focus: "allocation",
                  symbol: <FormulaVar base="x" sub="t" sup="m" />,
                  description: t("explainer.sections.wealth.var.allocation"),
                },
                {
                  focus: "withdrawal",
                  symbol: <FormulaVar base="D" sub="t" sup="m" />,
                  description: hasWithdrawals
                    ? t("explainer.sections.wealth.var.withdrawalActive")
                    : t("explainer.sections.wealth.var.withdrawalZero"),
                },
                {
                  focus: "return",
                  symbol: <FormulaVar base="R" sub="t" sup="m" />,
                  description: t("explainer.sections.wealth.var.return"),
                },
              ]}
            />
            <Callout>
              {t("explainer.sections.wealth.calloutPre")}
              <FocusToken focus="allocation" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>
                <InlineCode>x</InlineCode>
              </FocusToken>
              {t("explainer.sections.wealth.calloutPost", { count: profile.accounts_config.length })}
            </Callout>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem
          value="bands"
          className={cn(
            "overflow-hidden rounded-xl border px-4 transition-all duration-200",
            activeSection === "bands" && "border-primary/40 bg-primary/5 shadow-[0_0_0_4px_var(--color-primary)/0.08]",
          )}
        >
          <AccordionTrigger className="py-3 text-sm no-underline hover:no-underline">
            {t("explainer.sections.bands.title")}
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              {t("explainer.sections.bands.intro")}
            </p>
            <FormulaBlock>
              <FormulaLine
                left={<FocusToken focus="median" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>P50</FocusToken>}
                right={t("explainer.sections.bands.formula.median")}
              />
              <FormulaLine
                left={<FocusToken focus="likely-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>P25 - P75</FocusToken>}
                right={t("explainer.sections.bands.formula.likely")}
              />
              <FormulaLine
                left={<FocusToken focus="possible-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>P10 - P90</FocusToken>}
                right={t("explainer.sections.bands.formula.possible")}
              />
            </FormulaBlock>
            <VariableGrid
              activeFocus={activeFocus}
              pinnedFocus={pinnedFocus}
              onFocusChange={onFocusChange}
              onTogglePin={onTogglePin}
              items={[
                {
                  focus: "median",
                  symbol: "P50",
                  description: t("explainer.sections.bands.var.median"),
                },
                {
                  focus: "likely-band",
                  symbol: "P25-P75",
                  description: t("explainer.sections.bands.var.likely"),
                },
                {
                  focus: "possible-band",
                  symbol: "P10-P90",
                  description: t("explainer.sections.bands.var.possible"),
                },
              ]}
            />
            <Callout>
              {t("explainer.sections.bands.callout")}
            </Callout>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem
          value="goals"
          className={cn(
            "overflow-hidden rounded-xl border px-4 transition-all duration-200",
            activeSection === "goals" && "border-primary/40 bg-primary/5 shadow-[0_0_0_4px_var(--color-primary)/0.08]",
          )}
        >
          <AccordionTrigger className="py-3 text-sm no-underline hover:no-underline">
            {t("explainer.sections.goals.title")}
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              {t("explainer.sections.goals.intro")}
            </p>
            <FormulaBlock>
              <FormulaLine
                left={<FocusToken focus="goal-probability" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>Pr(W_t^m &gt;= target)</FocusToken>}
                right={t("explainer.sections.goals.formula.probability")}
              />
              <FormulaLine
                left={<FocusToken focus="horizon" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>T*</FocusToken>}
                right={t("explainer.sections.goals.formula.horizon")}
              />
            </FormulaBlock>
            <VariableGrid
              activeFocus={activeFocus}
              pinnedFocus={pinnedFocus}
              onFocusChange={onFocusChange}
              onTogglePin={onTogglePin}
              items={[
                {
                  focus: "goal-probability",
                  symbol: "Pr(.)",
                  description: t("explainer.sections.goals.var.probability"),
                },
                {
                  focus: "goal-target",
                  symbol: "target",
                  description: t("explainer.sections.goals.var.target"),
                },
                {
                  focus: "horizon",
                  symbol: "T*",
                  description: t("explainer.sections.goals.var.horizon", { horizon: result.optimal_horizon ?? "—" }),
                },
              ]}
            />
            <Callout>
              {t("explainer.sections.goals.callout")}
            </Callout>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </section>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border bg-muted/30 px-3 py-2">
      <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-semibold text-foreground">{value}</div>
    </div>
  );
}

function FormulaBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border bg-muted/20 px-4 py-3">
      <div className="min-w-max space-y-2 font-mono text-sm text-foreground">
        {children}
      </div>
    </div>
  );
}

function FormulaLine({
  left,
  right,
}: {
  left: React.ReactNode;
  right: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline gap-3 whitespace-nowrap">
      <span className="font-semibold">{left}</span>
      <span className="text-muted-foreground">=</span>
      <span>{right}</span>
    </div>
  );
}

function FormulaVar({
  base,
  sub,
  sup,
}: {
  base: string;
  sub?: string;
  sup?: string;
}) {
  return (
    <span>
      {base}
      {sub ? <sub>{sub}</sub> : null}
      {sup ? <sup>{sup}</sup> : null}
    </span>
  );
}

function VariableGrid({
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
  items,
}: {
  activeFocus: ExplainerFocus | null;
  pinnedFocus: ExplainerFocus | null;
  onFocusChange: (focus: ExplainerFocus | null) => void;
  onTogglePin: (focus: ExplainerFocus) => void;
  items: { focus: ExplainerFocus; symbol: React.ReactNode; description: string }[];
}) {
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {items.map((item, index) => (
        <FocusToken
          key={index}
          focus={item.focus}
          activeFocus={activeFocus}
          pinnedFocus={pinnedFocus}
          onFocusChange={onFocusChange}
          onTogglePin={onTogglePin}
          className="block rounded-xl border bg-background px-3 py-2 text-left"
        >
          <div className="font-mono text-sm font-semibold text-foreground">{item.symbol}</div>
          <p className="mt-1 text-sm text-muted-foreground">{item.description}</p>
        </FocusToken>
      ))}
    </div>
  );
}

function Callout({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm text-muted-foreground">
      {children}
    </div>
  );
}

function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.9em] text-foreground">
      {children}
    </span>
  );
}

function FocusToken({
  focus,
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
  className,
  children,
}: {
  focus: ExplainerFocus;
  activeFocus: ExplainerFocus | null;
  pinnedFocus: ExplainerFocus | null;
  onFocusChange: (focus: ExplainerFocus | null) => void;
  onTogglePin: (focus: ExplainerFocus) => void;
  className?: string;
  children: React.ReactNode;
}) {
  const active = activeFocus === focus;
  const pinned = pinnedFocus === focus;

  return (
    <button
      type="button"
      aria-pressed={pinned}
      onClick={() => onTogglePin(focus)}
      onMouseEnter={() => onFocusChange(focus)}
      onMouseLeave={() => onFocusChange(null)}
      onFocus={() => onFocusChange(focus)}
      onBlur={() => onFocusChange(null)}
      className={cn(
        "rounded transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
        pinned
          ? "bg-primary/14 text-foreground ring-1 ring-primary/30"
          : active
            ? "bg-primary/10 text-foreground"
            : "hover:bg-muted/70",
        className,
      )}
    >
      {children}
    </button>
  );
}
