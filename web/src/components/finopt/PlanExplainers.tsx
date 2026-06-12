import { useEffect, useMemo, useState } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import type { Profile, Result, Scenario } from "@/mocks/types";
import { formatPercent } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ExplainerFocus } from "./plan-explainer-focus";
import { focusToLabel, focusToSection, type ExplainerSection } from "./plan-explainer-focus";

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
  const goalsCount = scenario.terminal_goals.length + scenario.intermediate_goals.length;
  const confidenceValues = [
    ...scenario.terminal_goals.map((goal) => goal.confidence),
    ...scenario.intermediate_goals.map((goal) => goal.confidence),
  ];
  const confidenceFloor = confidenceValues.length > 0 ? Math.min(...confidenceValues) : null;
  const [openItems, setOpenItems] = useState<ExplainerSection[]>([]);
  const activeSection = focusToSection(activeFocus);
  const pinnedSection = focusToSection(pinnedFocus);
  const activeLabel = focusToLabel(activeFocus);

  // Only auto-open a section when the user *pins* a concept (a click), not on
  // hover — hovering still highlights the matching header/token, but the panel
  // should not expand on its own as the mouse moves across the charts.
  useEffect(() => {
    if (!pinnedSection) return;
    setOpenItems((current) => (current.includes(pinnedSection) ? current : [...current, pinnedSection]));
  }, [pinnedSection]);

  const spotlightText = useMemo(() => {
    if (!activeFocus || !activeLabel) return null;
    if (activeSection === "wealth") return `${activeLabel} is active in the wealth equation and cash-flow visuals.`;
    if (activeSection === "bands") return `${activeLabel} is active in the projected wealth distribution.`;
    return `${activeLabel} is active in the goal logic for this plan.`;
  }, [activeFocus, activeLabel, activeSection]);

  return (
    <section className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">How this plan works</h2>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            The math is here if you want it, but each section translates the equation into plain language first.
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            Hover or focus a symbol to highlight the matching part of the charts below. Click to pin it.
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
              {activeLabel ? `Linked concept: ${activeLabel}` : "Linked concept: hover a formula or chart element"}
            </span>
            {spotlightText ? <span className="ml-2 hidden text-muted-foreground sm:inline">{spotlightText}</span> : null}
            {pinnedFocus ? (
              <button
                type="button"
                onClick={onClearPin}
                className="ml-3 rounded-full border border-primary/25 px-2 py-0.5 text-[11px] font-medium text-foreground transition-colors hover:bg-primary/10"
              >
                Clear pin
              </button>
            ) : null}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-xs sm:min-w-[18rem]">
          <MiniStat label="Accounts" value={String(profile.accounts_config.length)} />
          <MiniStat label="Goals" value={String(goalsCount)} />
          <MiniStat
            label="Withdrawals"
            value={hasWithdrawals ? "Included" : "None"}
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
            Wealth evolution equation
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Each month, every account starts from its current balance, adds the planned contribution,
              subtracts any withdrawal, and then grows by that month's investment return.
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
                  description: "wealth in account m at month t",
                },
                {
                  focus: "contribution",
                  symbol: <FormulaVar base="A" sub="t" />,
                  description: "money available to invest in month t",
                },
                {
                  focus: "allocation",
                  symbol: <FormulaVar base="x" sub="t" sup="m" />,
                  description: "share of that month's contribution sent to account m",
                },
                {
                  focus: "withdrawal",
                  symbol: <FormulaVar base="D" sub="t" sup="m" />,
                  description: hasWithdrawals
                    ? "withdrawal taken from account m in month t"
                    : "withdrawal term; zero in this scenario",
                },
                {
                  focus: "return",
                  symbol: <FormulaVar base="R" sub="t" sup="m" />,
                  description: "return earned by account m during month t",
                },
              ]}
            />
            <Callout>
              In this plan, the optimizer is choosing the monthly split{" "}
              <FocusToken focus="allocation" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>
                <InlineCode>x</InlineCode>
              </FocusToken>{" "}
              across{" "}
              {profile.accounts_config.length} accounts while respecting your goals, withdrawals, and time horizon.
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
            Confidence bands and uncertainty
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              The projected wealth chart is not one future. It summarizes many simulated futures and shows where
              the middle of the distribution sits and how wide the range becomes over time.
            </p>
            <FormulaBlock>
              <FormulaLine
                left={<FocusToken focus="median" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>P50</FocusToken>}
                right="median path of simulated wealth"
              />
              <FormulaLine
                left={<FocusToken focus="likely-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>P25 - P75</FocusToken>}
                right="likely middle range"
              />
              <FormulaLine
                left={<FocusToken focus="possible-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>P10 - P90</FocusToken>}
                right="wider possible range"
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
                  description: "half of simulated outcomes are above it and half are below it",
                },
                {
                  focus: "likely-band",
                  symbol: "P25-P75",
                  description: "the darker band; a tighter middle slice of outcomes",
                },
                {
                  focus: "possible-band",
                  symbol: "P10-P90",
                  description: "the lighter band; a broader uncertainty envelope",
                },
              ]}
            />
            <Callout>
              Wider bands usually mean more uncertainty from returns, variable income, or withdrawals. Narrower
              bands mean outcomes are clustering more tightly.
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
            Goal probability and minimum horizon
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              FinOpt searches for the shortest horizon where your goals become achievable with the confidence
              levels you selected.
            </p>
            <FormulaBlock>
              <FormulaLine
                left={<FocusToken focus="goal-probability" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>Pr(W_t^m &gt;= target)</FocusToken>}
                right={confidenceFloor !== null ? `>= ${formatPercent(confidenceFloor)}` : ">= chosen confidence"}
              />
              <FormulaLine
                left={<FocusToken focus="horizon" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin}>T*</FocusToken>}
                right="smallest month count that satisfies all goals together"
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
                  description: "probability measured across simulated futures",
                },
                {
                  focus: "goal-target",
                  symbol: "target",
                  description: "the threshold attached to a goal",
                },
                {
                  focus: "horizon",
                  symbol: "T*",
                  description: `the minimum feasible horizon; current result = ${result.optimal_horizon ?? "—"} months`,
                },
              ]}
            />
            <Callout>
              The goal list below compares the achieved probability against each goal's required confidence.
              If one goal forces the plan to wait longer, it can determine the minimum horizon for the full plan.
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
