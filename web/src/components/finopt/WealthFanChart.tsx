import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AccountWealthSeries, WealthPercentiles } from "@/mocks/types";
import { formatCLP, formatCLPCompact, monthLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ExplainerFocus } from "./plan-explainer-focus";
import { isFocusActive } from "./plan-explainer-focus";

// Same palette as ContributionPlan so each account keeps its color across cards.
const SERIES_COLORS = ["var(--color-chart-1)", "var(--color-chart-2)", "var(--color-chart-4)", "var(--color-chart-5)"];

export type GoalLine = {
  account: string;
  threshold: number;
  label: string;
};

export type WithdrawalMarker = {
  month: number;
  label: string;
};

type Props = {
  /** Total (sum of accounts) percentiles. */
  percentiles: WealthPercentiles;
  /** Per-account percentiles (summary_stats.per_account). Optional for old results. */
  accounts?: AccountWealthSeries[];
  /** Goal thresholds to draw as dashed lines, colored per account. */
  goals?: GoalLine[];
  /** Plan anchor date for x-axis month labels. */
  startDate?: string;
  /** Dated withdrawals shown as vertical markers. */
  withdrawals?: WithdrawalMarker[];
  activeFocus?: ExplainerFocus | null;
  pinnedFocus?: ExplainerFocus | null;
  onFocusChange?: (focus: ExplainerFocus | null) => void;
  onTogglePin?: (focus: ExplainerFocus) => void;
};

/** "accounts" = all account medians; "total" = fan of the sum; otherwise an account slug. */
type View = string;

export function WealthFanChart({
  percentiles,
  accounts = [],
  goals = [],
  startDate,
  withdrawals = [],
  activeFocus = null,
  pinnedFocus = null,
  onFocusChange,
  onTogglePin,
}: Props) {
  const hasAccounts = accounts.length > 0;
  const [view, setView] = useState<View>(hasAccounts ? "accounts" : "total");

  const colorOf = (slug: string) => {
    const idx = accounts.findIndex((a) => a.account === slug);
    return SERIES_COLORS[(idx === -1 ? 0 : idx) % SERIES_COLORS.length];
  };

  const selected = view !== "accounts" && view !== "total"
    ? accounts.find((a) => a.account === view)
    : undefined;

  // Fan views (total / single account) share the band-trick row shape.
  const fanSource: WealthPercentiles | undefined =
    view === "total" ? percentiles : selected;

  const fanData = useMemo(() => {
    if (!fanSource) return [];
    return fanSource.p50.map((_, i) => ({
      month: i,
      label: monthLabel(i, startDate),
      p10: fanSource.p10[i],
      p25: fanSource.p25[i],
      p50: fanSource.p50[i],
      p75: fanSource.p75[i],
      p90: fanSource.p90[i],
      // for stacked-area "bands":
      band1090Lower: fanSource.p10[i],
      band1090Width: fanSource.p90[i] - fanSource.p10[i],
      band2575Lower: fanSource.p25[i],
      band2575Width: fanSource.p75[i] - fanSource.p25[i],
    }));
  }, [fanSource, startDate]);

  // "All accounts" view: median series per account plus its P25–P75 band
  // (same stacked-area trick as the fan views, one stack per account).
  const accountsData = useMemo(() => {
    if (!hasAccounts) return [];
    const n = Math.max(...accounts.map((a) => a.p50.length));
    return Array.from({ length: n }, (_, i) => {
      const row: Record<string, number | string> = { month: i, label: monthLabel(i, startDate) };
      for (const a of accounts) {
        const p50 = a.p50[i] ?? 0;
        const p25 = a.p25?.[i] ?? p50;
        const p75 = a.p75?.[i] ?? p50;
        row[a.account] = p50;
        row[`${a.account}__bandLower`] = p25;
        row[`${a.account}__bandWidth`] = Math.max(0, p75 - p25);
      }
      return row;
    });
  }, [accounts, hasAccounts, startDate]);

  // Goal lines visible in the current view. The total view sums accounts, so
  // per-account thresholds would be misleading there — goals show in account views.
  const visibleGoals =
    view === "accounts" ? goals : selected ? goals.filter((g) => g.account === selected.account) : [];

  const fanColor = selected ? colorOf(selected.account) : "var(--color-primary)";

  const subtitle =
    view === "total"
      ? "Likely range of your total savings over time. Darker band = more likely."
      : view === "accounts"
        ? "Median line for each account with its likely range (P25–P75) shaded. Dashed lines show your goals."
        : `Likely range for ${selected?.display_name ?? view}. Darker band = more likely.`;

  const tooltipStyle = {
    background: "var(--color-popover)",
    border: "1px solid var(--color-border)",
    borderRadius: 12,
    color: "var(--color-popover-foreground)",
    fontSize: 12,
    boxShadow: "0 10px 30px -10px oklch(0 0 0 / 0.25)",
  } as const;

  const axisProps = {
    stroke: "var(--color-muted-foreground)",
    tick: { fontSize: 11 },
    tickLine: false,
    axisLine: false,
  } as const;

  const highlightWealth = isFocusActive(activeFocus, ["wealth", "contribution", "allocation", "return", "median"]);
  const highlightGoals = isFocusActive(activeFocus, ["goal-target", "goal-probability"]);
  const highlightWithdrawals = isFocusActive(activeFocus, ["withdrawal"]);
  const highlightLikelyBand = isFocusActive(activeFocus, ["likely-band"]);
  const highlightPossibleBand = isFocusActive(activeFocus, ["possible-band"]);
  const dimForFocus = activeFocus !== null;
  const setHoverFocus = (focus: ExplainerFocus | null) => onFocusChange?.(focus);
  const togglePin = (focus: ExplainerFocus) => onTogglePin?.(focus);

  return (
    <div className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Projected wealth</h2>
          <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
        </div>
        {hasAccounts && (
          <SegmentedToggle
            value={view}
            onChange={setView}
            options={[
              { value: "accounts", label: "All accounts" },
              { value: "total", label: "Total" },
              ...accounts.map((a) => ({ value: a.account, label: a.display_name })),
            ]}
          />
        )}
      </div>

      <div className="mt-2">
        {view === "accounts" ? (
          <AccountsLegend
            accounts={accounts}
            colorOf={colorOf}
            showGoal={visibleGoals.length > 0}
            showWithdrawal={withdrawals.length > 0}
            activeFocus={activeFocus}
            pinnedFocus={pinnedFocus}
            onFocusChange={setHoverFocus}
            onTogglePin={togglePin}
          />
        ) : (
          <FanLegend
            color={fanColor}
            showGoal={visibleGoals.length > 0}
            showWithdrawal={withdrawals.length > 0}
            activeFocus={activeFocus}
            pinnedFocus={pinnedFocus}
            onFocusChange={setHoverFocus}
            onTogglePin={togglePin}
          />
        )}
      </div>

      <div className="mt-4 h-[320px] w-full sm:h-[380px]">
        <ResponsiveContainer width="100%" height="100%">
          {view === "accounts" ? (
            <ComposedChart data={accountsData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="month"
                tickFormatter={(m) => monthLabel(Number(m), startDate)}
                interval="preserveStartEnd"
                minTickGap={48}
                {...axisProps}
              />
              <YAxis tickFormatter={(v) => formatCLPCompact(Number(v))} width={56} {...axisProps} />
              <Tooltip
                cursor={{ stroke: "var(--color-muted-foreground)", strokeDasharray: "3 3" }}
                contentStyle={tooltipStyle}
                labelFormatter={(m) => monthLabel(Number(m), startDate)}
                formatter={(value: number, name: string) => {
                  const acc = accounts.find((a) => a.account === name);
                  return [formatCLP(Number(value)), acc?.display_name ?? name];
                }}
              />
              {/* P25–P75 band per account (stacked-area trick: invisible
                  baseline at P25 + filled width up to P75). Low-volatility
                  accounts have a band only a few px tall on the shared axis, so
                  we also stroke each edge (P25 on the baseline area, P75 on the
                  width area) — that keeps a thin band perceptible instead of
                  vanishing under the 2px median line, without inflating it. */}
              {accounts.map((a) => (
                <Area
                  key={`${a.account}-band-lower`}
                  type="monotone"
                  dataKey={`${a.account}__bandLower`}
                  stackId={`band-${a.account}`}
                  stroke={colorOf(a.account)}
                  strokeWidth={highlightLikelyBand ? 1.25 : 1}
                  strokeOpacity={highlightLikelyBand ? 0.7 : dimForFocus ? 0.25 : 0.45}
                  fill="transparent"
                  isAnimationActive={false}
                  activeDot={false}
                  legendType="none"
                  tooltipType="none"
                />
              ))}
              {accounts.map((a) => (
                <Area
                  key={`${a.account}-band-width`}
                  type="monotone"
                  dataKey={`${a.account}__bandWidth`}
                  stackId={`band-${a.account}`}
                  stroke={colorOf(a.account)}
                  strokeWidth={highlightLikelyBand ? 1.25 : 1}
                  strokeOpacity={highlightLikelyBand ? 0.7 : dimForFocus ? 0.25 : 0.45}
                  fill={colorOf(a.account)}
                  fillOpacity={highlightLikelyBand ? 0.32 : dimForFocus ? 0.08 : 0.16}
                  isAnimationActive={false}
                  activeDot={false}
                  legendType="none"
                  tooltipType="none"
                  onClick={() => togglePin("likely-band")}
                  onMouseEnter={() => setHoverFocus("likely-band")}
                  onMouseLeave={() => setHoverFocus(null)}
                />
              ))}
              {/* Median line per account, drawn above the bands. */}
              {accounts.map((a) => (
                <Line
                  key={a.account}
                  type="monotone"
                  dataKey={a.account}
                  stroke={colorOf(a.account)}
                  strokeWidth={highlightWealth ? 2.8 : 2}
                  opacity={highlightWithdrawals || highlightGoals ? 0.45 : dimForFocus && !highlightWealth ? 0.55 : 1}
                  dot={false}
                  isAnimationActive={false}
                  onClick={() => togglePin("wealth")}
                  onMouseEnter={() => setHoverFocus("wealth")}
                  onMouseLeave={() => setHoverFocus(null)}
                  activeDot={{ r: 4, stroke: colorOf(a.account), fill: "var(--color-background)", strokeWidth: 2 }}
                />
              ))}
              {visibleGoals.map((g) => (
                <ReferenceLine
                  key={`${g.account}-${g.threshold}`}
                  y={g.threshold}
                  stroke={colorOf(g.account)}
                  strokeDasharray="6 4"
                  strokeWidth={highlightGoals ? 2.5 : 1.5}
                  strokeOpacity={highlightWithdrawals ? 0.35 : dimForFocus && !highlightGoals ? 0.45 : 1}
                  ifOverflow="extendDomain"
                  label={{
                    value: `${g.label}: ${formatCLPCompact(g.threshold)}`,
                    position: "insideTopRight",
                    fill: colorOf(g.account),
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                  onClick={() => togglePin("goal-target")}
                  onMouseEnter={() => setHoverFocus("goal-target")}
                  onMouseLeave={() => setHoverFocus(null)}
                />
              ))}
              {withdrawals.map((w, idx) => (
                <ReferenceLine
                  key={`withdrawal-accounts-${idx}-${w.month}`}
                  x={w.month}
                  stroke="var(--color-danger)"
                  strokeDasharray="4 4"
                  strokeWidth={highlightWithdrawals ? 2 : 1}
                  strokeOpacity={dimForFocus && !highlightWithdrawals ? 0.35 : 1}
                  ifOverflow="extendDomain"
                  label={{
                    value: w.label,
                    position: "insideTopLeft",
                    fill: "var(--color-danger)",
                    fontSize: 10,
                    fontWeight: 600,
                  }}
                  onClick={() => togglePin("withdrawal")}
                  onMouseEnter={() => setHoverFocus("withdrawal")}
                  onMouseLeave={() => setHoverFocus(null)}
                />
              ))}
            </ComposedChart>
          ) : (
            <ComposedChart data={fanData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="band-outer" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={fanColor} stopOpacity={0.18} />
                  <stop offset="100%" stopColor={fanColor} stopOpacity={0.06} />
                </linearGradient>
                <linearGradient id="band-inner" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={fanColor} stopOpacity={0.32} />
                  <stop offset="100%" stopColor={fanColor} stopOpacity={0.14} />
                </linearGradient>
              </defs>

              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="month"
                tickFormatter={(m) => monthLabel(Number(m), startDate)}
                interval="preserveStartEnd"
                minTickGap={48}
                {...axisProps}
              />
              <YAxis tickFormatter={(v) => formatCLPCompact(Number(v))} width={56} {...axisProps} />

              <Tooltip
                cursor={{ stroke: "var(--color-muted-foreground)", strokeDasharray: "3 3" }}
                contentStyle={tooltipStyle}
                labelFormatter={(m) => monthLabel(Number(m), startDate)}
                formatter={(value: number, name: string) => {
                  const labels: Record<string, string> = {
                    p50: "Median (P50)",
                    p10: "Pessimistic (P10)",
                    p90: "Optimistic (P90)",
                    p25: "P25",
                    p75: "P75",
                  };
                  return [formatCLP(Number(value)), labels[name] ?? name];
                }}
                itemSorter={(item) => {
                  const order = ["p90", "p75", "p50", "p25", "p10"];
                  return order.indexOf(String(item.dataKey));
                }}
              />

              {/* P10–P90 outer band (stacked-area trick) */}
              <Area
                type="monotone"
                dataKey="band1090Lower"
                stackId="outer"
                stroke="transparent"
                fill="transparent"
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
              />
              <Area
                type="monotone"
                dataKey="band1090Width"
                stackId="outer"
                stroke="transparent"
                fill="url(#band-outer)"
                fillOpacity={highlightPossibleBand ? 1 : dimForFocus && !highlightLikelyBand ? 0.45 : 1}
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
                onClick={() => togglePin("possible-band")}
                onMouseEnter={() => setHoverFocus("possible-band")}
                onMouseLeave={() => setHoverFocus(null)}
              />

              {/* P25–P75 inner band */}
              <Area
                type="monotone"
                dataKey="band2575Lower"
                stackId="inner"
                stroke="transparent"
                fill="transparent"
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
              />
              <Area
                type="monotone"
                dataKey="band2575Width"
                stackId="inner"
                stroke="transparent"
                fill="url(#band-inner)"
                fillOpacity={highlightLikelyBand ? 1 : dimForFocus && !highlightPossibleBand ? 0.45 : 1}
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
                onClick={() => togglePin("likely-band")}
                onMouseEnter={() => setHoverFocus("likely-band")}
                onMouseLeave={() => setHoverFocus(null)}
              />

              {/* Hidden series for tooltip readouts */}
              <Area type="monotone" dataKey="p10" stroke="transparent" fill="transparent" isAnimationActive={false} activeDot={false} legendType="none" />
              <Area type="monotone" dataKey="p25" stroke="transparent" fill="transparent" isAnimationActive={false} activeDot={false} legendType="none" />
              <Area type="monotone" dataKey="p75" stroke="transparent" fill="transparent" isAnimationActive={false} activeDot={false} legendType="none" />
              <Area type="monotone" dataKey="p90" stroke="transparent" fill="transparent" isAnimationActive={false} activeDot={false} legendType="none" />

              {/* Median */}
              <Line
                type="monotone"
                dataKey="p50"
                stroke={fanColor}
                strokeWidth={highlightWealth ? 3.5 : 2.5}
                opacity={highlightGoals || highlightWithdrawals ? 0.45 : 1}
                dot={false}
                isAnimationActive={false}
                onClick={() => togglePin("median")}
                onMouseEnter={() => setHoverFocus("median")}
                onMouseLeave={() => setHoverFocus(null)}
                activeDot={{ r: 4, stroke: fanColor, fill: "var(--color-background)", strokeWidth: 2 }}
              />

              {visibleGoals.map((g) => (
                <ReferenceLine
                  key={`${g.account}-${g.threshold}`}
                  y={g.threshold}
                  stroke="var(--color-success)"
                  strokeDasharray="6 4"
                  strokeWidth={highlightGoals ? 2.5 : 1.5}
                  strokeOpacity={highlightWithdrawals ? 0.35 : dimForFocus && !highlightGoals ? 0.45 : 1}
                  ifOverflow="extendDomain"
                  label={{
                    value: `Goal: ${formatCLPCompact(g.threshold)}`,
                    position: "insideTopRight",
                    fill: "var(--color-success)",
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                  onClick={() => togglePin("goal-target")}
                  onMouseEnter={() => setHoverFocus("goal-target")}
                  onMouseLeave={() => setHoverFocus(null)}
                />
              ))}
              {withdrawals.map((w, idx) => (
                <ReferenceLine
                  key={`withdrawal-fan-${idx}-${w.month}`}
                  x={w.month}
                  stroke="var(--color-danger)"
                  strokeDasharray="4 4"
                  strokeWidth={highlightWithdrawals ? 2 : 1}
                  strokeOpacity={dimForFocus && !highlightWithdrawals ? 0.35 : 1}
                  ifOverflow="extendDomain"
                  label={{
                    value: w.label,
                    position: "insideTopLeft",
                    fill: "var(--color-danger)",
                    fontSize: 10,
                    fontWeight: 600,
                  }}
                  onClick={() => togglePin("withdrawal")}
                  onMouseEnter={() => setHoverFocus("withdrawal")}
                  onMouseLeave={() => setHoverFocus(null)}
                />
              ))}
            </ComposedChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function FanLegend({
  color,
  showGoal,
  showWithdrawal,
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
}: {
  color: string;
  showGoal: boolean;
  showWithdrawal: boolean;
  activeFocus: ExplainerFocus | null;
  pinnedFocus: ExplainerFocus | null;
  onFocusChange: (focus: ExplainerFocus | null) => void;
  onTogglePin: (focus: ExplainerFocus) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      <LegendItem color={color} label="Median" focus="median" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />
      <LegendItem color={color} opacity={0.3} label="Likely range (P25–P75)" focus="likely-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />
      <LegendItem color={color} opacity={0.15} label="Possible range (P10–P90)" focus="possible-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />
      {showGoal && <LegendItem color="var(--color-success)" label="Goal" dashed focus="goal-target" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />}
      {showWithdrawal && <LegendItem color="var(--color-danger)" label="Withdrawal" dashed focus="withdrawal" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />}
    </div>
  );
}

function AccountsLegend({
  accounts,
  colorOf,
  showGoal,
  showWithdrawal,
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
}: {
  accounts: AccountWealthSeries[];
  colorOf: (slug: string) => string;
  showGoal: boolean;
  showWithdrawal: boolean;
  activeFocus: ExplainerFocus | null;
  pinnedFocus: ExplainerFocus | null;
  onFocusChange: (focus: ExplainerFocus | null) => void;
  onTogglePin: (focus: ExplainerFocus) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      {accounts.map((a) => (
        <LegendItem key={a.account} color={colorOf(a.account)} label={a.display_name} focus="wealth" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />
      ))}
      <LegendItem color="var(--color-muted-foreground)" opacity={0.35} label="Likely range (P25–P75)" focus="likely-band" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />
      {showGoal && <LegendItem color="var(--color-muted-foreground)" label="Goal" dashed focus="goal-target" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />}
      {showWithdrawal && <LegendItem color="var(--color-danger)" label="Withdrawal" dashed focus="withdrawal" activeFocus={activeFocus} pinnedFocus={pinnedFocus} onFocusChange={onFocusChange} onTogglePin={onTogglePin} />}
    </div>
  );
}

function LegendItem({
  color,
  label,
  dashed,
  opacity = 1,
  focus,
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  opacity?: number;
  focus: ExplainerFocus;
  activeFocus: ExplainerFocus | null;
  pinnedFocus: ExplainerFocus | null;
  onFocusChange: (focus: ExplainerFocus | null) => void;
  onTogglePin: (focus: ExplainerFocus) => void;
}) {
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
        "inline-flex items-center gap-1.5 rounded px-1 py-0.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40",
        pinned ? "bg-primary/14 text-foreground ring-1 ring-primary/30" : activeFocus === focus ? "bg-primary/10 text-foreground" : "hover:bg-muted/60",
      )}
    >
      <span
        className={cn(
          "inline-block h-2 w-4 rounded-sm",
          dashed && "[mask-image:repeating-linear-gradient(90deg,#000_0_4px,transparent_4px_7px)]",
        )}
        style={{ background: color, opacity }}
        aria-hidden
      />
      {label}
    </button>
  );
}

function SegmentedToggle({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div role="tablist" className="inline-flex flex-wrap rounded-lg border bg-muted/40 p-0.5">
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "max-w-[10rem] truncate rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              active
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
            title={opt.label}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
