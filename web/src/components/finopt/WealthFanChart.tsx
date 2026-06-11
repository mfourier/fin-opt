import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Line,
} from "recharts";
import type { AccountWealthSeries, WealthPercentiles } from "@/mocks/types";
import { formatCLP, formatCLPCompact, monthLabel } from "@/lib/format";
import { cn } from "@/lib/utils";

// Same palette as ContributionPlan so each account keeps its color across cards.
const SERIES_COLORS = ["var(--color-chart-1)", "var(--color-chart-2)", "var(--color-chart-4)", "var(--color-chart-5)"];

export type GoalLine = {
  account: string;
  threshold: number;
  label: string;
};

type Props = {
  /** Total (sum of accounts) percentiles. */
  percentiles: WealthPercentiles;
  /** Per-account percentiles (summary_stats.per_account). Optional for old results. */
  accounts?: AccountWealthSeries[];
  /** Goal thresholds to draw as dashed lines, colored per account. */
  goals?: GoalLine[];
};

/** "accounts" = all account medians; "total" = fan of the sum; otherwise an account slug. */
type View = string;

export function WealthFanChart({ percentiles, accounts = [], goals = [] }: Props) {
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
      label: monthLabel(i),
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
  }, [fanSource]);

  // "All accounts" view: one median series per account.
  const accountsData = useMemo(() => {
    if (!hasAccounts) return [];
    const n = accounts[0].p50.length;
    return Array.from({ length: n }, (_, i) => {
      const row: Record<string, number | string> = { month: i, label: monthLabel(i) };
      for (const a of accounts) row[a.account] = a.p50[i];
      return row;
    });
  }, [accounts, hasAccounts]);

  // Goal lines visible in the current view. The total view sums accounts, so
  // per-account thresholds would be misleading there — goals show in account views.
  const visibleGoals =
    view === "accounts" ? goals : selected ? goals.filter((g) => g.account === selected.account) : [];

  const fanColor = selected ? colorOf(selected.account) : "var(--color-primary)";

  const subtitle =
    view === "total"
      ? "Likely range of your total savings over time. Darker band = more likely."
      : view === "accounts"
        ? "Median projected wealth for each account. Dashed lines mark your goals."
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
          <AccountsLegend accounts={accounts} colorOf={colorOf} showGoal={visibleGoals.length > 0} />
        ) : (
          <FanLegend color={fanColor} showGoal={visibleGoals.length > 0} />
        )}
      </div>

      <div className="mt-4 h-[320px] w-full sm:h-[380px]">
        <ResponsiveContainer width="100%" height="100%">
          {view === "accounts" ? (
            <AreaChart data={accountsData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="month"
                tickFormatter={(m) => monthLabel(Number(m))}
                interval="preserveStartEnd"
                minTickGap={48}
                {...axisProps}
              />
              <YAxis tickFormatter={(v) => formatCLPCompact(Number(v))} width={56} {...axisProps} />
              <Tooltip
                cursor={{ stroke: "var(--color-muted-foreground)", strokeDasharray: "3 3" }}
                contentStyle={tooltipStyle}
                labelFormatter={(m) => monthLabel(Number(m))}
                formatter={(value: number, name: string) => {
                  const acc = accounts.find((a) => a.account === name);
                  return [formatCLP(Number(value)), acc?.display_name ?? name];
                }}
              />
              {accounts.map((a) => (
                <Line
                  key={a.account}
                  type="monotone"
                  dataKey={a.account}
                  stroke={colorOf(a.account)}
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                  activeDot={{ r: 4, stroke: colorOf(a.account), fill: "var(--color-background)", strokeWidth: 2 }}
                />
              ))}
              {visibleGoals.map((g) => (
                <ReferenceLine
                  key={`${g.account}-${g.threshold}`}
                  y={g.threshold}
                  stroke={colorOf(g.account)}
                  strokeDasharray="6 4"
                  strokeWidth={1.5}
                  ifOverflow="extendDomain"
                  label={{
                    value: `${g.label}: ${formatCLPCompact(g.threshold)}`,
                    position: "insideTopRight",
                    fill: colorOf(g.account),
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                />
              ))}
            </AreaChart>
          ) : (
            <AreaChart data={fanData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
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
                tickFormatter={(m) => monthLabel(Number(m))}
                interval="preserveStartEnd"
                minTickGap={48}
                {...axisProps}
              />
              <YAxis tickFormatter={(v) => formatCLPCompact(Number(v))} width={56} {...axisProps} />

              <Tooltip
                cursor={{ stroke: "var(--color-muted-foreground)", strokeDasharray: "3 3" }}
                contentStyle={tooltipStyle}
                labelFormatter={(m) => monthLabel(Number(m))}
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
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
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
                isAnimationActive={false}
                activeDot={false}
                legendType="none"
              />

              {/* Hidden series for tooltip readouts */}
              <Line type="monotone" dataKey="p10" stroke="transparent" dot={false} isAnimationActive={false} activeDot={false} />
              <Line type="monotone" dataKey="p25" stroke="transparent" dot={false} isAnimationActive={false} activeDot={false} />
              <Line type="monotone" dataKey="p75" stroke="transparent" dot={false} isAnimationActive={false} activeDot={false} />
              <Line type="monotone" dataKey="p90" stroke="transparent" dot={false} isAnimationActive={false} activeDot={false} />

              {/* Median */}
              <Line
                type="monotone"
                dataKey="p50"
                stroke={fanColor}
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
                activeDot={{ r: 4, stroke: fanColor, fill: "var(--color-background)", strokeWidth: 2 }}
              />

              {visibleGoals.map((g) => (
                <ReferenceLine
                  key={`${g.account}-${g.threshold}`}
                  y={g.threshold}
                  stroke="var(--color-success)"
                  strokeDasharray="6 4"
                  strokeWidth={1.5}
                  ifOverflow="extendDomain"
                  label={{
                    value: `Goal: ${formatCLPCompact(g.threshold)}`,
                    position: "insideTopRight",
                    fill: "var(--color-success)",
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                />
              ))}
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function FanLegend({ color, showGoal }: { color: string; showGoal: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      <LegendItem color={color} label="Median" />
      <LegendItem color={color} opacity={0.3} label="Likely range (P25–P75)" />
      <LegendItem color={color} opacity={0.15} label="Possible range (P10–P90)" />
      {showGoal && <LegendItem color="var(--color-success)" label="Goal" dashed />}
    </div>
  );
}

function AccountsLegend({
  accounts,
  colorOf,
  showGoal,
}: {
  accounts: AccountWealthSeries[];
  colorOf: (slug: string) => string;
  showGoal: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      {accounts.map((a) => (
        <LegendItem key={a.account} color={colorOf(a.account)} label={a.display_name} />
      ))}
      {showGoal && <LegendItem color="var(--color-muted-foreground)" label="Goal" dashed />}
    </div>
  );
}

function LegendItem({
  color,
  label,
  dashed,
  opacity = 1,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  opacity?: number;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={cn(
          "inline-block h-2 w-4 rounded-sm",
          dashed && "[mask-image:repeating-linear-gradient(90deg,#000_0_4px,transparent_4px_7px)]",
        )}
        style={{ background: color, opacity }}
        aria-hidden
      />
      {label}
    </span>
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
