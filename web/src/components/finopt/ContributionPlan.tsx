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
import type { Result } from "@/mocks/types";
import { formatCLP, formatCLPCompact, monthLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ExplainerFocus } from "./plan-explainer-focus";
import { isFocusActive } from "./plan-explainer-focus";
import type { WithdrawalMarker } from "./WealthFanChart";

type Props = {
  cashFlow: NonNullable<NonNullable<Result["summary_stats"]>["cash_flow"]>;
  startDate?: string;
  withdrawals?: WithdrawalMarker[];
  activeFocus?: ExplainerFocus | null;
  onFocusChange?: (focus: ExplainerFocus | null) => void;
  pinnedFocus?: ExplainerFocus | null;
  onTogglePin?: (focus: ExplainerFocus) => void;
};

const SERIES_COLORS = ["var(--color-chart-1)", "var(--color-chart-2)", "var(--color-chart-4)", "var(--color-chart-5)"];

export function ContributionPlan({
  cashFlow,
  startDate,
  withdrawals = [],
  activeFocus = null,
  onFocusChange,
  pinnedFocus = null,
  onTogglePin,
}: Props) {
  const [mode, setMode] = useState<"amount" | "percent">("amount");
  const accounts = cashFlow.contributions_by_account;
  const total = cashFlow.contributions_mean;
  const withdrawalsTotal = useMemo(() => cashFlow.withdrawals_mean ?? [], [cashFlow.withdrawals_mean]);
  const withdrawalsByAccount = useMemo(
    () => cashFlow.withdrawals_by_account ?? [],
    [cashFlow.withdrawals_by_account],
  );
  const hasWithdrawals = withdrawalsTotal.some((v) => (v ?? 0) > 0);
  const highlightContrib = isFocusActive(activeFocus, ["contribution", "allocation"]);
  const highlightWithdrawals = isFocusActive(activeFocus, ["withdrawal"]);
  const dimForFocus = activeFocus !== null;
  const setHoverFocus = (focus: ExplainerFocus | null) => onFocusChange?.(focus);
  const togglePin = (focus: ExplainerFocus) => onTogglePin?.(focus);

  const data = useMemo(() => {
    return total.map((_, i) => {
      const row: Record<string, number | string> = {
        month: i,
        label: monthLabel(i, startDate),
        total: total[i],
        withdrawalsTotal: mode === "amount" ? (withdrawalsTotal[i] ?? 0) : 0,
        netFlow: mode === "amount" ? total[i] - (withdrawalsTotal[i] ?? 0) : 0,
      };
      for (const a of accounts) {
        const v = a.mean[i] ?? 0;
        row[a.account] = mode === "amount" ? v : total[i] === 0 ? 0 : v / total[i];
      }
      return row;
    });
  }, [total, accounts, mode, startDate, withdrawalsTotal]);

  // Quick summary numbers
  const monthlyAvg = total.reduce((s, v) => s + v, 0) / Math.max(1, total.length);
  const totalContrib = total.reduce((s, v) => s + v, 0);
  const totalWithdrawn = withdrawalsTotal.reduce((s, v) => s + (v ?? 0), 0);
  const netFlowTotal = totalContrib - totalWithdrawn;

  return (
    <div className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Month-by-month contributions</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {hasWithdrawals
              ? "How much goes into each account each month, and when money is expected to come back out."
              : "How much to put into each account, every month."}
          </p>
        </div>
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          options={[
            { value: "amount", label: "Amount" },
            { value: "percent", label: "Percent" },
          ]}
        />
      </div>

      <dl className={cn("mt-4 grid grid-cols-2 gap-3", hasWithdrawals ? "sm:grid-cols-4" : "sm:grid-cols-3")}>
        <SummaryStat label="Avg per month" value={formatCLP(monthlyAvg)} />
        <SummaryStat label="Total over plan" value={formatCLP(totalContrib)} />
        {hasWithdrawals ? (
          <>
            <SummaryStat label="Total withdrawn" value={formatCLP(totalWithdrawn)} />
            <SummaryStat label="Net cash flow" value={formatCLP(netFlowTotal)} />
          </>
        ) : (
          <SummaryStat label="Accounts in use" value={String(accounts.length)} />
        )}
      </dl>

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
        {accounts.map((a, idx) => (
          <LegendItem
            key={a.account}
            color={SERIES_COLORS[idx % SERIES_COLORS.length]}
            label={a.display_name}
            focus={mode === "amount" ? "contribution" : "allocation"}
            activeFocus={activeFocus}
            pinnedFocus={pinnedFocus}
            onFocusChange={setHoverFocus}
            onTogglePin={togglePin}
          />
        ))}
        {hasWithdrawals && (
          <LegendItem
            color="var(--color-danger)"
            label="Withdrawals"
            dashed
            focus="withdrawal"
            activeFocus={activeFocus}
            pinnedFocus={pinnedFocus}
            onFocusChange={setHoverFocus}
            onTogglePin={togglePin}
          />
        )}
      </div>

      <div className="mt-5 h-[260px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              {accounts.map((a, idx) => (
                <linearGradient key={a.account} id={`grad-${a.account}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={SERIES_COLORS[idx % SERIES_COLORS.length]} stopOpacity={0.55} />
                  <stop offset="100%" stopColor={SERIES_COLORS[idx % SERIES_COLORS.length]} stopOpacity={0.18} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="month"
              tickFormatter={(m) => monthLabel(Number(m), startDate)}
              interval="preserveStartEnd"
              minTickGap={48}
              stroke="var(--color-muted-foreground)"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tickFormatter={(v) =>
                mode === "amount" ? formatCLPCompact(Number(v)) : `${Math.round(Number(v) * 100)}%`
              }
              stroke="var(--color-muted-foreground)"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={56}
              domain={mode === "percent" ? [0, 1] : undefined}
            />
            <Tooltip
              contentStyle={{
                background: "var(--color-popover)",
                border: "1px solid var(--color-border)",
                borderRadius: 12,
                color: "var(--color-popover-foreground)",
                fontSize: 12,
              }}
              labelFormatter={(m) => monthLabel(Number(m), startDate)}
              formatter={(value: number, name: string) => {
                const acc = accounts.find((a) => a.account === name);
                if (name === "withdrawalsTotal") return [formatCLP(Number(value)), "Withdrawals"];
                if (name === "netFlow") return [formatCLP(Number(value)), "Net cash flow"];
                return [
                  mode === "amount" ? formatCLP(Number(value)) : `${(Number(value) * 100).toFixed(0)}%`,
                  acc?.display_name ?? name,
                ];
              }}
            />
            {accounts.map((a, idx) => (
              <Area
                key={a.account}
                type="monotone"
                dataKey={a.account}
                stackId="contrib"
                stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                strokeWidth={highlightContrib ? 2.2 : 1.5}
                fill={`url(#grad-${a.account})`}
                fillOpacity={highlightContrib ? 1 : dimForFocus ? 0.35 : 1}
                opacity={highlightWithdrawals ? 0.45 : 1}
                isAnimationActive={false}
                onClick={() => togglePin(mode === "amount" ? "contribution" : "allocation")}
                onMouseEnter={() => setHoverFocus(mode === "amount" ? "contribution" : "allocation")}
                onMouseLeave={() => setHoverFocus(null)}
              />
            ))}
            {mode === "amount" && hasWithdrawals && (
              <Line
                type="monotone"
                dataKey="withdrawalsTotal"
                stroke="var(--color-danger)"
                strokeWidth={highlightWithdrawals ? 3 : 2}
                strokeDasharray="6 4"
                opacity={dimForFocus && !highlightWithdrawals ? 0.35 : 1}
                dot={false}
                isAnimationActive={false}
                onClick={() => togglePin("withdrawal")}
                onMouseEnter={() => setHoverFocus("withdrawal")}
                onMouseLeave={() => setHoverFocus(null)}
                activeDot={{ r: 4, stroke: "var(--color-danger)", fill: "var(--color-background)", strokeWidth: 2 }}
              />
            )}
            {withdrawals.map((w, idx) => (
              <ReferenceLine
                key={`contrib-withdrawal-${idx}-${w.month}`}
                x={w.month}
                stroke="var(--color-danger)"
                strokeDasharray="4 4"
                strokeWidth={highlightWithdrawals ? 2 : 1}
                strokeOpacity={dimForFocus && !highlightWithdrawals ? 0.35 : 1}
                ifOverflow="extendDomain"
                label={
                  mode === "amount"
                    ? {
                        value: w.label,
                        position: "insideTopLeft",
                        fill: "var(--color-danger)",
                        fontSize: 10,
                        fontWeight: 600,
                      }
                    : undefined
                }
                onClick={() => togglePin("withdrawal")}
                onMouseEnter={() => setHoverFocus("withdrawal")}
                onMouseLeave={() => setHoverFocus(null)}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 overflow-x-auto rounded-xl border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-muted-foreground">
            <tr className="text-left">
              <th className="px-4 py-2 font-medium">Account</th>
              <th className="px-4 py-2 text-right font-medium">First month</th>
              <th className="px-4 py-2 text-right font-medium">Last month</th>
              <th className="px-4 py-2 text-right font-medium">Avg / month</th>
              <th className="px-4 py-2 text-right font-medium">Total</th>
              {hasWithdrawals && <th className="px-4 py-2 text-right font-medium">Withdrawn</th>}
              {hasWithdrawals && <th className="px-4 py-2 text-right font-medium">Net</th>}
            </tr>
          </thead>
          <tbody>
            {accounts.map((a, idx) => {
              const first = a.mean[0] ?? 0;
              const last = a.mean[a.mean.length - 1] ?? 0;
              const sum = a.mean.reduce((s, v) => s + v, 0);
              const avg = sum / Math.max(1, a.mean.length);
              const withdrawalsForAccount = withdrawalsByAccount.find((item) => item.account === a.account)?.mean ?? [];
              const withdrawn = withdrawalsForAccount.reduce((s, v) => s + v, 0);
              return (
                <tr key={a.account} className="border-t">
                  <td className="px-4 py-2.5">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="inline-block size-2.5 rounded-sm"
                        style={{ background: SERIES_COLORS[idx % SERIES_COLORS.length] }}
                        aria-hidden
                      />
                      <span className="font-medium text-foreground">{a.display_name}</span>
                    </span>
                  </td>
                  <td className="tabular px-4 py-2.5 text-right">{formatCLP(first)}</td>
                  <td className="tabular px-4 py-2.5 text-right">{formatCLP(last)}</td>
                  <td className="tabular px-4 py-2.5 text-right">{formatCLP(avg)}</td>
                  <td className="tabular px-4 py-2.5 text-right font-semibold">{formatCLP(sum)}</td>
                  {hasWithdrawals && <td className="tabular px-4 py-2.5 text-right">{formatCLP(withdrawn)}</td>}
                  {hasWithdrawals && <td className="tabular px-4 py-2.5 text-right font-semibold">{formatCLP(sum - withdrawn)}</td>}
                </tr>
              );
            })}
            <tr className="border-t bg-muted/30">
              <td className="px-4 py-2.5 font-medium">Total</td>
              <td className="tabular px-4 py-2.5 text-right">{formatCLP(total[0] ?? 0)}</td>
              <td className="tabular px-4 py-2.5 text-right">{formatCLP(total[total.length - 1] ?? 0)}</td>
              <td className="tabular px-4 py-2.5 text-right">{formatCLP(monthlyAvg)}</td>
              <td className="tabular px-4 py-2.5 text-right font-semibold">{formatCLP(totalContrib)}</td>
              {hasWithdrawals && <td className="tabular px-4 py-2.5 text-right">{formatCLP(totalWithdrawn)}</td>}
              {hasWithdrawals && <td className="tabular px-4 py-2.5 text-right font-semibold">{formatCLP(netFlowTotal)}</td>}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border bg-muted/30 p-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="tabular mt-1 text-lg font-semibold text-foreground">{value}</dd>
    </div>
  );
}

function LegendItem({
  color,
  label,
  dashed,
  focus,
  activeFocus,
  pinnedFocus,
  onFocusChange,
  onTogglePin,
}: {
  color: string;
  label: string;
  dashed?: boolean;
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
        style={{ background: color }}
        aria-hidden
      />
      {label}
    </button>
  );
}

function SegmentedToggle<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <div role="tablist" className="inline-flex rounded-lg border bg-muted/40 p-0.5">
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              active
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
