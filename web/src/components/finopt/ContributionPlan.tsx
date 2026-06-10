import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Result } from "@/mocks/types";
import { formatCLP, formatCLPCompact, monthLabel } from "@/lib/format";
import { cn } from "@/lib/utils";

type Props = {
  cashFlow: NonNullable<NonNullable<Result["summary_stats"]>["cash_flow"]>;
};

const SERIES_COLORS = ["var(--color-chart-1)", "var(--color-chart-2)", "var(--color-chart-4)", "var(--color-chart-5)"];

export function ContributionPlan({ cashFlow }: Props) {
  const [mode, setMode] = useState<"amount" | "percent">("amount");
  const accounts = cashFlow.contributions_by_account;
  const total = cashFlow.contributions_mean;

  const data = useMemo(() => {
    return total.map((_, i) => {
      const row: Record<string, number | string> = {
        month: i,
        label: monthLabel(i),
        total: total[i],
      };
      for (const a of accounts) {
        const v = a.mean[i] ?? 0;
        row[a.account] = mode === "amount" ? v : total[i] === 0 ? 0 : v / total[i];
      }
      return row;
    });
  }, [total, accounts, mode]);

  // Quick summary numbers
  const monthlyAvg = total.reduce((s, v) => s + v, 0) / Math.max(1, total.length);
  const totalContrib = total.reduce((s, v) => s + v, 0);

  return (
    <div className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Month-by-month contributions</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            How much to put into each account, every month.
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

      <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <SummaryStat label="Avg per month" value={formatCLP(monthlyAvg)} />
        <SummaryStat label="Total over plan" value={formatCLP(totalContrib)} />
        <SummaryStat label="Accounts in use" value={String(accounts.length)} />
      </dl>

      <div className="mt-5 h-[260px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
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
              tickFormatter={(m) => monthLabel(Number(m))}
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
              labelFormatter={(m) => monthLabel(Number(m))}
              formatter={(value: number, name: string) => {
                const acc = accounts.find((a) => a.account === name);
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
                strokeWidth={1.5}
                fill={`url(#grad-${a.account})`}
                isAnimationActive={false}
              />
            ))}
          </AreaChart>
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
            </tr>
          </thead>
          <tbody>
            {accounts.map((a, idx) => {
              const first = a.mean[0] ?? 0;
              const last = a.mean[a.mean.length - 1] ?? 0;
              const sum = a.mean.reduce((s, v) => s + v, 0);
              const avg = sum / Math.max(1, a.mean.length);
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
                </tr>
              );
            })}
            <tr className="border-t bg-muted/30">
              <td className="px-4 py-2.5 font-medium">Total</td>
              <td className="tabular px-4 py-2.5 text-right">{formatCLP(total[0] ?? 0)}</td>
              <td className="tabular px-4 py-2.5 text-right">{formatCLP(total[total.length - 1] ?? 0)}</td>
              <td className="tabular px-4 py-2.5 text-right">{formatCLP(monthlyAvg)}</td>
              <td className="tabular px-4 py-2.5 text-right font-semibold">{formatCLP(totalContrib)}</td>
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
