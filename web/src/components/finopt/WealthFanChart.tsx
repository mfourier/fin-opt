import { useMemo } from "react";
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
import type { WealthPercentiles } from "@/mocks/types";
import { formatCLP, formatCLPCompact, monthLabel } from "@/lib/format";

type Props = {
  percentiles: WealthPercentiles;
  goalThreshold?: number;
  goalLabel?: string;
};

export function WealthFanChart({ percentiles, goalThreshold, goalLabel }: Props) {
  const data = useMemo(() => {
    return percentiles.p50.map((_, i) => ({
      month: i,
      label: monthLabel(i),
      p10: percentiles.p10[i],
      p25: percentiles.p25[i],
      p50: percentiles.p50[i],
      p75: percentiles.p75[i],
      p90: percentiles.p90[i],
      // for stacked-area "bands":
      band1090Lower: percentiles.p10[i],
      band1090Width: percentiles.p90[i] - percentiles.p10[i],
      band2575Lower: percentiles.p25[i],
      band2575Width: percentiles.p75[i] - percentiles.p25[i],
    }));
  }, [percentiles]);

  return (
    <div className="rounded-2xl border bg-card p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Projected wealth</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Likely range of your total savings over time. Darker band = more likely.
          </p>
        </div>
        <Legend />
      </div>

      <div className="mt-4 h-[320px] w-full sm:h-[380px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="band-outer" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-primary)" stopOpacity={0.18} />
                <stop offset="100%" stopColor="var(--color-primary)" stopOpacity={0.06} />
              </linearGradient>
              <linearGradient id="band-inner" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-primary)" stopOpacity={0.32} />
                <stop offset="100%" stopColor="var(--color-primary)" stopOpacity={0.14} />
              </linearGradient>
            </defs>

            <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="month"
              tickFormatter={(m) => monthLabel(m)}
              interval="preserveStartEnd"
              minTickGap={48}
              stroke="var(--color-muted-foreground)"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tickFormatter={(v) => formatCLPCompact(Number(v))}
              stroke="var(--color-muted-foreground)"
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={56}
            />

            <Tooltip
              cursor={{ stroke: "var(--color-muted-foreground)", strokeDasharray: "3 3" }}
              contentStyle={{
                background: "var(--color-popover)",
                border: "1px solid var(--color-border)",
                borderRadius: 12,
                color: "var(--color-popover-foreground)",
                fontSize: 12,
                boxShadow: "0 10px 30px -10px oklch(0 0 0 / 0.25)",
              }}
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
              stroke="var(--color-primary)"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
              activeDot={{ r: 4, stroke: "var(--color-primary)", fill: "var(--color-background)", strokeWidth: 2 }}
            />

            {goalThreshold !== undefined && (
              <ReferenceLine
                y={goalThreshold}
                stroke="var(--color-success)"
                strokeDasharray="6 4"
                strokeWidth={1.5}
                label={{
                  value: goalLabel ?? `Goal: ${formatCLPCompact(goalThreshold)}`,
                  position: "insideTopRight",
                  fill: "var(--color-success)",
                  fontSize: 11,
                  fontWeight: 600,
                }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      <LegendItem swatchClass="bg-primary" label="Median" />
      <LegendItem swatchClass="bg-primary/30" label="Likely range (P25–P75)" />
      <LegendItem swatchClass="bg-primary/15" label="Possible range (P10–P90)" />
      <LegendItem swatchClass="bg-success" label="Goal" dashed />
    </div>
  );
}

function LegendItem({
  swatchClass,
  label,
  dashed,
}: {
  swatchClass: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={`inline-block h-2 w-4 rounded-sm ${swatchClass} ${
          dashed ? "[mask-image:repeating-linear-gradient(90deg,#000_0_4px,transparent_4px_7px)]" : ""
        }`}
        aria-hidden
      />
      {label}
    </span>
  );
}
