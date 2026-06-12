/**
 * AbstractWealthChart — lightweight, decorative SVG used on the login hero.
 *
 * Conveys the FinOpt idea (a wealth trajectory rising toward a goal, surrounded
 * by a soft probability band) without any charting library, real data, or
 * trading-style candlesticks. Pure inline SVG + CSS so it stays crisp and fast.
 */
import { cn } from "@/lib/utils";

interface AbstractWealthChartProps {
  className?: string;
}

// Median trajectory and the upper/lower edges of the confidence band.
const MEDIAN = "M 12 168 C 92 160, 150 128, 206 104 S 320 56, 384 40";
const UPPER = "M 12 150 C 96 138, 156 100, 210 76 S 322 30, 384 18";
const LOWER = "M 12 186 C 88 182, 146 156, 202 134 S 320 84, 384 64";
// Closed area between UPPER and LOWER (band fill).
const BAND = `${UPPER} L 384 64 C 320 84, 202 134, 12 186 Z`;

export function AbstractWealthChart({ className }: AbstractWealthChartProps) {
  return (
    <svg
      viewBox="0 0 400 220"
      fill="none"
      className={cn("h-auto w-full", className)}
      role="img"
      aria-label="A rising wealth trajectory reaching a financial goal, surrounded by a probability band"
    >
      <defs>
        <linearGradient id="finopt-band" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#12BFEF" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#008FD3" stopOpacity="0.04" />
        </linearGradient>
        <linearGradient id="finopt-line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#005C99" />
          <stop offset="100%" stopColor="#12BFEF" />
        </linearGradient>
      </defs>

      {/* Soft horizontal guides */}
      {[52, 96, 140].map((y) => (
        <line
          key={y}
          x1="12"
          y1={y}
          x2="388"
          y2={y}
          stroke="#ffffff"
          strokeOpacity="0.18"
          strokeWidth="1"
        />
      ))}

      {/* Confidence band */}
      <path d={BAND} fill="url(#finopt-band)" />
      <path d={UPPER} stroke="#ffffff" strokeOpacity="0.25" strokeWidth="1.5" />
      <path d={LOWER} stroke="#ffffff" strokeOpacity="0.25" strokeWidth="1.5" />

      {/* Goal level (dashed) */}
      <line
        x1="12"
        y1="40"
        x2="384"
        y2="40"
        stroke="#ffffff"
        strokeOpacity="0.5"
        strokeWidth="1.5"
        strokeDasharray="4 5"
      />

      {/* Median trajectory (animated draw) */}
      <path
        d={MEDIAN}
        stroke="url(#finopt-line)"
        strokeWidth="3.5"
        strokeLinecap="round"
        className="animate-draw"
        style={{ ["--draw-length" as string]: "520" }}
      />

      {/* Goal point reached */}
      <g className="animate-fade-in-up" style={{ animationDelay: "1.4s" }}>
        <circle cx="384" cy="40" r="11" fill="#ffffff" fillOpacity="0.18" />
        <circle cx="384" cy="40" r="6" fill="#ffffff" />
        <circle cx="384" cy="40" r="3" fill="#008FD3" />
      </g>
    </svg>
  );
}
