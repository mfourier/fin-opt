/**
 * FinOptHeroPanel — the left/visual side of the login screen.
 *
 * A calm, premium fintech hero: soft blue gradient, brand wordmark, headline +
 * subtitle in plain language, an abstract wealth chart, and a few floating
 * preview cards that hint at the product (goal, confidence, minimum horizon,
 * feasibility). Fully presentational and self-contained.
 */
import { CalendarClock, CheckCircle2 } from "lucide-react";
import { AbstractWealthChart } from "./AbstractWealthChart";
import { GoalPreviewCard } from "./GoalPreviewCard";
import { MiniMetricCard } from "./MiniMetricCard";
import { FinOptWordmark } from "./FinOptWordmark";

export function FinOptHeroPanel() {
  return (
    <div className="relative flex h-full w-full flex-col justify-between overflow-hidden bg-[#005C99] p-10 lg:p-14">
      {/* Brand gradient + ambient blobs */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "linear-gradient(135deg, #005C99 0%, #008FD3 55%, #12BFEF 130%)",
        }}
      />
      <div className="pointer-events-none absolute -right-24 -top-24 h-80 w-80 rounded-full bg-white/10 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-32 -left-16 h-80 w-80 rounded-full bg-[#12BFEF]/30 blur-3xl" />

      {/* Wordmark */}
      <div className="relative z-10 animate-fade-in-up">
        <FinOptWordmark className="text-white" />
      </div>

      {/* Headline + visual */}
      <div className="relative z-10 my-8">
        <h1
          className="max-w-md text-4xl font-extrabold leading-tight tracking-tight text-white animate-fade-in-up lg:text-5xl"
          style={{ animationDelay: "0.05s" }}
        >
          Your goals, optimized.
        </h1>
        <p
          className="mt-4 max-w-md text-base leading-relaxed text-white/85 animate-fade-in-up"
          style={{ animationDelay: "0.15s" }}
        >
          Understand when your financial goals become achievable — with clear
          scenarios and risk-aware planning.
        </p>

        {/* Chart + floating cards */}
        <div className="relative mt-10 max-w-lg">
          <div
            className="rounded-3xl border border-white/15 bg-white/10 p-5 shadow-2xl backdrop-blur-sm animate-fade-in-up"
            style={{ animationDelay: "0.25s" }}
          >
            <AbstractWealthChart />
          </div>

          {/* Floating preview cards */}
          <div
            className="absolute -right-4 -top-8 hidden animate-float sm:block"
            style={{ animationDelay: "0.2s" }}
          >
            <GoalPreviewCard goal="Home" status="Feasible" confidence={0.8} />
          </div>

          <div
            className="absolute -bottom-6 -left-4 hidden animate-float sm:block"
            style={{ animationDelay: "1.2s" }}
          >
            <MiniMetricCard
              label="Minimum horizon"
              value="42 months"
              icon={CalendarClock}
            />
          </div>

          <div
            className="absolute -bottom-10 right-6 hidden animate-float md:block"
            style={{ animationDelay: "2s" }}
          >
            <MiniMetricCard
              label="Status"
              value="On track"
              icon={CheckCircle2}
              tone="success"
            />
          </div>
        </div>
      </div>

      {/* Footer note */}
      <p
        className="relative z-10 text-xs text-white/70 animate-fade-in-up"
        style={{ animationDelay: "0.35s" }}
      >
        Clear scenarios · Confidence bands · Minimum horizon to reach your goals
      </p>
    </div>
  );
}
