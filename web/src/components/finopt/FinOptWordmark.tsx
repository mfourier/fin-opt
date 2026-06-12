/**
 * FinOptWordmark — lightweight inline brand mark (logo glyph + "FinOpt").
 *
 * Reusable across auth/marketing surfaces. The glyph is a small ascending-line
 * motif inside a rounded square; color is inherited via `currentColor` so it
 * works on both light cards and the blue hero. No external image needed.
 */
import { cn } from "@/lib/utils";

interface FinOptWordmarkProps {
  className?: string;
  /** Hide the text and render only the glyph. */
  glyphOnly?: boolean;
}

export function FinOptWordmark({ className, glyphOnly = false }: FinOptWordmarkProps) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <svg
        viewBox="0 0 40 40"
        className="h-9 w-9"
        role="img"
        aria-label="FinOpt logo"
      >
        <rect
          x="1.5"
          y="1.5"
          width="37"
          height="37"
          rx="11"
          fill="currentColor"
          fillOpacity="0.12"
          stroke="currentColor"
          strokeOpacity="0.35"
          strokeWidth="1.5"
        />
        <path
          d="M9 27 L17 19 L23 23 L31 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="31" cy="12" r="3.2" fill="currentColor" />
      </svg>
      {!glyphOnly && (
        <span className="text-xl font-extrabold tracking-tight">
          Fin<span className="font-bold opacity-90">Opt</span>
        </span>
      )}
    </span>
  );
}
