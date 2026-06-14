import { Loader2, AlertCircle, Clock } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { JobStatus } from "@/mocks/types";
import { Button } from "@/components/ui/button";

type Props = {
  status: Exclude<JobStatus, "completed">;
  progress?: number; // 0..1
  onAdjust?: () => void;
  onRetry?: () => void;
  errorMessage?: string;
};

export function JobStatusView({ status, progress = 0.45, onAdjust, onRetry, errorMessage }: Props) {
  const { t } = useTranslation("plan");

  if (status === "pending") {
    return (
      <Shell>
        <Clock className="size-8 text-muted-foreground" aria-hidden />
        <h2 className="mt-4 text-xl font-semibold text-foreground">{t("job.queuedTitle")}</h2>
        <p className="mt-1 max-w-md text-sm text-muted-foreground">
          {t("job.queuedBody")}
        </p>
      </Shell>
    );
  }

  if (status === "running") {
    const pct = Math.round(progress * 100);
    return (
      <Shell>
        <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
        <h2 className="mt-4 text-xl font-semibold text-foreground">{t("job.runningTitle")}</h2>
        <p className="mt-1 max-w-md text-sm text-muted-foreground">
          {t("job.runningBody")}
        </p>
        <div
          className="mt-6 h-2 w-full max-w-sm overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className="h-full rounded-full bg-primary transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="tabular mt-2 text-xs text-muted-foreground">{pct}%</p>
      </Shell>
    );
  }

  // failed
  return (
    <Shell>
      <div className="rounded-full bg-danger-soft p-3">
        <AlertCircle className="size-7 text-danger" aria-hidden />
      </div>
      <h2 className="mt-4 text-xl font-semibold text-foreground">{t("job.failedTitle")}</h2>
      <p className="mt-1 max-w-md text-sm text-muted-foreground">
        {errorMessage ?? t("job.failedBody")}
      </p>
      <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
        {onAdjust && (
          <Button onClick={onAdjust} variant="default">
            {t("job.adjustGoals")}
          </Button>
        )}
        {onRetry && (
          <Button onClick={onRetry} variant="outline">
            {t("job.tryAgain")}
          </Button>
        )}
      </div>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <section className="flex min-h-[420px] flex-col items-center justify-center rounded-2xl border bg-card p-10 text-center">
      {children}
    </section>
  );
}
