import * as React from "react";
import { Check, ChevronLeft, ChevronRight, Plus, Trash2, Calculator } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import {
  CONFIDENCE_PRESETS,
  DEFAULT_CONFIDENCE,
  formatCLP,
  presetForConfidence,
} from "@/lib/format";
import type {
  Profile,
  ScenarioDraft,
  TerminalGoal,
  IntermediateGoal,
  ScheduledWithdrawal,
  StochasticWithdrawal,
} from "@/mocks/types";
import { MoneyInput } from "./MoneyInput";

type Props = {
  profiles: Profile[];
  initialDraft?: Partial<ScenarioDraft>;
  onCalculate: (draft: ScenarioDraft) => void;
  onCancel?: () => void;
};

type Objective = ScenarioDraft["objective"];

// Display strings (title/subtitle/tag) come from the `common:objectives.<id>` keys;
// only the option ids (offered in the wizard) live here.
const OBJECTIVES: { id: Objective }[] = [
  { id: "risky" },
  { id: "proportional" },
  { id: "conservative" },
];

// Step ids; labels are resolved from `goals:steps.<id>`.
const STEPS = ["basics", "goals", "review"] as const;

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function emptyGoal(account: string): TerminalGoal {
  return { account, threshold: 0, confidence: DEFAULT_CONFIDENCE };
}

function emptyIntermediate(account: string, date: string): IntermediateGoal {
  return { account, threshold: 0, confidence: DEFAULT_CONFIDENCE, date };
}

/** Round a goal's stored confidence to the nearest preset level. */
function snapGoalConfidence<T extends { confidence: number }>(goal: T): T {
  return { ...goal, confidence: presetForConfidence(goal.confidence).value };
}

export function GoalsWizard({ profiles, initialDraft, onCalculate, onCancel }: Props) {
  const { t } = useTranslation("goals");
  const defaultProfileId = initialDraft?.profile_id ?? profiles[0]?.id ?? "";
  const [step, setStep] = React.useState(0);

  const [draft, setDraft] = React.useState<ScenarioDraft>(() => ({
    profile_id: defaultProfileId,
    name: initialDraft?.name ?? "",
    description: initialDraft?.description ?? "",
    start_date: initialDraft?.start_date ?? todayISO(),
    objective: initialDraft?.objective ?? "proportional",
    // Snap any stored confidence to the nearest preset so the chooser's selection
    // and the value we'll submit always agree (legacy plans used a free slider).
    terminal_goals: (initialDraft?.terminal_goals ?? []).map(snapGoalConfidence),
    intermediate_goals: (initialDraft?.intermediate_goals ?? []).map(snapGoalConfidence),
    withdrawals: initialDraft?.withdrawals ?? null,
  }));

  const profile = profiles.find((p) => p.id === draft.profile_id) ?? profiles[0];
  const accounts = profile?.accounts_config ?? [];
  const firstAccount = accounts[0]?.name ?? "";

  // Seed an initial terminal goal once a profile is known.
  React.useEffect(() => {
    if (draft.terminal_goals.length === 0 && firstAccount) {
      setDraft((d) => ({ ...d, terminal_goals: [emptyGoal(firstAccount)] }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [firstAccount]);

  const update = (patch: Partial<ScenarioDraft>) => setDraft((d) => ({ ...d, ...patch }));

  // ---------- Validation ----------
  const step1Valid =
    !!draft.profile_id && draft.name.trim().length > 0 && !!draft.start_date;

  // Certainty is now picked from 3 presets (always a valid level), so there is
  // no confidence range to validate — just require a goal with an amount/account.
  const step2Valid =
    draft.terminal_goals.length > 0 &&
    draft.terminal_goals.every((g) => g.threshold > 0 && !!g.account) &&
    draft.intermediate_goals.every(
      (g) => g.threshold > 0 && !!g.account && g.date > draft.start_date,
    ) &&
    (draft.withdrawals?.scheduled ?? []).every((w) => w.amount > 0 && !!w.account && !!w.date) &&
    // The backend requires either `date` or `month` on stochastic withdrawals;
    // the wizard always works with `date`.
    (draft.withdrawals?.stochastic ?? []).every((w) => w.base_amount > 0 && !!w.account && !!w.date);

  const canNext = step === 0 ? step1Valid : step === 1 ? step2Valid : true;

  return (
    <div className="space-y-6">
      <Stepper step={step} />

      {step === 0 && (
        <StepBasics
          profiles={profiles}
          draft={draft}
          onChange={update}
        />
      )}

      {step === 1 && (
        <StepGoals draft={draft} profile={profile} onChange={update} />
      )}

      {step === 2 && <StepReview draft={draft} profile={profile} />}

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card px-4 py-3">
        <div className="flex items-center gap-2">
          {onCancel && (
            <Button variant="ghost" size="sm" onClick={onCancel}>
              {t("nav.cancel")}
            </Button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
          >
            <ChevronLeft className="size-4" aria-hidden /> {t("nav.back")}
          </Button>
          {step < STEPS.length - 1 ? (
            <Button size="sm" onClick={() => setStep((s) => s + 1)} disabled={!canNext}>
              {t("nav.next")} <ChevronRight className="size-4" aria-hidden />
            </Button>
          ) : (
            <Button size="sm" onClick={() => onCalculate(draft)}>
              <Calculator className="size-4" aria-hidden /> {t("nav.calculate")}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Stepper ---------------- */

function Stepper({ step }: { step: number }) {
  const { t } = useTranslation("goals");
  return (
    <ol className="flex items-center gap-3 rounded-2xl border bg-card p-4">
      {STEPS.map((stepId, i) => {
        const isDone = i < step;
        const isCurrent = i === step;
        return (
          <li key={stepId} className="flex flex-1 items-center gap-3">
            <div
              className={cn(
                "grid size-8 shrink-0 place-items-center rounded-full text-xs font-semibold ring-1",
                isCurrent && "bg-primary text-primary-foreground ring-primary",
                isDone && "bg-success text-success-foreground ring-success",
                !isCurrent && !isDone && "bg-secondary text-muted-foreground ring-border",
              )}
              aria-current={isCurrent ? "step" : undefined}
            >
              {isDone ? <Check className="size-4" aria-hidden /> : i + 1}
            </div>
            <div className="min-w-0">
              <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                {t("stepper.stepOf", { current: i + 1, total: STEPS.length })}
              </p>
              <p
                className={cn(
                  "truncate text-sm font-medium",
                  isCurrent || isDone ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {t(`steps.${stepId}`)}
              </p>
            </div>
            {i < STEPS.length - 1 && (
              <div
                className={cn(
                  "ml-1 hidden h-px flex-1 sm:block",
                  isDone ? "bg-success/60" : "bg-border",
                )}
                aria-hidden
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}

/* ---------------- Step 1 ---------------- */

function StepBasics({
  profiles,
  draft,
  onChange,
}: {
  profiles: Profile[];
  draft: ScenarioDraft;
  onChange: (p: Partial<ScenarioDraft>) => void;
}) {
  const { t } = useTranslation(["goals", "common"]);
  return (
    <section className="space-y-6 rounded-2xl border bg-card p-5 sm:p-6">
      <header>
        <h2 className="text-base font-semibold text-foreground">{t("basics.title")}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("basics.subtitle")}
        </p>
      </header>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field label={t("basics.situationLabel")}>
          <Select
            value={draft.profile_id}
            onValueChange={(v) => onChange({ profile_id: v, terminal_goals: [] })}
          >
            <SelectTrigger className="h-10">
              <SelectValue placeholder={t("basics.situationPlaceholder")} />
            </SelectTrigger>
            <SelectContent>
              {profiles.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>

        <Field label={t("basics.nameLabel")}>
          <Input
            value={draft.name}
            onChange={(e) => onChange({ name: e.target.value })}
            placeholder={t("basics.namePlaceholder")}
            className="h-10"
          />
        </Field>

        <Field label={t("basics.startDateLabel")}>
          <Input
            type="date"
            value={draft.start_date}
            onChange={(e) => onChange({ start_date: e.target.value })}
            className="h-10"
          />
        </Field>

        <Field label={t("basics.descriptionLabel")}>
          <Textarea
            value={draft.description}
            onChange={(e) => onChange({ description: e.target.value })}
            placeholder={t("basics.descriptionPlaceholder")}
            rows={2}
          />
        </Field>
      </div>

      <div>
        <p className="text-sm font-medium text-foreground">{t("basics.styleTitle")}</p>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("basics.styleSubtitle")}
        </p>
        <div
          role="radiogroup"
          aria-label={t("basics.styleAria")}
          className="mt-3 grid gap-3 sm:grid-cols-3"
        >
          {OBJECTIVES.map((o) => {
            const selected = draft.objective === o.id;
            const tag = t(`common:objectives.${o.id}.tag`, { defaultValue: "" });
            return (
              <button
                type="button"
                key={o.id}
                role="radio"
                aria-checked={selected}
                onClick={() => onChange({ objective: o.id })}
                className={cn(
                  "group relative rounded-xl border bg-card p-4 text-left transition-all",
                  "hover:border-primary/40 hover:shadow-sm",
                  selected
                    ? "border-primary ring-2 ring-primary/30"
                    : "border-border",
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="font-medium text-foreground">{t(`common:objectives.${o.id}.title`)}</p>
                  {tag && (
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-[10px] font-medium ring-1",
                        o.id === "proportional"
                          ? "bg-success-soft text-success ring-success/30"
                          : "bg-secondary text-muted-foreground ring-border",
                      )}
                    >
                      {tag}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-sm text-muted-foreground">{t(`common:objectives.${o.id}.subtitle`)}</p>
                {selected && (
                  <span className="absolute right-3 top-3 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                    <Check className="size-3" aria-hidden />
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ---------------- Step 2 ---------------- */

function StepGoals({
  draft,
  profile,
  onChange,
}: {
  draft: ScenarioDraft;
  profile: Profile | undefined;
  onChange: (p: Partial<ScenarioDraft>) => void;
}) {
  const { t } = useTranslation("goals");
  const accounts = profile?.accounts_config ?? [];
  const firstAccount = accounts[0]?.name ?? "";

  const updateTerminal = (i: number, patch: Partial<TerminalGoal>) => {
    const next = draft.terminal_goals.map((g, idx) => (idx === i ? { ...g, ...patch } : g));
    onChange({ terminal_goals: next });
  };
  const removeTerminal = (i: number) => {
    onChange({ terminal_goals: draft.terminal_goals.filter((_, idx) => idx !== i) });
  };
  const addTerminal = () => {
    onChange({ terminal_goals: [...draft.terminal_goals, emptyGoal(firstAccount)] });
  };

  const intermediates = draft.intermediate_goals;
  const updateInter = (i: number, patch: Partial<IntermediateGoal>) => {
    const next = intermediates.map((g, idx) => (idx === i ? { ...g, ...patch } : g));
    onChange({ intermediate_goals: next });
  };
  const addInter = () => {
    const d = new Date(draft.start_date);
    d.setFullYear(d.getFullYear() + 1);
    onChange({
      intermediate_goals: [
        ...intermediates,
        emptyIntermediate(firstAccount, d.toISOString().slice(0, 10)),
      ],
    });
  };
  const removeInter = (i: number) => {
    onChange({ intermediate_goals: intermediates.filter((_, idx) => idx !== i) });
  };

  const w = draft.withdrawals ?? { scheduled: [], stochastic: [] };
  const setWithdrawals = (
    patch: Partial<{ scheduled: ScheduledWithdrawal[]; stochastic: StochasticWithdrawal[] }>,
  ) => {
    const next = { ...w, ...patch };
    const empty = next.scheduled.length === 0 && next.stochastic.length === 0;
    onChange({ withdrawals: empty ? null : next });
  };

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border bg-card p-5 sm:p-6">
        <header className="flex items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-foreground">{t("goals.title")}</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {t("goals.subtitle")}
            </p>
          </div>
        </header>

        <ul className="mt-4 space-y-4">
          {draft.terminal_goals.map((g, i) => (
            <li
              key={i}
              className="rounded-xl border bg-muted/30 p-4"
            >
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm font-medium text-foreground">{t("goals.goalN", { n: i + 1 })}</p>
                {draft.terminal_goals.length > 1 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeTerminal(i)}
                    aria-label={t("goals.removeGoal", { n: i + 1 })}
                  >
                    <Trash2 className="size-4" aria-hidden /> {t("remove")}
                  </Button>
                )}
              </div>
              <div className="mt-3 grid gap-4 sm:grid-cols-2">
                <Field label={t("goals.amountLabel")}>
                  <MoneyInput
                    value={g.threshold}
                    onChange={(v) => updateTerminal(i, { threshold: v })}
                  />
                </Field>
                <Field label={t("goals.accountLabel")}>
                  <Select
                    value={g.account}
                    onValueChange={(v) => updateTerminal(i, { account: v })}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue placeholder={t("goals.accountPlaceholder")} />
                    </SelectTrigger>
                    <SelectContent>
                      {accounts.map((a) => (
                        <SelectItem key={a.name} value={a.name}>
                          {a.display_name ?? a.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
              </div>
              <div className="mt-4">
                <ConfidenceChooser
                  value={g.confidence}
                  onChange={(v) => updateTerminal(i, { confidence: v })}
                />
              </div>
            </li>
          ))}
        </ul>

        <Button variant="outline" size="sm" className="mt-4" onClick={addTerminal}>
          <Plus className="size-4" aria-hidden /> {t("goals.addGoal")}
        </Button>
      </section>

      <section className="rounded-2xl border bg-card p-2 sm:p-3">
        <Accordion type="multiple" className="w-full">
          <AccordionItem value="dated" className="border-b last:border-b-0">
            <AccordionTrigger className="px-3">
              <div className="flex flex-col items-start text-left">
                <span className="text-sm font-medium text-foreground">{t("dated.title")}</span>
                <span className="text-xs text-muted-foreground">
                  {t("dated.subtitle")}
                </span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-3 pb-4">
              {intermediates.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  {t("dated.empty")}
                </p>
              )}
              <ul className="space-y-4">
                {intermediates.map((g, i) => (
                  <li key={i} className="rounded-xl border bg-muted/30 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-foreground">{t("dated.goalN", { n: i + 1 })}</p>
                      <Button variant="ghost" size="sm" onClick={() => removeInter(i)}>
                        <Trash2 className="size-4" aria-hidden /> {t("remove")}
                      </Button>
                    </div>
                    <div className="mt-3 grid gap-4 sm:grid-cols-3">
                      <Field label={t("goals.amountShort")}>
                        <MoneyInput
                          value={g.threshold}
                          onChange={(v) => updateInter(i, { threshold: v })}
                        />
                      </Field>
                      <Field label={t("goals.accountShort")}>
                        <Select
                          value={g.account}
                          onValueChange={(v) => updateInter(i, { account: v })}
                        >
                          <SelectTrigger className="h-10">
                            <SelectValue placeholder={t("goals.accountPlaceholder")} />
                          </SelectTrigger>
                          <SelectContent>
                            {accounts.map((a) => (
                              <SelectItem key={a.name} value={a.name}>
                                {a.display_name ?? a.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </Field>
                      <Field label={t("dated.byLabel")}>
                        <Input
                          type="date"
                          value={g.date}
                          min={draft.start_date}
                          onChange={(e) => updateInter(i, { date: e.target.value })}
                          className="h-10"
                        />
                      </Field>
                    </div>
                    <div className="mt-4">
                      <ConfidenceChooser
                        value={g.confidence}
                        onChange={(v) => updateInter(i, { confidence: v })}
                      />
                    </div>
                    {g.date && g.date <= draft.start_date && (
                      <p className="mt-2 text-xs text-danger">
                        {t("dated.dateError")}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
              <Button variant="outline" size="sm" className="mt-4" onClick={addInter}>
                <Plus className="size-4" aria-hidden /> {t("dated.add")}
              </Button>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="withdrawals" className="border-b-0">
            <AccordionTrigger className="px-3">
              <div className="flex flex-col items-start text-left">
                <span className="text-sm font-medium text-foreground">{t("withdrawals.title")}</span>
                <span className="text-xs text-muted-foreground">
                  {t("withdrawals.subtitle")}
                </span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-3 pb-4">
              <WithdrawalsEditor
                accounts={accounts}
                value={w}
                onChange={setWithdrawals}
                startDate={draft.start_date}
              />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </section>
    </div>
  );
}

function ConfidenceChooser({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const { t } = useTranslation(["goals", "common"]);
  const selectedId = presetForConfidence(value).id;
  return (
    <div>
      <Label className="text-sm">{t("confidence.question")}</Label>
      <div
        role="radiogroup"
        aria-label={t("confidence.question")}
        className="mt-2 grid gap-2 sm:grid-cols-3"
      >
        {CONFIDENCE_PRESETS.map((p) => {
          const selected = p.id === selectedId;
          return (
            <button
              type="button"
              key={p.id}
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(p.value)}
              className={cn(
                "rounded-xl border bg-card p-3 text-left transition-all",
                "hover:border-primary/40 hover:shadow-sm",
                selected ? "border-primary ring-2 ring-primary/30" : "border-border",
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-foreground">{t(`common:confidence.${p.id}.label`)}</span>
                {selected && (
                  <span className="grid size-4 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
                    <Check className="size-2.5" aria-hidden />
                  </span>
                )}
              </div>
              <p className="mt-1 text-xs text-muted-foreground">{t(`common:confidence.${p.id}.blurb`)}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function WithdrawalsEditor({
  accounts,
  value,
  onChange,
  startDate,
}: {
  accounts: Profile["accounts_config"];
  value: { scheduled: ScheduledWithdrawal[]; stochastic: StochasticWithdrawal[] };
  onChange: (p: Partial<{ scheduled: ScheduledWithdrawal[]; stochastic: StochasticWithdrawal[] }>) => void;
  startDate: string;
}) {
  const { t } = useTranslation("goals");
  const firstAccount = accounts[0]?.name ?? "";
  const addScheduled = () =>
    onChange({
      scheduled: [
        ...value.scheduled,
        { account: firstAccount, amount: 0, date: startDate, description: "" },
      ],
    });
  const updateScheduled = (i: number, patch: Partial<ScheduledWithdrawal>) =>
    onChange({
      scheduled: value.scheduled.map((s, idx) => (idx === i ? { ...s, ...patch } : s)),
    });
  const removeScheduled = (i: number) =>
    onChange({ scheduled: value.scheduled.filter((_, idx) => idx !== i) });

  const addStochastic = () =>
    onChange({
      stochastic: [
        ...value.stochastic,
        { account: firstAccount, base_amount: 0, sigma: 0.2, date: startDate, description: "" },
      ],
    });
  const updateStochastic = (i: number, patch: Partial<StochasticWithdrawal>) =>
    onChange({
      stochastic: value.stochastic.map((s, idx) => (idx === i ? { ...s, ...patch } : s)),
    });
  const removeStochastic = (i: number) =>
    onChange({ stochastic: value.stochastic.filter((_, idx) => idx !== i) });

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-medium text-foreground">{t("withdrawals.onetimeTitle")}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {t("withdrawals.onetimeSubtitle")}
        </p>
        <ul className="mt-3 space-y-3">
          {value.scheduled.map((s, i) => (
            <li key={i} className="rounded-xl border bg-muted/30 p-4">
              <div className="grid gap-4 sm:grid-cols-4">
                <Field label={t("goals.amountShort")}>
                  <MoneyInput
                    value={s.amount}
                    onChange={(v) => updateScheduled(i, { amount: v })}
                  />
                </Field>
                <Field label={t("goals.accountShort")}>
                  <Select
                    value={s.account}
                    onValueChange={(v) => updateScheduled(i, { account: v })}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {accounts.map((a) => (
                        <SelectItem key={a.name} value={a.name}>
                          {a.display_name ?? a.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                <Field label={t("goals.dateShort")}>
                  <Input
                    type="date"
                    value={s.date}
                    onChange={(e) => updateScheduled(i, { date: e.target.value })}
                    className="h-10"
                  />
                </Field>
                <Field label={t("withdrawals.noteLabel")}>
                  <Input
                    value={s.description ?? ""}
                    onChange={(e) => updateScheduled(i, { description: e.target.value })}
                    placeholder={t("withdrawals.notePlaceholderScheduled")}
                    className="h-10"
                  />
                </Field>
              </div>
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => removeScheduled(i)}>
                  <Trash2 className="size-4" aria-hidden /> {t("remove")}
                </Button>
              </div>
            </li>
          ))}
        </ul>
        <Button variant="outline" size="sm" className="mt-3" onClick={addScheduled}>
          <Plus className="size-4" aria-hidden /> {t("withdrawals.addOnetime")}
        </Button>
      </div>

      <div className="border-t pt-4">
        <p className="text-sm font-medium text-foreground">{t("withdrawals.variableTitle")}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {t("withdrawals.variableSubtitle")}
        </p>
        <ul className="mt-3 space-y-3">
          {value.stochastic.map((s, i) => (
            <li key={i} className="rounded-xl border bg-muted/30 p-4">
              <div className="grid gap-4 sm:grid-cols-3">
                <Field label={t("withdrawals.baseAmount")}>
                  <MoneyInput
                    value={s.base_amount}
                    onChange={(v) => updateStochastic(i, { base_amount: v })}
                  />
                </Field>
                <Field label={t("goals.accountShort")}>
                  <Select
                    value={s.account}
                    onValueChange={(v) => updateStochastic(i, { account: v })}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {accounts.map((a) => (
                        <SelectItem key={a.name} value={a.name}>
                          {a.display_name ?? a.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                <Field label={t("goals.dateShort")}>
                  <Input
                    type="date"
                    value={s.date ?? ""}
                    min={startDate}
                    onChange={(e) => updateStochastic(i, { date: e.target.value })}
                    className="h-10"
                  />
                </Field>
                <Field label={t("withdrawals.sigma")}>
                  <Input
                    type="number"
                    step={0.05}
                    min={0}
                    value={s.sigma}
                    onChange={(e) => updateStochastic(i, { sigma: Number(e.target.value) })}
                    className="h-10"
                  />
                </Field>
                <Field label={t("withdrawals.floor")}>
                  <MoneyInput
                    value={s.floor ?? 0}
                    onChange={(v) => updateStochastic(i, { floor: v || undefined })}
                  />
                </Field>
                <Field label={t("withdrawals.cap")}>
                  <MoneyInput
                    value={s.cap ?? 0}
                    onChange={(v) => updateStochastic(i, { cap: v || undefined })}
                  />
                </Field>
                <Field label={t("withdrawals.noteLabel")}>
                  <Input
                    value={s.description ?? ""}
                    onChange={(e) => updateStochastic(i, { description: e.target.value })}
                    placeholder={t("withdrawals.notePlaceholderVariable")}
                    className="h-10"
                  />
                </Field>
              </div>
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => removeStochastic(i)}>
                  <Trash2 className="size-4" aria-hidden /> {t("remove")}
                </Button>
              </div>
            </li>
          ))}
        </ul>
        <Button variant="outline" size="sm" className="mt-3" onClick={addStochastic}>
          <Plus className="size-4" aria-hidden /> {t("withdrawals.addVariable")}
        </Button>
      </div>
    </div>
  );
}

/* ---------------- Step 3 ---------------- */

function StepReview({
  draft,
  profile,
}: {
  draft: ScenarioDraft;
  profile: Profile | undefined;
}) {
  const { t } = useTranslation(["goals", "common"]);
  const accountName = (a: string) =>
    profile?.accounts_config.find((x) => x.name === a)?.display_name ?? a;
  const styleLabel = t(`common:objectives.${draft.objective}.title`, { defaultValue: draft.objective });
  const amountSpan = <span className="tabular font-medium text-foreground" />;
  const strongSpan = <span className="font-medium text-foreground" />;

  return (
    <section className="space-y-5 rounded-2xl border bg-card p-5 sm:p-6">
      <header>
        <h2 className="text-base font-semibold text-foreground">{t("review.title")}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("review.subtitle")}
        </p>
      </header>

      <dl className="grid gap-4 sm:grid-cols-2">
        <Summary label={t("review.situation")} value={profile?.name ?? "—"} />
        <Summary label={t("review.planName")} value={draft.name || "—"} />
        <Summary label={t("review.startDate")} value={draft.start_date} />
        <Summary label={t("review.style")} value={styleLabel} />
      </dl>

      <div className="rounded-xl border bg-muted/30 p-4">
        <p className="text-sm font-medium text-foreground">{t("review.goalsTitle")}</p>
        <ul className="mt-2 space-y-2 text-sm">
          {draft.terminal_goals.map((g, i) => (
            <li key={i} className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-muted-foreground">
                <Trans
                  i18nKey="review.reachLine"
                  t={t}
                  values={{ amount: formatCLP(g.threshold), account: accountName(g.account) }}
                  components={{ amount: amountSpan, acct: strongSpan }}
                />
              </span>
              <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-muted-foreground">
                {t(`common:confidence.${presetForConfidence(g.confidence).id}.label`)}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {draft.intermediate_goals.length > 0 && (
        <div className="rounded-xl border bg-muted/30 p-4">
          <p className="text-sm font-medium text-foreground">{t("dated.title")}</p>
          <ul className="mt-2 space-y-2 text-sm">
            {draft.intermediate_goals.map((g, i) => (
              <li key={i} className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-muted-foreground">
                  <Trans
                    i18nKey="review.datedLine"
                    t={t}
                    values={{
                      date: g.date,
                      amount: formatCLP(g.threshold),
                      account: accountName(g.account),
                    }}
                    components={{ date: strongSpan, amount: amountSpan, acct: strongSpan }}
                  />
                </span>
                <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  {t(`common:confidence.${presetForConfidence(g.confidence).id}.label`)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(draft.withdrawals?.scheduled?.length ?? 0) +
        (draft.withdrawals?.stochastic?.length ?? 0) >
        0 && (
        <div className="rounded-xl border bg-muted/30 p-4">
          <p className="text-sm font-medium text-foreground">{t("withdrawals.title")}</p>
          <ul className="mt-2 space-y-2 text-sm text-muted-foreground">
            {draft.withdrawals?.scheduled.map((s, i) => (
              <li key={`s-${i}`}>
                <Trans
                  i18nKey="review.scheduledLine"
                  t={t}
                  values={{ amount: formatCLP(s.amount), account: accountName(s.account), date: s.date }}
                  components={{ amount: amountSpan, acct: strongSpan, date: strongSpan }}
                />
                {s.description ? ` — ${s.description}` : ""}
              </li>
            ))}
            {draft.withdrawals?.stochastic.map((s, i) => (
              <li key={`v-${i}`}>
                <Trans
                  i18nKey="review.variableLine"
                  t={t}
                  values={{ amount: formatCLP(s.base_amount), account: accountName(s.account) }}
                  components={{ amount: amountSpan, acct: strongSpan }}
                />
                {s.description ? ` — ${s.description}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

/* ---------------- Small bits ---------------- */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-sm text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="tabular mt-1 text-sm font-medium text-foreground">{value}</dd>
    </div>
  );
}
