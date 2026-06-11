import * as React from "react";
import { Check, ChevronLeft, ChevronRight, Plus, Trash2, Calculator } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
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
import { describeConfidence, formatCLP, formatPercent } from "@/lib/format";
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

const OBJECTIVES: { id: Objective; title: string; subtitle: string; tag?: string }[] = [
  { id: "risky", title: "Maximum growth", subtitle: "Higher swings, higher long-term return." },
  {
    id: "proportional",
    title: "Steady & even",
    subtitle: "A stable, even monthly split across your accounts.",
    tag: "Recommended",
  },
  {
    id: "conservative",
    title: "Conservative",
    subtitle: "Smaller ups and downs, slower growth.",
    tag: "Lower risk",
  },
];

const STEPS = ["Basics", "Your goals", "Review"] as const;

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function emptyGoal(account: string): TerminalGoal {
  return { account, threshold: 0, confidence: 0.8 };
}

function emptyIntermediate(account: string, date: string): IntermediateGoal {
  return { account, threshold: 0, confidence: 0.8, date };
}

export function GoalsWizard({ profiles, initialDraft, onCalculate, onCancel }: Props) {
  const defaultProfileId = initialDraft?.profile_id ?? profiles[0]?.id ?? "";
  const [step, setStep] = React.useState(0);

  const [draft, setDraft] = React.useState<ScenarioDraft>(() => ({
    profile_id: defaultProfileId,
    name: initialDraft?.name ?? "",
    description: initialDraft?.description ?? "",
    start_date: initialDraft?.start_date ?? todayISO(),
    objective: initialDraft?.objective ?? "proportional",
    terminal_goals: initialDraft?.terminal_goals ?? [],
    intermediate_goals: initialDraft?.intermediate_goals ?? [],
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

  const step2Valid =
    draft.terminal_goals.length > 0 &&
    draft.terminal_goals.every(
      (g) => g.threshold > 0 && !!g.account && g.confidence >= 0.5 && g.confidence <= 0.95,
    ) &&
    draft.intermediate_goals.every(
      (g) =>
        g.threshold > 0 &&
        !!g.account &&
        g.confidence >= 0.5 &&
        g.confidence <= 0.95 &&
        g.date > draft.start_date,
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
              Cancel
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
            <ChevronLeft className="size-4" aria-hidden /> Back
          </Button>
          {step < STEPS.length - 1 ? (
            <Button size="sm" onClick={() => setStep((s) => s + 1)} disabled={!canNext}>
              Next <ChevronRight className="size-4" aria-hidden />
            </Button>
          ) : (
            <Button size="sm" onClick={() => onCalculate(draft)}>
              <Calculator className="size-4" aria-hidden /> Calculate my plan
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Stepper ---------------- */

function Stepper({ step }: { step: number }) {
  return (
    <ol className="flex items-center gap-3 rounded-2xl border bg-card p-4">
      {STEPS.map((label, i) => {
        const isDone = i < step;
        const isCurrent = i === step;
        return (
          <li key={label} className="flex flex-1 items-center gap-3">
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
                Step {i + 1} of {STEPS.length}
              </p>
              <p
                className={cn(
                  "truncate text-sm font-medium",
                  isCurrent || isDone ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {label}
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
  return (
    <section className="space-y-6 rounded-2xl border bg-card p-5 sm:p-6">
      <header>
        <h2 className="text-base font-semibold text-foreground">The basics</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Tell us which situation this plan is for and how you want to invest.
        </p>
      </header>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field label="Which situation is this plan for?">
          <Select
            value={draft.profile_id}
            onValueChange={(v) => onChange({ profile_id: v, terminal_goals: [] })}
          >
            <SelectTrigger className="h-10">
              <SelectValue placeholder="Pick a situation" />
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

        <Field label="Plan name">
          <Input
            value={draft.name}
            onChange={(e) => onChange({ name: e.target.value })}
            placeholder="e.g. Apartment down payment"
            className="h-10"
          />
        </Field>

        <Field label="Start date">
          <Input
            type="date"
            value={draft.start_date}
            onChange={(e) => onChange({ start_date: e.target.value })}
            className="h-10"
          />
        </Field>

        <Field label="Short description (optional)">
          <Textarea
            value={draft.description}
            onChange={(e) => onChange({ description: e.target.value })}
            placeholder="Anything helpful to remember about this plan."
            rows={2}
          />
        </Field>
      </div>

      <div>
        <p className="text-sm font-medium text-foreground">Investment style</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Pick how we balance growth and diversification across your accounts.
        </p>
        <div
          role="radiogroup"
          aria-label="Investment style"
          className="mt-3 grid gap-3 sm:grid-cols-3"
        >
          {OBJECTIVES.map((o) => {
            const selected = draft.objective === o.id;
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
                  <p className="font-medium text-foreground">{o.title}</p>
                  {o.tag && (
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-[10px] font-medium ring-1",
                        o.id === "proportional"
                          ? "bg-success-soft text-success-foreground ring-success/30"
                          : "bg-secondary text-muted-foreground ring-border",
                      )}
                    >
                      {o.tag}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-sm text-muted-foreground">{o.subtitle}</p>
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
            <h2 className="text-base font-semibold text-foreground">Your goals</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Tell us how much you want to reach and how sure you want to be.
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
                <p className="text-sm font-medium text-foreground">Goal {i + 1}</p>
                {draft.terminal_goals.length > 1 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeTerminal(i)}
                    aria-label={`Remove goal ${i + 1}`}
                  >
                    <Trash2 className="size-4" aria-hidden /> Remove
                  </Button>
                )}
              </div>
              <div className="mt-3 grid gap-4 sm:grid-cols-2">
                <Field label="How much do you want?">
                  <MoneyInput
                    value={g.threshold}
                    onChange={(v) => updateTerminal(i, { threshold: v })}
                  />
                </Field>
                <Field label="In which account?">
                  <Select
                    value={g.account}
                    onValueChange={(v) => updateTerminal(i, { account: v })}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue placeholder="Pick an account" />
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
                <ConfidenceSlider
                  value={g.confidence}
                  onChange={(v) => updateTerminal(i, { confidence: v })}
                />
              </div>
            </li>
          ))}
        </ul>

        <Button variant="outline" size="sm" className="mt-4" onClick={addTerminal}>
          <Plus className="size-4" aria-hidden /> Add another goal
        </Button>
      </section>

      <section className="rounded-2xl border bg-card p-2 sm:p-3">
        <Accordion type="multiple" className="w-full">
          <AccordionItem value="dated" className="border-b last:border-b-0">
            <AccordionTrigger className="px-3">
              <div className="flex flex-col items-start text-left">
                <span className="text-sm font-medium text-foreground">Goals with a deadline</span>
                <span className="text-xs text-muted-foreground">
                  Optional · e.g. an emergency fund ready by a certain date.
                </span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-3 pb-4">
              {intermediates.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No dated goals yet. These are useful when something must be ready by a specific
                  date.
                </p>
              )}
              <ul className="space-y-4">
                {intermediates.map((g, i) => (
                  <li key={i} className="rounded-xl border bg-muted/30 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-foreground">Dated goal {i + 1}</p>
                      <Button variant="ghost" size="sm" onClick={() => removeInter(i)}>
                        <Trash2 className="size-4" aria-hidden /> Remove
                      </Button>
                    </div>
                    <div className="mt-3 grid gap-4 sm:grid-cols-3">
                      <Field label="Amount">
                        <MoneyInput
                          value={g.threshold}
                          onChange={(v) => updateInter(i, { threshold: v })}
                        />
                      </Field>
                      <Field label="Account">
                        <Select
                          value={g.account}
                          onValueChange={(v) => updateInter(i, { account: v })}
                        >
                          <SelectTrigger className="h-10">
                            <SelectValue placeholder="Pick an account" />
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
                      <Field label="By">
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
                      <ConfidenceSlider
                        value={g.confidence}
                        onChange={(v) => updateInter(i, { confidence: v })}
                      />
                    </div>
                    {g.date && g.date <= draft.start_date && (
                      <p className="mt-2 text-xs text-danger-foreground">
                        The date must be after the plan's start date.
                      </p>
                    )}
                  </li>
                ))}
              </ul>
              <Button variant="outline" size="sm" className="mt-4" onClick={addInter}>
                <Plus className="size-4" aria-hidden /> Add a dated goal
              </Button>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="withdrawals" className="border-b-0">
            <AccordionTrigger className="px-3">
              <div className="flex flex-col items-start text-left">
                <span className="text-sm font-medium text-foreground">Planned withdrawals</span>
                <span className="text-xs text-muted-foreground">
                  Optional · money you'll take out along the way.
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

function ConfidenceSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-end justify-between">
        <Label className="text-sm">How sure do you want to be?</Label>
        <span className="tabular text-sm font-medium text-foreground">
          {formatPercent(value)}
        </span>
      </div>
      <Slider
        className="mt-2"
        min={50}
        max={95}
        step={5}
        value={[Math.round(value * 100)]}
        onValueChange={([v]) => onChange((v ?? 80) / 100)}
        aria-label="Confidence"
      />
      <p className="mt-1.5 text-xs text-muted-foreground">{describeConfidence(value)}.</p>
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
        <p className="text-sm font-medium text-foreground">One-time withdrawals</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          A specific amount taken out on a specific date.
        </p>
        <ul className="mt-3 space-y-3">
          {value.scheduled.map((s, i) => (
            <li key={i} className="rounded-xl border bg-muted/30 p-4">
              <div className="grid gap-4 sm:grid-cols-4">
                <Field label="Amount">
                  <MoneyInput
                    value={s.amount}
                    onChange={(v) => updateScheduled(i, { amount: v })}
                  />
                </Field>
                <Field label="Account">
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
                <Field label="Date">
                  <Input
                    type="date"
                    value={s.date}
                    onChange={(e) => updateScheduled(i, { date: e.target.value })}
                    className="h-10"
                  />
                </Field>
                <Field label="Note (optional)">
                  <Input
                    value={s.description ?? ""}
                    onChange={(e) => updateScheduled(i, { description: e.target.value })}
                    placeholder="e.g. tuition payment"
                    className="h-10"
                  />
                </Field>
              </div>
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => removeScheduled(i)}>
                  <Trash2 className="size-4" aria-hidden /> Remove
                </Button>
              </div>
            </li>
          ))}
        </ul>
        <Button variant="outline" size="sm" className="mt-3" onClick={addScheduled}>
          <Plus className="size-4" aria-hidden /> Add a one-time withdrawal
        </Button>
      </div>

      <div className="border-t pt-4">
        <p className="text-sm font-medium text-foreground">Recurring or variable withdrawals</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Advanced · for living expenses that vary month to month.
        </p>
        <ul className="mt-3 space-y-3">
          {value.stochastic.map((s, i) => (
            <li key={i} className="rounded-xl border bg-muted/30 p-4">
              <div className="grid gap-4 sm:grid-cols-3">
                <Field label="Base amount">
                  <MoneyInput
                    value={s.base_amount}
                    onChange={(v) => updateStochastic(i, { base_amount: v })}
                  />
                </Field>
                <Field label="Account">
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
                <Field label="Date">
                  <Input
                    type="date"
                    value={s.date ?? ""}
                    min={startDate}
                    onChange={(e) => updateStochastic(i, { date: e.target.value })}
                    className="h-10"
                  />
                </Field>
                <Field label="Variability (sigma)">
                  <Input
                    type="number"
                    step={0.05}
                    min={0}
                    value={s.sigma}
                    onChange={(e) => updateStochastic(i, { sigma: Number(e.target.value) })}
                    className="h-10"
                  />
                </Field>
                <Field label="Floor (optional)">
                  <MoneyInput
                    value={s.floor ?? 0}
                    onChange={(v) => updateStochastic(i, { floor: v || undefined })}
                  />
                </Field>
                <Field label="Cap (optional)">
                  <MoneyInput
                    value={s.cap ?? 0}
                    onChange={(v) => updateStochastic(i, { cap: v || undefined })}
                  />
                </Field>
                <Field label="Note (optional)">
                  <Input
                    value={s.description ?? ""}
                    onChange={(e) => updateStochastic(i, { description: e.target.value })}
                    placeholder="e.g. living expenses"
                    className="h-10"
                  />
                </Field>
              </div>
              <div className="mt-2 flex justify-end">
                <Button variant="ghost" size="sm" onClick={() => removeStochastic(i)}>
                  <Trash2 className="size-4" aria-hidden /> Remove
                </Button>
              </div>
            </li>
          ))}
        </ul>
        <Button variant="outline" size="sm" className="mt-3" onClick={addStochastic}>
          <Plus className="size-4" aria-hidden /> Add a variable withdrawal
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
  const accountName = (a: string) =>
    profile?.accounts_config.find((x) => x.name === a)?.display_name ?? a;
  const styleLabel = OBJECTIVES.find((o) => o.id === draft.objective)?.title ?? draft.objective;

  return (
    <section className="space-y-5 rounded-2xl border bg-card p-5 sm:p-6">
      <header>
        <h2 className="text-base font-semibold text-foreground">Review your plan</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          We'll find the shortest time to reach your goals.
        </p>
      </header>

      <dl className="grid gap-4 sm:grid-cols-2">
        <Summary label="Situation" value={profile?.name ?? "—"} />
        <Summary label="Plan name" value={draft.name || "—"} />
        <Summary label="Start date" value={draft.start_date} />
        <Summary label="Investment style" value={styleLabel} />
      </dl>

      <div className="rounded-xl border bg-muted/30 p-4">
        <p className="text-sm font-medium text-foreground">Goals</p>
        <ul className="mt-2 space-y-2 text-sm">
          {draft.terminal_goals.map((g, i) => (
            <li key={i} className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-muted-foreground">
                Reach{" "}
                <span className="tabular font-medium text-foreground">
                  {formatCLP(g.threshold)}
                </span>{" "}
                in <span className="font-medium text-foreground">{accountName(g.account)}</span>
              </span>
              <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-muted-foreground">
                {Math.round(g.confidence * 10)} out of 10
              </span>
            </li>
          ))}
        </ul>
      </div>

      {draft.intermediate_goals.length > 0 && (
        <div className="rounded-xl border bg-muted/30 p-4">
          <p className="text-sm font-medium text-foreground">Goals with a deadline</p>
          <ul className="mt-2 space-y-2 text-sm">
            {draft.intermediate_goals.map((g, i) => (
              <li key={i} className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-muted-foreground">
                  By <span className="font-medium text-foreground">{g.date}</span>: reach{" "}
                  <span className="tabular font-medium text-foreground">
                    {formatCLP(g.threshold)}
                  </span>{" "}
                  in <span className="font-medium text-foreground">{accountName(g.account)}</span>
                </span>
                <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  {Math.round(g.confidence * 10)} out of 10
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
          <p className="text-sm font-medium text-foreground">Planned withdrawals</p>
          <ul className="mt-2 space-y-2 text-sm text-muted-foreground">
            {draft.withdrawals?.scheduled.map((s, i) => (
              <li key={`s-${i}`}>
                <span className="tabular font-medium text-foreground">{formatCLP(s.amount)}</span>{" "}
                from <span className="font-medium text-foreground">{accountName(s.account)}</span>{" "}
                on <span className="font-medium text-foreground">{s.date}</span>
                {s.description ? ` — ${s.description}` : ""}
              </li>
            ))}
            {draft.withdrawals?.stochastic.map((s, i) => (
              <li key={`v-${i}`}>
                About{" "}
                <span className="tabular font-medium text-foreground">
                  {formatCLP(s.base_amount)}
                </span>{" "}
                from <span className="font-medium text-foreground">{accountName(s.account)}</span>{" "}
                (variable)
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
