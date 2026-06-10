import * as React from "react";
import { Plus, Trash2, Save, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import { formatCLP } from "@/lib/format";
import type {
  AccountConfig,
  IncomeConfig,
  ProfileDraft,
} from "@/mocks/types";
import { MoneyInput } from "./MoneyInput";

type Props = {
  initialProfile?: Partial<ProfileDraft>;
  onSave: (draft: ProfileDraft) => void;
  onCancel?: () => void;
};

/* ---------------- helpers ---------------- */

function slugify(s: string): string {
  const base = s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return base || "account";
}

function uniqueSlug(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base;
  let i = 2;
  while (taken.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

const RISK_PRESETS = [
  { id: "conservative", title: "Conservative", subtitle: "Smaller ups and downs.", ret: 0.08, vol: 0.09 },
  { id: "balanced", title: "Balanced", subtitle: "A steady mix.", ret: 0.1, vol: 0.12, tag: "Recommended" },
  { id: "aggressive", title: "Aggressive", subtitle: "Bigger swings, more growth.", ret: 0.14, vol: 0.18 },
  { id: "custom", title: "Custom", subtitle: "Set your own numbers." },
] as const;

type RiskId = (typeof RISK_PRESETS)[number]["id"];

function detectRisk(a: AccountConfig): RiskId {
  for (const p of RISK_PRESETS) {
    if (p.id === "custom") continue;
    if (
      Math.abs(a.annual_return - (p as { ret: number }).ret) < 1e-6 &&
      Math.abs(a.annual_volatility - (p as { vol: number }).vol) < 1e-6
    )
      return p.id;
  }
  return "custom";
}

const VARIABILITY = [
  { id: "low", title: "Low", subtitle: "Pretty steady", sigma: 0.1 },
  { id: "medium", title: "Medium", subtitle: "Some swings", sigma: 0.2 },
  { id: "high", title: "High", subtitle: "Big swings", sigma: 0.4 },
  { id: "custom", title: "Custom", subtitle: "Set your own" },
] as const;

type VarId = (typeof VARIABILITY)[number]["id"];

function detectVariability(sigma: number): VarId {
  for (const v of VARIABILITY) {
    if (v.id === "custom") continue;
    if (Math.abs(sigma - (v as { sigma: number }).sigma) < 1e-6) return v.id;
  }
  return "custom";
}

function emptyAccount(taken: Set<string>): AccountConfig {
  const name = uniqueSlug("account", taken);
  return {
    name,
    display_name: "",
    annual_return: 0.1,
    annual_volatility: 0.12,
    initial_wealth: 0,
  };
}

const DEFAULT_INCOME: IncomeConfig = {
  fixed: { base: 0, annual_growth: 0.03 },
  contribution_rate_fixed: 0.3,
  contribution_rate_variable: 1,
};

/* ---------------- main ---------------- */

export function SituationForm({ initialProfile, onSave, onCancel }: Props) {
  const [name, setName] = React.useState(initialProfile?.name ?? "");
  const [description, setDescription] = React.useState(initialProfile?.description ?? "");
  const [income, setIncome] = React.useState<IncomeConfig>(
    initialProfile?.income_config ?? DEFAULT_INCOME,
  );
  const [accounts, setAccounts] = React.useState<AccountConfig[]>(
    initialProfile?.accounts_config ?? [emptyAccount(new Set())],
  );
  const [correlation, setCorrelation] = React.useState<number[][] | null>(
    initialProfile?.correlation_matrix ?? null,
  );

  const updateIncome = (patch: Partial<IncomeConfig>) =>
    setIncome((p) => ({ ...p, ...patch }));

  /* ---- account ops ---- */
  const setAccount = (i: number, patch: Partial<AccountConfig>) => {
    setAccounts((arr) => arr.map((a, idx) => (idx === i ? { ...a, ...patch } : a)));
  };
  const addAccount = () => {
    setAccounts((arr) => {
      const taken = new Set(arr.map((a) => a.name));
      return [...arr, emptyAccount(taken)];
    });
  };
  const removeAccount = (i: number) => {
    if (accounts.length <= 1) return;
    setAccounts((arr) => arr.filter((_, idx) => idx !== i));
    setCorrelation(null);
  };

  /* ---- validation ---- */
  const nameValid = name.trim().length > 0;
  const salary = income.fixed?.base ?? 0;
  const salaryValid = salary >= 0;
  const rateFixed = Number(income.contribution_rate_fixed) || 0;
  const rateValid = rateFixed >= 0 && rateFixed <= 1;
  const duplicateNames = (() => {
    const seen = new Set<string>();
    for (const a of accounts) {
      const k = (a.display_name ?? "").trim().toLowerCase();
      if (!k) continue;
      if (seen.has(k)) return true;
      seen.add(k);
    }
    return false;
  })();
  const accountsValid =
    accounts.length >= 1 &&
    accounts.every(
      (a) =>
        (a.display_name ?? "").trim().length > 0 &&
        a.annual_return >= -1 &&
        a.annual_return <= 2 &&
        a.annual_volatility >= 0 &&
        a.annual_volatility <= 2,
    ) &&
    !duplicateNames;

  const canSave = nameValid && salaryValid && rateValid && accountsValid;

  const handleSave = () => {
    // Normalize account slugs from display names (preserve existing slug if already set sensibly).
    const taken = new Set<string>();
    const finalAccounts = accounts.map((a) => {
      const base = slugify(a.display_name ?? a.name);
      const slug = uniqueSlug(base, taken);
      taken.add(slug);
      return { ...a, name: slug, display_name: (a.display_name ?? "").trim() };
    });
    onSave({
      name: name.trim(),
      description: description.trim(),
      income_config: income,
      accounts_config: finalAccounts,
      correlation_matrix: finalAccounts.length >= 2 ? correlation : null,
    });
  };

  return (
    <div className="space-y-6">
      {/* ---- Basics ---- */}
      <section className="rounded-2xl border bg-card p-5 sm:p-6">
        <header>
          <h2 className="text-base font-semibold text-foreground">The basics</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Give this situation a name so you can reuse it across plans.
          </p>
        </header>
        <div className="mt-5 grid gap-5 sm:grid-cols-2">
          <Field label="Name" error={!nameValid && name.length > 0 ? "Give it a name" : undefined}>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. My finances"
              className="h-10"
            />
          </Field>
          <Field label="Short description (optional)">
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Anything helpful to remember about this situation."
              rows={2}
            />
          </Field>
        </div>
      </section>

      {/* ---- Income ---- */}
      <IncomeSection income={income} onChange={updateIncome} />

      {/* ---- Accounts ---- */}
      <section className="rounded-2xl border bg-card p-5 sm:p-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-foreground">Where you invest</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              List each account or portfolio you're saving into.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={addAccount}>
            <Plus className="size-4" aria-hidden /> Add account
          </Button>
        </header>

        <ul className="mt-4 space-y-4">
          {accounts.map((a, i) => (
            <AccountCard
              key={i}
              index={i}
              account={a}
              canRemove={accounts.length > 1}
              onChange={(p) => setAccount(i, p)}
              onRemove={() => removeAccount(i)}
            />
          ))}
        </ul>
        {duplicateNames && (
          <p className="mt-3 text-xs text-danger">Account names must be unique.</p>
        )}

        {accounts.length >= 2 && (
          <div className="mt-5 rounded-xl border bg-card">
            <Accordion type="single" collapsible>
              <AccordionItem value="corr" className="border-b-0">
                <AccordionTrigger className="px-4">
                  <div className="flex flex-col items-start text-left">
                    <span className="text-sm font-medium text-foreground">
                      How your accounts move together
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Optional · advanced. Leave alone if unsure.
                    </span>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="px-4 pb-4">
                  <CorrelationEditor
                    accounts={accounts}
                    value={correlation}
                    onChange={setCorrelation}
                  />
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        )}
      </section>

      {/* ---- Footer actions ---- */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card px-4 py-3">
        <div>
          {onCancel && (
            <Button variant="ghost" size="sm" onClick={onCancel}>
              <X className="size-4" aria-hidden /> Cancel
            </Button>
          )}
        </div>
        <Button size="sm" onClick={handleSave} disabled={!canSave}>
          <Save className="size-4" aria-hidden /> Save situation
        </Button>
      </div>
    </div>
  );
}

/* ---------------- Income section ---------------- */

function IncomeSection({
  income,
  onChange,
}: {
  income: IncomeConfig;
  onChange: (p: Partial<IncomeConfig>) => void;
}) {
  const salary = income.fixed?.base ?? 0;
  const rateFixed = Number(income.contribution_rate_fixed) || 0;
  const ratePct = Math.round(rateFixed * 100);
  const monthlySaved = Math.round(salary * rateFixed);

  const variable = income.variable;
  const variableEnabled = !!variable;
  const rateVar = Number(income.contribution_rate_variable) || 0;

  const setFixed = (patch: Partial<NonNullable<IncomeConfig["fixed"]>>) => {
    onChange({ fixed: { base: 0, annual_growth: 0.03, ...income.fixed, ...patch } });
  };
  const setVariable = (patch: Partial<NonNullable<IncomeConfig["variable"]>> | null) => {
    if (patch === null) {
      onChange({ variable: undefined });
      return;
    }
    onChange({
      variable: {
        base: 0,
        sigma: 0.2,
        ...variable,
        ...patch,
      },
    });
  };

  return (
    <section className="rounded-2xl border bg-card p-5 sm:p-6">
      <header>
        <h2 className="text-base font-semibold text-foreground">What you earn</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Your monthly salary and what share of it you set aside to invest.
        </p>
      </header>

      <div className="mt-5 grid gap-5 sm:grid-cols-2">
        <Field label="Monthly salary">
          <MoneyInput value={salary} onChange={(v) => setFixed({ base: v })} />
        </Field>
        <div>
          <div className="flex items-baseline justify-between">
            <Label className="text-sm font-medium">How much of it do you invest?</Label>
            <span className="tabular text-sm font-semibold text-foreground">{ratePct}%</span>
          </div>
          <Slider
            value={[ratePct]}
            min={0}
            max={100}
            step={1}
            onValueChange={([v]) => onChange({ contribution_rate_fixed: v / 100 })}
            className="mt-3"
          />
          <p className="mt-2 text-xs text-muted-foreground">
            You invest {ratePct}% — about{" "}
            <span className="tabular font-medium text-foreground">{formatCLP(monthlySaved)}</span>{" "}
            per month.
          </p>
        </div>
      </div>

      <Accordion type="multiple" className="mt-5 w-full">
        <AccordionItem value="growth">
          <AccordionTrigger>
            <div className="flex flex-col items-start text-left">
              <span className="text-sm font-medium text-foreground">Salary growth & raises</span>
              <span className="text-xs text-muted-foreground">
                Optional · expected yearly raise and one-off bumps.
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-5">
            <Field label="Expected yearly raise">
              <PercentInput
                value={income.fixed?.annual_growth ?? 0}
                onChange={(v) => setFixed({ annual_growth: v })}
              />
            </Field>
            <RaisesEditor
              raises={income.fixed?.salary_raises ?? {}}
              onChange={(r) => setFixed({ salary_raises: Object.keys(r).length ? r : undefined })}
            />
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="variable">
          <AccordionTrigger>
            <div className="flex flex-col items-start text-left">
              <span className="text-sm font-medium text-foreground">
                Bonuses or commissions? (variable income)
              </span>
              <span className="text-xs text-muted-foreground">
                Optional · income that changes month-to-month.
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-5">
            {!variableEnabled ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setVariable({ base: 0, sigma: 0.2 })}
              >
                <Plus className="size-4" aria-hidden /> Add variable income
              </Button>
            ) : (
              <>
                <div className="grid gap-5 sm:grid-cols-2">
                  <Field label="Typical amount">
                    <MoneyInput
                      value={variable!.base}
                      onChange={(v) => setVariable({ base: v })}
                    />
                  </Field>
                  <div>
                    <Label className="text-sm font-medium">How much does it vary?</Label>
                    <VariabilityPicker
                      value={variable!.sigma}
                      onChange={(v) => setVariable({ sigma: v })}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex items-baseline justify-between">
                    <Label className="text-sm font-medium">
                      How much of bonuses do you invest?
                    </Label>
                    <span className="tabular text-sm font-semibold text-foreground">
                      {Math.round(rateVar * 100)}%
                    </span>
                  </div>
                  <Slider
                    value={[Math.round(rateVar * 100)]}
                    min={0}
                    max={100}
                    step={1}
                    onValueChange={([v]) => onChange({ contribution_rate_variable: v / 100 })}
                    className="mt-3"
                  />
                </div>

                <Accordion type="single" collapsible>
                  <AccordionItem value="adv" className="border-b-0">
                    <AccordionTrigger className="text-sm">Advanced</AccordionTrigger>
                    <AccordionContent className="space-y-5">
                      <div className="grid gap-5 sm:grid-cols-2">
                        <Field label="Minimum per month (floor)">
                          <MoneyInput
                            value={variable!.floor ?? 0}
                            onChange={(v) =>
                              setVariable({ floor: v > 0 ? v : undefined })
                            }
                          />
                        </Field>
                        <Field label="Maximum per month (cap)">
                          <MoneyInput
                            value={variable!.cap ?? 0}
                            onChange={(v) => setVariable({ cap: v > 0 ? v : undefined })}
                          />
                        </Field>
                      </div>
                      <SeasonalityEditor
                        value={variable!.seasonality ?? null}
                        onChange={(s) => setVariable({ seasonality: s ?? undefined })}
                      />
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setVariable(null)}
                  className="text-muted-foreground"
                >
                  <Trash2 className="size-4" aria-hidden /> Remove variable income
                </Button>
              </>
            )}
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </section>
  );
}

/* ---------------- Account card ---------------- */

function AccountCard({
  index,
  account,
  canRemove,
  onChange,
  onRemove,
}: {
  index: number;
  account: AccountConfig;
  canRemove: boolean;
  onChange: (p: Partial<AccountConfig>) => void;
  onRemove: () => void;
}) {
  const risk = detectRisk(account);

  const pickRisk = (id: RiskId) => {
    const preset = RISK_PRESETS.find((p) => p.id === id);
    if (!preset || preset.id === "custom") {
      // Stay with current numbers (treated as custom)
      return;
    }
    onChange({
      annual_return: (preset as { ret: number }).ret,
      annual_volatility: (preset as { vol: number }).vol,
    });
  };

  // When user clicks Custom, we don't change numbers; we just want the UI to show custom inputs.
  // Detection above will show custom when numbers don't match any preset.
  const [forceCustom, setForceCustom] = React.useState(false);
  const showCustom = risk === "custom" || forceCustom;

  return (
    <li className="rounded-xl border bg-muted/30 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-foreground">Account {index + 1}</p>
        {canRemove && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRemove}
            aria-label={`Remove account ${index + 1}`}
          >
            <Trash2 className="size-4" aria-hidden /> Remove
          </Button>
        )}
      </div>

      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <Field label="Account name">
          <Input
            value={account.display_name ?? ""}
            onChange={(e) => onChange({ display_name: e.target.value })}
            placeholder="e.g. Safe savings"
            className="h-10"
          />
        </Field>
        <Field label="How much is already there?">
          <MoneyInput
            value={account.initial_wealth}
            onChange={(v) => onChange({ initial_wealth: v })}
          />
        </Field>
      </div>

      <div className="mt-4">
        <Label className="text-sm font-medium">Risk</Label>
        <div
          role="radiogroup"
          aria-label="Risk preset"
          className="mt-2 grid gap-2 sm:grid-cols-4"
        >
          {RISK_PRESETS.map((p) => {
            const selected =
              p.id === "custom" ? showCustom && risk === "custom" : !showCustom && risk === p.id;
            return (
              <button
                type="button"
                key={p.id}
                role="radio"
                aria-checked={selected}
                onClick={() => {
                  if (p.id === "custom") {
                    setForceCustom(true);
                  } else {
                    setForceCustom(false);
                    pickRisk(p.id);
                  }
                }}
                className={cn(
                  "rounded-lg border bg-card p-3 text-left transition-all",
                  "hover:border-primary/40",
                  selected ? "border-primary ring-2 ring-primary/30" : "border-border",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium text-foreground">{p.title}</p>
                  {"tag" in p && p.tag && (
                    <span className="rounded-full bg-success-soft px-1.5 py-0.5 text-[10px] font-medium text-success-foreground ring-1 ring-success/30">
                      {p.tag}
                    </span>
                  )}
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">{p.subtitle}</p>
              </button>
            );
          })}
        </div>

        {showCustom && (
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Field label="Expected annual return">
              <PercentInput
                value={account.annual_return}
                onChange={(v) => onChange({ annual_return: v })}
              />
            </Field>
            <Field label="Volatility (how bumpy)">
              <PercentInput
                value={account.annual_volatility}
                onChange={(v) => onChange({ annual_volatility: v })}
                min={0}
              />
            </Field>
          </div>
        )}
      </div>
    </li>
  );
}

/* ---------------- Variability picker ---------------- */

function VariabilityPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const current = detectVariability(value);
  const [forceCustom, setForceCustom] = React.useState(false);
  const showCustom = current === "custom" || forceCustom;

  return (
    <>
      <div role="radiogroup" className="mt-2 grid gap-2 sm:grid-cols-4">
        {VARIABILITY.map((v) => {
          const selected =
            v.id === "custom"
              ? showCustom && current === "custom"
              : !showCustom && current === v.id;
          return (
            <button
              type="button"
              key={v.id}
              role="radio"
              aria-checked={selected}
              onClick={() => {
                if (v.id === "custom") {
                  setForceCustom(true);
                } else {
                  setForceCustom(false);
                  onChange((v as { sigma: number }).sigma);
                }
              }}
              className={cn(
                "rounded-lg border bg-card p-2.5 text-left transition-all",
                "hover:border-primary/40",
                selected ? "border-primary ring-2 ring-primary/30" : "border-border",
              )}
            >
              <p className="text-sm font-medium text-foreground">{v.title}</p>
              <p className="text-xs text-muted-foreground">{v.subtitle}</p>
            </button>
          );
        })}
      </div>
      {showCustom && (
        <div className="mt-3">
          <Field label="Variability">
            <PercentInput value={value} onChange={onChange} min={0} />
          </Field>
        </div>
      )}
    </>
  );
}

/* ---------------- Raises editor ---------------- */

function RaisesEditor({
  raises,
  onChange,
}: {
  raises: Record<string, number>;
  onChange: (r: Record<string, number>) => void;
}) {
  const entries = Object.entries(raises).sort(([a], [b]) => a.localeCompare(b));
  const [date, setDate] = React.useState("");
  const [amount, setAmount] = React.useState(0);

  const add = () => {
    if (!date || amount <= 0) return;
    onChange({ ...raises, [date]: amount });
    setDate("");
    setAmount(0);
  };
  const remove = (k: string) => {
    const next = { ...raises };
    delete next[k];
    onChange(next);
  };

  return (
    <div>
      <Label className="text-sm font-medium">Specific raises</Label>
      {entries.length > 0 && (
        <ul className="mt-2 space-y-2">
          {entries.map(([d, amt]) => (
            <li
              key={d}
              className="flex items-center justify-between rounded-md border bg-card px-3 py-2"
            >
              <span className="text-sm text-foreground">{d}</span>
              <div className="flex items-center gap-3">
                <span className="tabular text-sm font-medium">{formatCLP(amt)}</span>
                <Button variant="ghost" size="sm" onClick={() => remove(d)}>
                  <Trash2 className="size-4" aria-hidden />
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-3 grid items-end gap-3 sm:grid-cols-[1fr_1fr_auto]">
        <Field label="Date">
          <Input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="h-10"
          />
        </Field>
        <Field label="New salary">
          <MoneyInput value={amount} onChange={setAmount} />
        </Field>
        <Button variant="outline" size="sm" onClick={add} disabled={!date || amount <= 0}>
          <Plus className="size-4" aria-hidden /> Add
        </Button>
      </div>
    </div>
  );
}

/* ---------------- Seasonality editor ---------------- */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function SeasonalityEditor({
  value,
  onChange,
}: {
  value: number[] | null;
  onChange: (v: number[] | null) => void;
}) {
  const enabled = !!value && value.length === 12;
  const arr = enabled ? value! : Array(12).fill(1);

  const setMonth = (i: number, v: number) => {
    const next = arr.slice();
    next[i] = v;
    onChange(next);
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-sm font-medium">Monthly pattern</Label>
          <p className="text-xs text-muted-foreground">
            1.0 = typical month. 1.5 = 50% higher than typical.
          </p>
        </div>
        {enabled ? (
          <Button variant="ghost" size="sm" onClick={() => onChange(null)}>
            <Trash2 className="size-4" aria-hidden /> Clear
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={() => onChange(Array(12).fill(1))}>
            <Plus className="size-4" aria-hidden /> Enable
          </Button>
        )}
      </div>
      {enabled && (
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {MONTHS.map((m, i) => (
            <div key={m} className="rounded-md border bg-card p-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">{m}</span>
                <span className="tabular text-xs font-medium">{arr[i].toFixed(2)}</span>
              </div>
              <Slider
                value={[Math.round(arr[i] * 100)]}
                min={0}
                max={300}
                step={5}
                onValueChange={([v]) => setMonth(i, v / 100)}
                className="mt-2"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------------- Correlation editor ---------------- */

function CorrelationEditor({
  accounts,
  value,
  onChange,
}: {
  accounts: AccountConfig[];
  value: number[][] | null;
  onChange: (v: number[][] | null) => void;
}) {
  const n = accounts.length;
  const matrix = React.useMemo(() => {
    if (value && value.length === n && value.every((r) => r.length === n)) return value;
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
    );
  }, [value, n]);

  const setPair = (i: number, j: number, v: number) => {
    const next = matrix.map((r) => r.slice());
    next[i][j] = v;
    next[j][i] = v;
    onChange(next);
  };

  const pairs: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        −1 means they move opposite, 0 means independent, +1 means they move together.
      </p>
      <ul className="space-y-3">
        {pairs.map(([i, j]) => {
          const v = matrix[i][j];
          return (
            <li key={`${i}-${j}`} className="rounded-md border bg-card p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">
                  {accounts[i].display_name || `Account ${i + 1}`} ↔{" "}
                  {accounts[j].display_name || `Account ${j + 1}`}
                </span>
                <span className="tabular text-sm font-medium">{v.toFixed(2)}</span>
              </div>
              <Slider
                value={[Math.round(v * 100)]}
                min={-100}
                max={100}
                step={5}
                onValueChange={([nv]) => setPair(i, j, nv / 100)}
                className="mt-3"
              />
            </li>
          );
        })}
      </ul>
      {value && (
        <Button variant="ghost" size="sm" onClick={() => onChange(null)}>
          Reset to independent
        </Button>
      )}
    </div>
  );
}

/* ---------------- shared atoms ---------------- */

function Field({
  label,
  error,
  children,
}: {
  label: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      <div className="mt-1.5">{children}</div>
      {error && <p className="mt-1 text-xs text-danger">{error}</p>}
    </div>
  );
}

function PercentInput({
  value,
  onChange,
  min,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
}) {
  const [text, setText] = React.useState((value * 100).toString());
  React.useEffect(() => {
    const parsed = parseFloat(text);
    if (!Number.isFinite(parsed) || Math.abs(parsed / 100 - value) > 1e-6) {
      setText((value * 100).toString());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <div className="flex h-10 items-center rounded-md border border-input bg-transparent text-sm shadow-sm focus-within:ring-1 focus-within:ring-ring">
      <input
        inputMode="decimal"
        value={text}
        onChange={(e) => {
          const t = e.target.value;
          setText(t);
          const n = parseFloat(t);
          if (Number.isFinite(n)) {
            const dec = n / 100;
            if (min !== undefined && dec < min) return;
            onChange(dec);
          } else if (t === "" || t === "-") {
            onChange(0);
          }
        }}
        className="tabular w-full bg-transparent px-3 py-2 text-right font-medium outline-none"
      />
      <span className="select-none pr-3 text-muted-foreground">%</span>
    </div>
  );
}
