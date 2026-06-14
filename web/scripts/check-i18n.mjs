#!/usr/bin/env node
/**
 * i18n guard. Three checks, all gating CI (non-zero exit on any failure):
 *
 *   1. Parity   — every language exposes the same flattened key set per
 *                 namespace (catches a string translated in one language only).
 *   2. Usage    — every literal key referenced in the code (`t('…')`,
 *                 `i18nKey="…"`) exists, scoped to the namespace(s) the file
 *                 actually declares via `useTranslation(...)`. Catches typos and
 *                 wrong-namespace bindings that render the raw key at runtime
 *                 (tsc/build never see these — keys are untyped strings).
 *   3. Dynamic  — template keys `t(`prefix.${id}.suffix`)` can't be read
 *                 statically, so each family is registered in DYNAMIC_KEY_FAMILIES
 *                 below and every id×template expansion is verified to exist.
 *                 ⚠ When you add a new dynamic-key pattern in the UI, register it
 *                 here or it goes unchecked.
 *
 * Run: `npm run i18n:check`.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, relative } from "node:path";
import { execSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = join(here, "..");
const localesDir = join(webRoot, "src", "i18n", "locales");

/* ------------------------------------------------------------------ *
 * Dynamic-key families: `t(`<prefix>.${id}.<suffix>`)` expansions.
 * Each entry: { ns, templates: ["a.{id}.b", …], ids: [...] }.
 * ------------------------------------------------------------------ */
const DYNAMIC_KEY_FAMILIES = [
  { ns: "situation", templates: ["risk.{id}.title", "risk.{id}.subtitle"], ids: ["savings", "conservative", "balanced", "aggressive", "custom"] },
  { ns: "situation", templates: ["risk.{id}.range"], ids: ["savings", "conservative", "balanced", "aggressive"] }, // custom has no range (guarded in UI)
  { ns: "situation", templates: ["variability.{id}.title", "variability.{id}.subtitle"], ids: ["low", "medium", "high", "custom"] },
  { ns: "common", templates: ["objectives.{id}.title"], ids: ["risky", "proportional", "conservative", "balanced", "risky_turnover"] },
  { ns: "common", templates: ["objectives.{id}.subtitle"], ids: ["risky", "proportional", "conservative", "balanced"] },
  { ns: "common", templates: ["confidence.{id}.label", "confidence.{id}.blurb"], ids: ["reach_sooner", "balanced", "play_it_safe"] },
  { ns: "common", templates: ["profileRisk.{id}"], ids: ["none", "growth", "balanced", "lower"] },
  { ns: "goals", templates: ["steps.{id}"], ids: ["basics", "goals", "review"] },
  { ns: "plan", templates: ["hero.status.{id}.label", "hero.status.{id}.description"], ids: ["feasible", "tight", "infeasible"] },
  { ns: "plan", templates: ["explainer.concepts.{id}"], ids: ["wealth", "contribution", "allocation", "withdrawal", "return", "median", "likely-band", "possible-band", "goal-probability", "goal-target", "horizon"] },
  { ns: "scenarios", templates: ["status.{id}"], ids: ["needsChanges", "tight", "failed", "calculating", "queued", "completed", "onTrack", "needsRun"] },
  { ns: "results", templates: ["status.{id}"], ids: ["completed", "running", "failed", "pending"] },
];

/* ------------------------------------------------------------------ *
 * Load locales
 * ------------------------------------------------------------------ */
function flatten(obj, prefix = "", out = new Set()) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) flatten(v, key, out);
    else out.add(key);
  }
  return out;
}

function loadLang(lang) {
  const dir = join(localesDir, lang);
  const namespaces = {};
  for (const file of readdirSync(dir)) {
    if (!file.endsWith(".json")) continue;
    namespaces[file.replace(/\.json$/, "")] = flatten(JSON.parse(readFileSync(join(dir, file), "utf8")));
  }
  return namespaces;
}

const langs = readdirSync(localesDir).filter((d) => statSync(join(localesDir, d)).isDirectory());
if (langs.length === 0) {
  console.error("i18n: no locale directories found.");
  process.exit(1);
}
const byLang = Object.fromEntries(langs.map((l) => [l, loadLang(l)]));
const [base, ...others] = langs;
const baseNs = byLang[base];

/** A key resolves if it (or its plural variants) exists in the namespace. */
const hasKey = (nsSet, key) => !!nsSet && (nsSet.has(key) || nsSet.has(key + "_one") || nsSet.has(key + "_other"));

let problems = 0;
const fail = (msg) => { console.error("  ✗ " + msg); problems++; };

/* ------------------------------------------------------------------ *
 * Check 1 — parity across languages
 * ------------------------------------------------------------------ */
for (const lang of others) {
  const a = baseNs, b = byLang[lang];
  const names = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const name of names) {
    if (!a[name]) { fail(`[parity] namespace "${name}" only in [${lang}]`); continue; }
    if (!b[name]) { fail(`[parity] namespace "${name}" only in [${base}]`); continue; }
    for (const key of a[name]) if (!b[name].has(key)) fail(`[parity] ${name}:${key} in [${base}] but missing in [${lang}]`);
    for (const key of b[name]) if (!a[name].has(key)) fail(`[parity] ${name}:${key} in [${lang}] but missing in [${base}]`);
  }
}

/* ------------------------------------------------------------------ *
 * Check 2 — every literal key referenced in code exists (ns-scoped)
 * ------------------------------------------------------------------ */
const sourceFiles = execSync(
  'grep -rlE "useTranslation|i18nKey|\\bt\\(" src --include=*.tsx --include=*.ts',
  { cwd: webRoot },
).toString().trim().split("\n").filter(Boolean);

const reUse = /useTranslation\(\s*(\[[^\]]*\]|["'][^"']*["'])\s*\)/g;
const reLit = /(?:\bt\(|i18nKey=)\s*["']([^"'`${}]+)["']/g;
const DEFAULT_NS = "common"; // i18next defaultNS in config.ts

let usageChecked = 0;
for (const rel of sourceFiles) {
  const src = readFileSync(join(webRoot, rel), "utf8");

  const declared = new Set();
  let u;
  while ((u = reUse.exec(src))) for (const q of u[1].match(/["']([^"']+)["']/g) || []) declared.add(q.slice(1, -1));
  // Files that receive `t` as a param (helpers) declare no namespace; their keys
  // use explicit `ns:` prefixes, so fall back to all namespaces for bare keys.
  const scope = declared.size > 0 ? [...declared] : [DEFAULT_NS, ...Object.keys(baseNs)];

  let m;
  while ((m = reLit.exec(src))) {
    const raw = m[1];
    // Heuristic: keys contain a "." or ":"; plain words that happen to be keys
    // (e.g. "remove") are covered too as long as they live in a declared ns.
    usageChecked++;
    let ok;
    if (raw.includes(":")) {
      const [ns, ...r] = raw.split(":");
      ok = hasKey(baseNs[ns], r.join(":"));
    } else {
      ok = scope.some((ns) => hasKey(baseNs[ns], raw));
    }
    if (!ok) fail(`[usage] ${relative(webRoot, join(webRoot, rel))}: "${raw}" not found in ns [${raw.includes(":") ? raw.split(":")[0] : scope.join(", ")}]`);
  }
}

/* ------------------------------------------------------------------ *
 * Check 3 — dynamic key families
 * ------------------------------------------------------------------ */
let dynamicChecked = 0;
for (const { ns, templates, ids } of DYNAMIC_KEY_FAMILIES) {
  for (const id of ids) for (const tmpl of templates) {
    dynamicChecked++;
    if (!hasKey(baseNs[ns], tmpl.replace("{id}", id))) fail(`[dynamic] ${ns}:${tmpl.replace("{id}", id)} (registered family) missing`);
  }
}

/* ------------------------------------------------------------------ */
if (problems > 0) {
  console.error(`\ni18n check FAILED: ${problems} problem(s).`);
  process.exit(1);
}
console.log(
  `✓ i18n check passed — parity [${langs.join(", ")}], ${usageChecked} literal + ${dynamicChecked} dynamic keys verified against locales.`,
);
