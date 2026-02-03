# Plan: Mejoras API y Frontend - FinOpt Web

## Estado Actual

**Completado:**
- [x] Supabase configurado (proyecto, migrations, auth)
- [x] API Python funcionando localmente
- [x] Frontend React funcionando localmente
- [x] Flujo básico: Login → Profile → Scenario → Run Optimization

**Problemas Identificados:**
- Mismatch entre formatos frontend ↔ API (ya parcialmente corregido)
- Falta validación en frontend antes de enviar datos
- UI/UX básica sin feedback adecuado
- Sin manejo de errores robusto
- Frontend no captura toda la complejidad del modelo (income variable, withdrawals, etc.)

---

## Modelo de Datos Completo

### 1. Income Model (Perfil de Ingresos)

Total monthly income: $Y_t = y_t^{\text{fixed}} + Y_t^{\text{variable}}$

#### 1.1 Fixed Income (Salario)

```python
fixed_income = FixedIncome(
    base=1_495_000.0,           # CLP/month (base salary)
    annual_growth=0.03,         # 3% nominal annual growth
    salary_raises={
        date(2026, 3, 1): 400_000,  # +400k promotion
        date(2028, 6, 1): 500_000,  # +500k tenure
    },
    name="fixed"
)
```

**Campos requeridos en frontend:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `base` | number | Salario base mensual |
| `annual_growth` | number (0-0.2) | Crecimiento anual (ej: 0.03 = 3%) |
| `salary_raises` | array | Lista de aumentos programados: `[{date, amount}]` |

#### 1.2 Variable Income (Bonos, Comisiones)

$$Y_t^{\text{variable}} = \text{clip}\big(\mu_t (1 + \epsilon_t), \text{floor}, \text{cap}\big)$$

```python
seasonality = [0.0, 0.0, 0.0, 0.6, 1.0, 1.16, 1.0, 1.1, 0.5, 0.9, 0.85, 1.0]

variable_income = VariableIncome(
    base=40_000.0,              # Base mensual esperado
    seasonality=seasonality,    # Factores por mes (Ene-Dic)
    sigma=0.1,                  # 10% ruido mensual
    floor=0.0,                  # Mínimo
    cap=400_000.0,              # Máximo
    annual_growth=0.0,
    name="variable"
)
```

**Campos requeridos en frontend:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `base` | number | Ingreso variable base |
| `sigma` | number (0-0.5) | Volatilidad (ruido) |
| `seasonality` | number[12] | Factor por mes (0=nada, 1=normal, 2=doble) |
| `floor` | number | Mínimo garantizado |
| `cap` | number | Máximo posible |
| `annual_growth` | number | Crecimiento anual |

#### 1.3 Contribution Rates (Tasas de Ahorro)

$$A_t = \alpha_{(t \mod 12)}^{f} \cdot y_t^{\text{fixed}} + \alpha_{(t \mod 12)}^{v} \cdot Y_t^{\text{variable}}$$

```python
monthly_contribution = {
    "fixed": [0.45]*11 + [0.30],    # 45% excepto Dic (30%)
    "variable": [1.0] * 12          # 100% de variable
}
```

**Campos requeridos en frontend:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `contribution_rate_fixed` | number \| number[12] | % del salario a invertir |
| `contribution_rate_variable` | number \| number[12] | % del variable a invertir |

---

### 2. Investment Accounts (Cuentas de Inversión)

```python
accounts = [
    Account.from_annual(
        name="Vivienda",
        display_name="Cuenta Ahorro Vivienda (BE)",
        annual_return=0.025,
        annual_volatility=0.01,
        initial_wealth=1_600_000
    ),
    Account.from_annual(
        name="Reserva",
        display_name="Fondo Reserva Fintual",
        annual_return=0.042,
        annual_volatility=0.01,
        initial_wealth=700_000
    ),
    Account.from_annual(
        name="VT",
        display_name="VT Global ETF",
        annual_return=0.14,
        annual_volatility=0.15,
        initial_wealth=1_842_158
    )
]
```

**Campos por cuenta:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `name` | string | Identificador único (usado en goals) |
| `display_name` | string (opcional) | Nombre para mostrar en UI |
| `annual_return` | number (0-0.3) | Retorno anual esperado |
| `annual_volatility` | number (0-0.3) | Volatilidad anual |
| `initial_wealth` | number | Riqueza inicial |

#### 2.1 Correlation Matrix (Opcional)

```python
correlation_dict = {
    ('VT', 'Reserva'): 0.3,
    ('VT', 'Vivienda'): 0.1,
    ('Reserva', 'Vivienda'): 0.2
}
```

**Representación en DB:** Matriz NxN o diccionario de pares.

---

### 3. Withdrawals (Retiros Planificados)

Dinámica con retiros:
$$W_{t+1}^m = (W_t^m + A_t \cdot x_t^m - D_t^m)(1 + R_t^m)$$

Restricción de factibilidad:
$$\mathbb{P}(W_t^m \geq D_t^m) \geq 1 - \delta$$

#### 3.1 WithdrawalEvent (Determinístico)

```python
withdrawals = [
    WithdrawalEvent(
        account="Vivienda",
        amount=15_000_000,
        date=date(2025, 7, 1),
        description="Pie departamento"
    ),
    WithdrawalEvent(
        account="Reserva",
        amount=2_000_000,
        date=date(2026, 1, 1),
        description="Vacaciones"
    )
]
```

**Campos por retiro:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `account` | string | Nombre de cuenta |
| `amount` | number | Monto a retirar |
| `date` | date | Fecha del retiro |
| `description` | string (opcional) | Descripción |

#### 3.2 StochasticWithdrawal (Con incertidumbre)

```python
stochastic_withdrawals = [
    StochasticWithdrawal(
        account="Reserva",
        expected_amount=500_000,
        volatility=0.2,
        date=date(2025, 12, 1),
        description="Gastos navidad"
    )
]
```

---

### 4. Financial Goals (Metas Financieras)

#### 4.1 IntermediateGoal (Checkpoint intermedio)

$$\mathbb{P}(W_{t_{\text{fixed}}}^m \geq b) \geq 1 - \varepsilon$$

```python
intermediate_goals = [
    IntermediateGoal(
        account="Vivienda",
        threshold=20_000_000,
        confidence=0.95,
        date=date(2025, 6, 1)
    )
]
```

#### 4.2 TerminalGoal (Meta al final del horizonte)

$$\mathbb{P}(W_T^m \geq b) \geq 1 - \varepsilon$$

```python
terminal_goals = [
    TerminalGoal(
        account="VT",
        threshold=50_000_000,
        confidence=0.80
    )
]
```

**Campos por goal:**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `account` | string | Nombre de cuenta |
| `threshold` | number | Monto objetivo |
| `confidence` | number (0.5-0.99) | Probabilidad requerida |
| `date` | date (solo intermediate) | Fecha del checkpoint |

---

### 5. Optimization Problem

**Problema bilevel:**
$$\min_{T \in \mathbb{N}} T \quad \text{s.t.} \quad \mathcal{F}_T \neq \emptyset$$

Con CVaR reformulation para constraints probabilísticos.

**Parámetros de optimización:**
| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `t_max` | number | 240 | Horizonte máximo (meses) |
| `t_min` | number | 12 | Horizonte mínimo |
| `n_sims` | number | 500 | Simulaciones Monte Carlo |
| `seed` | number | null | Semilla para reproducibilidad |
| `solver` | string | "ECOS" | Solver CVXPY |
| `objective` | string | "balanced" | Función objetivo |
| `withdrawal_epsilon` | number | 0.05 | Tolerancia retiros |

---

## Fase A: Correcciones Críticas de Compatibilidad

### A.1 Actualizar tipos en Frontend

**Archivo:** `web/src/types/database.ts`

```typescript
export interface IncomeConfig {
  fixed?: {
    base: number
    annual_growth: number
    salary_raises?: Array<{ date: string; amount: number }>
  }
  variable?: {
    base: number
    sigma: number
    annual_growth?: number
    seasonality?: number[]  // 12 elementos
    floor?: number
    cap?: number
  }
  contribution_rate_fixed: number | number[]
  contribution_rate_variable: number | number[]
}

export interface AccountConfig {
  name: string
  display_name?: string
  annual_return: number
  annual_volatility: number
  initial_wealth: number
}

export interface WithdrawalEvent {
  account: string
  amount: number
  date: string
  description?: string
}

export interface StochasticWithdrawal {
  account: string
  expected_amount: number
  volatility: number
  date: string
  description?: string
}

export interface Goal {
  account: string
  threshold: number
  confidence: number
  date?: string  // Solo para IntermediateGoal
}
```

### A.2 Actualizar schema Supabase

Verificar que `scenarios.withdrawals` soporte:
```json
{
  "scheduled": [{"account": "X", "amount": 1000, "date": "2025-07-01"}],
  "stochastic": [{"account": "Y", "expected_amount": 500, "volatility": 0.2, "date": "2025-12-01"}]
}
```

---

## Fase B: Mejoras de API

### B.1 Validación robusta

**Archivo:** `api/services/reconstruction.py`

- [ ] Validar que `account` en goals existe en `accounts_config`
- [ ] Validar que `account` en withdrawals existe en `accounts_config`
- [ ] Validar fechas de withdrawals < horizonte máximo
- [ ] Validar fechas de intermediate_goals coherentes

### B.2 Mensajes de error descriptivos

```python
class OptimizationError(Exception):
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code

# Códigos:
# INVALID_ACCOUNT: cuenta referenciada no existe
# INFEASIBLE: no hay solución factible
# SOLVER_ERROR: error en CVXPY
# TIMEOUT: optimización excedió tiempo
```

### B.3 Progress callbacks

Modificar `model.optimize()` para aceptar callback de progreso:
```python
def progress_callback(step: str, progress: int):
    update_job(job_id, progress=progress, step=step)

result = model.optimize(
    ...,
    progress_callback=progress_callback
)
```

---

## Fase C: Mejoras de Frontend

### C.1 ProfileForm Completo

**Nueva estructura de secciones:**

```
┌─────────────────────────────────────────────┐
│ Profile Name & Description                   │
├─────────────────────────────────────────────┤
│ 1. FIXED INCOME                             │
│    ├─ Base salary                           │
│    ├─ Annual growth %                       │
│    └─ Scheduled raises (add/remove)         │
├─────────────────────────────────────────────┤
│ 2. VARIABLE INCOME (optional)               │
│    ├─ Base amount                           │
│    ├─ Volatility (sigma)                    │
│    ├─ Seasonality (12 sliders or chart)     │
│    ├─ Floor / Cap                           │
│    └─ Annual growth                         │
├─────────────────────────────────────────────┤
│ 3. CONTRIBUTION RATES                       │
│    ├─ Fixed income: single % or 12 months   │
│    └─ Variable income: single % or 12 months│
├─────────────────────────────────────────────┤
│ 4. INVESTMENT ACCOUNTS (add/remove)         │
│    └─ Per account:                          │
│        ├─ Name (unique)                     │
│        ├─ Display name                      │
│        ├─ Annual return %                   │
│        ├─ Annual volatility %               │
│        └─ Initial wealth                    │
├─────────────────────────────────────────────┤
│ 5. CORRELATION (optional, advanced)         │
│    └─ Matrix or pair inputs                 │
└─────────────────────────────────────────────┘
```

### C.2 ScenarioForm Completo

**Nueva estructura:**

```
┌─────────────────────────────────────────────┐
│ Scenario Name & Profile Selection            │
├─────────────────────────────────────────────┤
│ 1. SIMULATION PARAMETERS                    │
│    ├─ Start date                            │
│    ├─ Number of simulations                 │
│    └─ Random seed (optional)                │
├─────────────────────────────────────────────┤
│ 2. OPTIMIZATION PARAMETERS                  │
│    ├─ T_max (max horizon)                   │
│    ├─ Solver (ECOS, SCS, CLARABEL)          │
│    └─ Objective (balanced, risky, conserv.) │
├─────────────────────────────────────────────┤
│ 3. WITHDRAWALS (add/remove)                 │
│    └─ Per withdrawal:                       │
│        ├─ Account (dropdown from profile)   │
│        ├─ Amount                            │
│        ├─ Date                              │
│        ├─ Type: Deterministic/Stochastic    │
│        └─ Volatility (if stochastic)        │
├─────────────────────────────────────────────┤
│ 4. INTERMEDIATE GOALS (add/remove)          │
│    └─ Per goal:                             │
│        ├─ Account (dropdown)                │
│        ├─ Target amount                     │
│        ├─ Confidence %                      │
│        └─ Target date                       │
├─────────────────────────────────────────────┤
│ 5. TERMINAL GOALS (add/remove)              │
│    └─ Per goal:                             │
│        ├─ Account (dropdown)                │
│        ├─ Target amount                     │
│        └─ Confidence %                      │
└─────────────────────────────────────────────┘
```

### C.3 Componentes UI Nuevos

| Componente | Descripción |
|------------|-------------|
| `SeasonalityEditor.tsx` | 12 sliders o bar chart editable |
| `MonthlyRatesEditor.tsx` | Editor de tasas por mes |
| `AccountCard.tsx` | Card con info de cuenta |
| `GoalCard.tsx` | Card con info de goal |
| `WithdrawalCard.tsx` | Card con info de retiro |
| `CorrelationMatrix.tsx` | Editor visual de correlaciones |

### C.4 Results Page Mejorada

- [ ] Gráfico de trayectorias de riqueza por cuenta
- [ ] Timeline visual de withdrawals y goals
- [ ] Allocation heatmap mejorado con nombres de cuenta
- [ ] Resumen de goal satisfaction con probabilidades

---

## Fase D: Deploy a Producción

(Sin cambios respecto al plan original)

---

## Fase E: Mejoras Futuras

### E.1 Wizard de creación

Guía paso a paso para usuarios nuevos:
1. "¿Tienes ingresos fijos?" → Configura salario
2. "¿Tienes ingresos variables?" → Configura bonos
3. "¿En qué cuentas inviertes?" → Agrega cuentas
4. "¿Qué retiros planeas?" → Agrega withdrawals
5. "¿Cuáles son tus metas?" → Agrega goals

### E.2 Templates predefinidos

- "Empleado promedio": Salario fijo, 2 cuentas (conservadora + agresiva)
- "Freelancer": Alto variable, volatilidad alta
- "Ahorro vivienda": Meta intermedia para pie

### E.3 Simulación rápida

Antes de optimizar, mostrar:
- Proyección de ingresos totales
- Estimación de horizonte mínimo
- Probabilidad de éxito con allocation ingenua

---

## Prioridades Actualizadas

| Prioridad | Fase | Descripción | Esfuerzo |
|-----------|------|-------------|----------|
| 🔴 Alta | A.1 | Actualizar tipos TypeScript | 2h |
| 🔴 Alta | C.1 | ProfileForm completo | 6h |
| 🔴 Alta | C.2 | ScenarioForm completo | 6h |
| 🟡 Media | B.1-B.2 | Validación y errores API | 3h |
| 🟡 Media | C.3 | Componentes UI nuevos | 4h |
| 🟡 Media | D.* | Deploy producción | 2h |
| 🟢 Baja | C.4 | Results mejorado | 4h |
| 🟢 Baja | E.* | Wizard y templates | 8h |

---

## Comandos Útiles

```bash
# Desarrollo local
cd /home/mlioi/fin-opt
uvicorn api.main:app --reload --port 8000  # API
cd web && npm run dev                       # Frontend

# Build producción
cd web && npm run build

# Type check
cd web && npx tsc --noEmit

# Ver datos en Supabase
python3 -c "
from supabase import create_client
import os
s = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])
print(s.table('profiles').select('*').execute().data)
"
```

---

## Contexto para Futuro Prompt

```
Continúa el deploy de FinOpt. Ver docs/deployment/improvement-plan.md.

Estado: [indicar fase actual]

Modelo matemático:
- Income: Y_t = fixed + variable (con seasonality, raises)
- Wealth: W_{t+1} = (W_t + A_t·x_t - D_t)(1 + R_t)
- Goals: P(W_t >= b) >= 1 - ε
- Optimization: min T s.t. CVaR constraints

Contexto:
- Proyecto: /home/mlioi/fin-opt
- Supabase: https://gbnfipjxwmjjqpuxqkjo.supabase.co
- API: api/ (FastAPI + finopt core)
- Frontend: web/ (React + Vite + Tailwind)
```
