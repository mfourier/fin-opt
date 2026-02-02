# Plan: FinOpt Web Application & Deployment

## Resumen Ejecutivo

Implementar una aplicación web para FinOpt usando una arquitectura simplificada:

- **Supabase**: PostgreSQL + Auth + API REST automática + Realtime
- **Render**: Servicio Python mínimo solo para simulación/optimización
- **Frontend**: React + Vite + Tailwind desplegado en Render/Vercel

---

## 1. Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│              React + Vite + Tailwind (Render/Vercel)            │
└─────────────────────┬───────────────────────┬───────────────────┘
                      │                       │
                      ▼                       ▼
┌─────────────────────────────┐   ┌───────────────────────────────┐
│         SUPABASE            │   │      PYTHON SERVICE           │
│  ┌───────────────────────┐  │   │         (Render)              │
│  │ PostgreSQL (DB)       │  │   │                               │
│  ├───────────────────────┤  │   │  POST /simulate               │
│  │ PostgREST (Auto API)  │  │   │  POST /optimize               │
│  │  - Profiles CRUD      │  │   │  GET  /jobs/{id}/status       │
│  │  - Scenarios CRUD     │  │   │  WS   /ws/jobs/{id}           │
│  │  - Jobs CRUD          │  │   │                               │
│  │  - Results CRUD       │  │   │  (Solo lógica que requiere    │
│  ├───────────────────────┤  │   │   NumPy, CVXPY, FinOpt core)  │
│  │ Auth (usuarios)       │  │   │                               │
│  ├───────────────────────┤  │   └───────────────────────────────┘
│  │ Realtime (WebSocket)  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

### Ventajas de esta arquitectura:
1. **Menos código**: Supabase genera CRUD automáticamente
2. **Auth gratis**: Supabase Auth con magic links, OAuth
3. **Realtime incluido**: WebSockets para progreso sin implementación extra
4. **Escalable**: Servicio Python stateless, fácil de escalar
5. **Costo bajo**: Free tier generoso en Supabase + Render

---

## 2. Estructura de Directorios

```
fin-opt/
├── src/                          # FinOpt core (sin cambios)
├── api/                          # NUEVO: Servicio Python mínimo
│   ├── __init__.py
│   ├── main.py                   # FastAPI app (solo 4 endpoints)
│   ├── config.py                 # Settings (Supabase URL, keys)
│   ├── services/
│   │   ├── simulation.py         # Lógica de simulación
│   │   └── optimization.py       # Lógica de optimización
│   └── supabase_client.py        # Cliente Supabase para jobs/results
├── web/                          # NUEVO: Frontend React
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── lib/
│   │   │   └── supabase.ts       # Cliente Supabase
│   │   ├── components/
│   │   │   ├── Layout/
│   │   │   ├── ProfileForm.tsx
│   │   │   ├── ScenarioForm.tsx
│   │   │   ├── WealthChart.tsx
│   │   │   └── AllocationHeatmap.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Profiles.tsx
│   │   │   ├── Scenarios.tsx
│   │   │   └── Results.tsx
│   │   └── hooks/
│   │       ├── useProfiles.ts    # React Query + Supabase
│   │       └── useJobProgress.ts # Supabase Realtime
│   └── public/
├── supabase/                     # NUEVO: Configuración Supabase
│   ├── migrations/               # SQL migrations
│   │   └── 001_initial_schema.sql
│   └── seed.sql                  # Datos de ejemplo
├── docker/
│   └── Dockerfile.api            # Solo para servicio Python
├── render.yaml                   # Blueprint de Render
└── .env.example
```

---

## 3. Schema de Base de Datos (Supabase)

```sql
-- supabase/migrations/001_initial_schema.sql

-- Profiles (configuración de FinancialModel)
CREATE TABLE profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT DEFAULT '',

    -- JSON columns (mismo formato que serialization.py)
    income_config JSONB NOT NULL,
    accounts_config JSONB NOT NULL,
    correlation_matrix JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scenarios (metas + parámetros)
CREATE TABLE scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT DEFAULT '',

    -- Goals
    intermediate_goals JSONB DEFAULT '[]',
    terminal_goals JSONB DEFAULT '[]',
    withdrawals JSONB,

    -- Parameters
    start_date DATE NOT NULL,
    n_sims INTEGER DEFAULT 500,
    seed INTEGER,
    t_max INTEGER DEFAULT 240,
    solver VARCHAR(20) DEFAULT 'ECOS',
    objective VARCHAR(30) DEFAULT 'balanced',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Jobs (tracking de tareas async)
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID REFERENCES scenarios(id) ON DELETE CASCADE,
    job_type VARCHAR(20) NOT NULL, -- 'simulation' | 'optimization'
    status VARCHAR(20) DEFAULT 'pending', -- pending|running|completed|failed

    progress INTEGER DEFAULT 0,
    current_step VARCHAR(100),
    error_message TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Results
CREATE TABLE results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    result_type VARCHAR(20) NOT NULL,

    -- Optimization results
    allocation_policy JSONB,
    optimal_horizon INTEGER,
    objective_value DOUBLE PRECISION,
    feasible BOOLEAN,
    solve_time DOUBLE PRECISION,

    -- Simulation results (summary stats, not full trajectories)
    summary_stats JSONB,
    goal_status JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS (Row Level Security)
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE results ENABLE ROW LEVEL SECURITY;

-- Policies (users can only see their own data)
CREATE POLICY "Users can CRUD own profiles" ON profiles
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can CRUD own scenarios" ON scenarios
    FOR ALL USING (profile_id IN (
        SELECT id FROM profiles WHERE user_id = auth.uid()
    ));

-- Enable Realtime for jobs table (progress updates)
ALTER PUBLICATION supabase_realtime ADD TABLE jobs;
```

---

## 4. Servicio Python (Render)

### Archivo principal: `api/main.py`

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
import os

app = FastAPI(title="FinOpt Compute API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurar para producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client para actualizar jobs
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

class SimulateRequest(BaseModel):
    scenario_id: str
    job_id: str

class OptimizeRequest(BaseModel):
    scenario_id: str
    job_id: str

@app.post("/simulate")
async def simulate(request: SimulateRequest, background_tasks: BackgroundTasks):
    """Queue simulation job."""
    background_tasks.add_task(run_simulation, request.scenario_id, request.job_id)
    return {"status": "queued", "job_id": request.job_id}

@app.post("/optimize")
async def optimize(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """Queue optimization job."""
    background_tasks.add_task(run_optimization, request.scenario_id, request.job_id)
    return {"status": "queued", "job_id": request.job_id}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

### Lógica de optimización: `api/services/optimization.py`

```python
from finopt import FinancialModel, Account, IncomeModel
from finopt.goals import IntermediateGoal, TerminalGoal
from finopt.optimization import CVaROptimizer
from finopt.serialization import income_from_dict, account_from_dict

async def run_optimization(scenario_id: str, job_id: str):
    """Run optimization and update Supabase with progress."""
    try:
        # 1. Fetch scenario + profile from Supabase
        scenario = supabase.table("scenarios").select("*, profiles(*)").eq("id", scenario_id).single().execute()

        # 2. Update job status
        update_job(job_id, status="running", progress=5, step="Loading model")

        # 3. Reconstruct FinancialModel
        profile = scenario.data["profiles"]
        model = reconstruct_model(profile)
        goals = reconstruct_goals(scenario.data)

        update_job(job_id, progress=10, step="Starting optimization")

        # 4. Run optimization
        optimizer = CVaROptimizer(
            n_accounts=len(profile["accounts_config"]),
            objective=scenario.data["objective"]
        )

        result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=scenario.data["t_max"],
            n_sims=scenario.data["n_sims"],
            seed=scenario.data["seed"]
        )

        update_job(job_id, progress=90, step="Saving results")

        # 5. Save result to Supabase
        supabase.table("results").insert({
            "job_id": job_id,
            "result_type": "optimization",
            "allocation_policy": result.X.tolist(),
            "optimal_horizon": result.T,
            "objective_value": result.objective_value,
            "feasible": result.feasible,
            "solve_time": result.solve_time
        }).execute()

        # 6. Mark job complete
        update_job(job_id, status="completed", progress=100, step="Done")

    except Exception as e:
        update_job(job_id, status="failed", error_message=str(e))
```

---

## 5. Frontend (React + Vite + Tailwind)

### Supabase Client: `web/src/lib/supabase.ts`

```typescript
import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
)
```

### Hook para progreso en tiempo real: `web/src/hooks/useJobProgress.ts`

```typescript
import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'

export function useJobProgress(jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null)

  useEffect(() => {
    if (!jobId) return

    // Initial fetch
    supabase.from('jobs').select('*').eq('id', jobId).single()
      .then(({ data }) => setJob(data))

    // Subscribe to realtime updates
    const subscription = supabase
      .channel(`job:${jobId}`)
      .on('postgres_changes', {
        event: 'UPDATE',
        schema: 'public',
        table: 'jobs',
        filter: `id=eq.${jobId}`
      }, (payload) => {
        setJob(payload.new as Job)
      })
      .subscribe()

    return () => { subscription.unsubscribe() }
  }, [jobId])

  return job
}
```

### Componentes principales:

| Componente | Descripción |
|------------|-------------|
| `ProfileForm.tsx` | Formulario para crear/editar perfiles (income, accounts) |
| `ScenarioForm.tsx` | Formulario para metas y parámetros de optimización |
| `WealthChart.tsx` | Gráfico de trayectorias de riqueza (Recharts) |
| `AllocationHeatmap.tsx` | Heatmap de política de asignación |
| `JobProgress.tsx` | Barra de progreso con Supabase Realtime |

---

## 6. Configuración de Deploy

### render.yaml (Blueprint)

```yaml
services:
  # Python API
  - type: web
    name: finopt-api
    runtime: python
    buildCommand: pip install -e ".[web]"
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_SERVICE_KEY
        sync: false
    healthCheckPath: /health

  # React Frontend
  - type: web
    name: finopt-web
    buildCommand: cd web && npm ci && npm run build
    staticPublishPath: web/dist
    envVars:
      - key: VITE_SUPABASE_URL
        sync: false
      - key: VITE_SUPABASE_ANON_KEY
        sync: false
      - key: VITE_API_URL
        fromService:
          type: web
          name: finopt-api
          property: host
```

### Variables de entorno (.env.example)

```bash
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...         # Para frontend
SUPABASE_SERVICE_KEY=eyJ...      # Para backend (más permisos)

# API
VITE_API_URL=https://finopt-api.onrender.com
```

---

## 7. Fases de Implementación

### Fase 1: Setup Supabase (1-2 días)
- [ ] Crear proyecto en Supabase
- [ ] Ejecutar migration SQL
- [ ] Configurar Auth (email/password o magic link)
- [ ] Habilitar Realtime en tabla jobs
- [ ] Probar CRUD desde Supabase Studio

**Validación**: Puedo crear/leer profiles y scenarios desde Supabase Studio

### Fase 2: Servicio Python (2-3 días)
- [ ] Crear estructura `api/`
- [ ] Implementar `main.py` con endpoints básicos
- [ ] Implementar `services/optimization.py`
- [ ] Implementar `services/simulation.py`
- [ ] Probar localmente con Supabase

**Validación**: POST /optimize actualiza job en Supabase con progreso

### Fase 3: Frontend Base (3-4 días)
- [ ] Setup React + Vite + Tailwind
- [ ] Configurar Supabase client
- [ ] Implementar auth (login/signup)
- [ ] Crear layout y navegación
- [ ] Implementar CRUD de profiles
- [ ] Implementar CRUD de scenarios

**Validación**: Puedo crear profile y scenario desde el navegador

### Fase 4: Ejecución y Resultados (3-4 días)
- [ ] Implementar botón "Run Optimization"
- [ ] Hook useJobProgress con Realtime
- [ ] Componente de progreso
- [ ] Página de resultados
- [ ] Gráfico de riqueza (Recharts)
- [ ] Heatmap de asignación

**Validación**: Puedo ejecutar optimización y ver resultados con gráficos

### Fase 5: Deploy (1-2 días)
- [ ] Crear render.yaml
- [ ] Deploy API a Render
- [ ] Deploy Frontend a Render
- [ ] Configurar variables de entorno
- [ ] Probar flujo completo en producción

**Validación**: App funcionando en URL pública

### Fase 6: Polish (2-3 días)
- [ ] Manejo de errores y feedback al usuario
- [ ] Loading states
- [ ] Responsive design
- [ ] Export de resultados (JSON/CSV)
- [ ] Documentación básica

---

## 8. Archivos Críticos a Modificar/Crear

| Archivo | Acción | Propósito |
|---------|--------|-----------|
| `pyproject.toml` | Modificar | Agregar dependencias: `supabase`, `python-dotenv` |
| `api/main.py` | Crear | FastAPI app con 4 endpoints |
| `api/services/optimization.py` | Crear | Lógica de optimización + updates a Supabase |
| `api/services/simulation.py` | Crear | Lógica de simulación |
| `supabase/migrations/001_initial_schema.sql` | Crear | Schema de DB |
| `web/src/lib/supabase.ts` | Crear | Cliente Supabase |
| `web/src/hooks/useJobProgress.ts` | Crear | Realtime hook |
| `render.yaml` | Crear | Deploy config |

---

## 9. Verificación End-to-End

1. **Crear profile**: Frontend → Supabase → Verificar en DB
2. **Crear scenario**: Frontend → Supabase → Verificar en DB
3. **Ejecutar optimización**:
   - Frontend crea job en Supabase
   - Frontend llama POST /optimize a Python service
   - Python service actualiza job.progress en Supabase
   - Frontend recibe updates via Realtime
   - Python service guarda result en Supabase
   - Frontend muestra resultado con gráficos
4. **Auth**: Login → Acceso solo a datos propios (RLS)
