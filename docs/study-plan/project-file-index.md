# Índice de Archivos del Proyecto

Referencia rápida de cada archivo relevante para el estudio.

---

## Backend (api/)

### api/main.py
**Propósito**: Entry point de FastAPI, configura la aplicación.

**Conceptos clave**:
- `FastAPI()` - Crea la aplicación
- `@app.get()`, `@app.post()` - Decoradores de rutas
- `CORSMiddleware` - Permite requests desde el frontend
- `@app.on_event("startup")` - Código que corre al iniciar

**Preguntas de estudio**:
- ¿Qué URLs están permitidas en CORS?
- ¿Qué endpoints están definidos?
- ¿Dónde se inicializa el cliente de Supabase?

---

### api/routes/optimization.py
**Propósito**: Endpoints para optimización y simulación.

**Endpoints**:
| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/optimize` | Ejecuta optimización |
| POST | `/simulate` | Ejecuta simulación |
| GET | `/health` | Health check |

**Conceptos clave**:
- `APIRouter` - Agrupa rutas relacionadas
- `BackgroundTasks` - Ejecuta código en background
- Pydantic models para validación

**Preguntas de estudio**:
- ¿Qué datos recibe `/optimize`?
- ¿Cómo se actualiza el progreso del job?
- ¿Dónde se llama a `model.optimize()`?

---

### api/services/reconstruction.py
**Propósito**: Reconstruye objetos FinOpt desde datos de Supabase.

**Funciones principales**:
```python
reconstruct_income(income_config: dict) -> IncomeModel
reconstruct_accounts(accounts_config: list) -> list[Account]
reconstruct_goals(terminal, intermediate) -> list[Goal]
reconstruct_withdrawals(withdrawals_config: dict) -> list[WithdrawalEvent]
```

**Conceptos clave**:
- Serialización/Deserialización
- Validación de datos
- Mapeo entre formatos DB ↔ Python

**Preguntas de estudio**:
- ¿Cómo se convierte `salary_raises` de JSON a dict de dates?
- ¿Qué pasa si un campo es `None`?

---

### api/services/supabase.py
**Propósito**: Cliente de Supabase para el backend.

**Conceptos clave**:
- `create_client(url, key)` - Inicializa cliente
- `service_role` key vs `anon` key
- Operaciones CRUD

---

## Frontend (web/src/)

### web/src/main.tsx
**Propósito**: Entry point de React, monta la app.

```tsx
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

**Conceptos clave**:
- `createRoot` - React 18 API
- `StrictMode` - Detecta problemas

---

### web/src/App.tsx
**Propósito**: Configuración de routing y providers.

**Estructura**:
```tsx
<QueryClientProvider>       // React Query
  <BrowserRouter>           // React Router
    <AuthProvider>          // Autenticación
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route path="profiles" element={<ProfilesPage />} />
          <Route path="scenarios" element={<ScenariosPage />} />
          ...
        </Route>
      </Routes>
    </AuthProvider>
  </BrowserRouter>
</QueryClientProvider>
```

**Conceptos clave**:
- Providers (Context API)
- Nested routes
- Protected routes

---

### web/src/lib/supabase.ts
**Propósito**: Cliente de Supabase para el frontend.

```typescript
import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
)
```

**Uso**:
```typescript
// SELECT
const { data } = await supabase.from('profiles').select('*')

// INSERT
await supabase.from('profiles').insert({ name: 'Test' })

// Auth
await supabase.auth.signInWithPassword({ email, password })
```

---

### web/src/lib/api.ts
**Propósito**: Funciones para llamar al backend FastAPI.

```typescript
const API_URL = import.meta.env.VITE_API_URL

export async function queueOptimization(params: {...}) {
  const response = await fetch(`${API_URL}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  })
  return response.json()
}
```

**Preguntas de estudio**:
- ¿Por qué usamos `fetch` en vez del cliente Supabase?
- ¿Qué headers son necesarios?

---

### web/src/lib/store.ts
**Propósito**: Estado global con Zustand.

```typescript
interface AuthStore {
  user: User | null
  session: Session | null
  setUser: (user: User | null) => void
  setSession: (session: Session | null) => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  session: null,
  setUser: (user) => set({ user }),
  setSession: (session) => set({ session })
}))
```

**Uso en componentes**:
```typescript
const user = useAuthStore(state => state.user)
const setUser = useAuthStore(state => state.setUser)
```

---

### web/src/types/database.ts
**Propósito**: Tipos TypeScript que matchean con la base de datos.

**Tipos principales**:
```typescript
interface Profile {
  id: string
  user_id: string
  name: string
  income_config: IncomeConfig
  accounts_config: AccountConfig[]
  ...
}

interface IncomeConfig {
  fixed?: FixedIncomeConfig
  variable?: VariableIncomeConfig
  contribution_rate_fixed: number | number[]
  contribution_rate_variable: number | number[]
}
```

**Importancia**: Estos tipos aseguran que el frontend envíe datos en el formato correcto a Supabase y la API.

---

### web/src/pages/ProfilesPage.tsx
**Propósito**: CRUD completo de perfiles.

**Estructura típica de una página**:
```
1. Imports
2. Tipos/Interfaces locales
3. Valores por defecto
4. Componente principal:
   a. Estado local (useState)
   b. Queries (useQuery)
   c. Mutations (useMutation)
   d. Handlers
   e. Return JSX
```

**Patterns importantes**:
- Form state management
- Conditional rendering
- List rendering con `.map()`
- Event handlers

**Secciones del formulario**:
1. Basic info (name, description)
2. Fixed Income (base, growth, raises)
3. Variable Income (toggleable)
4. Contribution Rates (simple o monthly)
5. Accounts (add/remove)

---

### web/src/pages/ScenariosPage.tsx
**Propósito**: CRUD de escenarios + trigger de optimización.

**Diferencias con ProfilesPage**:
- Depende de `profiles` (foreign key)
- Tiene secciones toggleables (withdrawals, intermediate goals)
- Llama a la API para ejecutar optimización

**Flujo de "Run Optimization"**:
```
1. Crear job en Supabase (status: pending)
2. Llamar a POST /optimize con job_id y scenario_id
3. Navegar a /results/:jobId
4. ResultsPage muestra progreso y resultados
```

---

### web/src/pages/ResultsPage.tsx
**Propósito**: Muestra resultados de optimización.

**Secciones**:
1. Job status (pending, running, completed, failed)
2. Optimal horizon
3. Allocation policy heatmap
4. Goal status
5. Export options

---

### web/src/pages/LoginPage.tsx
**Propósito**: Autenticación de usuarios.

**Flujo**:
```
1. Usuario ingresa email/password
2. supabase.auth.signInWithPassword()
3. Si éxito → setUser() y redirect a /profiles
4. Si error → mostrar mensaje
```

---

## Configuración

### .env (raíz)
**Propósito**: Variables de entorno para el backend.

```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...  # SECRETO
CORS_ORIGINS=http://localhost:5173
```

### web/.env
**Propósito**: Variables de entorno para el frontend.

```bash
VITE_SUPABASE_URL=https://xxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
VITE_API_URL=http://localhost:8000
```

**Nota**: El prefijo `VITE_` es requerido para que Vite exponga la variable al código.

---

### render.yaml
**Propósito**: Configuración de deployment para Render.

```yaml
services:
  - type: web
    name: finopt-api
    runtime: python
    ...

  - type: web
    name: finopt-web
    runtime: static
    ...
```

---

### supabase/migrations/001_initial_schema.sql
**Propósito**: Define el schema de la base de datos.

**Tablas**:
- `profiles` - Configuración financiera del usuario
- `scenarios` - Escenarios de optimización
- `jobs` - Estado de ejecución
- `results` - Resultados de optimización

**Conceptos**:
- Primary keys (UUID)
- Foreign keys
- JSONB columns
- Row Level Security (RLS)
- Triggers (updated_at)

---

## Flujo de Datos Completo

```
Usuario hace click "Run Optimization"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Frontend (ScenariosPage.tsx)                                 │
│ 1. supabase.from('jobs').insert({status:'pending'})         │
│ 2. await queueOptimization({job_id, scenario_id})           │
│ 3. navigate(`/results/${job.id}`)                           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ HTTP POST /optimize
┌─────────────────────────────────────────────────────────────┐
│ Backend (api/routes/optimization.py)                         │
│ 1. Fetch scenario and profile from Supabase                 │
│ 2. reconstruct_income(), reconstruct_accounts(), etc.       │
│ 3. model = FinancialModel(income, accounts)                 │
│ 4. result = model.optimize(goals=goals, ...)                │
│ 5. supabase.table('results').insert({...})                  │
│ 6. supabase.table('jobs').update({status:'completed'})      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Frontend (ResultsPage.tsx)                                   │
│ 1. useQuery(['job', jobId]) - polling status                │
│ 2. When completed, fetch result                             │
│ 3. Render heatmap, goal status, etc.                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Comandos Útiles para Debugging

```bash
# Ver logs del backend
uvicorn api.main:app --reload --log-level debug

# Ver requests de red en browser
# Chrome: F12 → Network tab

# Ver estado de Supabase
# Dashboard: https://app.supabase.com → Table Editor

# Type check del frontend
cd web && npx tsc --noEmit

# Build de producción
cd web && npm run build

# Ver variables de entorno cargadas
python -c "import os; print(os.getenv('SUPABASE_URL'))"
```
