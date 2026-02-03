# Plan de Estudios: Backend y Frontend de FinOpt

Este documento es una guía de estudio para entender todo el código "vibecodeado" en el proyecto FinOpt, desde el backend (FastAPI + Supabase) hasta el frontend (React + Vite + Tailwind).

## Prerequisitos

Asumimos que ya entiendes:
- Python básico/intermedio
- El core de FinOpt (`/src/finopt/`) - income, portfolio, optimization, goals

## Mapa del Proyecto

```
fin-opt/
├── src/finopt/          # Core (YA LO CONOCES)
│   ├── income.py        # Modelos de ingreso
│   ├── portfolio.py     # Simulación de wealth
│   ├── optimization.py  # CVaR optimizer
│   └── ...
│
├── api/                 # BACKEND (POR APRENDER)
│   ├── main.py          # Entry point FastAPI
│   ├── routes/          # Endpoints HTTP
│   └── services/        # Lógica de negocio
│
├── web/                 # FRONTEND (POR APRENDER)
│   ├── src/
│   │   ├── pages/       # Componentes de página
│   │   ├── components/  # Componentes reutilizables
│   │   ├── lib/         # Utilidades (API client, store)
│   │   └── types/       # TypeScript types
│   └── ...
│
└── supabase/            # BASE DE DATOS
    └── migrations/      # Schema SQL
```

---

## Módulo 1: Fundamentos Web (1-2 días)

### 1.1 Modelo Cliente-Servidor

**Concepto clave**: Tu aplicación tiene 3 partes que se comunican:

```
┌─────────────┐      HTTP/JSON      ┌─────────────┐      SQL      ┌──────────────┐
│   BROWSER   │ ◄──────────────────►│   API       │ ◄────────────►│   DATABASE   │
│  (Frontend) │                     │  (Backend)  │               │  (Supabase)  │
│   React     │                     │  FastAPI    │               │  PostgreSQL  │
└─────────────┘                     └─────────────┘               └──────────────┘
```

**Flujo de una optimización:**
1. Usuario hace click en "Run Optimization" (Frontend)
2. Frontend envía HTTP POST a `/optimize` con JSON (API call)
3. Backend recibe request, lee datos de Supabase
4. Backend ejecuta `model.optimize()` (tu core de FinOpt)
5. Backend guarda resultados en Supabase
6. Backend responde con JSON al frontend
7. Frontend muestra resultados

### 1.2 HTTP Basics

**Métodos HTTP** (verbos):
| Método | Uso | Ejemplo |
|--------|-----|---------|
| GET | Leer datos | `GET /profiles` → lista de perfiles |
| POST | Crear datos | `POST /optimize` → ejecutar optimización |
| PUT/PATCH | Actualizar | `PUT /profiles/123` → actualizar perfil |
| DELETE | Eliminar | `DELETE /profiles/123` |

**Status Codes**:
- `200 OK` - Éxito
- `201 Created` - Recurso creado
- `400 Bad Request` - Error del cliente (datos inválidos)
- `401 Unauthorized` - No autenticado
- `404 Not Found` - Recurso no existe
- `500 Internal Server Error` - Error del servidor

**JSON** (JavaScript Object Notation):
```json
{
  "name": "Mi Perfil",
  "income_config": {
    "fixed": { "base": 1500000, "annual_growth": 0.03 }
  }
}
```

### 📚 Recursos Módulo 1
- [ ] Video: [HTTP Crash Course](https://www.youtube.com/watch?v=iYM2zFP3Zn0) (35 min)
- [ ] Leer: [MDN - HTTP Overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [ ] Practicar: Usar `curl` para hacer requests manuales

---

## Módulo 2: Backend con FastAPI (2-3 días)

### 2.1 ¿Qué es FastAPI?

FastAPI es un framework Python para crear APIs web. Es:
- **Rápido**: Basado en Starlette (async)
- **Tipado**: Usa type hints de Python para validación automática
- **Documentado**: Genera docs automáticos en `/docs`

### 2.2 Anatomía de un Endpoint

**Archivo**: `api/main.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")           # Decorador: GET request a /health
async def health_check():     # Función async (puede ser sync también)
    return {                  # Retorna dict → se convierte a JSON
        "status": "ok",
        "version": "0.1.0"
    }
```

**Probar**:
```bash
# Levantar servidor
uvicorn api.main:app --reload --port 8000

# En otra terminal
curl http://localhost:8000/health
# → {"status":"ok","version":"0.1.0"}

# Ver documentación automática
# Abrir: http://localhost:8000/docs
```

### 2.3 Request Body con Pydantic

FastAPI usa Pydantic para validar datos de entrada:

```python
from pydantic import BaseModel

class OptimizationRequest(BaseModel):
    scenario_id: str
    job_id: str

@app.post("/optimize")
async def run_optimization(request: OptimizationRequest):
    # FastAPI automáticamente:
    # 1. Parsea el JSON del body
    # 2. Valida que tenga scenario_id y job_id
    # 3. Retorna 422 si falta algo
    return {"status": "queued", "job_id": request.job_id}
```

### 2.4 Estructura del Backend FinOpt

```
api/
├── main.py              # Entry point, configura app
├── routes/
│   └── optimization.py  # Endpoints de optimización
└── services/
    ├── reconstruction.py  # Reconstruye objetos FinOpt desde DB
    └── supabase.py        # Cliente de Supabase
```

**Ejercicio**: Lee `api/main.py` y identifica:
- [ ] ¿Dónde se configura CORS?
- [ ] ¿Qué endpoints existen?
- [ ] ¿Cómo se conecta a Supabase?

### 2.5 Flujo de `/optimize`

```python
# api/routes/optimization.py (simplificado)

@router.post("/optimize")
async def optimize(request: OptimizationRequest):
    # 1. Obtener datos de Supabase
    scenario = supabase.table("scenarios").select("*").eq("id", request.scenario_id).single()
    profile = supabase.table("profiles").select("*").eq("id", scenario.profile_id).single()

    # 2. Reconstruir objetos FinOpt
    income = reconstruct_income(profile.income_config)
    accounts = reconstruct_accounts(profile.accounts_config)
    goals = reconstruct_goals(scenario.terminal_goals, scenario.intermediate_goals)

    # 3. Crear modelo y optimizar (TU CÓDIGO CORE)
    model = FinancialModel(income, accounts)
    result = model.optimize(goals=goals, T_max=scenario.t_max)

    # 4. Guardar resultados en Supabase
    supabase.table("results").insert({
        "job_id": request.job_id,
        "allocation_policy": result.allocation.tolist(),
        "optimal_horizon": result.optimal_horizon,
        ...
    })

    return {"status": "completed"}
```

### 📚 Recursos Módulo 2
- [ ] Tutorial oficial: [FastAPI First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [ ] Video: [FastAPI Full Course](https://www.youtube.com/watch?v=tLKKmouUams) (primeros 60 min)
- [ ] Ejercicio: Crear un endpoint `/echo` que devuelva lo que recibe

---

## Módulo 3: Base de Datos con Supabase (1-2 días)

### 3.1 ¿Qué es Supabase?

Supabase es un "Backend as a Service" que provee:
- **PostgreSQL**: Base de datos relacional
- **Auth**: Sistema de autenticación (login, signup)
- **Realtime**: WebSockets para actualizaciones en vivo
- **Storage**: Almacenamiento de archivos
- **REST API**: Acceso a datos via HTTP (PostgREST)

### 3.2 Modelo de Datos FinOpt

**Archivo**: `supabase/migrations/001_initial_schema.sql`

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│  profiles   │       │  scenarios  │       │    jobs     │       │   results   │
├─────────────┤       ├─────────────┤       ├─────────────┤       ├─────────────┤
│ id (PK)     │◄──────│ profile_id  │       │ id (PK)     │◄──────│ job_id      │
│ user_id     │       │ id (PK)     │◄──────│ scenario_id │       │ id (PK)     │
│ name        │       │ name        │       │ status      │       │ allocation  │
│ income_conf │       │ goals       │       │ progress    │       │ horizon     │
│ accounts    │       │ withdrawals │       │ ...         │       │ ...         │
└─────────────┘       └─────────────┘       └─────────────┘       └─────────────┘
```

**Relaciones**:
- Un `profile` tiene muchos `scenarios`
- Un `scenario` tiene muchos `jobs`
- Un `job` tiene un `result`

### 3.3 JSONB para Datos Complejos

PostgreSQL soporta `JSONB` para almacenar objetos complejos:

```sql
CREATE TABLE profiles (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    name TEXT NOT NULL,
    income_config JSONB NOT NULL,  -- Objeto complejo como JSON
    accounts_config JSONB NOT NULL
);
```

Esto permite guardar estructuras como:
```json
{
  "fixed": {"base": 1500000, "annual_growth": 0.03},
  "variable": {"base": 40000, "sigma": 0.1, "seasonality": [0,0,0,0.6,...]}
}
```

### 3.4 Row Level Security (RLS)

RLS asegura que cada usuario solo vea sus datos:

```sql
-- Solo el dueño puede ver sus perfiles
CREATE POLICY "Users can view own profiles"
ON profiles FOR SELECT
USING (auth.uid() = user_id);
```

### 3.5 Cliente Python

```python
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# SELECT * FROM profiles WHERE user_id = '...'
data = supabase.table("profiles").select("*").execute()

# INSERT INTO profiles VALUES (...)
supabase.table("profiles").insert({"name": "Test", ...}).execute()

# UPDATE profiles SET name = '...' WHERE id = '...'
supabase.table("profiles").update({"name": "New"}).eq("id", "123").execute()
```

### 📚 Recursos Módulo 3
- [ ] Tutorial: [Supabase Quickstart](https://supabase.com/docs/guides/getting-started)
- [ ] Video: [Supabase Crash Course](https://www.youtube.com/watch?v=7uKQBl9uZ00)
- [ ] SQL básico: [SQLBolt Interactive](https://sqlbolt.com/)
- [ ] Ejercicio: Ir al Dashboard de Supabase y explorar las tablas

---

## Módulo 4: Frontend con React (3-5 días)

### 4.1 ¿Qué es React?

React es una librería JavaScript para construir interfaces de usuario. Conceptos clave:

**Componentes**: Bloques de UI reutilizables
```jsx
function Button({ label, onClick }) {
  return <button onClick={onClick}>{label}</button>
}

// Uso
<Button label="Click me" onClick={() => alert('Hi!')} />
```

**JSX**: HTML dentro de JavaScript
```jsx
const name = "Juan"
return <h1>Hola, {name}!</h1>  // Renderiza: <h1>Hola, Juan!</h1>
```

**State**: Datos que cambian y causan re-render
```jsx
function Counter() {
  const [count, setCount] = useState(0)  // Hook de estado

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  )
}
```

### 4.2 TypeScript en React

TypeScript añade tipos a JavaScript:

```typescript
// JavaScript
function add(a, b) {
  return a + b
}

// TypeScript
function add(a: number, b: number): number {
  return a + b
}
```

**Interfaces para props**:
```typescript
interface ButtonProps {
  label: string
  onClick: () => void
  disabled?: boolean  // Opcional
}

function Button({ label, onClick, disabled = false }: ButtonProps) {
  return <button onClick={onClick} disabled={disabled}>{label}</button>
}
```

### 4.3 Estructura del Frontend FinOpt

```
web/src/
├── main.tsx            # Entry point
├── App.tsx             # Router principal
├── pages/              # Componentes de página (una por ruta)
│   ├── LoginPage.tsx
│   ├── ProfilesPage.tsx
│   ├── ScenariosPage.tsx
│   └── ResultsPage.tsx
├── components/         # Componentes reutilizables
│   ├── Layout.tsx
│   └── Navbar.tsx
├── lib/                # Utilidades
│   ├── supabase.ts     # Cliente Supabase
│   ├── api.ts          # Cliente API (fetch a FastAPI)
│   └── store.ts        # Estado global (Zustand)
└── types/              # Definiciones TypeScript
    └── database.ts     # Tipos que matchean con Supabase
```

### 4.4 Hooks Importantes

**useState**: Estado local del componente
```jsx
const [name, setName] = useState('')
```

**useEffect**: Efectos secundarios (API calls, subscriptions)
```jsx
useEffect(() => {
  fetchProfiles()  // Se ejecuta al montar el componente
}, [])  // [] = solo una vez
```

**useQuery** (React Query): Data fetching con cache
```jsx
const { data, isLoading, error } = useQuery({
  queryKey: ['profiles'],
  queryFn: async () => {
    const { data } = await supabase.from('profiles').select('*')
    return data
  }
})
```

**useMutation** (React Query): Modificar datos
```jsx
const mutation = useMutation({
  mutationFn: async (newProfile) => {
    await supabase.from('profiles').insert(newProfile)
  },
  onSuccess: () => {
    queryClient.invalidateQueries(['profiles'])  // Refetch
  }
})
```

### 4.5 Anatomía de ProfilesPage

**Archivo**: `web/src/pages/ProfilesPage.tsx`

```tsx
export default function ProfilesPage() {
  // 1. Estado local para el formulario
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({...})

  // 2. Query para obtener perfiles de Supabase
  const { data: profiles, isLoading } = useQuery({
    queryKey: ['profiles'],
    queryFn: async () => {
      const { data } = await supabase.from('profiles').select('*')
      return data
    }
  })

  // 3. Mutation para crear perfil
  const createMutation = useMutation({
    mutationFn: async (profile) => {
      await supabase.from('profiles').insert(profile)
    }
  })

  // 4. Handler del formulario
  const handleSubmit = (e) => {
    e.preventDefault()
    createMutation.mutate(formData)
  }

  // 5. Render
  return (
    <div>
      {/* Lista de perfiles */}
      {profiles?.map(p => <ProfileCard key={p.id} profile={p} />)}

      {/* Formulario */}
      {showForm && (
        <form onSubmit={handleSubmit}>
          <input value={formData.name} onChange={...} />
          <button type="submit">Create</button>
        </form>
      )}
    </div>
  )
}
```

### 📚 Recursos Módulo 4
- [ ] Tutorial oficial: [React Quick Start](https://react.dev/learn)
- [ ] Video: [React en 1 hora](https://www.youtube.com/watch?v=SqcY0GlETPk)
- [ ] TypeScript: [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [ ] Ejercicio: Modificar un componente y ver el hot reload

---

## Módulo 5: Herramientas del Ecosistema (1 día)

### 5.1 Vite (Build Tool)

**¿Qué hace?**
- Development server con Hot Module Replacement (HMR)
- Bundling para producción (tree-shaking, minification)
- Soporte nativo para TypeScript, JSX, CSS

**Comandos**:
```bash
npm run dev      # Development server (http://localhost:5173)
npm run build    # Build para producción (genera /dist)
npm run preview  # Preview del build de producción
```

### 5.2 Tailwind CSS

Framework de CSS "utility-first":

```html
<!-- CSS tradicional -->
<button class="btn-primary">Click</button>
<style>
.btn-primary {
  background-color: blue;
  color: white;
  padding: 8px 16px;
  border-radius: 4px;
}
</style>

<!-- Tailwind -->
<button class="bg-blue-500 text-white px-4 py-2 rounded">Click</button>
```

**Clases comunes**:
| Clase | CSS |
|-------|-----|
| `p-4` | `padding: 1rem` |
| `mt-2` | `margin-top: 0.5rem` |
| `flex` | `display: flex` |
| `text-gray-500` | `color: #6b7280` |
| `rounded-lg` | `border-radius: 0.5rem` |

### 5.3 React Query (TanStack Query)

Manejo de server state (datos de APIs):

**Sin React Query**:
```jsx
const [data, setData] = useState(null)
const [loading, setLoading] = useState(true)
const [error, setError] = useState(null)

useEffect(() => {
  fetch('/api/profiles')
    .then(res => res.json())
    .then(data => { setData(data); setLoading(false) })
    .catch(err => { setError(err); setLoading(false) })
}, [])
```

**Con React Query**:
```jsx
const { data, isLoading, error } = useQuery({
  queryKey: ['profiles'],
  queryFn: () => fetch('/api/profiles').then(r => r.json())
})
```

Beneficios:
- Caching automático
- Refetch en background
- Retry automático
- Deduplicación de requests

### 5.4 Zustand (State Management)

Estado global simple:

```typescript
// lib/store.ts
import { create } from 'zustand'

interface AuthStore {
  user: User | null
  setUser: (user: User | null) => void
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  setUser: (user) => set({ user })
}))

// En cualquier componente
function Navbar() {
  const user = useAuthStore(state => state.user)
  return <div>Hello, {user?.email}</div>
}
```

### 📚 Recursos Módulo 5
- [ ] Tailwind: [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [ ] React Query: [TanStack Query Overview](https://tanstack.com/query/latest/docs/react/overview)
- [ ] Ejercicio: Cambiar colores de un botón con Tailwind

---

## Módulo 6: Deployment (1 día)

### 6.1 ¿Qué es deployment?

Mover tu código de "mi computador" a "internet":

```
Local Development          →        Production
─────────────────                   ─────────────────
localhost:8000 (API)        →       finopt-api.onrender.com
localhost:5173 (Frontend)   →       finopt-web.onrender.com
Supabase (ya en cloud)      →       (mismo)
```

### 6.2 Render

**render.yaml** define los servicios:

```yaml
services:
  # Backend Python
  - type: web
    name: finopt-api
    runtime: python
    buildCommand: pip install -e ".[web]"
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: SUPABASE_URL
        sync: false  # Configurar manualmente

  # Frontend Static
  - type: web
    name: finopt-web
    runtime: static
    buildCommand: cd web && npm install && npm run build
    staticPublishPath: web/dist
```

### 6.3 Variables de Entorno

**¿Qué son?** Configuración que cambia entre ambientes:

```bash
# .env (local)
SUPABASE_URL=https://xxx.supabase.co
API_URL=http://localhost:8000

# Producción (en Render dashboard)
SUPABASE_URL=https://xxx.supabase.co
API_URL=https://finopt-api.onrender.com
```

**Acceso en código**:
```python
# Python
import os
url = os.getenv("SUPABASE_URL")

# JavaScript (Vite)
const url = import.meta.env.VITE_SUPABASE_URL
```

### 📚 Recursos Módulo 6
- [ ] Render docs: [Deploy a FastAPI app](https://render.com/docs/deploy-fastapi)
- [ ] Video: [Deploy Full Stack App](https://www.youtube.com/watch?v=...)

---

## Ruta de Estudio Recomendada

### Semana 1: Fundamentos
| Día | Módulo | Actividad |
|-----|--------|-----------|
| 1 | 1 | HTTP basics, curl, JSON |
| 2 | 2 | FastAPI tutorial, crear endpoint de prueba |
| 3 | 2 | Leer `api/main.py` y `api/routes/` |
| 4 | 3 | Supabase dashboard, SQL básico |
| 5 | 3 | Entender RLS, leer `migrations/` |

### Semana 2: Frontend
| Día | Módulo | Actividad |
|-----|--------|-----------|
| 1 | 4 | React basics, componentes, JSX |
| 2 | 4 | useState, useEffect |
| 3 | 4 | Leer `ProfilesPage.tsx` línea por línea |
| 4 | 5 | Tailwind, React Query |
| 5 | 5 | Modificar UI, ver cambios en hot reload |

### Semana 3: Integración
| Día | Módulo | Actividad |
|-----|--------|-----------|
| 1 | - | Trazar flujo completo: click → API → DB → response |
| 2 | - | Agregar un campo nuevo end-to-end |
| 3 | 6 | Entender render.yaml |
| 4 | 6 | Deploy manual a Render |
| 5 | - | Revisión, preguntas, cleanup |

---

## Ejercicios Prácticos

### Ejercicio 1: Agregar campo al Profile
**Objetivo**: Entender el flujo completo
1. Agregar campo `notes: string` al tipo en `database.ts`
2. Agregar input en `ProfilesPage.tsx`
3. Verificar que se guarda en Supabase
4. (Opcional) Mostrarlo en la lista

### Ejercicio 2: Nuevo Endpoint API
**Objetivo**: Entender FastAPI
1. Crear endpoint `GET /api/stats` que retorne número de profiles
2. Crear query en frontend que lo consuma
3. Mostrar en dashboard

### Ejercicio 3: Tracing del Flujo de Optimización
**Objetivo**: Entender la integración completa
1. Poner `console.log` en frontend antes del fetch
2. Poner `print()` en API al recibir request
3. Verificar flujo: Frontend → API → Core → DB → API → Frontend

---

## Glosario Rápido

| Término | Definición |
|---------|------------|
| **API** | Application Programming Interface - interfaz para comunicar sistemas |
| **REST** | Estilo de arquitectura para APIs usando HTTP |
| **Endpoint** | URL específica de una API (ej: `/profiles`) |
| **JSON** | Formato de datos (JavaScript Object Notation) |
| **CORS** | Cross-Origin Resource Sharing - permisos entre dominios |
| **JWT** | JSON Web Token - token de autenticación |
| **Hook** | Función de React que "engancha" funcionalidad (useState, useEffect) |
| **Component** | Bloque de UI reutilizable en React |
| **Props** | Propiedades pasadas a un componente |
| **State** | Datos internos de un componente que pueden cambiar |
| **Query** | Petición de datos (lectura) |
| **Mutation** | Petición que modifica datos (escritura) |
| **ORM** | Object-Relational Mapping - abstracción sobre SQL |
| **Middleware** | Código que se ejecuta entre request y response |
| **Build** | Proceso de compilar/empaquetar código para producción |
| **Hot Reload** | Actualización automática al guardar archivos |

---

## Archivos Clave para Estudiar

### Backend (en orden)
1. `api/main.py` - Entry point, configuración
2. `api/routes/optimization.py` - Endpoints principales
3. `api/services/reconstruction.py` - Conecta DB con FinOpt core

### Frontend (en orden)
1. `web/src/main.tsx` - Entry point
2. `web/src/App.tsx` - Router
3. `web/src/lib/supabase.ts` - Cliente DB
4. `web/src/lib/api.ts` - Cliente API
5. `web/src/pages/ProfilesPage.tsx` - Ejemplo completo de CRUD
6. `web/src/types/database.ts` - Tipos compartidos

### Configuración
1. `.env` y `web/.env` - Variables de entorno
2. `render.yaml` - Configuración de deployment
3. `supabase/migrations/001_initial_schema.sql` - Schema DB

---

## Próximos Pasos

Una vez domines estos conceptos, podrás:

1. **Modificar el frontend** con confianza
2. **Agregar nuevos endpoints** a la API
3. **Debuggear problemas** entendiendo el flujo completo
4. **Desplegar cambios** a producción
5. **Extender el modelo** agregando nuevas features

¡Éxito en tu aprendizaje! 🚀
