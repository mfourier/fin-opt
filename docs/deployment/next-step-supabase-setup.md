# Plan: Configurar Supabase y Deploy a Render

## Estado Actual

**Completado:**
- [x] API Python (`api/`) - FastAPI con endpoints `/simulate`, `/optimize`, `/health`
- [x] Frontend React (`web/`) - Vite + Tailwind + Supabase client
- [x] Schema SQL (`supabase/migrations/001_initial_schema.sql`)
- [x] `render.yaml` configurado para ambos servicios

**Pendiente:**
- [ ] Crear proyecto en Supabase y ejecutar migrations
- [ ] Configurar Auth en Supabase
- [ ] Deploy a Render con variables de entorno
- [ ] Verificar flujo end-to-end

---

## Paso 1: Crear Proyecto en Supabase

### 1.1 Crear proyecto
1. Ir a [supabase.com](https://supabase.com) y crear cuenta/login
2. Click "New Project"
3. Configurar:
   - **Name**: `finopt` (o nombre preferido)
   - **Database Password**: Guardar en lugar seguro
   - **Region**: Elegir la más cercana (ej: `us-east-1`)
4. Esperar ~2 minutos mientras se provisiona

### 1.2 Obtener credenciales
En Project Settings > API:
- **Project URL**: `https://xxxxx.supabase.co` → `SUPABASE_URL`
- **anon public key**: `eyJhbGc...` → `SUPABASE_ANON_KEY`
- **service_role key**: `eyJhbGc...` → `SUPABASE_SERVICE_KEY` (SECRETO)

### 1.3 Ejecutar migration
En SQL Editor, copiar y ejecutar el contenido de:
```
supabase/migrations/001_initial_schema.sql
```

**Verificar** que se crearon:
- 4 tablas: `profiles`, `scenarios`, `jobs`, `results`
- RLS policies habilitadas
- Realtime habilitado para `jobs`

---

## Paso 2: Configurar Auth en Supabase

### 2.1 Habilitar Email Auth
En Authentication > Providers:
- Email: **Enabled**
- Confirm email: Opcional (deshabilitar para desarrollo rápido)

### 2.2 Configurar URLs (opcional para producción)
En Authentication > URL Configuration:
- Site URL: `https://finopt-web.onrender.com` (después del deploy)
- Redirect URLs: Agregar URL del frontend

---

## Paso 3: Crear archivo .env local para testing

Crear `web/.env`:
```bash
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGc...
VITE_API_URL=http://localhost:8000
```

Crear `.env` en raíz (para API local):
```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGc...
SUPABASE_SERVICE_KEY=eyJhbGc...
ENVIRONMENT=development
```

---

## Paso 4: Test Local

### 4.1 Levantar API
```bash
cd /home/mlioi/fin-opt
uvicorn api.main:app --reload --port 8000
```

### 4.2 Levantar Frontend
```bash
cd /home/mlioi/fin-opt/web
npm run dev
```

### 4.3 Verificar flujo
1. Abrir http://localhost:5173
2. Crear cuenta (Sign Up)
3. Crear Profile con income y accounts
4. Crear Scenario con goals
5. Click "Run Optimization"
6. Verificar que el job aparece y progresa
7. Ver resultados con allocation heatmap

---

## Paso 5: Deploy a Render

### 5.1 Conectar repositorio
1. Ir a [render.com](https://render.com) y login
2. New > Blueprint
3. Conectar repositorio GitHub `fin-opt`
4. Render detectará `render.yaml` automáticamente

### 5.2 Configurar variables de entorno
En Render Dashboard, para **finopt-api**:
| Variable | Valor |
|----------|-------|
| `SUPABASE_URL` | `https://xxxxx.supabase.co` |
| `SUPABASE_ANON_KEY` | `eyJhbGc...` |
| `SUPABASE_SERVICE_KEY` | `eyJhbGc...` (SECRETO) |
| `CORS_ORIGINS` | `https://finopt-web.onrender.com` |

Para **finopt-web**:
| Variable | Valor |
|----------|-------|
| `VITE_SUPABASE_URL` | `https://xxxxx.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | `eyJhbGc...` |
| `VITE_API_URL` | (auto desde `fromService`) |

### 5.3 Deploy
Click "Apply" - Render construirá y desplegará ambos servicios.

---

## Paso 6: Verificación Post-Deploy

### 6.1 Health check API
```bash
curl https://finopt-api.onrender.com/health
# Debe retornar: {"status":"ok","version":"0.1.0","environment":"production"}
```

### 6.2 Test frontend
1. Abrir https://finopt-web.onrender.com
2. Crear cuenta y login
3. Crear profile → scenario → run optimization
4. Verificar resultados

### 6.3 Verificar Supabase
En Supabase Dashboard > Table Editor:
- Ver registros en `profiles`, `scenarios`, `jobs`, `results`
- Verificar RLS funciona (cada usuario ve solo sus datos)

---

## Troubleshooting

### API no responde
- Verificar logs en Render Dashboard
- Confirmar variables de entorno están seteadas
- Verificar health check: `GET /health`

### Frontend no conecta a Supabase
- Abrir DevTools > Console
- Verificar que `VITE_SUPABASE_URL` está definido
- Confirmar CORS está configurado en API

### Jobs no actualizan progreso
- Verificar Realtime está habilitado en Supabase para tabla `jobs`
- Verificar `SUPABASE_SERVICE_KEY` tiene permisos

### Optimization falla
- Revisar logs de la API en Render
- Verificar que `cvxpy` está instalado (parte de `.[web]`)
- Confirmar schema de goals coincide con lo esperado

---

## Comandos Útiles

```bash
# Test API local
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "uuid", "job_id": "uuid"}'

# Ver logs Render (requiere CLI)
render logs finopt-api

# Verificar build frontend
cd web && npm run build && npx serve dist
```
