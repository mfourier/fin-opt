# MVP Go-Live Checklist

Fecha de auditoria: 2026-06-14

Este checklist aterriza el estado actual del repo en una salida de MVP realista. La idea no es convertir FinOpt en una plataforma "enterprise", sino cerrar lo minimo para publicar una version usable sin sorpresas evitables.

## Alcance del MVP

- [x] Frontend web autenticado con Supabase
- [x] CRUD de perfiles y escenarios
- [x] Lanzamiento de simulaciones y optimizaciones desde la app
- [x] Persistencia de jobs y resultados en Supabase
- [x] Progreso en tiempo real via Realtime
- [x] Deploy base en Render definido en `render.yaml`
- [ ] Decision explicita sobre que queda fuera del MVP

Antes de publicar, dejar por escrito estas dos decisiones:

- [ ] El CLI queda fuera del MVP web, o se completa el flujo de goal-seeking en `src/finopt/cli.py`
- [ ] Se acepta como limitacion del MVP que los jobs corren in-process y se pierden si el servidor reinicia

## Estado actual del repo

Listo hoy:

- [x] Login, signup y reset password en la app
- [x] Rutas privadas para usuarios autenticados
- [x] API con bearer token de Supabase
- [x] Resultados reales renderizados en frontend
- [x] Reaper de jobs huerfanos al reiniciar el backend
- [x] Build de frontend pasando

Pendiente o con riesgo:

- [ ] Suite completa de tests verificada en entorno limpio
- [ ] Smoke test manual completo contra un proyecto Supabase real
- [ ] Checklist operativo versionado en el repo
- [ ] Reproducibilidad local full-stack con un solo comando
- [ ] Monitoreo basico y plan de soporte para el primer release

## 1. Configuracion y secretos

- [ ] Completar `.env` para API con `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `ENVIRONMENT`, `CORS_ORIGINS`
- [ ] Completar `web/.env` con `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`, `VITE_API_URL`
- [ ] Confirmar que `ENVIRONMENT=production` en Render
- [ ] Confirmar que `CORS_ORIGINS` contiene solo los dominios reales del frontend
- [ ] Verificar que no existe `FINOPT_DEBUG` invalido en shell, CI o plataforma
- [ ] Guardar las variables productivas solo en Render/Supabase, nunca en el repo

## 2. Supabase

- [ ] Crear o validar el proyecto Supabase de produccion
- [ ] Aplicar las migraciones de `supabase/migrations/`
- [ ] Ejecutar `supabase/seed.sql` solo si el demo debe existir en produccion
- [ ] Confirmar politicas RLS para `profiles`, `scenarios`, `jobs`, `results`
- [ ] Confirmar que `jobs` esta publicado en Realtime
- [ ] Validar redirects de Auth para login, signup y reset password
- [ ] Validar emails de Auth y experiencia de confirmacion de cuenta

## 3. Backend API

- [ ] Deployar `finopt-api` desde `render.yaml`
- [ ] Confirmar `/health` responde 200 en produccion
- [ ] Confirmar que `/simulate` y `/optimize` aceptan bearer token valido
- [ ] Verificar tiempo de cold start aceptable para el uso esperado
- [ ] Verificar logs de errores y trazas en Render
- [ ] Definir timeout operativo aceptable para jobs largos

Limitacion aceptada si el MVP sale asi:

- [ ] Documentar que los jobs usan `BackgroundTasks` dentro del mismo proceso
- [ ] Documentar que un restart puede marcar jobs en ejecucion como fallidos
- [ ] Definir como se respondera al usuario si un job falla por restart

## 4. Frontend web

- [ ] Confirmar build productivo con `cd web && npm ci && npm run build`
- [ ] Validar `VITE_API_URL` apuntando a la API real
- [ ] Validar `VITE_SUPABASE_URL` y `VITE_SUPABASE_ANON_KEY`
- [ ] Revisar que el flujo `/login -> dashboard -> profiles -> scenarios -> results` funciona sin datos manuales
- [ ] Revisar UX en mobile y desktop
- [ ] Limpiar copy o comentarios heredados de preview si afectan percepcion del producto

## 5. QA minima antes de release

## Tests automatizados

- [ ] Correr `ruff check src/ api/ tests/`
- [ ] Correr `pytest` en entorno limpio con variables de prueba
- [ ] Confirmar que CI en `.github/workflows/test.yml` queda verde en `main`

## Smoke test manual

- [ ] Crear un usuario nuevo en produccion
- [ ] Iniciar sesion
- [ ] Crear un perfil
- [ ] Crear un escenario
- [ ] Ejecutar una optimizacion
- [ ] Esperar progreso en tiempo real
- [ ] Abrir el resultado final
- [ ] Recalcular el mismo escenario
- [ ] Borrar un job o resultado si ese flujo forma parte del MVP
- [ ] Validar acceso al escenario demo, si sigue habilitado

## 6. Operacion y soporte

- [ ] Definir quien monitorea el primer release
- [ ] Definir donde mirar logs de API y errores de frontend
- [ ] Definir un canal unico para reportes de usuarios
- [ ] Definir procedimiento de rollback
- [ ] Definir procedimiento para limpiar jobs fallidos o demos rotos

## 7. Criterio de salida

FinOpt puede considerarse "MVP listo" cuando se cumplan estas condiciones:

- [ ] CI verde
- [ ] Build de frontend verde
- [ ] Smoke test real completo aprobado
- [ ] Variables y redirects productivos confirmados
- [ ] Limitaciones del compute in-process documentadas y aceptadas
- [ ] Un responsable de soporte del release asignado

## Comandos utiles

```bash
# Backend + frontend local
./run-local.sh --install

# Tests Python
export SUPABASE_URL=https://test.supabase.co
export SUPABASE_ANON_KEY=test-anon-key
export SUPABASE_SERVICE_KEY=test-service-key
export ENVIRONMENT=development
pytest

# Frontend
cd web
npm ci
npm run build
```
