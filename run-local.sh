#!/usr/bin/env bash
# ============================================================================
# run-local.sh — Levanta el backend (FastAPI) y el frontend (Vite) en local
# ----------------------------------------------------------------------------
# Uso:
#   ./run-local.sh            # arranca API + web
#   ./run-local.sh --install  # instala/actualiza dependencias y luego arranca
#
# Backend:  http://localhost:8000   (docs en /docs)
# Frontend: http://localhost:5173
#
# Detiene ambos procesos con Ctrl+C.
# ============================================================================
set -euo pipefail

# --- Ubicarse en la raíz del repo (donde vive este script) -----------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${FINOPT_CONDA_ENV:-finance}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
INSTALL=0
[[ "${1:-}" == "--install" ]] && INSTALL=1

# --- Resolver el Python del entorno conda ----------------------------------
if command -v conda >/dev/null 2>&1; then
  ENV_PREFIX="$(conda env list | awk -v e="$CONDA_ENV" '$1==e {print $NF}')"
fi
if [[ -n "${ENV_PREFIX:-}" && -x "$ENV_PREFIX/bin/python" ]]; then
  PYTHON="$ENV_PREFIX/bin/python"
else
  echo "⚠️  No se encontró el entorno conda '$CONDA_ENV'; usando 'python3' del PATH."
  PYTHON="$(command -v python3)"
fi
echo "🐍 Python backend: $PYTHON"

# --- Verificaciones / instalación de dependencias --------------------------
if [[ "$INSTALL" -eq 1 ]]; then
  echo "📦 Instalando dependencias del backend (finopt[web])..."
  "$PYTHON" -m pip install -e ".[web]"
  echo "📦 Instalando dependencias del frontend (npm install)..."
  (cd web && npm install)
fi

if ! "$PYTHON" -c "import fastapi, uvicorn, finopt" >/dev/null 2>&1; then
  echo "❌ Faltan dependencias del backend. Ejecuta: ./run-local.sh --install"
  exit 1
fi

if [[ ! -d web/node_modules ]]; then
  echo "📦 web/node_modules no existe; ejecutando npm install..."
  (cd web && npm install)
fi

# --- Avisos sobre variables de entorno -------------------------------------
[[ -f .env ]]     || echo "⚠️  No existe .env (copia .env.example y completa Supabase)."
[[ -f web/.env ]] || echo "⚠️  No existe web/.env (copia web/.env.example y completa Supabase)."

# --- Limpieza al salir ------------------------------------------------------
PIDS=()
cleanup() {
  echo ""
  echo "🛑 Deteniendo servicios..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "✅ Listo."
}
trap cleanup INT TERM EXIT

# --- Backend (FastAPI / uvicorn con reload) --------------------------------
echo "🚀 Backend  → http://localhost:$API_PORT  (docs en /docs)"
API_PORT="$API_PORT" "$PYTHON" -m uvicorn api.main:app \
  --host 0.0.0.0 --port "$API_PORT" --reload &
PIDS+=($!)

# --- Frontend (Vite dev server) --------------------------------------------
echo "🚀 Frontend → http://localhost:$WEB_PORT"
(cd web && npm run dev -- --port "$WEB_PORT") &
PIDS+=($!)

echo ""
echo "Ambos servicios en marcha. Pulsa Ctrl+C para detenerlos."
wait
