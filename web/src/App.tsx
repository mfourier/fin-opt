import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './lib/store'
import { ToastProvider } from './components/Toast'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import RouteLoader from './components/RouteLoader'

const ENABLE_PREVIEW_ROUTES = import.meta.env.DEV
const PlanPreviewPage = ENABLE_PREVIEW_ROUTES ? lazy(() => import('./pages/PlanPreviewPage')) : null
const GoalsPreviewPage = ENABLE_PREVIEW_ROUTES ? lazy(() => import('./pages/GoalsPreviewPage')) : null
const SituationPreviewPage = ENABLE_PREVIEW_ROUTES ? lazy(() => import('./pages/SituationPreviewPage')) : null
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const ProfilesPage = lazy(() => import('./pages/ProfilesPage'))
const ScenariosPage = lazy(() => import('./pages/ScenariosPage'))
const ResultsPage = lazy(() => import('./pages/ResultsPage'))

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const user = useAuthStore((state) => state.user)
  const loading = useAuthStore((state) => state.loading)

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background text-foreground">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

function LazyPage({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<RouteLoader fullScreen />}>
      {children}
    </Suspense>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          {ENABLE_PREVIEW_ROUTES && PlanPreviewPage && (
            <Route
              path="/plan-preview"
              element={
                <LazyPage>
                  <PlanPreviewPage />
                </LazyPage>
              }
            />
          )}
          {ENABLE_PREVIEW_ROUTES && GoalsPreviewPage && (
            <Route
              path="/goals-preview"
              element={
                <LazyPage>
                  <GoalsPreviewPage />
                </LazyPage>
              }
            />
          )}
          {ENABLE_PREVIEW_ROUTES && SituationPreviewPage && (
            <Route
              path="/situation-preview"
              element={
                <LazyPage>
                  <SituationPreviewPage />
                </LazyPage>
              }
            />
          )}
          <Route
            path="/"
            element={
              <PrivateRoute>
                <Layout />
              </PrivateRoute>
            }
          >
            <Route index element={<DashboardPage />} />
            <Route path="profiles" element={<ProfilesPage />} />
            <Route path="scenarios" element={<ScenariosPage />} />
            <Route path="results/:jobId" element={<ResultsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ToastProvider>
  )
}
