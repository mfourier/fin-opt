import { useEffect, useState, createContext, useContext, useCallback, ReactNode } from 'react'
import { AlertCircle, CheckCircle2, Info, TriangleAlert, X } from 'lucide-react'

// ============================================================================
// Types
// ============================================================================

type ToastType = 'success' | 'error' | 'warning' | 'info'

interface Toast {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
}

interface ToastContextType {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
  success: (title: string, message?: string) => void
  error: (title: string, message?: string) => void
  warning: (title: string, message?: string) => void
  info: (title: string, message?: string) => void
}

// ============================================================================
// Context
// ============================================================================

const ToastContext = createContext<ToastContextType | undefined>(undefined)

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

// ============================================================================
// Provider
// ============================================================================

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }, [])

  const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).substring(2, 9)
    setToasts((prev) => [...prev, { ...toast, id }])

    // Auto-remove after duration
    const duration = toast.duration ?? 5000
    if (duration > 0) {
      setTimeout(() => removeToast(id), duration)
    }
  }, [removeToast])

  const success = useCallback((title: string, message?: string) => {
    addToast({ type: 'success', title, message })
  }, [addToast])

  const error = useCallback((title: string, message?: string) => {
    addToast({ type: 'error', title, message, duration: 8000 })
  }, [addToast])

  const warning = useCallback((title: string, message?: string) => {
    addToast({ type: 'warning', title, message })
  }, [addToast])

  const info = useCallback((title: string, message?: string) => {
    addToast({ type: 'info', title, message })
  }, [addToast])

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, success, error, warning, info }}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  )
}

// ============================================================================
// Toast Container
// ============================================================================

function ToastContainer({ toasts, removeToast }: { toasts: Toast[]; removeToast: (id: string) => void }) {
  return (
    <div className="pointer-events-none fixed bottom-0 right-0 z-50 flex flex-col gap-2 p-4">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
      ))}
    </div>
  )
}

// ============================================================================
// Toast Item
// ============================================================================

function ToastItem({ toast, onClose }: { toast: Toast; onClose: () => void }) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // Trigger enter animation
    requestAnimationFrame(() => setIsVisible(true))
  }, [])

  const icons: Record<ToastType, ReactNode> = {
    success: (
      <span className="rounded-full bg-success-soft p-1.5 text-success">
        <CheckCircle2 className="h-4 w-4" />
      </span>
    ),
    error: (
      <span className="rounded-full bg-danger-soft p-1.5 text-danger">
        <AlertCircle className="h-4 w-4" />
      </span>
    ),
    warning: (
      <span className="rounded-full bg-warning-soft p-1.5 text-warning">
        <TriangleAlert className="h-4 w-4" />
      </span>
    ),
    info: (
      <span className="rounded-full bg-accent p-1.5 text-primary">
        <Info className="h-4 w-4" />
      </span>
    ),
  }

  const surfaces: Record<ToastType, string> = {
    success: 'border-success/30',
    error: 'border-danger/30',
    warning: 'border-warning/30',
    info: 'border-primary/20',
  }

  return (
    <div
      className={`pointer-events-auto w-80 transform rounded-2xl border bg-card text-card-foreground shadow-lg transition-all duration-300 ${
        isVisible ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'
      } ${surfaces[toast.type]}`}
    >
      <div className="p-4">
        <div className="flex items-start">
          <div className="flex-shrink-0">{icons[toast.type]}</div>
          <div className="ml-3 flex-1">
            <p className="text-sm font-medium text-foreground">{toast.title}</p>
            {toast.message && (
              <p className="mt-1 text-sm text-muted-foreground">{toast.message}</p>
            )}
          </div>
          <button
            onClick={onClose}
            className="ml-4 inline-flex rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Close notification"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
