import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import { useAuthStore } from '../lib/store'
import { supabase } from '../lib/supabase'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FinOptHeroPanel } from '@/components/finopt/FinOptHeroPanel'
import { FinOptWordmark } from '@/components/finopt/FinOptWordmark'
import { ThemeToggle } from '@/components/ThemeToggle'

type Notice = { kind: 'error' | 'success'; message: string } | null

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [notice, setNotice] = useState<Notice>(null)
  const [loading, setLoading] = useState(false)

  const navigate = useNavigate()
  const { signIn, signUp, user } = useAuthStore()

  useEffect(() => {
    if (user) {
      navigate('/', { replace: true })
    }
  }, [user, navigate])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setNotice(null)
    setLoading(true)

    try {
      if (isSignUp) {
        await signUp(email, password)
        setNotice({ kind: 'success', message: 'Check your email for a confirmation link.' })
      } else {
        await signIn(email, password)
        navigate('/')
      }
    } catch (err) {
      setNotice({ kind: 'error', message: err instanceof Error ? err.message : 'An error occurred' })
    } finally {
      setLoading(false)
    }
  }

  const handleForgotPassword = async () => {
    if (!email) {
      setNotice({ kind: 'error', message: 'Enter your email above, then click “Forgot password?”' })
      return
    }
    setNotice(null)
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: window.location.origin,
      })
      if (error) throw error
      setNotice({ kind: 'success', message: 'Password reset link sent. Check your email.' })
    } catch (err) {
      setNotice({ kind: 'error', message: err instanceof Error ? err.message : 'Could not send reset email' })
    }
  }

  return (
    <div className="relative grid min-h-screen w-full bg-background lg:grid-cols-[3fr_2fr]">
      <div className="absolute right-5 top-5 z-20">
        <ThemeToggle className="rounded-lg bg-card/90 backdrop-blur-sm" />
      </div>

      {/* Left: brand hero (below the form on mobile) */}
      <div className="relative order-2 hidden min-h-[16rem] lg:order-1 lg:block">
        <FinOptHeroPanel />
      </div>

      {/* Right: login card */}
      <div className="order-1 flex items-center justify-center px-5 py-10 sm:px-8 lg:order-2">
        <div className="w-full max-w-md animate-fade-in-up">
          {/* Mobile-only wordmark (hero is hidden on small screens) */}
          <div className="mb-8 flex justify-center lg:hidden">
            <FinOptWordmark className="text-primary" />
          </div>

          <div className="rounded-2xl border border-border bg-card p-7 shadow-xl shadow-primary/5 sm:p-9">
            <div className="mb-7 text-center">
              <h2 className="text-2xl font-bold tracking-tight text-foreground">
                {isSignUp ? 'Create your account' : 'Welcome back'}
              </h2>
              <p className="mt-1.5 text-sm text-muted-foreground">
                {isSignUp
                  ? 'Start planning your financial goals.'
                  : 'Continue to your financial planning workspace.'}
              </p>
            </div>

            {notice && (
              <div
                role="alert"
                className={`mb-5 flex items-start gap-2.5 rounded-xl border p-3 text-sm ${
                  notice.kind === 'success'
                    ? 'border-success/30 bg-success-soft text-success'
                    : 'border-danger/30 bg-danger-soft text-danger'
                }`}
              >
                {notice.kind === 'success' ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
                ) : (
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                )}
                <span>{notice.message}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="h-11 rounded-xl"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label htmlFor="password">Password</Label>
                  {!isSignUp && (
                    <button
                      type="button"
                      onClick={handleForgotPassword}
                      className="text-xs font-medium text-primary transition-colors hover:text-primary/80"
                    >
                      Forgot password?
                    </button>
                  )}
                </div>
                <Input
                  id="password"
                  type="password"
                  autoComplete={isSignUp ? 'new-password' : 'current-password'}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  className="h-11 rounded-xl"
                />
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="h-11 w-full rounded-xl text-sm font-semibold"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                {isSignUp ? 'Create account' : 'Sign in'}
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-muted-foreground">
              {isSignUp ? 'Already have an account?' : 'New to FinOpt?'}{' '}
              <button
                type="button"
                onClick={() => {
                  setIsSignUp(!isSignUp)
                  setNotice(null)
                }}
                className="font-semibold text-primary transition-colors hover:text-primary/80"
              >
                {isSignUp ? 'Sign in' : 'Create an account'}
              </button>
            </p>
          </div>

          <p className="mt-6 text-center text-xs text-muted-foreground">
            For educational and research purposes. Not financial advice.
          </p>
        </div>
      </div>
    </div>
  )
}
