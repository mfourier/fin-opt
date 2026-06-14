import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
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
  const { t } = useTranslation('login')
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
        setNotice({ kind: 'success', message: t('notice.checkEmail') })
      } else {
        await signIn(email, password)
        navigate('/')
      }
    } catch (err) {
      setNotice({ kind: 'error', message: err instanceof Error ? err.message : t('notice.genericError') })
    } finally {
      setLoading(false)
    }
  }

  const handleForgotPassword = async () => {
    if (!email) {
      setNotice({ kind: 'error', message: t('notice.enterEmailFirst') })
      return
    }
    setNotice(null)
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: window.location.origin,
      })
      if (error) throw error
      setNotice({ kind: 'success', message: t('notice.resetSent') })
    } catch (err) {
      setNotice({ kind: 'error', message: err instanceof Error ? err.message : t('notice.resetError') })
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
                {isSignUp ? t('title.signUp') : t('title.signIn')}
              </h2>
              <p className="mt-1.5 text-sm text-muted-foreground">
                {isSignUp ? t('subtitle.signUp') : t('subtitle.signIn')}
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
                <Label htmlFor="email">{t('email')}</Label>
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
                  <Label htmlFor="password">{t('password')}</Label>
                  {!isSignUp && (
                    <button
                      type="button"
                      onClick={handleForgotPassword}
                      className="text-xs font-medium text-primary transition-colors hover:text-primary/80"
                    >
                      {t('forgotPassword')}
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
                {isSignUp ? t('submit.signUp') : t('submit.signIn')}
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-muted-foreground">
              {isSignUp ? t('switchPrompt.signUp') : t('switchPrompt.signIn')}{' '}
              <button
                type="button"
                onClick={() => {
                  setIsSignUp(!isSignUp)
                  setNotice(null)
                }}
                className="font-semibold text-primary transition-colors hover:text-primary/80"
              >
                {isSignUp ? t('switchAction.signUp') : t('switchAction.signIn')}
              </button>
            </p>
          </div>

          <p className="mt-6 text-center text-xs text-muted-foreground">
            {t('disclaimer')}
          </p>
        </div>
      </div>
    </div>
  )
}
