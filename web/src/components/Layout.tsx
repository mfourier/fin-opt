import { Suspense } from 'react'
import { Outlet, Link, useLocation } from 'react-router-dom'
import { LogOut } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useAuthStore } from '../lib/store'
import { FinOptWordmark } from './finopt/FinOptWordmark'
import { Button } from './ui/button'
import { ThemeToggle } from './ThemeToggle'
import { LanguageToggle } from './LanguageToggle'
import RouteLoader from './RouteLoader'

const navigation = [
  { key: 'nav.dashboard', href: '/' },
  { key: 'nav.situation', href: '/profiles' },
  { key: 'nav.plans', href: '/scenarios' },
]

export default function Layout() {
  const location = useLocation()
  const { t } = useTranslation(['layout', 'common'])
  const signOut = useAuthStore((state) => state.signOut)
  const user = useAuthStore((state) => state.user)

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <nav className="sticky top-0 z-30 border-b border-border bg-card/80 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between gap-4">
            <div className="flex items-center gap-8">
              <Link to="/" className="text-primary transition-opacity hover:opacity-80">
                <FinOptWordmark />
              </Link>
              <div className="hidden items-center gap-1 sm:flex">
                {navigation.map((item) => {
                  const isActive = location.pathname === item.href
                  return (
                    <Link
                      key={item.key}
                      to={item.href}
                      className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-accent text-accent-foreground'
                          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                      }`}
                    >
                      {t(item.key)}
                    </Link>
                  )
                })}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <LanguageToggle className="rounded-lg" />
              <ThemeToggle className="rounded-lg" />
              <span className="hidden max-w-[14rem] truncate text-sm text-muted-foreground md:inline">
                {user?.email}
              </span>
              <Button variant="outline" size="sm" className="rounded-lg" onClick={signOut}>
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">{t('common:signOut')}</span>
              </Button>
            </div>
          </div>

          {/* Mobile nav row */}
          <div className="flex items-center gap-1 border-t border-border py-2 sm:hidden">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.key}
                  to={item.href}
                  className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                  }`}
                >
                  {t(item.key)}
                </Link>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Suspense fallback={<RouteLoader />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  )
}
