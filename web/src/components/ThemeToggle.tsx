import { LaptopMinimal, Moon, Sun } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useTheme, type ThemePreference } from './theme'

const NEXT_THEME: Record<ThemePreference, ThemePreference> = {
  light: 'dark',
  dark: 'system',
  system: 'light',
}

export function ThemeToggle({ className }: { className?: string }) {
  const { theme, resolvedTheme, setTheme } = useTheme()

  const nextTheme = NEXT_THEME[theme]

  const label =
    theme === 'system'
      ? `System (${resolvedTheme === 'dark' ? 'dark' : 'light'})`
      : theme === 'dark'
        ? 'Dark'
        : 'Light'

  const Icon =
    theme === 'system'
      ? LaptopMinimal
      : theme === 'dark'
        ? Moon
        : Sun

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={className}
      onClick={() => setTheme(nextTheme)}
      title={`Theme: ${label}. Click to switch to ${nextTheme}.`}
      aria-label={`Theme: ${label}. Click to switch to ${nextTheme}.`}
    >
      <Icon className="h-4 w-4" />
      <span className="hidden sm:inline">{label}</span>
    </Button>
  )
}
