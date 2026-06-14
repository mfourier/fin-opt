import { Languages } from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Button } from '@/components/ui/button'
import type { Language } from '@/i18n/config'

const NEXT_LANGUAGE: Record<Language, Language> = {
  es: 'en',
  en: 'es',
}

export function LanguageToggle({ className }: { className?: string }) {
  const { t, i18n } = useTranslation('common')

  const current: Language = i18n.language === 'en' ? 'en' : 'es'
  const next = NEXT_LANGUAGE[current]
  const nextLabel = t(`language.${next}`)

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={className}
      onClick={() => void i18n.changeLanguage(next)}
      title={t('language.switchTo', { lang: nextLabel })}
      aria-label={t('language.switchTo', { lang: nextLabel })}
    >
      <Languages className="h-4 w-4" />
      <span className="hidden sm:inline">{t(`language.${current}`)}</span>
    </Button>
  )
}
