import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

export const SUPPORTED_LANGUAGES = ['es', 'en'] as const
export type Language = (typeof SUPPORTED_LANGUAGES)[number]

export const LANGUAGE_STORAGE_KEY = 'finopt-lang'
export const DEFAULT_LANGUAGE: Language = 'es'

// Auto-load every locale JSON so adding a namespace = dropping a file under
// locales/<lang>/<namespace>.json, no edits here. Path shape is fixed by the
// glob below: ./locales/<lang>/<namespace>.json
const modules = import.meta.glob('./locales/*/*.json', { eager: true }) as Record<
  string,
  { default: Record<string, unknown> }
>

const resources: Record<string, Record<string, Record<string, unknown>>> = {}
const namespaces = new Set<string>()

for (const [path, mod] of Object.entries(modules)) {
  const match = path.match(/\.\/locales\/([^/]+)\/([^/]+)\.json$/)
  if (!match) continue
  const [, lang, ns] = match
  resources[lang] ??= {}
  resources[lang][ns] = mod.default
  namespaces.add(ns)
}

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    // Spanish is the product default: only an explicit prior choice in
    // localStorage should override it (we intentionally do not infer from the
    // browser, to keep the default predictable for the Chilean audience).
    fallbackLng: DEFAULT_LANGUAGE,
    supportedLngs: SUPPORTED_LANGUAGES as unknown as string[],
    ns: Array.from(namespaces),
    defaultNS: 'common',
    detection: {
      order: ['localStorage'],
      lookupLocalStorage: LANGUAGE_STORAGE_KEY,
      caches: ['localStorage'],
    },
    interpolation: {
      escapeValue: false, // React already escapes against XSS.
    },
  })

// Keep <html lang> in sync for accessibility/SEO, mirroring how ThemeProvider
// toggles the `dark` class on documentElement.
function syncHtmlLang(lng: string) {
  if (typeof document !== 'undefined') {
    document.documentElement.lang = lng
  }
}
syncHtmlLang(i18n.language)
i18n.on('languageChanged', syncHtmlLang)

export default i18n
