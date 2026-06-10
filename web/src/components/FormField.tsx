import { ReactNode, useEffect, useRef, useState } from 'react'

interface FormFieldProps {
  label: string
  error?: string
  hint?: string
  required?: boolean
  children: ReactNode
}

export function FormField({ label, error, hint, required, children }: FormFieldProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700">
        {label}
        {required && <span className="ml-1 text-red-500">*</span>}
      </label>
      <div className="mt-1">{children}</div>
      {hint && !error && (
        <p className="mt-1 text-xs text-gray-500">{hint}</p>
      )}
      {error && (
        <p className="mt-1 text-xs text-red-600">{error}</p>
      )}
    </div>
  )
}

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean
}

export function Input({ error, className = '', ...props }: InputProps) {
  return (
    <input
      {...props}
      className={`block w-full rounded-md border px-3 py-2 focus:outline-none focus:ring-1 ${
        error
          ? 'border-red-300 focus:border-red-500 focus:ring-red-500'
          : 'border-gray-300 focus:border-primary-500 focus:ring-primary-500'
      } ${className}`}
    />
  )
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  error?: boolean
}

export function Select({ error, className = '', children, ...props }: SelectProps) {
  return (
    <select
      {...props}
      className={`block w-full rounded-md border px-3 py-2 focus:outline-none focus:ring-1 ${
        error
          ? 'border-red-300 focus:border-red-500 focus:ring-red-500'
          : 'border-gray-300 focus:border-primary-500 focus:ring-primary-500'
      } ${className}`}
    >
      {children}
    </select>
  )
}

// -----------------------------------------------------------------------------
// CurrencyInput — money field with thousands separators (es-CL) and $ prefix.
// Stores a plain number; renders a formatted string so large amounts (e.g.
// 50.000.000) are readable and hard to mis-enter by an extra zero.
// Emits `null` when the field is cleared so optional amounts can be unset.
// -----------------------------------------------------------------------------

const currencyFormatter = new Intl.NumberFormat('en-US')

function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return ''
  return currencyFormatter.format(value)
}

interface CurrencyInputProps {
  value: number | null | undefined
  onChange: (value: number | null) => void
  error?: boolean
  placeholder?: string
  id?: string
  className?: string
  'aria-label'?: string
}

export function CurrencyInput({
  value,
  onChange,
  error,
  placeholder = '0',
  id,
  className = '',
  'aria-label': ariaLabel,
}: CurrencyInputProps) {
  const [display, setDisplay] = useState(() => formatCurrency(value))
  const focused = useRef(false)

  // Keep the displayed text in sync with external value changes while not typing.
  useEffect(() => {
    if (!focused.current) setDisplay(formatCurrency(value))
  }, [value])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const digits = e.target.value.replace(/[^\d]/g, '')
    if (digits === '') {
      setDisplay('')
      onChange(null)
      return
    }
    const parsed = Number(digits)
    setDisplay(currencyFormatter.format(parsed))
    onChange(parsed)
  }

  return (
    <div className="relative">
      <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-gray-500">
        $
      </span>
      <input
        id={id}
        type="text"
        inputMode="numeric"
        value={display}
        onChange={handleChange}
        onFocus={() => (focused.current = true)}
        onBlur={() => {
          focused.current = false
          setDisplay(formatCurrency(value))
        }}
        placeholder={placeholder}
        aria-label={ariaLabel}
        className={`block w-full rounded-md border py-2 pl-7 pr-3 focus:outline-none focus:ring-1 ${
          error
            ? 'border-red-300 focus:border-red-500 focus:ring-red-500'
            : 'border-gray-300 focus:border-primary-500 focus:ring-primary-500'
        } ${className}`}
      />
    </div>
  )
}

interface ValidationSummaryProps {
  errors: Array<{ field: string; message: string }>
  title?: string
}

export function ValidationSummary({ errors, title = 'Please fix the following errors:' }: ValidationSummaryProps) {
  if (errors.length === 0) return null

  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-4">
      <div className="flex">
        <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
        </svg>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-red-800">{title}</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-red-700">
            {errors.map((error, i) => (
              <li key={i}>{error.message}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
