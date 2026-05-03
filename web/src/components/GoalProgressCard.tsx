import { useState } from 'react'
import type { GoalStatus } from '../types/database'

interface GoalProgressCardProps {
  goals: GoalStatus[]
}

// Formats a number in [0,1] as a percentage string with N decimal places.
function pct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`
}

// Confidence-gap badge: colour + label depend on sign and magnitude.
function ConfidenceGapBadge({ gap }: { gap: number }) {
  if (gap > 0.01) {
    // Significant CVaR conservatism (>1% above specification)
    return (
      <span className="inline-flex items-center gap-1 rounded bg-sky-50 px-2 py-0.5 text-xs font-medium text-sky-700">
        <span>+{pct(gap)} CVaR margin</span>
      </span>
    )
  }
  if (gap >= 0) {
    // Mild conservatism (0–1%)
    return (
      <span className="inline-flex items-center gap-1 rounded bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
        <span>+{pct(gap)} margin</span>
      </span>
    )
  }
  // Violation (shouldn't occur with a feasible CVaR solution)
  return (
    <span className="inline-flex items-center gap-1 rounded bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700">
      <span>{pct(gap)} below target</span>
    </span>
  )
}

// Small info icon that shows the CVaR note in a tooltip on hover.
function CVaRInfoTooltip({ note }: { note: string }) {
  const [visible, setVisible] = useState(false)

  return (
    <span className="relative inline-block">
      <button
        type="button"
        aria-label="CVaR explanation"
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        className="ml-1 text-gray-400 hover:text-gray-600 focus:outline-none"
      >
        {/* Info circle icon */}
        <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {visible && (
        <div
          role="tooltip"
          className="absolute bottom-full left-1/2 z-10 mb-2 w-64 -translate-x-1/2 rounded-lg border border-sky-100 bg-sky-50 px-3 py-2 text-xs text-sky-800 shadow-lg"
        >
          {note}
          {/* Caret */}
          <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-sky-100" />
        </div>
      )}
    </span>
  )
}

export default function GoalProgressCard({ goals }: GoalProgressCardProps) {
  if (!goals || goals.length === 0) {
    return <p className="text-gray-500">No goals defined</p>
  }

  const satisfiedCount = goals.filter(g => g.satisfied).length
  const totalCount = goals.length
  const allSatisfied = satisfiedCount === totalCount

  return (
    <div className="space-y-4">
      {/* Summary Header */}
      <div className="flex items-center justify-between rounded-lg bg-gray-50 p-4">
        <div>
          <p className="text-sm font-medium text-gray-700">Goal Achievement</p>
          <p className="text-2xl font-bold text-gray-900">
            {satisfiedCount} / {totalCount}
          </p>
        </div>
        <div className={`rounded-full px-4 py-2 text-sm font-medium ${
          allSatisfied
            ? 'bg-green-100 text-green-800'
            : satisfiedCount > 0
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-red-100 text-red-800'
        }`}>
          {allSatisfied ? 'All Goals Met' : satisfiedCount > 0 ? 'Partial Success' : 'Goals Not Met'}
        </div>
      </div>

      {/* Individual Goals */}
      <div className="space-y-3">
        {goals.map((goal, index) => {
          // Prefer the explicit empirical_probability field; fall back to actual_probability.
          const empirical = goal.empirical_probability ?? goal.actual_probability
          const gap = goal.confidence_gap

          return (
            <div
              key={index}
              className={`rounded-lg border p-4 ${
                goal.satisfied ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
              }`}
            >
              {/* ── Top row: account name + type badge ── */}
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    {goal.satisfied ? (
                      <svg className="h-5 w-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      <svg className="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    )}
                    <span className="font-medium text-gray-900">{goal.account}</span>
                    <span className={`rounded px-2 py-0.5 text-xs font-medium ${
                      goal.type === 'terminal'
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-purple-100 text-purple-700'
                    }`}>
                      {goal.type}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-gray-600">
                    Target: <span className="font-medium">${goal.threshold.toLocaleString()}</span>
                  </p>
                </div>

                {/* ── Right column: probabilities ── */}
                <div className="text-right">
                  <div className="text-sm text-gray-500">
                    Required: {pct(goal.required_confidence, 0)}
                  </div>
                  {empirical !== undefined && (
                    <div className={`text-lg font-semibold ${
                      empirical >= goal.required_confidence ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {pct(empirical)}
                    </div>
                  )}
                </div>
              </div>

              {/* ── Progress bar ── */}
              {empirical !== undefined && (
                <div className="mt-3">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                    <div
                      className={`h-full transition-all ${
                        empirical >= goal.required_confidence ? 'bg-green-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${Math.min(empirical * 100, 100)}%` }}
                    />
                  </div>
                  {/* Confidence threshold marker */}
                  <div className="relative h-1">
                    <div
                      className="absolute -top-3 h-4 w-0.5 bg-gray-600"
                      style={{ left: `${goal.required_confidence * 100}%` }}
                      title={`Required: ${pct(goal.required_confidence, 0)}`}
                    />
                  </div>
                </div>
              )}

              {/* ── CVaR transparency row ── */}
              {gap !== undefined && gap !== null && (
                <div className="mt-3 flex items-center gap-2 border-t border-gray-200 pt-2">
                  <span className="text-xs text-gray-500">CVaR transparency:</span>
                  <ConfidenceGapBadge gap={gap} />
                  {goal.note && <CVaRInfoTooltip note={goal.note} />}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* CVaR footnote — only when any goal has dual metrics */}
      {goals.some(g => g.confidence_gap !== undefined) && (
        <p className="text-xs text-gray-400">
          CVaR optimisation provides conservative guarantees: actual confidence typically exceeds
          the specified level. The &quot;margin&quot; above reflects this inherent safety buffer.
        </p>
      )}
    </div>
  )
}
