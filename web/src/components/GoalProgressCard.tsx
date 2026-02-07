import type { GoalStatus } from '../types/database'

interface GoalProgressCardProps {
  goals: GoalStatus[]
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
        {goals.map((goal, index) => (
          <div
            key={index}
            className={`rounded-lg border p-4 ${
              goal.satisfied ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
            }`}
          >
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
                  <span className="font-medium text-gray-900">
                    {goal.account}
                  </span>
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
              <div className="text-right">
                <div className="text-sm text-gray-500">
                  Required: {(goal.required_confidence * 100).toFixed(0)}%
                </div>
                {goal.actual_probability !== undefined && (
                  <div className={`text-lg font-semibold ${
                    goal.actual_probability >= goal.required_confidence
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}>
                    {(goal.actual_probability * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            {goal.actual_probability !== undefined && (
              <div className="mt-3">
                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                  <div
                    className={`h-full transition-all ${
                      goal.actual_probability >= goal.required_confidence
                        ? 'bg-green-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(goal.actual_probability * 100, 100)}%` }}
                  />
                </div>
                {/* Confidence threshold marker */}
                <div className="relative h-1">
                  <div
                    className="absolute -top-3 h-4 w-0.5 bg-gray-600"
                    style={{ left: `${goal.required_confidence * 100}%` }}
                    title={`Required: ${(goal.required_confidence * 100).toFixed(0)}%`}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
