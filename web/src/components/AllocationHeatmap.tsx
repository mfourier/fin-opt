interface AllocationHeatmapProps {
  allocation: number[][]
  accountNames: string[]
}

export default function AllocationHeatmap({ allocation, accountNames }: AllocationHeatmapProps) {
  if (!allocation || allocation.length === 0) {
    return <p className="text-gray-500">No allocation data available</p>
  }

  const T = allocation.length
  const M = allocation[0]?.length ?? 0

  // Generate color based on allocation percentage
  const getColor = (value: number) => {
    // Blue gradient: higher = darker blue
    const intensity = Math.round(value * 255)
    return `rgb(${255 - intensity}, ${255 - intensity * 0.5}, 255)`
  }

  // Show only every nth month for large horizons
  const step = T > 60 ? Math.ceil(T / 30) : T > 24 ? 2 : 1
  const displayMonths = Array.from({ length: Math.ceil(T / step) }, (_, i) => i * step)

  return (
    <div className="overflow-x-auto">
      <div className="min-w-fit">
        {/* Legend */}
        <div className="mb-4 flex items-center gap-2 text-xs text-gray-500">
          <span>0%</span>
          <div className="flex h-3">
            {Array.from({ length: 10 }).map((_, i) => (
              <div
                key={i}
                className="w-4"
                style={{ backgroundColor: getColor(i / 9) }}
              />
            ))}
          </div>
          <span>100%</span>
        </div>

        {/* Heatmap */}
        <div className="flex">
          {/* Y-axis labels (accounts) */}
          <div className="flex flex-col pr-2">
            <div className="h-6" /> {/* Spacer for x-axis */}
            {accountNames.map((name, m) => (
              <div key={m} className="flex h-8 items-center text-xs text-gray-600">
                {name || `Account ${m}`}
              </div>
            ))}
          </div>

          {/* Grid */}
          <div>
            {/* X-axis labels (months) */}
            <div className="flex">
              {displayMonths.map((t) => (
                <div key={t} className="h-6 w-4 text-center text-[10px] text-gray-500">
                  {t % 12 === 0 ? `Y${t / 12}` : ''}
                </div>
              ))}
            </div>

            {/* Cells */}
            {Array.from({ length: M }).map((_, m) => (
              <div key={m} className="flex">
                {displayMonths.map((t) => {
                  const value = allocation[t]?.[m] ?? 0
                  return (
                    <div
                      key={t}
                      className="h-8 w-4 border border-white"
                      style={{ backgroundColor: getColor(value) }}
                      title={`Month ${t}, ${accountNames[m] || `Account ${m}`}: ${(value * 100).toFixed(1)}%`}
                    />
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Summary stats */}
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
          {accountNames.map((name, m) => {
            const accountAllocation = allocation.map(row => row[m] ?? 0)
            const avg = accountAllocation.reduce((a, b) => a + b, 0) / T
            const min = Math.min(...accountAllocation)
            const max = Math.max(...accountAllocation)
            return (
              <div key={m} className="rounded-md bg-gray-50 p-3">
                <p className="font-medium text-gray-700">{name || `Account ${m}`}</p>
                <p className="text-xs text-gray-500">
                  Avg: {(avg * 100).toFixed(1)}% | Range: {(min * 100).toFixed(0)}-{(max * 100).toFixed(0)}%
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
