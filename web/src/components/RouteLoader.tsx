import { cn } from '@/lib/utils'

type RouteLoaderProps = {
  className?: string
  fullScreen?: boolean
  label?: string
}

export default function RouteLoader({
  className,
  fullScreen = false,
  label = 'Loading page…',
}: RouteLoaderProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-center',
        fullScreen ? 'min-h-screen bg-background text-foreground' : 'min-h-[16rem] text-foreground',
        className,
      )}
    >
      <div className="text-center">
        <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <p className="mt-4 text-sm text-muted-foreground">{label}</p>
      </div>
    </div>
  )
}
