import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  CheckCircle2,
  FolderKanban,
  Loader2,
  Plus,
  Target,
  Users,
} from 'lucide-react'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import type { Profile, Scenario, Job } from '../types/database'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

const statusStyles: Record<string, string> = {
  completed: 'bg-success-soft text-success',
  running: 'bg-accent text-accent-foreground',
  failed: 'bg-danger-soft text-danger',
  pending: 'bg-muted text-muted-foreground',
}

export default function DashboardPage() {
  const user = useAuthStore((state) => state.user)

  const { data: profiles, isLoading: profilesLoading } = useQuery({
    // 'dashboard' segment: this query only fetches the 5 most recent profiles,
    // so it must not share a cache entry with the full list in ProfilesPage.
    queryKey: ['profiles', 'dashboard', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(5)
      if (error) throw error
      return data as Profile[]
    },
    enabled: !!user,
  })

  const { data: recentJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['recent-jobs', user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('jobs')
        .select('*, scenarios(name, profiles(name))')
        .order('created_at', { ascending: false })
        .limit(10)
      if (error) throw error
      return data as (Job & { scenarios: Scenario & { profiles: Profile } })[]
    },
    enabled: !!user,
  })

  const activeJobs = recentJobs?.filter((j) => j.status === 'running').length ?? 0
  const completedToday =
    recentJobs?.filter(
      (j) =>
        j.status === 'completed' &&
        new Date(j.completed_at ?? '').toDateString() === new Date().toDateString(),
    ).length ?? 0

  const stats = [
    {
      label: 'Profiles',
      value: profilesLoading ? '—' : profiles?.length ?? 0,
      icon: Users,
    },
    {
      label: 'Active plans',
      value: jobsLoading ? '—' : activeJobs,
      icon: Loader2,
    },
    {
      label: 'Completed today',
      value: jobsLoading ? '—' : completedToday,
      icon: CheckCircle2,
    },
  ]

  return (
    <div className="space-y-8">
      <div className="animate-fade-in-up">
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Dashboard</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Welcome back. Here's an overview of your goal-based plans.
        </p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {stats.map(({ label, value, icon: Icon }) => (
          <Card key={label} className="flex items-center gap-4 p-5 shadow-sm">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent text-primary">
              <Icon className="h-5 w-5" />
            </span>
            <div>
              <p className="text-sm font-medium text-muted-foreground">{label}</p>
              <p className="mt-0.5 text-2xl font-semibold tabular text-foreground">{value}</p>
            </div>
          </Card>
        ))}
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap gap-3">
        <Button asChild className="rounded-xl">
          <Link to="/profiles">
            <Plus className="h-4 w-4" />
            Add a situation
          </Link>
        </Button>
        <Button asChild variant="outline" className="rounded-xl">
          <Link to="/scenarios">
            <Target className="h-4 w-4" />
            New plan
          </Link>
        </Button>
      </div>

      {/* Recent jobs */}
      <Card className="overflow-hidden">
        <div className="flex items-center gap-2 border-b border-border px-6 py-4">
          <FolderKanban className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-base font-semibold text-foreground">Recent plans</h2>
        </div>
        <div className="divide-y divide-border">
          {jobsLoading ? (
            <div className="flex items-center gap-2 px-6 py-5 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading…
            </div>
          ) : recentJobs?.length === 0 ? (
            <div className="px-6 py-10 text-center">
              <p className="text-sm text-muted-foreground">
                No plans yet. Create one to see when your goals become achievable.
              </p>
              <Button asChild variant="outline" size="sm" className="mt-4 rounded-xl">
                <Link to="/scenarios">
                  <Target className="h-4 w-4" />
                  Create your first plan
                </Link>
              </Button>
            </div>
          ) : (
            recentJobs?.map((job) => (
              <div
                key={job.id}
                className="flex flex-wrap items-center justify-between gap-3 px-6 py-4 transition-colors hover:bg-muted/50"
              >
                <div className="min-w-0">
                  <p className="truncate font-medium text-foreground">
                    {job.scenarios?.name ?? 'Untitled plan'}
                  </p>
                  <p className="truncate text-sm text-muted-foreground">
                    {job.job_type} · {job.scenarios?.profiles?.name ?? 'Unknown situation'}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {job.status === 'running' && (
                    <span className="tabular text-sm text-muted-foreground">{job.progress}%</span>
                  )}
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${
                      statusStyles[job.status] ?? statusStyles.pending
                    }`}
                  >
                    {job.status}
                  </span>
                  {job.status === 'completed' && (
                    <Link
                      to={`/results/${job.id}`}
                      className="inline-flex items-center gap-1 text-sm font-medium text-primary transition-colors hover:text-primary/80"
                    >
                      View
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  )
}
