import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { supabase } from '../lib/supabase'
import { useAuthStore } from '../lib/store'
import type { Profile, Scenario, Job } from '../types/database'

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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'running': return 'bg-blue-100 text-blue-800'
      case 'failed': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Welcome back! Here's an overview of your portfolio optimization work.
        </p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-lg bg-white p-6 shadow">
          <p className="text-sm font-medium text-gray-500">Total Profiles</p>
          <p className="mt-2 text-3xl font-semibold text-gray-900">
            {profilesLoading ? '...' : profiles?.length ?? 0}
          </p>
        </div>
        <div className="rounded-lg bg-white p-6 shadow">
          <p className="text-sm font-medium text-gray-500">Active Jobs</p>
          <p className="mt-2 text-3xl font-semibold text-gray-900">
            {jobsLoading ? '...' : recentJobs?.filter(j => j.status === 'running').length ?? 0}
          </p>
        </div>
        <div className="rounded-lg bg-white p-6 shadow">
          <p className="text-sm font-medium text-gray-500">Completed Today</p>
          <p className="mt-2 text-3xl font-semibold text-gray-900">
            {jobsLoading ? '...' : recentJobs?.filter(j =>
              j.status === 'completed' &&
              new Date(j.completed_at ?? '').toDateString() === new Date().toDateString()
            ).length ?? 0}
          </p>
        </div>
      </div>

      {/* Quick actions */}
      <div className="flex gap-4">
        <Link
          to="/profiles"
          className="rounded-md bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700"
        >
          Create Profile
        </Link>
        <Link
          to="/scenarios"
          className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
        >
          New Scenario
        </Link>
      </div>

      {/* Recent jobs */}
      <div className="rounded-lg bg-white shadow">
        <div className="border-b border-gray-200 px-6 py-4">
          <h2 className="text-lg font-medium text-gray-900">Recent Jobs</h2>
        </div>
        <div className="divide-y divide-gray-200">
          {jobsLoading ? (
            <div className="px-6 py-4 text-sm text-gray-500">Loading...</div>
          ) : recentJobs?.length === 0 ? (
            <div className="px-6 py-4 text-sm text-gray-500">No jobs yet. Create a scenario to get started.</div>
          ) : (
            recentJobs?.map((job) => (
              <div key={job.id} className="flex items-center justify-between px-6 py-4">
                <div>
                  <p className="font-medium text-gray-900">
                    {job.scenarios?.name ?? 'Unknown Scenario'}
                  </p>
                  <p className="text-sm text-gray-500">
                    {job.job_type} - {job.scenarios?.profiles?.name ?? 'Unknown Profile'}
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  {job.status === 'running' && (
                    <span className="text-sm text-gray-500">{job.progress}%</span>
                  )}
                  <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                  {job.status === 'completed' && (
                    <Link
                      to={`/results/${job.id}`}
                      className="text-sm text-primary-600 hover:text-primary-500"
                    >
                      View Results
                    </Link>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
