import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import type { Job } from '../types/database'

export function useJobProgress(jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) {
      setLoading(false)
      return
    }

    // Initial fetch
    const fetchJob = async () => {
      const { data, error } = await supabase
        .from('jobs')
        .select('*')
        .eq('id', jobId)
        .single()

      if (error) {
        setError(error.message)
      } else {
        setJob(data as Job)
      }
      setLoading(false)
    }

    fetchJob()

    // Subscribe to realtime updates
    const channel = supabase
      .channel(`job:${jobId}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'jobs',
          filter: `id=eq.${jobId}`,
        },
        (payload) => {
          setJob(payload.new as Job)
        }
      )
      .subscribe()

    return () => {
      channel.unsubscribe()
    }
  }, [jobId])

  return { job, loading, error }
}
