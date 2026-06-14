import { supabase } from './supabase'

const _rawApiUrl = import.meta.env.VITE_API_URL
if (!_rawApiUrl) {
  throw new Error(
    'Missing VITE_API_URL. Set the compute API URL before loading the app.'
  )
}

const API_URL = _rawApiUrl.startsWith('http') ? _rawApiUrl : `https://${_rawApiUrl}`

interface JobRequest {
  scenario_id: string
  job_id: string
}

interface JobResponse {
  status: string
  job_id: string
  message: string
}

interface ApiErrorPayload {
  detail?: string
}

async function getAuthHeaders(): Promise<Record<string, string>> {
  const { data, error } = await supabase.auth.getSession()
  if (error) {
    throw new Error(`Failed to read session: ${error.message}`)
  }

  const accessToken = data.session?.access_token
  if (!accessToken) {
    throw new Error('You must be signed in to start a compute job.')
  }

  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${accessToken}`,
  }
}

async function parseApiError(response: Response, fallbackMessage: string): Promise<never> {
  let detail = fallbackMessage

  try {
    const payload = (await response.json()) as ApiErrorPayload
    if (typeof payload.detail === 'string' && payload.detail.trim()) {
      detail = payload.detail
    }
  } catch {
    // Ignore non-JSON error bodies and keep the fallback.
  }

  throw new Error(detail)
}

export async function queueSimulation(request: JobRequest): Promise<JobResponse> {
  const headers = await getAuthHeaders()
  const response = await fetch(`${API_URL}/simulate`, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    return parseApiError(response, `Failed to queue simulation: ${response.statusText}`)
  }

  return response.json()
}

export async function queueOptimization(request: JobRequest): Promise<JobResponse> {
  const headers = await getAuthHeaders()
  const response = await fetch(`${API_URL}/optimize`, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    return parseApiError(response, `Failed to queue optimization: ${response.statusText}`)
  }

  return response.json()
}

export async function checkHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_URL}/health`)
  return response.json()
}
