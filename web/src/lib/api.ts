const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface JobRequest {
  scenario_id: string
  job_id: string
}

interface JobResponse {
  status: string
  job_id: string
  message: string
}

export async function queueSimulation(request: JobRequest): Promise<JobResponse> {
  const response = await fetch(`${API_URL}/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Failed to queue simulation: ${response.statusText}`)
  }

  return response.json()
}

export async function queueOptimization(request: JobRequest): Promise<JobResponse> {
  const response = await fetch(`${API_URL}/optimize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Failed to queue optimization: ${response.statusText}`)
  }

  return response.json()
}

export async function checkHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_URL}/health`)
  return response.json()
}
