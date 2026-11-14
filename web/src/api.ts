import type {
  Finding,
  JobStatus,
  ModelMap,
  Plan,
  ResultRecord,
  Series,
} from './contracts'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export interface ModelSummary {
  model_id: string
  revision?: string | null
  arch_family?: string | null
  n_layers?: number
  confirmed_by_user?: boolean
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  step?: number
  total?: number
  message?: string | null
  question?: string | null
}

export interface ApprovePlanResponse {
  plan_id: string
  approved: boolean
  plan?: Plan
}

export interface ClaudeCodeSetupResponse {
  command: string
  description?: string
  mcp_url?: string
}

export interface ResultsQueryParams {
  run_id?: string
  check?: string
  metric?: string
  checkpoint_id?: string
}

interface BrowserStorageWindowLike {
  location?: {
    search?: string
  }
  sessionStorage?: {
    getItem?: (key: string) => string | null
  }
}

export function getAuthToken(): string | null {
  const win = (globalThis as { window?: BrowserStorageWindowLike }).window
  const search = win?.location?.search
  if (search) {
    const params = new URLSearchParams(search)
    const queryToken = params.get('token')
    if (queryToken) {
      return queryToken
    }
  }
  try {
    return win?.sessionStorage?.getItem?.('tool_local_token') ?? null
  } catch {
    return null
  }
}

function buildHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...(extra ?? {}) }
  if (!headers.Accept && !headers.accept) {
    headers.Accept = 'application/json'
  }
  const token = getAuthToken()
  if (token && !headers.Authorization && !headers.authorization) {
    headers.Authorization = `Bearer ${token}`
  }
  return headers
}

interface FetchOptions {
  method?: string
  headers?: Record<string, string>
  body?: string
}

async function requestJson<T>(url: string, init?: FetchOptions): Promise<T> {
  const response = await fetch(url, {
    method: init?.method,
    body: init?.body,
    headers: buildHeaders(init?.headers),
  })
  if (!response.ok) {
    const bodyText = await response.text().catch(() => '')
    throw new ApiError(
      response.status,
      bodyText
        ? `Request to ${url} failed (${response.status}): ${bodyText}`
        : `Request to ${url} failed with status ${response.status}`,
    )
  }
  return (await response.json()) as T
}

function unwrapList<T>(raw: unknown, key: string): T[] {
  if (Array.isArray(raw)) {
    return raw as T[]
  }
  if (typeof raw === 'object' && raw !== null && key in raw) {
    const candidate = (raw as Record<string, unknown>)[key]
    if (Array.isArray(candidate)) {
      return candidate as T[]
    }
  }
  return []
}

export async function fetchModels(): Promise<ModelSummary[]> {
  const raw = await requestJson<unknown>('/api/models')
  return unwrapList<ModelSummary>(raw, 'models')
}

export async function fetchModelMap(modelId: string): Promise<ModelMap> {
  return requestJson<ModelMap>(`/api/models/${encodeURIComponent(modelId)}/map`)
}

export async function fetchSeries(seriesId: string): Promise<Series> {
  return requestJson<Series>(`/api/series/${encodeURIComponent(seriesId)}`)
}

export async function fetchResults(params?: ResultsQueryParams): Promise<ResultRecord[]> {
  const search = new URLSearchParams()
  if (params?.run_id) search.set('run_id', params.run_id)
  if (params?.check) search.set('check', params.check)
  if (params?.metric) search.set('metric', params.metric)
  if (params?.checkpoint_id) search.set('checkpoint_id', params.checkpoint_id)
  const qs = search.toString()
  const raw = await requestJson<unknown>(qs ? `/api/results?${qs}` : '/api/results')
  return unwrapList<ResultRecord>(raw, 'results')
}

export async function fetchFindings(runId?: string): Promise<Finding[]> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : ''
  const raw = await requestJson<unknown>(`/api/findings${qs}`)
  return unwrapList<Finding>(raw, 'findings')
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`)
}

export async function approvePlan(planId: string): Promise<ApprovePlanResponse> {
  return requestJson<ApprovePlanResponse>(
    `/api/plans/${encodeURIComponent(planId)}/approve`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ approved: true }),
    },
  )
}

export function getReportUrl(reportId: string): string {
  const base = `/api/reports/${encodeURIComponent(reportId)}`
  const token = getAuthToken()
  return token ? `${base}?token=${encodeURIComponent(token)}` : base
}

export async function fetchReportHtml(reportId: string): Promise<string> {
  const response = await fetch(`/api/reports/${encodeURIComponent(reportId)}`, {
    headers: buildHeaders({ Accept: 'text/html' }),
  })
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `Failed to load report ${reportId} (status ${response.status})`,
    )
  }
  return response.text()
}

export async function fetchClaudeCodeSetup(): Promise<ClaudeCodeSetupResponse> {
  return requestJson<ClaudeCodeSetupResponse>('/api/setup/claude-code')
}
