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
  http_command?: string
  stdio_command?: string
  plugin_command?: string
  mcp_url?: string
  workspace?: string
  notes?: string
  description?: string
}

export interface ReportSummary {
  report_id: string
  run_id?: string
  created_at?: string
  path?: string
  url?: string
  [key: string]: unknown
}

export interface ResultsQueryParams {
  run_id?: string
  check?: string
  metric?: string
  checkpoint_id?: string
}

export interface SpotCheckExample {
  dataset_id: string
  index: number
  prompt: string
  response: string
  redacted?: boolean
  contains_harmful?: boolean
  is_placeholder?: boolean
  reference_response?: string | null
  steered_response?: string | null
  label?: string | null
  [key: string]: unknown
}

export interface EvidenceBundle {
  finding: Finding | null
  results: ResultRecord[]
  examples: SpotCheckExample[]
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

export interface ActiveModelState {
  /** The model shown in this engine session, or null when none is loaded yet. */
  modelId: string | null
  /** Models currently being loaded. */
  loading: string[]
  /** Checkpoint series of the current check, if any. */
  seriesId: string | null
}

export async function fetchActiveModel(): Promise<ActiveModelState> {
  const raw = await requestJson<{
    model_id?: string | null
    loading?: unknown
    series_id?: string | null
  }>('/api/models/active')
  const modelId = raw && typeof raw.model_id === 'string' && raw.model_id ? raw.model_id : null
  const loading = Array.isArray(raw?.loading)
    ? raw.loading.filter((s): s is string => typeof s === 'string')
    : []
  const seriesId =
    raw && typeof raw.series_id === 'string' && raw.series_id ? raw.series_id : null
  return { modelId, loading, seriesId }
}

const DEMO_MODEL_NAMES: ReadonlyArray<readonly [string, string]> = [
  ['qwen2.5-1.5b-sleeper-canary', 'Qwen2.5-1.5B (community fine-tune)'],
]

/**
 * A plain name for a model, for display. Internal IDs such as
 * "hf:Qwen/Qwen2.5-1.5B-Instruct@main+adapter:my_adapter" become
 * "Qwen2.5-1.5B-Instruct (fine-tuned)".
 */
export function displayModelName(modelId: string): string {
  // Demo test models whose repository names describe the problem built into them.
  // Show a neutral name so the check, not the name, reveals what is wrong.
  const lower = modelId.toLowerCase()
  for (const [needle, label] of DEMO_MODEL_NAMES) {
    if (lower.includes(needle)) return label
  }
  let id = modelId.replace(/^(hf|local):/, '')
  let fineTuned = false
  const adapterIdx = id.indexOf('+adapter:')
  if (adapterIdx >= 0) {
    id = id.slice(0, adapterIdx)
    fineTuned = true
  }
  id = id.replace(/@[^/]*$/, '')
  // Local folders look like "Org__Name/main"; drop the trailing revision folder.
  id = id.replace(/\/(main|default)$/, '')
  const last = id.split('/').pop() ?? id
  const name = last.includes('__') ? last.split('__').pop() ?? last : last
  return fineTuned ? `${name} (fine-tuned)` : name
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

export async function fetchExamples(runId: string, n = 5): Promise<SpotCheckExample[]> {
  const safeN = Math.max(1, Math.min(50, Math.round(n)))
  const qs = `?run_id=${encodeURIComponent(runId)}&n=${safeN}`
  const raw = await requestJson<unknown>(`/api/examples${qs}`)
  return unwrapList<SpotCheckExample>(raw, 'examples')
}

export async function fetchEvidence(
  findingOrRunId: Finding | string,
  exampleCount = 5,
): Promise<EvidenceBundle> {
  const runId = typeof findingOrRunId === 'string' ? findingOrRunId : findingOrRunId.run_id
  const [results, examples, findings] = await Promise.all([
    fetchResults({ run_id: runId }),
    fetchExamples(runId, exampleCount).catch(() => [] as SpotCheckExample[]),
    typeof findingOrRunId === 'string'
      ? fetchFindings(runId).catch(() => [] as Finding[])
      : Promise.resolve([findingOrRunId]),
  ])
  const finding =
    typeof findingOrRunId === 'string' ? (findings[0] ?? null) : findingOrRunId
  return {
    finding,
    results,
    examples,
  }
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

export async function fetchReports(): Promise<ReportSummary[]> {
  const raw = await requestJson<unknown>('/api/reports')
  return unwrapList<ReportSummary>(raw, 'reports')
}

export async function buildReportForRun(runId: string): Promise<ReportSummary> {
  return requestJson<ReportSummary>(`/api/reports/${encodeURIComponent(runId)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ run_id: runId }),
  })
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
