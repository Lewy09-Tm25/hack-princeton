/**
 * Shared data formats matching engine/contracts.py and docs/contracts.md.
 * Update all three files together whenever any format changes.
 */

export type ComponentKind =
  | 'embedding'
  | 'layer'
  | 'attention'
  | 'attention_head'
  | 'mlp'
  | 'norm'
  | 'unembedding'
  | 'other'

export interface Component {
  id: string
  kind: ComponentKind
  index?: number | null
  head?: number | null
  n_heads?: number | null
  module?: string | null
  children: string[]
}

export interface ModelMap {
  model_id: string
  revision: string | null
  arch_family: string | null
  n_layers: number
  hidden_size: number | null
  confirmed_by_user: boolean
  needs_confirmation: string[]
  components: Component[]
}

export type SeriesKind = 'pretraining' | 'post_training'

export interface SeriesItem {
  checkpoint_id: string
  label: string
  order: number
  source: string
}

export interface Series {
  series_id: string
  kind: SeriesKind
  model_family: string
  items: SeriesItem[]
}

export interface Plan {
  plan_id?: string | null
  experiment_id?: string | null
  version?: number
  check: string
  model_id: string
  reference_model_id?: string | null
  series_id?: string | null
  checkpoints?: string[] | null
  datasets: string[]
  layers: number[] | 'all'
  controls: string[]
  shift_tests: string[]
  user_context: Record<string, unknown>
  estimate_seconds: number
  coverage: string
  not_covered: string
  maturity: MaturityLevel
  papers: string[]
  approved: boolean
  is_placeholder?: boolean
}

export interface ControlResult {
  kind: string
  value?: number | null
  passed?: boolean | null
  notes?: string | null
}

export type SplitKind = 'train' | 'held_out'

export interface ResultRecord {
  run_id: string
  check: string
  component_id: string
  checkpoint_id: string | null
  metric: string
  value: number
  control: ControlResult | null
  dataset_id: string
  split: SplitKind
  seed: number
  device: string
  precision: string
  quantized: boolean
  is_placeholder?: boolean
}

export type StrengthLevel = 'established' | 'suggestive' | 'inconclusive'

export type MaturityLevel = 'established' | 'experimental'

export interface ReviewerVerdict {
  agrees: boolean
  notes?: string
  corrected_claim?: string | null
  lower_strength_to?: 'suggestive' | 'inconclusive' | null
  [key: string]: unknown
}

export interface Finding {
  finding_id: string
  run_id: string
  check: string
  claim: string
  strength: StrengthLevel
  maturity: MaturityLevel
  evidence: string[]
  controls: ControlResult[]
  shift_tests: ControlResult[]
  coverage: string
  not_shown: string
  verified?: boolean
  is_placeholder?: boolean
  reviewer_verdict?: ReviewerVerdict | null
}

export type JobStatus =
  | 'queued'
  | 'running'
  | 'needs_decision'
  | 'completed'
  | 'stopped'
  | 'failed'

export type PaintScale = 'sequential' | 'diverging'

export type EventType =
  | 'focus'
  | 'paint'
  | 'timeline'
  | 'progress'
  | 'question'
  | 'finding'
  | 'job_status'

export interface FocusPayload {
  component_ids: string[]
  reason: string
}

export interface PaintPayload {
  metric: string
  values: Record<string, number>
  scale: PaintScale
  checkpoint_id?: string | null
  run_id?: string | null
}

export interface TimelinePayload {
  checkpoint_id: string
}

export interface ProgressPayload {
  job_id: string
  step: number
  total: number
  message: string
}

export interface QuestionPayload {
  job_id: string
  text: string
}

export interface FindingEventPayload {
  finding_id: string
}

export interface JobStatusPayload {
  job_id: string
  status: JobStatus
  message?: string | null
}

export type TypedEvent =
  | { type: 'focus'; payload: FocusPayload }
  | { type: 'paint'; payload: PaintPayload }
  | { type: 'timeline'; payload: TimelinePayload }
  | { type: 'progress'; payload: ProgressPayload }
  | { type: 'question'; payload: QuestionPayload }
  | { type: 'finding'; payload: FindingEventPayload }
  | { type: 'job_status'; payload: JobStatusPayload }

export interface Event {
  type: EventType
  payload: Record<string, unknown>
}

export function isValidEvent(value: unknown): value is TypedEvent {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const candidate = value as Record<string, unknown>
  if (typeof candidate.type !== 'string' || typeof candidate.payload !== 'object' || candidate.payload === null) {
    return false
  }
  const p = candidate.payload as Record<string, unknown>
  switch (candidate.type) {
    case 'focus':
      return Array.isArray(p.component_ids) && typeof p.reason === 'string'
    case 'paint':
      return (
        typeof p.metric === 'string' &&
        typeof p.values === 'object' &&
        p.values !== null &&
        (p.scale === 'sequential' || p.scale === 'diverging')
      )
    case 'timeline':
      return typeof p.checkpoint_id === 'string'
    case 'progress':
      return (
        typeof p.job_id === 'string' &&
        typeof p.step === 'number' &&
        typeof p.total === 'number' &&
        typeof p.message === 'string'
      )
    case 'question':
      return typeof p.job_id === 'string' && typeof p.text === 'string'
    case 'finding':
      return typeof p.finding_id === 'string'
    case 'job_status':
      return (
        typeof p.job_id === 'string' &&
        typeof p.status === 'string' &&
        ['queued', 'running', 'needs_decision', 'completed', 'stopped', 'failed'].includes(p.status)
      )
    default:
      return false
  }
}
