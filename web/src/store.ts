import { create } from 'zustand'
import type { SpotCheckExample } from './api'
import type {
  Finding,
  JobStatus,
  ModelMap,
  PaintScale,
  Plan,
  ResultRecord,
  Series,
  TypedEvent,
} from './contracts'

export const SINGLE_CHECKPOINT_KEY = '__single__'

export const HONESTY_BANNER_TEXT =
  'Placeholder / unverified test data — not real model evaluation results'

export type ViewMode = '3d' | '2d'
export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'fake'
export type DrawerTab = 'closed' | 'charts' | 'report' | 'claude-code'

export interface PaintEntry {
  metric: string
  scale: PaintScale
  values: Record<string, number>
}

export interface JobState {
  status: JobStatus
  step?: number
  total?: number
  message?: string
  question?: string
}

export type ChatMessageKind = 'text' | 'plan' | 'finding' | 'question' | 'tool_activity'

export interface ToolActivityInfo {
  summary: string
  detail?: string
  done?: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  kind: ChatMessageKind
  text: string
  streaming?: boolean
  contextComponentId?: string | null
  planId?: string
  plan?: Plan
  findingId?: string
  finding?: Finding
  questionJobId?: string
  toolActivity?: ToolActivityInfo
  isPlaceholder?: boolean
  timestamp: number
}

export interface AppState {
  modelMap: ModelMap | null
  series: Series | null
  currentCheckpointId: string | null
  paintByCheckpoint: Record<string, PaintEntry>
  focusedComponentIds: string[]
  /** The strongest component as chosen by the check itself (from its own focus event). */
  peakComponentId: string | null
  /** The layer a running check is measuring right now, from its progress messages. */
  scanningComponentId: string | null
  focusReason: string | null
  selectedComponentId: string | null
  expandedLayerIds: string[]
  viewMode: ViewMode
  jobs: Record<string, JobState>
  plans: Record<string, Plan>
  findings: Finding[]
  /** Display name of the model each finding was about, recorded when it arrived. */
  findingModelNames: Record<string, string>
  resultsByRun: Record<string, ResultRecord[]>
  examplesByRun: Record<string, SpotCheckExample[]>
  chatMessages: ChatMessage[]
  draftInput: string
  isFakeMode: boolean
  connectionStatus: ConnectionStatus
  drawerTab: DrawerTab
  activeEvidenceFindingId: string | null

  setModelMap: (modelMap: ModelMap | null) => void
  setSeries: (series: Series | null) => void
  setCurrentCheckpointId: (checkpointId: string | null) => void
  selectComponent: (componentId: string | null) => void
  focusLayerFromChart: (layerId: string, reason?: string) => void
  toggleLayerExpanded: (layerId: string) => void
  setExpandedLayerIds: (layerIds: string[]) => void
  setViewMode: (mode: ViewMode) => void
  upsertPlan: (plan: Plan) => string
  markPlanApproved: (planId: string) => void
  upsertFinding: (finding: Finding) => void
  setFindingModelName: (findingId: string, modelName: string) => void
  setFindings: (findings: Finding[]) => void
  setResultsForRun: (runId: string, results: ResultRecord[]) => void
  setExamplesForRun: (runId: string, examples: SpotCheckExample[]) => void
  addChatMessage: (message: ChatMessage) => void
  updateChatMessage: (id: string, patch: Partial<ChatMessage>) => void
  appendChatStreamText: (id: string, delta: string) => void
  upsertChatToolActivity: (jobId: string, summary: string, done?: boolean, detail?: string) => void
  setChatMessages: (messages: ChatMessage[]) => void
  setDraftInput: (input: string) => void
  setFakeMode: (enabled: boolean) => void
  setConnectionStatus: (status: ConnectionStatus) => void
  setDrawerTab: (tab: DrawerTab) => void
  openEvidenceForFinding: (findingId: string) => void
  applyEvent: (event: TypedEvent) => void
  resetStore: (overrides?: Partial<Pick<AppState, 'isFakeMode' | 'viewMode'>>) => void
}

interface BrowserWindowLike {
  location?: {
    search?: string
  }
  document?: {
    createElement?: (tag: string) => {
      getContext?: (contextId: string) => unknown
    }
  }
}

function getBrowserWindow(): BrowserWindowLike | undefined {
  const g = globalThis as { window?: BrowserWindowLike }
  return g.window
}

export function detectInitialFakeMode(): boolean {
  const win = getBrowserWindow()
  const search = win?.location?.search
  if (!search) {
    return false
  }
  const params = new URLSearchParams(search)
  const fake = params.get('fake')
  return fake === '1' || fake === 'true'
}

export function detectWebGLSupport(): boolean {
  const win = getBrowserWindow()
  const createElement = win?.document?.createElement
  if (!createElement) {
    return false
  }
  try {
    const canvas = createElement.call(win.document, 'canvas')
    const gl =
      canvas.getContext?.('webgl2') ??
      canvas.getContext?.('webgl') ??
      canvas.getContext?.('experimental-webgl')
    return Boolean(gl)
  } catch {
    return false
  }
}

export function extractParentLayerId(componentId: string): string | null {
  const match = /^(L\d+)\./.exec(componentId)
  return match ? match[1] : null
}

export function formatComponentLabel(componentId: string): string {
  if (componentId === 'embed') return 'Embedding'
  if (componentId === 'unembed') return 'Unembedding'
  const headMatch = /^L(\d+)\.attn\.h(\d+)$/.exec(componentId)
  if (headMatch) {
    return `Layer ${headMatch[1]} attention head ${headMatch[2]}`
  }
  const attnMatch = /^L(\d+)\.attn$/.exec(componentId)
  if (attnMatch) {
    return `Layer ${attnMatch[1]} attention`
  }
  const mlpMatch = /^L(\d+)\.mlp$/.exec(componentId)
  if (mlpMatch) {
    return `Layer ${mlpMatch[1]} MLP`
  }
  const normMatch = /^L(\d+)\.(.+)$/.exec(componentId)
  if (normMatch) {
    return `Layer ${normMatch[1]} ${normMatch[2].replace(/_/g, ' ')}`
  }
  const layerMatch = /^L(\d+)$/.exec(componentId)
  if (layerMatch) {
    return `Layer ${layerMatch[1]}`
  }
  return componentId
}

export function hasPlaceholderOrUnverifiedData(state: {
  isFakeMode: boolean
  plans: Record<string, Plan>
  findings: Finding[]
  chatMessages: ChatMessage[]
  resultsByRun?: Record<string, ResultRecord[]>
  examplesByRun?: Record<string, SpotCheckExample[]>
}): boolean {
  if (state.isFakeMode) return true
  for (const plan of Object.values(state.plans)) {
    if (plan.is_placeholder === true) return true
  }
  for (const finding of state.findings) {
    if (finding.is_placeholder === true || finding.verified !== true) return true
  }
  if (state.resultsByRun) {
    for (const rows of Object.values(state.resultsByRun)) {
      if (rows.some((r) => r.is_placeholder === true)) return true
    }
  }
  if (state.examplesByRun) {
    for (const items of Object.values(state.examplesByRun)) {
      if (items.some((ex) => ex.is_placeholder === true)) return true
    }
  }
  for (const msg of state.chatMessages) {
    if (msg.isPlaceholder === true) return true
    if (msg.plan && msg.plan.is_placeholder === true) return true
    if (msg.finding && (msg.finding.is_placeholder === true || msg.finding.verified !== true)) {
      return true
    }
  }
  return false
}

export function getActivePaint(state: {
  currentCheckpointId: string | null
  paintByCheckpoint: Record<string, PaintEntry>
}): PaintEntry | null {
  if (state.currentCheckpointId !== null) {
    return state.paintByCheckpoint[state.currentCheckpointId] ?? null
  }
  return state.paintByCheckpoint[SINGLE_CHECKPOINT_KEY] ?? null
}

function createInitialDataState(overrides?: Partial<Pick<AppState, 'isFakeMode' | 'viewMode'>>) {
  const initialFake = overrides?.isFakeMode ?? detectInitialFakeMode()
  const hasWindow = Boolean(getBrowserWindow())
  const initialViewMode: ViewMode =
    overrides?.viewMode ?? (hasWindow && !detectWebGLSupport() ? '2d' : '3d')

  return {
    modelMap: null as ModelMap | null,
    series: null as Series | null,
    currentCheckpointId: null as string | null,
    paintByCheckpoint: {} as Record<string, PaintEntry>,
    focusedComponentIds: [] as string[],
    peakComponentId: null as string | null,
    scanningComponentId: null as string | null,
    focusReason: null as string | null,
    selectedComponentId: null as string | null,
    expandedLayerIds: [] as string[],
    viewMode: initialViewMode,
    jobs: {} as Record<string, JobState>,
    plans: {} as Record<string, Plan>,
    findings: [] as Finding[],
    findingModelNames: {} as Record<string, string>,
    resultsByRun: {} as Record<string, ResultRecord[]>,
    examplesByRun: {} as Record<string, SpotCheckExample[]>,
    chatMessages: [] as ChatMessage[],
    draftInput: '',
    isFakeMode: initialFake,
    connectionStatus: (initialFake ? 'fake' : 'disconnected') as ConnectionStatus,
    drawerTab: 'closed' as DrawerTab,
    activeEvidenceFindingId: null as string | null,
  }
}

let fallbackPlanCounter = 1

export const useAppStore = create<AppState>((set) => ({
  ...createInitialDataState(),

  setModelMap: (modelMap) =>
    set((state) => {
      const switched = (state.modelMap?.model_id ?? null) !== (modelMap?.model_id ?? null)
      return {
        modelMap,
        expandedLayerIds: [],
        selectedComponentId: null,
        focusedComponentIds: [],
        focusReason: null,
        // A different model: the previous model's colours and timeline no longer apply.
        ...(switched
          ? {
              paintByCheckpoint: {},
              series: null,
              currentCheckpointId: null,
              peakComponentId: null,
              scanningComponentId: null,
            }
          : {}),
      }
    }),

  setSeries: (series) =>
    set(() => ({
      series,
      currentCheckpointId:
        series && series.items.length > 0 ? series.items[0].checkpoint_id : null,
    })),

  setCurrentCheckpointId: (currentCheckpointId) => set({ currentCheckpointId }),

  selectComponent: (selectedComponentId) => set({ selectedComponentId }),

  focusLayerFromChart: (layerId, reason) =>
    set((state) => {
      const parentLayersToExpand = new Set(state.expandedLayerIds)
      const parentLayer = extractParentLayerId(layerId)
      if (parentLayer) {
        parentLayersToExpand.add(parentLayer)
      }
      return {
        focusedComponentIds: [layerId],
        focusReason: reason ?? `Focused ${formatComponentLabel(layerId)} from layer curve`,
        selectedComponentId: layerId,
        expandedLayerIds: Array.from(parentLayersToExpand),
      }
    }),

  toggleLayerExpanded: (layerId) =>
    set((state) => {
      const exists = state.expandedLayerIds.includes(layerId)
      return {
        expandedLayerIds: exists
          ? state.expandedLayerIds.filter((id) => id !== layerId)
          : [...state.expandedLayerIds, layerId],
      }
    }),

  setExpandedLayerIds: (expandedLayerIds) => set({ expandedLayerIds }),

  setViewMode: (viewMode) => set({ viewMode }),

  upsertPlan: (plan) => {
    const id = plan.plan_id ?? `plan_local_${fallbackPlanCounter++}`
    const normalized: Plan = { ...plan, plan_id: id }
    set((state) => ({
      plans: {
        ...state.plans,
        [id]: normalized,
      },
      chatMessages: state.chatMessages.map((msg) =>
        msg.planId === id || msg.plan?.plan_id === id
          ? { ...msg, planId: id, plan: normalized }
          : msg,
      ),
    }))
    return id
  },

  markPlanApproved: (planId) =>
    set((state) => {
      const existing = state.plans[planId]
      const updatedPlan = existing ? { ...existing, approved: true } : undefined
      return {
        plans: updatedPlan
          ? {
              ...state.plans,
              [planId]: updatedPlan,
            }
          : state.plans,
        chatMessages: state.chatMessages.map((msg) => {
          if (msg.planId === planId || msg.plan?.plan_id === planId) {
            return {
              ...msg,
              plan: msg.plan ? { ...msg.plan, approved: true } : updatedPlan,
            }
          }
          return msg
        }),
      }
    }),

  setFindingModelName: (findingId, modelName) =>
    set((state) =>
      state.findingModelNames[findingId]
        ? {}
        : { findingModelNames: { ...state.findingModelNames, [findingId]: modelName } },
    ),

  upsertFinding: (finding) =>
    set((state) => {
      const idx = state.findings.findIndex((f) => f.finding_id === finding.finding_id)
      const nextFindings =
        idx >= 0
          ? state.findings.map((f, i) => (i === idx ? finding : f))
          : [...state.findings, finding]
      return {
        findings: nextFindings,
        chatMessages: state.chatMessages.map((msg) =>
          msg.findingId === finding.finding_id || msg.finding?.finding_id === finding.finding_id
            ? { ...msg, finding }
            : msg,
        ),
      }
    }),

  setFindings: (findings) => set({ findings }),

  setResultsForRun: (runId, results) =>
    set((state) => ({
      resultsByRun: {
        ...state.resultsByRun,
        [runId]: results,
      },
    })),

  setExamplesForRun: (runId, examples) =>
    set((state) => ({
      examplesByRun: {
        ...state.examplesByRun,
        [runId]: examples,
      },
    })),

  addChatMessage: (message) =>
    set((state) => ({
      chatMessages: [...state.chatMessages, message],
    })),

  updateChatMessage: (id, patch) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg)),
    })),

  appendChatStreamText: (id, delta) =>
    set((state) => ({
      chatMessages: state.chatMessages.map((msg) =>
        msg.id === id ? { ...msg, text: msg.text + delta } : msg,
      ),
    })),

  upsertChatToolActivity: (jobId, summary, done = false, detail) =>
    set((state) => {
      const msgId = `tool_activity_${jobId}`
      const existing = state.chatMessages.find((m) => m.id === msgId)
      if (existing) {
        const prevDetail = existing.toolActivity?.detail ?? ''
        const nextDetail =
          detail ??
          (prevDetail ? `${prevDetail}\n${summary}` : summary)
        return {
          chatMessages: state.chatMessages.map((m) =>
            m.id === msgId
              ? {
                  ...m,
                  text: summary,
                  toolActivity: {
                    summary,
                    detail: nextDetail,
                    done,
                  },
                }
              : m,
          ),
        }
      }
      return {
        chatMessages: [
          ...state.chatMessages,
          {
            id: msgId,
            role: 'assistant',
            kind: 'tool_activity',
            text: summary,
            toolActivity: {
              summary,
              detail: detail ?? summary,
              done,
            },
            isPlaceholder: state.isFakeMode,
            timestamp: Date.now(),
          },
        ],
      }
    }),

  setChatMessages: (chatMessages) => set({ chatMessages }),

  setDraftInput: (draftInput) => set({ draftInput }),

  setFakeMode: (enabled) =>
    set(() => ({
      isFakeMode: enabled,
      connectionStatus: enabled ? 'fake' : 'disconnected',
    })),

  setConnectionStatus: (connectionStatus) => set({ connectionStatus }),

  setDrawerTab: (drawerTab) => set({ drawerTab }),

  openEvidenceForFinding: (findingId) =>
    set({
      activeEvidenceFindingId: findingId,
      drawerTab: 'charts',
    }),

  applyEvent: (event) =>
    set((state) => {
      switch (event.type) {
        case 'focus': {
          const parentLayersToExpand = new Set(state.expandedLayerIds)
          for (const cid of event.payload.component_ids) {
            const parentLayer = extractParentLayerId(cid)
            if (parentLayer) {
              parentLayersToExpand.add(parentLayer)
            }
          }
          const fromCheck = (event.payload as { source?: string }).source === 'check'
          return {
            focusedComponentIds: [...event.payload.component_ids],
            focusReason: event.payload.reason,
            expandedLayerIds: Array.from(parentLayersToExpand),
            // A check's focus names its own strongest component: highlight that one.
            ...(fromCheck
              ? { peakComponentId: event.payload.component_ids[0] ?? null, scanningComponentId: null }
              : {}),
          }
        }
        case 'paint': {
          const key = event.payload.checkpoint_id ?? SINGLE_CHECKPOINT_KEY
          return {
            paintByCheckpoint: {
              ...state.paintByCheckpoint,
              [key]: {
                metric: event.payload.metric,
                scale: event.payload.scale,
                values: { ...event.payload.values },
              },
            },
          }
        }
        case 'timeline': {
          return {
            currentCheckpointId: event.payload.checkpoint_id,
          }
        }
        case 'progress': {
          const { job_id, step, total, message } = event.payload
          const prev = state.jobs[job_id]
          // Progress messages name the layer being measured ("... at L5"): mark it in the view.
          const scanned = typeof message === 'string' ? /\bat (L\d+)\b/.exec(message) : null
          // A new run starts: forget the previous run's strongest component.
          const isNewJob = prev === undefined
          return {
            ...(scanned ? { scanningComponentId: scanned[1] } : {}),
            ...(isNewJob ? { peakComponentId: null } : {}),
            jobs: {
              ...state.jobs,
              [job_id]: {
                ...prev,
                status: prev?.status ?? 'running',
                step,
                total,
                message,
              },
            },
          }
        }
        case 'question': {
          const { job_id, text } = event.payload
          const prev = state.jobs[job_id]
          const alreadyHasQuestion = state.chatMessages.some(
            (m) => m.kind === 'question' && m.questionJobId === job_id && m.text === text,
          )
          const nextMessages = alreadyHasQuestion
            ? state.chatMessages
            : [
                ...state.chatMessages,
                {
                  id: `question_${job_id}_${state.chatMessages.length}`,
                  role: 'assistant' as const,
                  kind: 'question' as const,
                  text,
                  questionJobId: job_id,
                  isPlaceholder: state.isFakeMode,
                  timestamp: Date.now(),
                },
              ]
          return {
            jobs: {
              ...state.jobs,
              [job_id]: {
                ...prev,
                status: 'needs_decision',
                question: text,
                message: text,
              },
            },
            chatMessages: nextMessages,
          }
        }
        case 'finding': {
          return {}
        }
        case 'job_status': {
          const { job_id, status, message } = event.payload
          const prev = state.jobs[job_id]
          const nextJob: JobState = {
            ...prev,
            status,
          }
          if (message !== undefined && message !== null) {
            nextJob.message = message
          }
          if (status !== 'needs_decision') {
            delete nextJob.question
          }
          return {
            jobs: {
              ...state.jobs,
              [job_id]: nextJob,
            },
            // The run has finished (or paused): stop marking a layer as being scanned.
            ...(status !== 'running' && status !== 'queued' ? { scanningComponentId: null } : {}),
          }
        }
      }
    }),

  resetStore: (overrides) => set(() => createInitialDataState(overrides)),
}))
