import { isValidEvent, type TypedEvent } from './contracts'
import { displayModelName, fetchFindings, getAuthToken } from './api'
import { useAppStore } from './store'

export interface SocketLike {
  onopen: (() => void) | null
  onmessage: ((event: { data: unknown }) => void) | null
  onclose: (() => void) | null
  onerror: (() => void) | null
  close: () => void
}

export interface EventsClientOptions {
  url?: string
  initialReconnectDelayMs?: number
  maxReconnectDelayMs?: number
  createWebSocket?: (url: string) => SocketLike
  onFindingEvent?: (findingId: string) => Promise<void> | void
}

interface BrowserLocationWindowLike {
  location?: {
    protocol?: string
    host?: string
  }
}

export function buildWsUrl(path: string): string {
  const win = (globalThis as { window?: BrowserLocationWindowLike }).window
  if (!win?.location?.host) {
    return `ws://127.0.0.1:8000${path}`
  }
  const protocol = win.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const base = `${protocol}//${win.location.host}${path}`
  const token = getAuthToken()
  if (!token) {
    return base
  }
  const sep = base.includes('?') ? '&' : '?'
  return `${base}${sep}token=${encodeURIComponent(token)}`
}

export function computeBackoffDelay(
  attempt: number,
  initialDelayMs = 500,
  maxDelayMs = 15000,
): number {
  const safeAttempt = Math.max(0, Math.min(attempt, 20))
  const raw = initialDelayMs * Math.pow(2, safeAttempt)
  return Math.min(raw, maxDelayMs)
}

export function parseAndValidateEventMessage(rawData: unknown): TypedEvent | null {
  if (typeof rawData !== 'string') {
    return null
  }
  try {
    const parsed: unknown = JSON.parse(rawData)
    if (isValidEvent(parsed)) {
      return parsed
    }
    return null
  } catch {
    return null
  }
}

export class EventsClient {
  private readonly url: string
  private readonly initialDelayMs: number
  private readonly maxDelayMs: number
  private readonly createWs: (url: string) => SocketLike
  private readonly onFindingEvent: (findingId: string) => Promise<void> | void

  private socket: SocketLike | null = null
  private reconnectAttempt = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private stopped = true

  constructor(options?: EventsClientOptions) {
    this.url = options?.url ?? buildWsUrl('/ws/events')
    this.initialDelayMs = options?.initialReconnectDelayMs ?? 500
    this.maxDelayMs = options?.maxReconnectDelayMs ?? 15000
    this.createWs =
      options?.createWebSocket ??
      ((u) => {
        const WsCtor = (globalThis as unknown as { WebSocket: new (wsUrl: string) => SocketLike })
          .WebSocket
        return new WsCtor(u)
      })
    this.onFindingEvent =
      options?.onFindingEvent ??
      (async (findingId: string) => {
        try {
          const allFindings = await fetchFindings()
          const store = useAppStore.getState()
          // Add only the finding this event is about. The engine's record also holds
          // findings from earlier sessions, which do not belong in this conversation.
          const matched = allFindings.find((f) => f.finding_id === findingId)
          if (matched) {
            store.upsertFinding(matched)
            // Remember which model it was about, for the Evidence panel's list of findings.
            const shownModel = store.modelMap?.model_id
            if (shownModel) store.setFindingModelName(matched.finding_id, displayModelName(shownModel))
            const alreadyInChat = store.chatMessages.some(
              (m) => m.kind === 'finding' && m.findingId === matched.finding_id,
            )
            if (!alreadyInChat) {
              store.addChatMessage({
                id: `finding_msg_${matched.finding_id}`,
                role: 'assistant',
                kind: 'finding',
                text: matched.claim,
                findingId: matched.finding_id,
                finding: matched,
                isPlaceholder: Boolean(matched.is_placeholder),
                timestamp: Date.now(),
              })
            }
          }
        } catch {
          // Findings will be fetched on next update or chat stream
        }
      })
  }

  start(): void {
    if (useAppStore.getState().isFakeMode) {
      useAppStore.getState().setConnectionStatus('fake')
      return
    }
    this.stopped = false
    this.openSocket()
  }

  stop(): void {
    this.stopped = true
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.socket) {
      const ws = this.socket
      this.socket = null
      ws.onopen = null
      ws.onmessage = null
      ws.onclose = null
      ws.onerror = null
      try {
        ws.close()
      } catch {
        // Ignore close errors
      }
    }
    const store = useAppStore.getState()
    if (!store.isFakeMode) {
      store.setConnectionStatus('disconnected')
    }
  }

  private openSocket(): void {
    if (this.stopped || useAppStore.getState().isFakeMode) {
      return
    }
    useAppStore.getState().setConnectionStatus('connecting')

    let ws: SocketLike
    try {
      ws = this.createWs(this.url)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.socket = ws

    ws.onopen = () => {
      if (this.stopped) return
      this.reconnectAttempt = 0
      useAppStore.getState().setConnectionStatus('connected')
    }

    ws.onmessage = (msgEvent) => {
      if (this.stopped) return
      const validEvent = parseAndValidateEventMessage(msgEvent.data)
      if (!validEvent) {
        return
      }
      useAppStore.getState().applyEvent(validEvent)
      if (validEvent.type === 'finding') {
        void this.onFindingEvent(validEvent.payload.finding_id)
      }
    }

    ws.onerror = () => {
      // onclose handles reconnect scheduling
    }

    ws.onclose = () => {
      this.socket = null
      if (this.stopped || useAppStore.getState().isFakeMode) {
        return
      }
      useAppStore.getState().setConnectionStatus('disconnected')
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.reconnectTimer !== null || useAppStore.getState().isFakeMode) {
      return
    }
    const delay = computeBackoffDelay(this.reconnectAttempt, this.initialDelayMs, this.maxDelayMs)
    this.reconnectAttempt += 1
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.openSocket()
    }, delay)
  }
}
