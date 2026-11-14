import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  FakeEventPlayer,
  createPlaceholderSycophancyFinding,
  createPlaceholderSycophancyPlan,
  getFixtureSycophancyEvents,
} from '../dev/fakeEvents'
import {
  EventsClient,
  computeBackoffDelay,
  parseAndValidateEventMessage,
  type SocketLike,
} from '../events'
import {
  HONESTY_BANNER_TEXT,
  SINGLE_CHECKPOINT_KEY,
  formatComponentLabel,
  hasPlaceholderOrUnverifiedData,
  useAppStore,
} from '../store'

describe('W1 App Store, Events Client, and Honesty Guards', () => {
  beforeEach(() => {
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('dispatches paint events under __single__ when no checkpoint_id is set, and under checkpoint_id when set', () => {
    const store = useAppStore.getState()
    store.applyEvent({
      type: 'paint',
      payload: {
        metric: 'probe_accuracy',
        scale: 'sequential',
        values: { L0: 0.52, L16: 0.89 },
      },
    })

    const singlePaint = useAppStore.getState().paintByCheckpoint[SINGLE_CHECKPOINT_KEY]
    expect(singlePaint).toBeDefined()
    expect(singlePaint.metric).toBe('probe_accuracy')
    expect(singlePaint.scale).toBe('sequential')
    expect(singlePaint.values.L16).toBe(0.89)

    store.applyEvent({
      type: 'paint',
      payload: {
        metric: 'harm_recognition',
        scale: 'sequential',
        checkpoint_id: 'step8000',
        values: { L2: 0.58 },
      },
    })
    expect(useAppStore.getState().paintByCheckpoint.step8000?.values.L2).toBe(0.58)
  })

  it('handles focus, timeline, progress, question, and job_status events cleanly', () => {
    const store = useAppStore.getState()

    store.applyEvent({
      type: 'focus',
      payload: {
        component_ids: ['L16', 'L16.attn', 'L16.mlp'],
        reason: 'Peak sycophancy diff at layer 16',
      },
    })
    expect(useAppStore.getState().focusedComponentIds).toEqual(['L16', 'L16.attn', 'L16.mlp'])
    expect(useAppStore.getState().focusReason).toBe('Peak sycophancy diff at layer 16')
    expect(useAppStore.getState().expandedLayerIds).toContain('L16')

    store.applyEvent({
      type: 'timeline',
      payload: { checkpoint_id: 'step32000' },
    })
    expect(useAppStore.getState().currentCheckpointId).toBe('step32000')

    store.applyEvent({
      type: 'progress',
      payload: { job_id: 'job_1', step: 5, total: 20, message: 'Layer 5 probe' },
    })
    expect(useAppStore.getState().jobs.job_1).toEqual({
      status: 'running',
      step: 5,
      total: 20,
      message: 'Layer 5 probe',
    })

    store.applyEvent({
      type: 'question',
      payload: { job_id: 'job_1', text: 'Effect is weak at layer 5; continue?' },
    })
    expect(useAppStore.getState().jobs.job_1.status).toBe('needs_decision')
    expect(useAppStore.getState().jobs.job_1.question).toBe('Effect is weak at layer 5; continue?')
    expect(useAppStore.getState().chatMessages.some((m) => m.kind === 'question')).toBe(true)

    store.applyEvent({
      type: 'job_status',
      payload: { job_id: 'job_1', status: 'completed', message: 'Done' },
    })
    expect(useAppStore.getState().jobs.job_1.status).toBe('completed')
    expect(useAppStore.getState().jobs.job_1.question).toBeUndefined()
  })

  it('formats component labels plainly to match agent references', () => {
    expect(formatComponentLabel('embed')).toBe('Embedding')
    expect(formatComponentLabel('unembed')).toBe('Unembedding')
    expect(formatComponentLabel('L14')).toBe('Layer 14')
    expect(formatComponentLabel('L14.attn')).toBe('Layer 14 attention')
    expect(formatComponentLabel('L14.mlp')).toBe('Layer 14 MLP')
    expect(formatComponentLabel('L14.attn.h3')).toBe('Layer 14 attention head 3')
  })

  it('enforces placeholder / unverified honesty banner detection', () => {
    expect(HONESTY_BANNER_TEXT).toBe(
      'Placeholder / unverified test data — not real model evaluation results',
    )

    expect(
      hasPlaceholderOrUnverifiedData({
        isFakeMode: false,
        plans: {},
        findings: [],
        chatMessages: [],
      }),
    ).toBe(false)

    expect(
      hasPlaceholderOrUnverifiedData({
        isFakeMode: true,
        plans: {},
        findings: [],
        chatMessages: [],
      }),
    ).toBe(true)

    const placeholderPlan = createPlaceholderSycophancyPlan()
    expect(placeholderPlan.is_placeholder).toBe(true)
    expect(
      hasPlaceholderOrUnverifiedData({
        isFakeMode: false,
        plans: { p1: placeholderPlan },
        findings: [],
        chatMessages: [],
      }),
    ).toBe(true)

    const placeholderFinding = createPlaceholderSycophancyFinding()
    expect(placeholderFinding.is_placeholder).toBe(true)
    expect(placeholderFinding.verified).toBe(false)
    expect(
      hasPlaceholderOrUnverifiedData({
        isFakeMode: false,
        plans: {},
        findings: [placeholderFinding],
        chatMessages: [],
      }),
    ).toBe(true)

    const verifiedFinding = {
      ...placeholderFinding,
      is_placeholder: false,
      verified: true,
    }
    expect(
      hasPlaceholderOrUnverifiedData({
        isFakeMode: false,
        plans: {},
        findings: [verifiedFinding],
        chatMessages: [],
      }),
    ).toBe(false)
  })

  it('validates incoming WebSocket events and reconnects with exponential backoff', () => {
    vi.useFakeTimers()
    try {
      expect(computeBackoffDelay(0, 500, 15000)).toBe(500)
      expect(computeBackoffDelay(1, 500, 15000)).toBe(1000)
      expect(computeBackoffDelay(2, 500, 15000)).toBe(2000)
      expect(computeBackoffDelay(10, 500, 15000)).toBe(15000)

      expect(parseAndValidateEventMessage('not json')).toBeNull()
      expect(parseAndValidateEventMessage(JSON.stringify({ type: 'focus', payload: {} }))).toBeNull()

      const sockets: SocketLike[] = []
      const findingIds: string[] = []

      const client = new EventsClient({
        url: 'ws://127.0.0.1:8000/ws/events',
        initialReconnectDelayMs: 200,
        maxReconnectDelayMs: 1000,
        createWebSocket: () => {
          const sock: SocketLike = {
            onopen: null,
            onmessage: null,
            onclose: null,
            onerror: null,
            close: () => {},
          }
          sockets.push(sock)
          return sock
        },
        onFindingEvent: (id) => {
          findingIds.push(id)
        },
      })

      client.start()
      expect(sockets).toHaveLength(1)
      expect(useAppStore.getState().connectionStatus).toBe('connecting')

      sockets[0].onopen?.()
      expect(useAppStore.getState().connectionStatus).toBe('connected')

      sockets[0].onmessage?.({ data: JSON.stringify({ type: 'unknown', payload: {} }) })
      sockets[0].onmessage?.({
        data: JSON.stringify({
          type: 'focus',
          payload: { component_ids: ['L4'], reason: 'Testing focus' },
        }),
      })
      expect(useAppStore.getState().focusedComponentIds).toEqual(['L4'])

      sockets[0].onmessage?.({
        data: JSON.stringify({
          type: 'finding',
          payload: { finding_id: 'finding_123' },
        }),
      })
      expect(findingIds).toEqual(['finding_123'])

      // Simulate unexpected close -> reconnect after 200ms
      sockets[0].onclose?.()
      expect(useAppStore.getState().connectionStatus).toBe('disconnected')
      vi.advanceTimersByTime(200)
      expect(sockets).toHaveLength(2)

      // Second close -> reconnect after 400ms
      sockets[1].onclose?.()
      vi.advanceTimersByTime(399)
      expect(sockets).toHaveLength(2)
      vi.advanceTimersByTime(1)
      expect(sockets).toHaveLength(3)

      client.stop()
    } finally {
      vi.useRealTimers()
    }
  })

  it('replays fixture sycophancy stream and marks all fixture findings as placeholder and unverified', () => {
    const events = getFixtureSycophancyEvents()
    expect(events.length).toBeGreaterThanOrEqual(60)

    const player = new FakeEventPlayer()
    player.replaySynchronously('qwen25_sycophancy')

    const state = useAppStore.getState()
    expect(state.isFakeMode).toBe(true)
    expect(state.connectionStatus).toBe('fake')
    expect(state.modelMap?.n_layers).toBe(24)
    expect(state.jobs.job_fixture_syco_001.status).toBe('completed')
    expect(state.paintByCheckpoint[SINGLE_CHECKPOINT_KEY].scale).toBe('diverging')
    expect(state.paintByCheckpoint[SINGLE_CHECKPOINT_KEY].values.L16).toBe(0.18)
    expect(state.findings).toHaveLength(1)
    expect(state.findings[0].is_placeholder).toBe(true)
    expect(state.findings[0].verified).toBe(false)
  })
})
