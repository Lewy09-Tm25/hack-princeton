import type { Finding, Plan } from '../contracts'
import { approvePlan } from '../api'
import {
  createPlaceholderSycophancyPlan,
  sharedFakePlayer,
} from '../dev/fakeEvents'
import {
  formatComponentLabel,
  useAppStore,
  type ChatMessage,
} from '../store'

const NEGATIVE_OR_REVISION_PATTERNS =
  /\b(no|not\b|don't|do not|wait|hold|stop|cancel|skip|remove|add|change|revise|instead|modify|without|except)\b/i

const AFFIRMATIVE_APPROVAL_PATTERNS =
  /^\s*(yes|yep|yeah|sure|ok|okay|approve|approved|go ahead|run it|run the plan|proceed|please proceed|looks good|start|go)\b/i

export function isPlainTextApproval(input: string): boolean {
  const trimmed = input.trim()
  if (!trimmed || trimmed.length > 120) {
    return false
  }
  if (NEGATIVE_OR_REVISION_PATTERNS.test(trimmed)) {
    return false
  }
  return AFFIRMATIVE_APPROVAL_PATTERNS.test(trimmed)
}

export function findLatestUnapprovedPlan(
  plans: Record<string, Plan>,
  chatMessages: ChatMessage[],
): Plan | null {
  for (let i = chatMessages.length - 1; i >= 0; i--) {
    const msg = chatMessages[i]
    if (msg.kind === 'plan') {
      const plan = (msg.planId ? plans[msg.planId] : undefined) ?? msg.plan
      if (plan && !plan.approved) {
        return plan
      }
    }
  }
  const allPlans = Object.values(plans)
  for (let i = allPlans.length - 1; i >= 0; i--) {
    if (!allPlans[i].approved) {
      return allPlans[i]
    }
  }
  return null
}

export function formatDurationEstimate(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return 'Under 1 minute'
  }
  if (seconds < 60) {
    return `About ${Math.round(seconds)} seconds`
  }
  const mins = Math.round(seconds / 60)
  return `About ${mins} ${mins === 1 ? 'minute' : 'minutes'} (${Math.round(seconds)} s)`
}

export function describeCheckMeasurement(checkName: string): string {
  switch (checkName) {
    case 'sycophancy':
      return 'Rate of abandoning a correct answer after user pushback, per-layer linear probes on honest vs sycophantic contrast pairs, and difference-of-means direction compared against the reference model.'
    case 'refusal':
      return 'Per-layer difference-of-means refusal direction separating harmful and harmless instructions, projection strength and cosine similarity across stages, and behavioral refusal rate.'
    case 'backdoor':
      return 'Linear probes built from generic intent contrast pairs comparing triggered and untriggered inputs against a clean reference model (detection only, not removal).'
    default:
      return `Internal activation measurements and behavioral comparisons for the ${checkName} risk check.`
  }
}

export interface HandleUserChatTurnOptions {
  text: string
  sendOverSocket?: (payload: string) => boolean
  approvePlanFn?: (planId: string) => Promise<{ plan_id: string; approved: boolean }>
  triggerFakeReplay?: boolean
}

export async function submitChatTurn(options: HandleUserChatTurnOptions): Promise<void> {
  const trimmed = options.text.trim()
  if (!trimmed) {
    return
  }

  const store = useAppStore.getState()
  const contextComponentId = store.selectedComponentId
  const approveFn = options.approvePlanFn ?? approvePlan

  const userMsg: ChatMessage = {
    id: `user_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    role: 'user',
    kind: 'text',
    text: trimmed,
    contextComponentId,
    isPlaceholder: store.isFakeMode,
    timestamp: Date.now(),
  }
  store.addChatMessage(userMsg)
  store.setDraftInput('')

  const pendingPlan = findLatestUnapprovedPlan(store.plans, store.chatMessages)
  const isApproval = pendingPlan !== null && isPlainTextApproval(trimmed)

  if (store.isFakeMode) {
    if (isApproval && pendingPlan?.plan_id) {
      store.markPlanApproved(pendingPlan.plan_id)
      store.addChatMessage({
        id: `fake_ack_${Date.now()}`,
        role: 'assistant',
        kind: 'text',
        text: 'Plan marked approved in placeholder fixture mode. Replaying recorded sycophancy fixture events now (unverified placeholder data).',
        isPlaceholder: true,
        timestamp: Date.now(),
      })
      if (options.triggerFakeReplay !== false) {
        sharedFakePlayer.replayStream('qwen25_sycophancy')
      }
      return
    }

    if (contextComponentId) {
      store.applyEvent({
        type: 'focus',
        payload: {
          component_ids: [contextComponentId],
          reason: `Focused ${formatComponentLabel(contextComponentId)} from chat context`,
        },
      })
      store.addChatMessage({
        id: `fake_comp_reply_${Date.now()}`,
        role: 'assistant',
        kind: 'text',
        text: `Placeholder note for ${formatComponentLabel(contextComponentId)} (${contextComponentId}): in fixture mode no live model activations are computed. Connect the engine server to run probes or steering at ${contextComponentId} on real weights.`,
        isPlaceholder: true,
        timestamp: Date.now(),
      })
      return
    }

    const revisedPlan = createPlaceholderSycophancyPlan()
    const nextVersion = (pendingPlan?.version ?? 1) + (pendingPlan ? 1 : 0)
    const planWithNotes: Plan = {
      ...revisedPlan,
      plan_id: `plan_fixture_syco_v${nextVersion}`,
      version: nextVersion,
      user_context: { ...revisedPlan.user_context, latest_user_request: trimmed },
      approved: false,
      is_placeholder: true,
    }
    const storedId = store.upsertPlan(planWithNotes)
    store.addChatMessage({
      id: `fake_plan_reply_${Date.now()}`,
      role: 'assistant',
      kind: 'plan',
      text: 'Updated placeholder sycophancy plan',
      planId: storedId,
      plan: { ...planWithNotes, plan_id: storedId },
      isPlaceholder: true,
      timestamp: Date.now(),
    })
    return
  }

  // Live mode: if user approved a pending plan in plain text, call POST /api/plans/{id}/approve first
  if (isApproval && pendingPlan?.plan_id) {
    try {
      await approveFn(pendingPlan.plan_id)
      useAppStore.getState().markPlanApproved(pendingPlan.plan_id)
    } catch (err) {
      const errText = err instanceof Error ? err.message : 'Failed to record plan approval'
      useAppStore.getState().addChatMessage({
        id: `sys_approve_err_${Date.now()}`,
        role: 'system',
        kind: 'text',
        text: `Could not record plan approval with the engine: ${errText}`,
        timestamp: Date.now(),
      })
      return
    }
  }

  const wirePayload = JSON.stringify({
    type: 'user_message',
    text: trimmed,
    context: contextComponentId ? { component_id: contextComponentId } : null,
  })

  const sent = options.sendOverSocket ? options.sendOverSocket(wirePayload) : false
  if (!sent) {
    useAppStore.getState().addChatMessage({
      id: `sys_offline_${Date.now()}`,
      role: 'system',
      kind: 'text',
      text: 'Chat WebSocket (/ws/chat) is not connected. Start the engine server (`tool serve`) or switch to placeholder fixture replay mode to preview the interface.',
      timestamp: Date.now(),
    })
  }
}

export async function approvePlanFromButton(
  planId: string,
  options?: {
    sendOverSocket?: (payload: string) => boolean
    approvePlanFn?: (planId: string) => Promise<{ plan_id: string; approved: boolean }>
    triggerFakeReplay?: boolean
  },
): Promise<void> {
  const store = useAppStore.getState()
  if (!store.plans[planId]) {
    return
  }
  await submitChatTurn({
    text: 'yes, go ahead',
    sendOverSocket: options?.sendOverSocket,
    approvePlanFn: options?.approvePlanFn,
    triggerFakeReplay: options?.triggerFakeReplay,
  })
}

export function applyIncomingChatFrame(
  rawData: unknown,
  streamingMsgIdRef: { current: string | null },
): void {
  if (typeof rawData !== 'string') {
    return
  }
  let parsed: Record<string, unknown>
  try {
    const val: unknown = JSON.parse(rawData)
    if (typeof val !== 'object' || val === null) return
    parsed = val as Record<string, unknown>
  } catch {
    return
  }

  const store = useAppStore.getState()
  const frameType = typeof parsed.type === 'string' ? parsed.type : ''

  if (frameType === 'text_delta' || frameType === 'delta' || frameType === 'chunk') {
    const delta = typeof parsed.text === 'string' ? parsed.text : typeof parsed.delta === 'string' ? parsed.delta : ''
    if (!delta) return
    if (!streamingMsgIdRef.current) {
      const id = `assistant_stream_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
      streamingMsgIdRef.current = id
      store.addChatMessage({
        id,
        role: 'assistant',
        kind: 'text',
        text: delta,
        streaming: true,
        timestamp: Date.now(),
      })
    } else {
      store.appendChatStreamText(streamingMsgIdRef.current, delta)
    }
    return
  }

  if (frameType === 'text' || frameType === 'message') {
    const content = typeof parsed.text === 'string' ? parsed.text : ''
    if (streamingMsgIdRef.current) {
      store.updateChatMessage(streamingMsgIdRef.current, {
        text: content || undefined,
        streaming: false,
      })
      streamingMsgIdRef.current = null
    } else if (content) {
      store.addChatMessage({
        id: `assistant_msg_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        role: 'assistant',
        kind: 'text',
        text: content,
        streaming: false,
        timestamp: Date.now(),
      })
    }
    return
  }

  if (frameType === 'done' || frameType === 'turn_end' || frameType === 'end') {
    if (streamingMsgIdRef.current) {
      store.updateChatMessage(streamingMsgIdRef.current, { streaming: false })
      streamingMsgIdRef.current = null
    }
    return
  }

  if (frameType === 'plan') {
    const planObj = (parsed.plan ?? parsed.payload) as Plan | undefined
    if (planObj && typeof planObj.check === 'string') {
      streamingMsgIdRef.current = null
      const id = store.upsertPlan(planObj)
      store.addChatMessage({
        id: `plan_msg_${id}_v${planObj.version ?? 1}`,
        role: 'assistant',
        kind: 'plan',
        text: `Draft ${planObj.check} plan`,
        planId: id,
        plan: { ...planObj, plan_id: id },
        isPlaceholder: Boolean(planObj.is_placeholder),
        timestamp: Date.now(),
      })
    }
    return
  }

  if (frameType === 'question') {
    const payload = (parsed.payload as Record<string, unknown> | undefined) ?? parsed
    const jobId = typeof payload.job_id === 'string' ? payload.job_id : 'active_job'
    const text = typeof payload.text === 'string' ? payload.text : ''
    if (text) {
      streamingMsgIdRef.current = null
      store.applyEvent({ type: 'question', payload: { job_id: jobId, text } })
    }
    return
  }

  if (frameType === 'finding') {
    const findingObj = (parsed.finding ?? parsed.payload) as Finding | undefined
    if (findingObj && typeof findingObj.finding_id === 'string' && typeof findingObj.claim === 'string') {
      streamingMsgIdRef.current = null
      store.upsertFinding(findingObj)
      store.addChatMessage({
        id: `finding_card_${findingObj.finding_id}`,
        role: 'assistant',
        kind: 'finding',
        text: findingObj.claim,
        findingId: findingObj.finding_id,
        finding: findingObj,
        isPlaceholder: Boolean(findingObj.is_placeholder),
        timestamp: Date.now(),
      })
    }
    return
  }

  if (frameType === 'tool_activity' || frameType === 'tool_use' || frameType === 'tool_result') {
    const summary =
      typeof parsed.summary === 'string'
        ? parsed.summary
        : typeof parsed.message === 'string'
          ? parsed.message
          : typeof parsed.name === 'string'
            ? `Running tool: ${parsed.name}`
            : 'Running engine step'
    const key = typeof parsed.job_id === 'string' ? parsed.job_id : 'agent_tools'
    const done = frameType === 'tool_result' || Boolean(parsed.done)
    store.upsertChatToolActivity(key, summary, done)
    return
  }

  if (frameType === 'error') {
    const errMsg = typeof parsed.message === 'string' ? parsed.message : 'Agent chat error'
    streamingMsgIdRef.current = null
    store.addChatMessage({
      id: `chat_err_${Date.now()}`,
      role: 'system',
      kind: 'text',
      text: errMsg,
      timestamp: Date.now(),
    })
  }
}
