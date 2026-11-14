import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  applyIncomingChatFrame,
  approvePlanFromButton,
  describeCheckMeasurement,
  findLatestUnapprovedPlan,
  formatDurationEstimate,
  isPlainTextApproval,
  submitChatTurn,
} from '../chat/chatLogic'
import type { Finding, Plan } from '../contracts'
import { FakeEventPlayer } from '../dev/fakeEvents'
import { hasPlaceholderOrUnverifiedData, useAppStore } from '../store'

function makeLivePlan(planId = 'plan_live_001'): Plan {
  return {
    plan_id: planId,
    experiment_id: 'exp_live_001',
    version: 1,
    check: 'sycophancy',
    model_id: 'hf:Qwen/Qwen2.5-0.5B-Instruct@main',
    reference_model_id: 'hf:Qwen/Qwen2.5-0.5B-Instruct@base',
    series_id: null,
    checkpoints: null,
    datasets: ['syco/are_you_sure', 'syco/pairs'],
    layers: 'all',
    controls: ['shuffled_labels', 'random_direction', 'held_out'],
    shift_tests: ['paraphrase', 'multilingual', 'multi_turn'],
    user_context: { domain: 'general' },
    estimate_seconds: 240,
    coverage: 'Layers 0-23, 200 English prompts, seed 0.',
    not_covered: 'Does not cover multi-session memory.',
    maturity: 'established',
    papers: ['Sharma et al. 2023', 'Rimsky et al. 2024'],
    approved: false,
    is_placeholder: false,
  }
}

describe('W2 Chat Panel logic, plan approval, streaming frames, and evidence drawer', () => {
  beforeEach(() => {
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('recognizes plain-language approval replies and rejects revision or refusal replies', () => {
    expect(isPlainTextApproval('yes, go ahead')).toBe(true)
    expect(isPlainTextApproval('Yes')).toBe(true)
    expect(isPlainTextApproval('go ahead and run it')).toBe(true)
    expect(isPlainTextApproval('approve plan')).toBe(true)
    expect(isPlainTextApproval('looks good, proceed')).toBe(true)

    expect(isPlainTextApproval('no')).toBe(false)
    expect(isPlainTextApproval('not yet, wait')).toBe(false)
    expect(isPlainTextApproval('skip the multilingual tests, add medical cases')).toBe(false)
    expect(isPlainTextApproval('yes, but remove layer 0')).toBe(false)
    expect(isPlainTextApproval('')).toBe(false)
  })

  it('triggers POST /api/plans/{id}/approve and sends reply with component context on "yes, go ahead"', async () => {
    const store = useAppStore.getState()
    const plan = makeLivePlan('plan_syco_42')
    const id = store.upsertPlan(plan)
    store.addChatMessage({
      id: 'msg_plan_1',
      role: 'assistant',
      kind: 'plan',
      text: 'Draft sycophancy plan',
      planId: id,
      plan,
      timestamp: 1000,
    })

    // Select Layer 16 in view -> attaches removable context chip
    store.selectComponent('L16')
    expect(findLatestUnapprovedPlan(useAppStore.getState().plans, useAppStore.getState().chatMessages)?.plan_id).toBe(
      'plan_syco_42',
    )

    const approveCalls: string[] = []
    const sentPayloads: string[] = []

    await submitChatTurn({
      text: 'yes, go ahead',
      approvePlanFn: async (pid) => {
        approveCalls.push(pid)
        return { plan_id: pid, approved: true }
      },
      sendOverSocket: (payload) => {
        sentPayloads.push(payload)
        return true
      },
    })

    expect(approveCalls).toEqual(['plan_syco_42'])
    expect(useAppStore.getState().plans.plan_syco_42.approved).toBe(true)
    expect(sentPayloads).toHaveLength(1)

    const wire = JSON.parse(sentPayloads[0]) as {
      type: string
      text: string
      context: { component_id: string } | null
    }
    expect(wire.type).toBe('user_message')
    expect(wire.text).toBe('yes, go ahead')
    expect(wire.context).toEqual({ component_id: 'L16' })

    // Removing context chip clears context on subsequent turns
    useAppStore.getState().selectComponent(null)
    await submitChatTurn({
      text: 'What is the status?',
      sendOverSocket: (payload) => {
        sentPayloads.push(payload)
        return true
      },
    })
    const secondWire = JSON.parse(sentPayloads[1]) as { context: unknown }
    expect(secondWire.context).toBeNull()
  })

  it('supports clicking Approve plan button with identical approval + reply behavior', async () => {
    const store = useAppStore.getState()
    const plan = makeLivePlan('plan_btn_7')
    store.upsertPlan(plan)
    store.addChatMessage({
      id: 'msg_plan_btn',
      role: 'assistant',
      kind: 'plan',
      text: 'Draft plan',
      planId: 'plan_btn_7',
      plan,
      timestamp: 1000,
    })

    const approveFn = vi.fn(async (pid: string) => ({ plan_id: pid, approved: true }))
    const sendSocket = vi.fn(() => true)

    await approvePlanFromButton('plan_btn_7', {
      approvePlanFn: approveFn,
      sendOverSocket: sendSocket,
    })

    expect(approveFn).toHaveBeenCalledWith('plan_btn_7')
    expect(useAppStore.getState().plans.plan_btn_7.approved).toBe(true)
    expect(sendSocket).toHaveBeenCalledTimes(1)
  })

  it('fills draftInput without auto-submitting when suggestion text is set', () => {
    const store = useAppStore.getState()
    store.setDraftInput('Did my fine-tune make the model more sycophantic?')
    expect(useAppStore.getState().draftInput).toBe(
      'Did my fine-tune make the model more sycophantic?',
    )
    expect(useAppStore.getState().chatMessages).toHaveLength(0)
  })

  it('handles streaming /ws/chat frames (deltas, plans, questions, tool activity, findings) and opens evidence drawer', () => {
    const streamingRef = { current: null as string | null }

    applyIncomingChatFrame(JSON.stringify({ type: 'text_delta', text: 'Checking ' }), streamingRef)
    applyIncomingChatFrame(JSON.stringify({ type: 'text_delta', text: 'model map.' }), streamingRef)
    applyIncomingChatFrame(JSON.stringify({ type: 'done' }), streamingRef)

    expect(useAppStore.getState().chatMessages).toHaveLength(1)
    expect(useAppStore.getState().chatMessages[0].text).toBe('Checking model map.')
    expect(useAppStore.getState().chatMessages[0].streaming).toBe(false)

    applyIncomingChatFrame(
      JSON.stringify({
        type: 'tool_activity',
        job_id: 'job_99',
        summary: 'Running sycophancy probes, layer 7 of 24',
      }),
      streamingRef,
    )
    expect(useAppStore.getState().chatMessages.some((m) => m.kind === 'tool_activity')).toBe(true)

    const unverifiedFinding: Finding = {
      finding_id: 'finding_99',
      run_id: 'run_99',
      check: 'sycophancy',
      claim: 'Probe accuracy peaked at layer 16.',
      strength: 'suggestive',
      maturity: 'established',
      evidence: ['run_99:probe_accuracy@L16'],
      controls: [{ kind: 'shuffled_labels', value: 0.51, passed: true }],
      shift_tests: [{ kind: 'multilingual', value: null, passed: null, notes: 'not run' }],
      coverage: 'Layers 0-23, English only.',
      not_shown: 'Does not test causal steering.',
      verified: false,
      is_placeholder: true,
      reviewer_verdict: null,
    }

    applyIncomingChatFrame(
      JSON.stringify({ type: 'finding', finding: unverifiedFinding }),
      streamingRef,
    )

    const stateAfterFinding = useAppStore.getState()
    expect(stateAfterFinding.findings).toHaveLength(1)
    expect(
      hasPlaceholderOrUnverifiedData({
        isFakeMode: stateAfterFinding.isFakeMode,
        plans: stateAfterFinding.plans,
        findings: stateAfterFinding.findings,
        chatMessages: stateAfterFinding.chatMessages,
      }),
    ).toBe(true)

    // Clicking "Show evidence" opens bottom drawer on charts tab
    stateAfterFinding.openEvidenceForFinding('finding_99')
    expect(useAppStore.getState().drawerTab).toBe('charts')
    expect(useAppStore.getState().activeEvidenceFindingId).toBe('finding_99')

    expect(formatDurationEstimate(180)).toContain('3 minutes')
    expect(describeCheckMeasurement('backdoor')).toContain('detection only')
  })

  it('supports mock conversation approval in ?fake=1 mode while keeping placeholder flags true', async () => {
    const player = new FakeEventPlayer()
    player.loadPresetModel('qwen25_sycophancy')
    player.seedInitialMockConversation()

    expect(useAppStore.getState().isFakeMode).toBe(true)
    expect(useAppStore.getState().plans.plan_fixture_syco_001.approved).toBe(false)
    expect(useAppStore.getState().plans.plan_fixture_syco_001.is_placeholder).toBe(true)

    await submitChatTurn({
      text: 'yes, go ahead',
      triggerFakeReplay: false,
    })

    expect(useAppStore.getState().plans.plan_fixture_syco_001.approved).toBe(true)
    expect(useAppStore.getState().plans.plan_fixture_syco_001.is_placeholder).toBe(true)
  })
})
