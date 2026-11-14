/// <reference types="vite/client" />
import qwenMapRaw from '../../../tests/fixtures/modelmap_qwen25_05b.json?raw'
import pythiaMapRaw from '../../../tests/fixtures/modelmap_pythia70m.json?raw'
import pythiaSeriesRaw from '../../../tests/fixtures/series_pythia70m.json?raw'
import sycoEventsRaw from '../../../tests/fixtures/events_sycophancy_run.jsonl?raw'

import {
  isValidEvent,
  type Finding,
  type ModelMap,
  type Plan,
  type Series,
  type TypedEvent,
} from '../contracts'
import { useAppStore } from '../store'

export type FakeFixturePreset = 'qwen25_sycophancy' | 'pythia70m_series'

export function getFixtureQwenModelMap(): ModelMap {
  return JSON.parse(qwenMapRaw) as ModelMap
}

export function getFixturePythiaModelMap(): ModelMap {
  return JSON.parse(pythiaMapRaw) as ModelMap
}

export function getFixturePythiaSeries(): Series {
  return JSON.parse(pythiaSeriesRaw) as Series
}

export function getFixtureSycophancyEvents(): TypedEvent[] {
  const lines = sycoEventsRaw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  const events: TypedEvent[] = []
  for (const line of lines) {
    const parsed: unknown = JSON.parse(line)
    if (isValidEvent(parsed)) {
      events.push(parsed)
    }
  }
  return events
}

export function createPlaceholderSycophancyPlan(): Plan {
  return {
    plan_id: 'plan_fixture_syco_001',
    experiment_id: 'exp_fixture_syco_001',
    version: 1,
    check: 'sycophancy',
    model_id: 'hf:Qwen/Qwen2.5-0.5B-Instruct@main',
    reference_model_id: 'hf:Qwen/Qwen2.5-0.5B-Instruct@base',
    series_id: null,
    checkpoints: null,
    datasets: ['syco/are_you_sure', 'syco/pairs', 'syco/paraphrase', 'syco/es', 'syco/multiturn'],
    layers: 'all',
    controls: ['shuffled_labels', 'random_direction', 'held_out'],
    shift_tests: ['paraphrase', 'multilingual', 'multi_turn'],
    user_context: { domain: 'customer_support', note: 'placeholder fixture replay' },
    estimate_seconds: 180,
    coverage:
      'Placeholder fixture plan covering layers 0 to 23 on synthetic sycophancy pushback items (English, paraphrased, Spanish, and multi-turn variants, seed 0).',
    not_covered:
      'Placeholder fixture replay only. Does not evaluate real model weights, domain-specific medical or legal advice, or open-ended multi-session conversations.',
    maturity: 'established',
    papers: [
      'Sharma et al. (2023) Towards Understanding Sycophancy in Language Models',
      'Rimsky et al. (2024) Steering Llama 2 via Contrastive Activation Addition',
    ],
    approved: false,
    is_placeholder: true,
  }
}

export function createPlaceholderSycophancyFinding(): Finding {
  return {
    finding_id: 'finding_fixture_syco_001',
    run_id: 'run_fixture_syco_001',
    check: 'sycophancy',
    claim:
      'Placeholder fixture result: linear probes in layers 14 to 17 separate sycophantic from honest continuations in recorded fixture data (+0.18 over reference at layer 16), with multi-turn shift performance weaker than single-turn.',
    strength: 'suggestive',
    maturity: 'established',
    evidence: [
      'run_fixture_syco_001:probe_accuracy@L16=0.89',
      'run_fixture_syco_001:probe_accuracy_diff_vs_reference@L16=0.18',
    ],
    controls: [
      { kind: 'shuffled_labels', value: 0.52, passed: true, notes: '20 label shuffles (fixture)' },
      { kind: 'random_direction', value: 0.04, passed: true, notes: '10 random draws (fixture)' },
      { kind: 'held_out', value: 0.89, passed: true, notes: '25% held-out split (fixture)' },
    ],
    shift_tests: [
      { kind: 'paraphrase', value: 0.84, passed: true, notes: 'Held on reworded items (fixture)' },
      { kind: 'multilingual', value: 0.78, passed: true, notes: 'Held on Spanish items (fixture)' },
      {
        kind: 'multi_turn',
        value: 0.56,
        passed: false,
        notes: 'Below 50% relative effect threshold on 3-turn conversations (fixture)',
      },
    ],
    coverage:
      'Placeholder fixture replay over layers 0 to 23, 200 English pushback items, 50 paraphrased, 50 Spanish, and 50 multi-turn items, seed 0.',
    not_shown:
      'Recorded fixture data only — no real model weights or real evaluation datasets were run. Does not establish causal use of the layer 16 direction without verified live steering.',
    verified: false,
    is_placeholder: true,
    reviewer_verdict: {
      agrees: true,
      notes:
        'Placeholder fixture verdict: claim wording notes the failed multi-turn shift test and explicitly marks that this is unverified fixture data.',
      corrected_claim: null,
      lower_strength_to: null,
    },
  }
}

export function buildPythiaPlaceholderPaintEvents(series: Series): TypedEvent[] {
  const events: TypedEvent[] = []
  const baseCurves: Record<string, number[]> = {
    step1000: [0.51, 0.52, 0.53, 0.52, 0.51, 0.51],
    step8000: [0.53, 0.56, 0.58, 0.57, 0.55, 0.54],
    step16000: [0.54, 0.59, 0.64, 0.63, 0.6, 0.58],
    step32000: [0.55, 0.62, 0.69, 0.71, 0.67, 0.64],
    step64000: [0.56, 0.65, 0.74, 0.77, 0.73, 0.69],
    step143000: [0.57, 0.67, 0.78, 0.82, 0.79, 0.74],
  }
  for (const item of series.items) {
    const curve = baseCurves[item.checkpoint_id] ?? [0.5, 0.55, 0.6, 0.6, 0.58, 0.55]
    const values: Record<string, number> = {}
    curve.forEach((v, idx) => {
      values[`L${idx}`] = v
    })
    events.push({
      type: 'timeline',
      payload: { checkpoint_id: item.checkpoint_id },
    })
    events.push({
      type: 'paint',
      payload: {
        metric: 'harm_recognition_probe_accuracy (placeholder)',
        scale: 'sequential',
        checkpoint_id: item.checkpoint_id,
        run_id: 'run_fixture_pythia_001',
        values,
      },
    })
  }
  events.push({
    type: 'focus',
    payload: {
      component_ids: ['L3'],
      reason: 'Placeholder fixture peak layer at step 143,000 (0.82)',
    },
  })
  return events
}

export interface FakeEventPlayerOptions {
  baseIntervalMs?: number
  speedMultiplier?: number
  onComplete?: () => void
}

export class FakeEventPlayer {
  private timers: Array<ReturnType<typeof setTimeout>> = []
  private speedMultiplier: number
  private readonly baseIntervalMs: number
  private running = false

  constructor(options?: FakeEventPlayerOptions) {
    this.baseIntervalMs = options?.baseIntervalMs ?? 120
    this.speedMultiplier = options?.speedMultiplier ?? 1
  }

  setSpeed(multiplier: number): void {
    this.speedMultiplier = Math.max(0.25, Math.min(multiplier, 20))
  }

  getSpeed(): number {
    return this.speedMultiplier
  }

  isRunning(): boolean {
    return this.running
  }

  stop(): void {
    for (const timer of this.timers) {
      clearTimeout(timer)
    }
    this.timers = []
    this.running = false
  }

  loadPresetModel(preset: FakeFixturePreset = 'qwen25_sycophancy'): void {
    const store = useAppStore.getState()
    store.setFakeMode(true)
    if (preset === 'pythia70m_series') {
      const pythiaMap = getFixturePythiaModelMap()
      const pythiaSeries = getFixturePythiaSeries()
      store.setModelMap(pythiaMap)
      store.setSeries(pythiaSeries)
    } else {
      const qwenMap = getFixtureQwenModelMap()
      store.setSeries(null)
      store.setModelMap(qwenMap)
    }
  }

  seedInitialMockConversation(): void {
    const store = useAppStore.getState()
    const plan = createPlaceholderSycophancyPlan()
    const planId = store.upsertPlan(plan)
    store.setChatMessages([
      {
        id: 'fake_welcome_sys',
        role: 'system',
        kind: 'text',
        text: 'Fixture replay mode is active. All messages, plans, measurements, and findings shown here are recorded placeholder test data (`is_placeholder: true, verified: false`), not real model evaluations.',
        isPlaceholder: true,
        timestamp: Date.now() - 3000,
      },
      {
        id: 'fake_user_1',
        role: 'user',
        kind: 'text',
        text: 'Did my customer-support fine-tune make Qwen2.5-0.5B-Instruct more sycophantic?',
        isPlaceholder: true,
        timestamp: Date.now() - 2000,
      },
      {
        id: 'fake_assistant_intro',
        role: 'assistant',
        kind: 'text',
        text: 'Here is a draft placeholder plan for checking sycophancy on your customer-support fine-tune against its base model. Review the controls, shift tests, and coverage limits below, then reply "yes, go ahead" or click Approve plan to replay the recorded fixture stream.',
        isPlaceholder: true,
        timestamp: Date.now() - 1500,
      },
      {
        id: 'fake_plan_msg_1',
        role: 'assistant',
        kind: 'plan',
        text: 'Draft sycophancy check plan (placeholder fixture)',
        planId,
        plan: { ...plan, plan_id: planId },
        isPlaceholder: true,
        timestamp: Date.now() - 1000,
      },
    ])
  }

  replayStream(preset: FakeFixturePreset = 'qwen25_sycophancy', onComplete?: () => void): void {
    this.stop()
    const store = useAppStore.getState()
    store.setFakeMode(true)
    if (!store.modelMap || (preset === 'pythia70m_series' && !store.series) || (preset === 'qwen25_sycophancy' && store.series)) {
      this.loadPresetModel(preset)
    }

    const events =
      preset === 'pythia70m_series'
        ? buildPythiaPlaceholderPaintEvents(getFixturePythiaSeries())
        : getFixtureSycophancyEvents()

    const stepDelay = Math.max(10, Math.round(this.baseIntervalMs / this.speedMultiplier))
    this.running = true

    events.forEach((evt, index) => {
      const timer = setTimeout(() => {
        const currentStore = useAppStore.getState()
        currentStore.applyEvent(evt)

        if (evt.type === 'progress') {
          currentStore.upsertChatToolActivity(evt.payload.job_id, evt.payload.message, false)
        } else if (evt.type === 'job_status' && evt.payload.status === 'completed') {
          currentStore.upsertChatToolActivity(
            evt.payload.job_id,
            evt.payload.message ?? 'Fixture run completed',
            true,
          )
        } else if (evt.type === 'finding') {
          const placeholderFinding = createPlaceholderSycophancyFinding()
          currentStore.upsertFinding(placeholderFinding)
          const alreadyExists = currentStore.chatMessages.some(
            (m) => m.kind === 'finding' && m.findingId === placeholderFinding.finding_id,
          )
          if (!alreadyExists) {
            currentStore.addChatMessage({
              id: `fake_finding_${placeholderFinding.finding_id}`,
              role: 'assistant',
              kind: 'finding',
              text: placeholderFinding.claim,
              findingId: placeholderFinding.finding_id,
              finding: placeholderFinding,
              isPlaceholder: true,
              timestamp: Date.now(),
            })
          }
        }

        if (index === events.length - 1) {
          this.running = false
          onComplete?.()
        }
      }, index * stepDelay)
      this.timers.push(timer)
    })
  }

  replaySynchronously(preset: FakeFixturePreset = 'qwen25_sycophancy'): void {
    this.stop()
    this.loadPresetModel(preset)
    const events =
      preset === 'pythia70m_series'
        ? buildPythiaPlaceholderPaintEvents(getFixturePythiaSeries())
        : getFixtureSycophancyEvents()

    for (const evt of events) {
      const store = useAppStore.getState()
      store.applyEvent(evt)
      if (evt.type === 'finding') {
        const placeholderFinding = createPlaceholderSycophancyFinding()
        store.upsertFinding(placeholderFinding)
      }
    }
  }
}

export const sharedFakePlayer = new FakeEventPlayer()
