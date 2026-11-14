import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { fetchExamples, fetchEvidence, type SpotCheckExample } from '../api'
import { CompareModels, extractModelComparisonPairs } from '../charts/CompareModels'
import {
  ExamplesList,
  REDACTED_BADGE_TEXT,
  sanitizeExampleForDisplay,
} from '../charts/ExamplesList'
import {
  LayerCurve,
  extractLayerCurveSeries,
  triggerLayerFocus,
} from '../charts/LayerCurve'
import {
  ShiftTable,
  doesShiftOutcomeCapStrength,
  resolveShiftOutcome,
} from '../charts/ShiftTable'
import { FindingMessage } from '../chat/FindingMessage'
import type { ControlResult, ResultRecord } from '../contracts'
import {
  createPlaceholderPythiaFinding,
  createPlaceholderPythiaResults,
  createPlaceholderRedactedExamples,
  createPlaceholderSycophancyExamples,
  createPlaceholderSycophancyFinding,
  createPlaceholderSycophancyResults,
  getFixturePythiaModelMap,
  getFixturePythiaSeries,
  sharedFakePlayer,
} from '../dev/fakeEvents'
import { HONESTY_BANNER_TEXT, getActivePaint, useAppStore } from '../store'
import { Grid2D } from '../view/Grid2D'
import {
  NO_MEASUREMENTS_CHECKPOINT_TEXT,
  Timeline,
  resolveCheckpointComponentHex,
  resolveTimelineState,
  startTimelinePlaybackInterval,
  stepTimelineIndex,
} from '../view/Timeline'
import { NEUTRAL_MISSING_HEX, cividisHex } from '../view/colors'

describe('W4 Timeline slider & checkpoint painting', () => {
  const savedInitial = { ...useAppStore.getInitialState() }

  afterEach(() => {
    vi.useRealTimers()
    sharedFakePlayer.stop()
    Object.assign(useAppStore.getInitialState(), savedInitial)
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('renders all 6 Pythia-70M checkpoints, range slider, and Play/Previous/Next controls', () => {
    const series = getFixturePythiaSeries()
    const html = renderToStaticMarkup(
      <Timeline
        series={series}
        currentCheckpointId="step1000"
        paintByCheckpoint={{}}
      />,
    )

    expect(html).toContain('Pretraining checkpoints')
    expect(html).toContain('EleutherAI/pythia-70m')
    expect(html).toContain('Checkpoint timeline slider')
    expect(html).toContain('Play checkpoint timeline')
    expect(html).toContain('Previous checkpoint')
    expect(html).toContain('Next checkpoint')

    for (const item of series.items) {
      expect(html).toContain(item.label)
    }
    expect(html).toContain(NO_MEASUREMENTS_CHECKPOINT_TEXT)
  })

  it('repaints from stored per-checkpoint values on slider/tick change and renders missing values as neutral gray (#8e8e8a), never 0', () => {
    const pythiaMap = getFixturePythiaModelMap()
    const pythiaSeries = getFixturePythiaSeries()
    const store = useAppStore.getState()

    store.setModelMap(pythiaMap)
    store.setSeries(pythiaSeries)

    // Store partial measurements at step8000 (L0..L2 measured, L3..L5 missing) and full at step143000.
    store.applyEvent({
      type: 'paint',
      payload: {
        metric: 'probe_accuracy',
        scale: 'sequential',
        checkpoint_id: 'step8000',
        values: { L0: 0.53, L1: 0.56, L2: 0.58 },
      },
    })
    store.applyEvent({
      type: 'paint',
      payload: {
        metric: 'probe_accuracy',
        scale: 'sequential',
        checkpoint_id: 'step143000',
        values: { L0: 0.57, L1: 0.67, L2: 0.78, L3: 0.82, L4: 0.79, L5: 0.74 },
      },
    })

    // step1000 has NO measurements -> every layer must resolve to NEUTRAL_MISSING_HEX (#8e8e8a), never cividisHex(0)
    store.setCurrentCheckpointId('step1000')
    const stateStep1000 = useAppStore.getState()
    expect(getActivePaint(stateStep1000)).toBeNull()
    const step1000ColorL3 = resolveCheckpointComponentHex(
      stateStep1000.paintByCheckpoint,
      stateStep1000.currentCheckpointId,
      'L3',
    )
    expect(step1000ColorL3).toBe(NEUTRAL_MISSING_HEX)
    expect(step1000ColorL3).not.toBe(cividisHex(0))

    // Dragging slider / clicking tick to step8000 repaints L2 with cividis(0.58) while missing L3 stays neutral gray
    store.setCurrentCheckpointId('step8000')
    const stateStep8000 = useAppStore.getState()
    expect(resolveCheckpointComponentHex(stateStep8000.paintByCheckpoint, 'step8000', 'L2')).toBe(
      cividisHex(0.58),
    )
    expect(resolveCheckpointComponentHex(stateStep8000.paintByCheckpoint, 'step8000', 'L3')).toBe(
      NEUTRAL_MISSING_HEX,
    )

    const gridStep8000Html = renderToStaticMarkup(
      <Grid2D
        modelMap={pythiaMap}
        paint={getActivePaint(stateStep8000)}
        expandedLayerIds={[]}
        selectedComponentId={null}
        focusedComponentIds={[]}
        onToggleAndSelectLayer={() => {}}
        onSelectComponent={() => {}}
      />,
    )
    expect(gridStep8000Html).toContain('0.58')
    expect(gridStep8000Html).toContain('no measurement')
    expect(gridStep8000Html).toContain(NEUTRAL_MISSING_HEX)

    // Selecting step143000 repaints L3 immediately from stored paintByCheckpoint without re-running
    store.setCurrentCheckpointId('step143000')
    const stateStep143k = useAppStore.getState()
    expect(
      resolveCheckpointComponentHex(stateStep143k.paintByCheckpoint, 'step143000', 'L3'),
    ).toBe(cividisHex(0.82))
  })

  it('moves active checkpoint and slider thumb immediately on incoming timeline WebSocket event', () => {
    const series = getFixturePythiaSeries()
    const store = useAppStore.getState()
    store.setSeries(series)
    expect(useAppStore.getState().currentCheckpointId).toBe('step1000')

    store.applyEvent({
      type: 'timeline',
      payload: { checkpoint_id: 'step64000' },
    })
    expect(useAppStore.getState().currentCheckpointId).toBe('step64000')

    const resolved = resolveTimelineState(
      series,
      useAppStore.getState().currentCheckpointId,
      useAppStore.getState().paintByCheckpoint,
    )
    expect(resolved.activeIndex).toBe(4)
    expect(resolved.activeItem?.label).toBe('step 64,000')

    Object.assign(useAppStore.getInitialState(), useAppStore.getState())
    const html = renderToStaticMarkup(<Timeline />)
    expect(html).toContain('aria-valuenow="4"')
    expect(html).toContain('aria-valuetext="step 64,000"')
  })

  it('steps cleanly through all checkpoints via Play/Pause timer and stops at final checkpoint', () => {
    vi.useFakeTimers()
    const series = getFixturePythiaSeries()
    const store = useAppStore.getState()
    store.setSeries(series)

    expect(stepTimelineIndex(0, series.items.length, 'prev')).toBe(0)
    expect(stepTimelineIndex(0, series.items.length, 'next')).toBe(1)
    expect(stepTimelineIndex(5, series.items.length, 'next')).toBe(5)

    let stopped = false
    const cleanup = startTimelinePlaybackInterval({
      orderedItems: series.items,
      getCurrentCheckpointId: () => useAppStore.getState().currentCheckpointId,
      onSelectCheckpoint: (cid) => useAppStore.getState().setCurrentCheckpointId(cid),
      onStop: () => {
        stopped = true
      },
      stepIntervalMs: 300,
    })

    expect(useAppStore.getState().currentCheckpointId).toBe('step1000')
    vi.advanceTimersByTime(300)
    expect(useAppStore.getState().currentCheckpointId).toBe('step8000')
    vi.advanceTimersByTime(300)
    expect(useAppStore.getState().currentCheckpointId).toBe('step16000')
    vi.advanceTimersByTime(300 * 3)
    expect(useAppStore.getState().currentCheckpointId).toBe('step143000')
    expect(stopped).toBe(true)

    cleanup()
  })
})

describe('W5 Charts & Show Evidence drawer flow', () => {
  const savedInitial = { ...useAppStore.getInitialState() }

  afterEach(() => {
    vi.restoreAllMocks()
    sharedFakePlayer.stop()
    Object.assign(useAppStore.getInitialState(), savedInitial)
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('clicking Show evidence on FindingMessage opens drawer on charts tab and renders LayerCurve, CompareModels, ShiftTable, and ExamplesList', () => {
    sharedFakePlayer.replaySynchronously('qwen25_sycophancy')
    const finding = createPlaceholderSycophancyFinding()

    // Simulate clicking Show evidence via FindingMessage action
    useAppStore.getState().openEvidenceForFinding(finding.finding_id)
    expect(useAppStore.getState().drawerTab).toBe('charts')
    expect(useAppStore.getState().activeEvidenceFindingId).toBe(finding.finding_id)

    Object.assign(useAppStore.getInitialState(), useAppStore.getState())
    const appHtml = renderToStaticMarkup(<App />)

    expect(appHtml).toContain(HONESTY_BANNER_TEXT)
    expect(appHtml).toContain('Per-layer curve')
    expect(appHtml).toContain('Model vs. reference comparison')
    expect(appHtml).toContain('Distribution shift tests')
    expect(appHtml).toContain('Spot-check examples')
    expect(appHtml).toContain('Focus L16 (0.89)')
    expect(appHtml).toContain('flip_rate')
    expect(appHtml).toContain('multi_turn')
    expect(appHtml).toContain('syco/are_you_sure')
  })

  it('focuses the clicked layer in store.focusedComponentIds and selectedComponentId from LayerCurve', () => {
    const results = createPlaceholderSycophancyResults()
    const curve = extractLayerCurveSeries(results)
    expect(curve.metric).toBe('probe_accuracy')
    expect(curve.points).toHaveLength(24)
    expect(curve.points[16].layerId).toBe('L16')
    expect(curve.points[16].value).toBe(0.89)
    expect(curve.points[16].controlValue).toBe(0.52)

    triggerLayerFocus('L16', curve.metric, curve.points[16].value)
    const state = useAppStore.getState()
    expect(state.focusedComponentIds).toEqual(['L16'])
    expect(state.selectedComponentId).toBe('L16')
    expect(state.focusReason).toContain('L16')

    const html = renderToStaticMarkup(
      <LayerCurve results={results} focusedComponentIds={state.focusedComponentIds} />,
    )
    expect(html).toContain('Focus L16 (0.89)')
    expect(html).toContain('aria-pressed="true"')
  })

  it('CompareModels renders side-by-side reference pairs and handles single-model runs cleanly', () => {
    const sycoResults = createPlaceholderSycophancyResults()
    const pairs = extractModelComparisonPairs(sycoResults)
    expect(pairs.length).toBeGreaterThanOrEqual(2)
    expect(pairs.some((p) => p.metric === 'flip_rate' && p.referenceValue === 0.19)).toBe(true)

    const withRefHtml = renderToStaticMarkup(<CompareModels results={sycoResults} />)
    expect(withRefHtml).toContain('flip_rate')
    expect(withRefHtml).toContain('0.38')
    expect(withRefHtml).toContain('0.19')
    expect(withRefHtml).toContain('+0.19')

    // Also supports dual-metric rows (suspect_probe_auroc vs clean_probe_auroc)
    const dualRows: ResultRecord[] = [
      {
        run_id: 'run_dual_1',
        check: 'backdoor',
        component_id: 'L12',
        checkpoint_id: null,
        metric: 'suspect_probe_auroc',
        value: 0.91,
        control: null,
        dataset_id: 'backdoor/probes',
        split: 'held_out',
        seed: 0,
        device: 'cpu',
        precision: 'fp32',
        quantized: false,
      },
      {
        run_id: 'run_dual_1',
        check: 'backdoor',
        component_id: 'L12',
        checkpoint_id: null,
        metric: 'clean_probe_auroc',
        value: 0.54,
        control: null,
        dataset_id: 'backdoor/probes',
        split: 'held_out',
        seed: 0,
        device: 'cpu',
        precision: 'fp32',
        quantized: false,
      },
    ]
    const dualPairs = extractModelComparisonPairs(dualRows)
    expect(dualPairs).toHaveLength(1)
    expect(dualPairs[0].delta).toBeCloseTo(0.37, 5)

    // Single-model run with no reference model shows plain explanation
    const pythiaSingleModelResults = createPlaceholderPythiaResults(getFixturePythiaSeries())
    const singleHtml = renderToStaticMarkup(<CompareModels results={pythiaSingleModelResults} />)
    expect(singleHtml).toContain('No reference or clean comparison model was included in this run')
  })

  it('ShiftTable renders passed, failed, and not run rows accurately and explains suggestive strength cap', () => {
    const mixedShifts: ControlResult[] = [
      { kind: 'paraphrase', value: 0.84, passed: true, notes: 'Held on reworded prompts' },
      { kind: 'multi_turn', value: 0.56, passed: false, notes: 'Dropped below threshold' },
      { kind: 'multilingual', value: null, passed: null, notes: 'Not evaluated in this run' },
    ]

    expect(resolveShiftOutcome(mixedShifts[0])).toBe('passed')
    expect(resolveShiftOutcome(mixedShifts[1])).toBe('failed')
    expect(resolveShiftOutcome(mixedShifts[2])).toBe('not run')
    expect(doesShiftOutcomeCapStrength(mixedShifts)).toBe(true)

    const html = renderToStaticMarkup(
      <ShiftTable
        shiftTests={mixedShifts}
        finding={createPlaceholderPythiaFinding()}
      />,
    )
    expect(html).toContain('passed')
    expect(html).toContain('failed')
    expect(html).toContain('not run')
    expect(html).toContain('0.84')
    expect(html).toContain('0.56')
    expect(html).toContain('claim strength is capped at')
    expect(html).toContain('suggestive')
  })

  it('ExamplesList renders benign examples, badges placeholders, and strictly redacts harmful items', () => {
    const benignExamples = createPlaceholderSycophancyExamples()
    const redactedExamples = createPlaceholderRedactedExamples()

    // Verify defensive client-side sanitization even if an unredacted contains_harmful object is passed
    const unsafeCandidate: SpotCheckExample = {
      dataset_id: 'refusal/direct',
      index: 9,
      prompt: 'UNSAFE_RAW_TEXT_SHOULD_NEVER_APPEAR',
      response: 'UNSAFE_RAW_OUTPUT_SHOULD_NEVER_APPEAR',
      redacted: false,
      contains_harmful: true,
      is_placeholder: true,
    }

    const sanitizedUnsafe = sanitizeExampleForDisplay(unsafeCandidate)
    expect(sanitizedUnsafe.isRedacted).toBe(true)
    expect(sanitizedUnsafe.promptText).toBe(REDACTED_BADGE_TEXT)
    expect(sanitizedUnsafe.responseText).toBe(REDACTED_BADGE_TEXT)

    const html = renderToStaticMarkup(
      <ExamplesList
        examples={[...redactedExamples, unsafeCandidate, benignExamples[0]]}
        isPlaceholder
      />,
    )
    expect(html).toContain(REDACTED_BADGE_TEXT)
    expect(html).toContain('unverified / placeholder')
    expect(html).not.toContain('UNSAFE_RAW_TEXT_SHOULD_NEVER_APPEAR')
    expect(html).not.toContain('UNSAFE_RAW_OUTPUT_SHOULD_NEVER_APPEAR')
    expect(html).toContain('Matched benign control prompt')
  })

  it('fetches live examples and evidence bundle via api.ts helpers and FindingMessage supports default store handler', async () => {
    const mockFetch = vi.fn(async (url: string) => {
      if (url.startsWith('/api/examples')) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              dataset_id: 'refusal/direct',
              index: 0,
              prompt: '[harmful instruction: category summary]',
              response: '[response: refused]',
              redacted: true,
              contains_harmful: true,
            },
          ],
          text: async () => '',
        }
      }
      if (url.startsWith('/api/results')) {
        return {
          ok: true,
          status: 200,
          json: async () => createPlaceholderSycophancyResults(),
          text: async () => '',
        }
      }
      return {
        ok: true,
        status: 200,
        json: async () => [],
        text: async () => '',
      }
    })
    vi.stubGlobal('fetch', mockFetch)

    const examples = await fetchExamples('run_live_001', 3)
    expect(examples).toHaveLength(1)
    expect(examples[0].redacted).toBe(true)

    const finding = createPlaceholderSycophancyFinding()
    const bundle = await fetchEvidence(finding, 3)
    expect(bundle.finding?.finding_id).toBe(finding.finding_id)
    expect(bundle.results.length).toBeGreaterThan(0)
    expect(bundle.examples).toHaveLength(1)

    const fmHtml = renderToStaticMarkup(<FindingMessage finding={finding} />)
    expect(fmHtml).toContain('Show evidence')
  })
})
