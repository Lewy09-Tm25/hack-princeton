import { useEffect, useMemo, useState } from 'react'
import {
  buildReportForRun,
  displayModelName,
  fetchActiveModel,
  fetchExamples,
  fetchModelMap,
  fetchResults,
  fetchSeries,
  getReportUrl,
} from './api'
import { CompareModels, extractModelComparisonPairs } from './charts/CompareModels'
import { ExamplesList } from './charts/ExamplesList'
import { LayerCurve } from './charts/LayerCurve'
import { ShiftTable } from './charts/ShiftTable'
import { ChatPanel } from './chat/ChatPanel'
import type { ResultRecord } from './contracts'
import {
  createPlaceholderPythiaResults,
  createPlaceholderRedactedExamples,
  createPlaceholderSycophancyExamples,
  createPlaceholderSycophancyResults,
  getFixturePythiaSeries,
  sharedFakePlayer,
  type FakeFixturePreset,
} from './dev/fakeEvents'
import { plainDatasetText } from './datasetNames'
import { EventsClient } from './events'
import { CloseIcon } from './icons'
import { ClaudeCodePanel } from './setup/ClaudeCodePanel'
import {
  HONESTY_BANNER_TEXT,
  formatComponentLabel,
  getActivePaint,
  hasPlaceholderOrUnverifiedData,
  useAppStore,
  type DrawerTab,
} from './store'
import { ModelScene } from './view/ModelScene'
import { Timeline } from './view/Timeline'

export { HONESTY_BANNER_TEXT }

const CHECK_NAMES: Readonly<Record<string, string>> = {
  sycophancy: 'sycophancy',
  refusal: 'weakened refusals',
  backdoor: 'backdoor',
}

function defaultDrawerHeight(): number {
  const h = typeof window !== 'undefined' ? window.innerHeight : 900
  return Math.round(h * 0.5)
}

function clampDrawerHeight(height: number): number {
  const h = typeof window !== 'undefined' ? window.innerHeight : 900
  return Math.round(Math.min(h - 120, Math.max(160, height)))
}

/** Short job status for the floating chip. */
function jobLabel(status: string): string {
  if (status === 'completed' || status === 'done' || status === 'succeeded') return 'done'
  if (status === 'failed' || status === 'error') return 'failed'
  if (status === 'cancelled' || status === 'stopped') return 'stopped'
  if (status === 'needs_decision') return 'waiting for you'
  if (status === 'queued' || status === 'pending') return 'starting'
  return 'running'
}

const CHAT_WIDTH_KEY = 'chatPanelWidth'
const DEFAULT_CHAT_WIDTH = 420
const MIN_CHAT_WIDTH = 280
const MIN_VIEW_WIDTH = 320

function clampChatWidth(width: number): number {
  const viewport = typeof window !== 'undefined' ? window.innerWidth : 1440
  const max = Math.max(MIN_CHAT_WIDTH, viewport - MIN_VIEW_WIDTH)
  return Math.round(Math.min(max, Math.max(MIN_CHAT_WIDTH, width)))
}

export function App() {
  const modelMap = useAppStore((s) => s.modelMap)
  const series = useAppStore((s) => s.series)
  const currentCheckpointId = useAppStore((s) => s.currentCheckpointId)
  const paintByCheckpoint = useAppStore((s) => s.paintByCheckpoint)
  const viewMode = useAppStore((s) => s.viewMode)
  const jobs = useAppStore((s) => s.jobs)
  const plans = useAppStore((s) => s.plans)
  const findings = useAppStore((s) => s.findings)
  const findingModelNames = useAppStore((s) => s.findingModelNames)
  const resultsByRun = useAppStore((s) => s.resultsByRun)
  const examplesByRun = useAppStore((s) => s.examplesByRun)
  const chatMessages = useAppStore((s) => s.chatMessages)
  const isFakeMode = useAppStore((s) => s.isFakeMode)
  const connectionStatus = useAppStore((s) => s.connectionStatus)
  const drawerTab = useAppStore((s) => s.drawerTab)
  const activeEvidenceFindingId = useAppStore((s) => s.activeEvidenceFindingId)

  const setModelMap = useAppStore((s) => s.setModelMap)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const setFakeMode = useAppStore((s) => s.setFakeMode)
  const setDrawerTab = useAppStore((s) => s.setDrawerTab)
  const openEvidenceForFinding = useAppStore((s) => s.openEvidenceForFinding)
  const setResultsForRun = useAppStore((s) => s.setResultsForRun)
  const setExamplesForRun = useAppStore((s) => s.setExamplesForRun)

  const [replaySpeed, setReplaySpeed] = useState<number>(1)
  const [preset, setPreset] = useState<FakeFixturePreset>('qwen25_sycophancy')
  const [failedEvidenceRuns, setFailedEvidenceRuns] = useState<Record<string, boolean>>({})
  // Chat panel width in pixels. The user drags the divider to change it; the choice is
  // remembered in this browser only.
  const [chatWidth, setChatWidth] = useState<number>(() => {
    try {
      const saved = Number(window.localStorage.getItem(CHAT_WIDTH_KEY))
      return Number.isFinite(saved) && saved > 0 ? clampChatWidth(saved) : DEFAULT_CHAT_WIDTH
    } catch {
      return DEFAULT_CHAT_WIDTH
    }
  })
  const updateChatWidth = (next: number) => {
    const clamped = clampChatWidth(next)
    setChatWidth(clamped)
    try {
      window.localStorage.setItem(CHAT_WIDTH_KEY, String(clamped))
    } catch {
      // Storage unavailable (private window); the width just is not remembered.
    }
  }
  const handleResizeStart = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    const startX = e.clientX
    const startWidth = chatWidth
    document.body.classList.add('is-resizing')
    const onMove = (ev: PointerEvent) => updateChatWidth(startWidth + ev.clientX - startX)
    const onUp = () => {
      document.body.classList.remove('is-resizing')
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }
  const handleResizeKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowLeft') updateChatWidth(chatWidth - 32)
    if (e.key === 'ArrowRight') updateChatWidth(chatWidth + 32)
  }

  // Models the engine is loading right now, so the view can say so instead of sitting empty.
  const [loadingModels, setLoadingModels] = useState<string[]>([])

  // Placeholder replay controls are for development only. Open the app with `?fixtures=1`
  // to show them; users never see them otherwise.
  const showFixtureControls = useMemo(() => {
    try {
      return new URLSearchParams(window.location.search).get('fixtures') === '1'
    } catch {
      return false
    }
  }, [])

  const showHonestyBanner = hasPlaceholderOrUnverifiedData({
    isFakeMode,
    plans,
    findings,
    chatMessages,
    resultsByRun,
    examplesByRun,
  })

  useEffect(() => {
    if (isFakeMode) {
      sharedFakePlayer.setSpeed(replaySpeed)
      if (!useAppStore.getState().modelMap) {
        sharedFakePlayer.loadPresetModel(preset)
      }
      if (useAppStore.getState().chatMessages.length === 0) {
        sharedFakePlayer.seedInitialMockConversation()
      }
      return () => {
        sharedFakePlayer.stop()
      }
    }

    const client = new EventsClient()
    client.start()

    // Show only the model loaded in this engine session. A fresh start has no model:
    // the user tells the agent (here or in Claude Code) which model to check, and the
    // view follows as soon as the engine loads it.
    let stopped = false
    let shownModelId: string | null = null
    let shownSeriesId: string | null = null
    let inFlight = false
    const syncActiveModel = async () => {
      // Never stack requests: if the engine is busy, wait for the previous check to return.
      if (inFlight) return
      inFlight = true
      try {
        const { modelId: activeId, loading, seriesId } = await fetchActiveModel()
        if (stopped) return
        setLoadingModels(loading)
        // Load the checkpoint series for the timeline slider when the check runs over one.
        if (seriesId !== shownSeriesId) {
          shownSeriesId = seriesId
          useAppStore.getState().setSeries(seriesId ? await fetchSeries(seriesId) : null)
        }
        if (!activeId) {
          // The engine has no model (for example after a restart): clear the view too.
          if (shownModelId !== null) {
            shownModelId = null
            shownSeriesId = null
            setModelMap(null)
          }
          return
        }
        if (activeId === shownModelId) return
        const map = await fetchModelMap(activeId)
        if (stopped) return
        shownModelId = activeId
        setModelMap(map)
        // Switching models clears the series in the store; fetch it again on the next poll.
        shownSeriesId = null
      } catch {
        // Engine not reachable yet; the connection status already says so.
      } finally {
        inFlight = false
      }
    }
    void syncActiveModel()
    const pollTimer = setInterval(() => void syncActiveModel(), 2000)

    return () => {
      stopped = true
      clearInterval(pollTimer)
      client.stop()
    }
  }, [isFakeMode, preset, replaySpeed, setModelMap])

  useEffect(() => {
    sharedFakePlayer.setSpeed(replaySpeed)
  }, [replaySpeed])

  const selectedEvidenceFinding =
    findings.find((f) => f.finding_id === activeEvidenceFindingId) ?? findings[0] ?? null

  useEffect(() => {
    if (drawerTab !== 'charts' || !selectedEvidenceFinding) {
      return
    }
    const runId = selectedEvidenceFinding.run_id
    if (isFakeMode || selectedEvidenceFinding.is_placeholder) {
      return
    }
    const state = useAppStore.getState()
    if (state.resultsByRun[runId] !== undefined) {
      return
    }
    let cancelled = false
    void Promise.all([
      fetchResults({ run_id: runId }),
      fetchExamples(runId, 5).catch(() => []),
    ])
      .then(([resRows, exRows]) => {
        if (cancelled) return
        setResultsForRun(runId, resRows)
        setExamplesForRun(runId, exRows)
      })
      .catch(() => {
        if (cancelled) return
        setFailedEvidenceRuns((prev) => ({ ...prev, [runId]: true }))
      })
    return () => {
      cancelled = true
    }
  }, [
    drawerTab,
    selectedEvidenceFinding,
    isFakeMode,
    setResultsForRun,
    setExamplesForRun,
  ])

  const handleStartFixtureReplay = () => {
    setFakeMode(true)
    sharedFakePlayer.setSpeed(replaySpeed)
    sharedFakePlayer.loadPresetModel(preset)
    if (useAppStore.getState().chatMessages.length === 0) {
      sharedFakePlayer.seedInitialMockConversation()
    }
    sharedFakePlayer.replayStream(preset)
  }

  const handleExitFakeMode = () => {
    sharedFakePlayer.stop()
    useAppStore.getState().resetStore({ isFakeMode: false })
  }

  const handlePresetChange = (nextPreset: FakeFixturePreset) => {
    setPreset(nextPreset)
    if (isFakeMode) {
      sharedFakePlayer.loadPresetModel(nextPreset)
    }
  }

  const toggleDrawer = (tab: Exclude<DrawerTab, 'closed'>) => {
    setDrawerTab(drawerTab === tab ? 'closed' : tab)
  }

  const activePaint = getActivePaint({ currentCheckpointId, paintByCheckpoint })
  const activeJobEntries = Object.entries(jobs)
  const expandedCount = useAppStore((s) => s.expandedLayerIds.length)

  // Evidence panel height; the user drags its top edge to change it.
  const [drawerHeight, setDrawerHeight] = useState<number>(() => defaultDrawerHeight())
  const handleDrawerResizeStart = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    const startY = e.clientY
    const startHeight = drawerHeight
    document.body.classList.add('is-resizing-rows')
    const onMove = (ev: PointerEvent) =>
      setDrawerHeight(clampDrawerHeight(startHeight + startY - ev.clientY))
    const onUp = () => {
      document.body.classList.remove('is-resizing-rows')
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  // The full HTML report for the finding's run, built on request and opened in a new tab.
  const [reportBusy, setReportBusy] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)
  const openReport = async (runId: string) => {
    setReportBusy(true)
    setReportError(null)
    const tab = window.open('', '_blank')
    try {
      const summary = await buildReportForRun(runId)
      const url = getReportUrl(summary.report_id)
      if (tab) tab.location.href = url
      else window.open(url, '_blank')
    } catch {
      tab?.close()
      setReportError('The report could not be prepared.')
    } finally {
      setReportBusy(false)
    }
  }
  const latestJob = activeJobEntries.length > 0 ? activeJobEntries[activeJobEntries.length - 1][1] : null

  // The component with the strongest measurement in the current colouring (largest value,
  // or largest magnitude on a two-sided scale). Shown in the top-left instead of whatever
  // the agent last pointed at.
  const checkPeakId = useAppStore((s) => s.peakComponentId)
  const scanningId = useAppStore((s) => s.scanningComponentId)
  const strongest = useMemo(() => {
    // Same rule as the 3D marker: the check's own choice, else the largest value.
    if (checkPeakId) return { id: checkPeakId, metric: activePaint?.metric ?? '' }
    if (scanningId) return null
    if (!activePaint) return null
    let best: { id: string; value: number } | null = null
    for (const [id, raw] of Object.entries(activePaint.values)) {
      const v = activePaint.scale === 'diverging' ? Math.abs(raw) : raw
      if (best === null || v > best.value) best = { id, value: v }
    }
    return best ? { id: best.id, metric: activePaint.metric } : null
  }, [activePaint, checkPeakId, scanningId])

  const activeRunId = selectedEvidenceFinding?.run_id ?? null
  const isEvidencePlaceholder = Boolean(
    isFakeMode ||
      selectedEvidenceFinding?.is_placeholder ||
      (selectedEvidenceFinding && !selectedEvidenceFinding.verified),
  )

  const resolvedEvidenceResults = useMemo<ResultRecord[]>(() => {
    if (activeRunId && resultsByRun[activeRunId] !== undefined) {
      return resultsByRun[activeRunId]
    }
    if (selectedEvidenceFinding && (isFakeMode || selectedEvidenceFinding.is_placeholder)) {
      if (selectedEvidenceFinding.run_id === 'run_fixture_pythia_001') {
        return createPlaceholderPythiaResults(series ?? getFixturePythiaSeries())
      }
      return createPlaceholderSycophancyResults()
    }
    if (activePaint) {
      return Object.entries(activePaint.values).map(([cid, val]) => ({
        run_id: activeRunId ?? 'paint_preview',
        check: selectedEvidenceFinding?.check ?? 'painted_metric',
        component_id: cid,
        checkpoint_id: currentCheckpointId,
        metric: activePaint.metric,
        value: val,
        control: null,
        dataset_id: 'painted',
        split: 'held_out',
        seed: 0,
        device: 'cpu',
        precision: 'fp32',
        quantized: false,
        is_placeholder: isFakeMode,
      }))
    }
    return []
  }, [
    activeRunId,
    resultsByRun,
    selectedEvidenceFinding,
    isFakeMode,
    series,
    activePaint,
    currentCheckpointId,
  ])

  // Only show the model comparison chart when the check compared two models.
  const hasComparison = useMemo(
    () => extractModelComparisonPairs(resolvedEvidenceResults).length > 0,
    [resolvedEvidenceResults],
  )

  const resolvedEvidenceExamples = useMemo(() => {
    if (activeRunId && examplesByRun[activeRunId] !== undefined) {
      return examplesByRun[activeRunId]
    }
    if (selectedEvidenceFinding && (isFakeMode || selectedEvidenceFinding.is_placeholder)) {
      if (selectedEvidenceFinding.run_id === 'run_fixture_pythia_001') {
        return createPlaceholderRedactedExamples()
      }
      return createPlaceholderSycophancyExamples()
    }
    return []
  }, [activeRunId, examplesByRun, selectedEvidenceFinding, isFakeMode])

  const isLoadingLiveEvidence = Boolean(
    activeRunId &&
      !isFakeMode &&
      selectedEvidenceFinding &&
      !selectedEvidenceFinding.is_placeholder &&
      resultsByRun[activeRunId] === undefined &&
      !failedEvidenceRuns[activeRunId],
  )

  return (
    <div className="app-shell">
      {showHonestyBanner && (
        <div className="honesty-banner" role="status" aria-live="polite">
          <span>{HONESTY_BANNER_TEXT}</span>
          {isFakeMode && (
            <button type="button" className="btn" onClick={handleExitFakeMode}>
              Exit fixture mode
            </button>
          )}
        </div>
      )}

      <div className="workspace-split">
        <section className="pane-chat" aria-label="Conversation" style={{ width: chatWidth }}>
          <ChatPanel />
        </section>

        <div
          className="pane-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize the chat panel"
          aria-valuenow={chatWidth}
          tabIndex={0}
          onPointerDown={handleResizeStart}
          onKeyDown={handleResizeKey}
          onDoubleClick={() => updateChatWidth(DEFAULT_CHAT_WIDTH)}
        />

        <section className="pane-view" aria-label="Model structure view">
          {/* Floating controls over the model view (no top bar). */}
          <div className="floating-bar floating-top-left">
            {modelMap && (
              <span className="floating-chip model-name-chip" title={modelMap.model_id}>
                <strong>{displayModelName(modelMap.model_id)}</strong> · {modelMap.n_layers}{' '}
                layers
              </span>
            )}
            {strongest && (
              <span className="floating-chip strongest-chip" title={strongest.metric}>
                Strongest signal: <strong>{formatComponentLabel(strongest.id)}</strong>
              </span>
            )}
            {loadingModels.length > 0 && (
              <span className="floating-chip loading-chip" role="status">
                <span className="loading-dot" aria-hidden="true" />
                Loading {displayModelName(loadingModels[0])}…
              </span>
            )}
          </div>

          <div className="floating-bar floating-top-right">
            {connectionStatus === 'disconnected' && !isFakeMode && (
              <span className="floating-chip floating-warning" role="status">
                Engine not connected
              </span>
            )}
            {isFakeMode && (
              <span className="floating-chip">Fixture replay mode (placeholder)</span>
            )}
          {latestJob && (
            <span className={`floating-chip job-chip job-${jobLabel(latestJob.status)}`}>
              Job: {jobLabel(latestJob.status)}
              {jobLabel(latestJob.status) === 'running' &&
              latestJob.step !== undefined &&
              latestJob.total !== undefined &&
              latestJob.total > 0
                ? ` · ${Math.min(99, Math.round((100 * latestJob.step) / latestJob.total))}%`
                : ''}
            </span>
          )}
          {(showFixtureControls || isFakeMode) && (
            <>
              <select
                aria-label="Fixture preset"
                className="select-compact"
                value={preset}
                onChange={(e) => handlePresetChange(e.target.value as FakeFixturePreset)}
              >
                <option value="qwen25_sycophancy">
                  Qwen2.5-0.5B fixture (24 layers, placeholder)
                </option>
                <option value="pythia70m_series">
                  Pythia-70M series fixture (6 layers, placeholder)
                </option>
              </select>
              <select
                aria-label="Replay speed"
                className="select-compact"
                value={replaySpeed}
                onChange={(e) => setReplaySpeed(Number(e.target.value))}
              >
                <option value={0.5}>0.5x speed</option>
                <option value={1}>1x speed</option>
                <option value={2}>2x speed</option>
                <option value={4}>4x speed</option>
              </select>
              <button type="button" className="btn" onClick={handleStartFixtureReplay}>
                Replay fixture stream (placeholder)
              </button>
            </>
          )}

          {findings.length > 0 && (
            <button
              type="button"
              className={`btn ${drawerTab === 'charts' ? 'btn-active' : ''}`}
              title="Evidence for the findings in this session"
              onClick={() => {
                if (drawerTab === 'charts') {
                  setDrawerTab('closed')
                } else if (selectedEvidenceFinding) {
                  setDrawerTab('charts')
                } else {
                  openEvidenceForFinding(findings[findings.length - 1].finding_id)
                }
              }}
            >
              Evidence ({findings.length})
            </button>
          )}
          <button
            type="button"
            className={`btn ${drawerTab === 'claude-code' ? 'btn-active' : ''}`}
            onClick={() => toggleDrawer('claude-code')}
          >
            Claude Code setup
          </button>
          </div>

          <div className="view-stage">
            <div className="floating-bar floating-bottom-right">
              {modelMap && expandedCount > 0 && (
                <button
                  type="button"
                  className="btn"
                  onClick={() => useAppStore.setState({ expandedLayerIds: [] })}
                >
                  Collapse all
                </button>
              )}
              <button
                type="button"
                className={`btn ${viewMode === '3d' ? 'btn-active' : ''}`}
                onClick={() => setViewMode('3d')}
              >
                3D
              </button>
              <button
                type="button"
                className={`btn ${viewMode === '2d' ? 'btn-active' : ''}`}
                onClick={() => setViewMode('2d')}
              >
                2D
              </button>
            </div>
            {!modelMap ? (
              <div className="view-empty-overlay">
                {connectionStatus === 'disconnected' ? (
                  <p style={{ margin: 0 }}>
                    The engine is not running. Start the tool with <code>tool serve</code>, then
                    reload this page.
                  </p>
                ) : (
                  <p style={{ margin: 0 }}>
                    {loadingModels.length > 0
                      ? `Loading ${displayModelName(loadingModels[0])}…`
                      : 'No model loaded.'}
                  </p>
                )}
              </div>
            ) : (
              <ModelScene modelMap={modelMap} />
            )}
          </div>

          {series && <Timeline />}
        </section>
      </div>

      {drawerTab !== 'closed' && (
        <aside
          className="bottom-drawer"
          aria-label={drawerTab === 'claude-code' ? 'Claude Code setup' : 'Evidence'}
          style={{ height: drawerHeight, maxHeight: 'none' }}
        >
          <div
            className="drawer-resizer"
            role="separator"
            aria-orientation="horizontal"
            aria-label="Resize the panel"
            tabIndex={0}
            onPointerDown={handleDrawerResizeStart}
            onKeyDown={(e) => {
              if (e.key === 'ArrowUp') setDrawerHeight(clampDrawerHeight(drawerHeight + 40))
              if (e.key === 'ArrowDown') setDrawerHeight(clampDrawerHeight(drawerHeight - 40))
            }}
            onDoubleClick={() => setDrawerHeight(defaultDrawerHeight())}
          />
          <div className="drawer-header">
            <strong className="drawer-title">
              {drawerTab === 'claude-code' ? 'Claude Code setup' : 'Evidence'}
            </strong>
            <button
              type="button"
              className="icon-btn"
              aria-label="Close panel"
              title="Close"
              onClick={() => setDrawerTab('closed')}
            >
              <CloseIcon />
            </button>
          </div>

          <div className="drawer-body">
            {drawerTab !== 'claude-code' && (
              <div>
                {selectedEvidenceFinding ? (
                  <>
                    <div className="evidence-summary-bar">
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="evidence-claim">
                          {plainDatasetText(
                            selectedEvidenceFinding.reviewer_verdict?.corrected_claim ??
                              selectedEvidenceFinding.claim,
                          )}
                        </div>
                      </div>
                      <div className="badge-row">
                        {findings.length > 1 && (
                          <select
                            aria-label="Choose which finding to show"
                            className="select-compact"
                            value={selectedEvidenceFinding.finding_id}
                            onChange={(e) => openEvidenceForFinding(e.target.value)}
                          >
                            {findings.map((f, i) => (
                              <option key={f.finding_id} value={f.finding_id}>
                                {`${i + 1}. ${CHECK_NAMES[f.check] ?? f.check}${
                                  findingModelNames[f.finding_id]
                                    ? ` · ${findingModelNames[f.finding_id]}`
                                    : ''
                                }`}
                              </option>
                            ))}
                          </select>
                        )}
                        {!isFakeMode && !selectedEvidenceFinding.is_placeholder && (
                          <button
                            type="button"
                            className="btn"
                            disabled={reportBusy}
                            onClick={() => void openReport(selectedEvidenceFinding.run_id)}
                          >
                            {reportBusy ? 'Preparing report…' : 'Open full report'}
                          </button>
                        )}
                        {isEvidencePlaceholder && (
                          <span className="badge badge-placeholder">unverified / placeholder</span>
                        )}
                      </div>
                    </div>
                    {reportError && <p className="drawer-note">{reportError}</p>}

                    <div className="evidence-grid">
                      <LayerCurve
                        results={resolvedEvidenceResults}
                        checkpointId={currentCheckpointId}
                      />
                      {hasComparison && <CompareModels results={resolvedEvidenceResults} />}
                      <ShiftTable
                        shiftTests={selectedEvidenceFinding.shift_tests}
                        finding={selectedEvidenceFinding}
                        isPlaceholder={isEvidencePlaceholder}
                      />
                      <ExamplesList
                        examples={resolvedEvidenceExamples}
                        loading={isLoadingLiveEvidence}
                        isPlaceholder={isEvidencePlaceholder}
                      />
                    </div>
                  </>
                ) : (
                  <p className="drawer-empty">No evidence yet.</p>
                )}
              </div>
            )}

            {drawerTab === 'claude-code' && <ClaudeCodePanel isFakeMode={isFakeMode} />}
          </div>
        </aside>
      )}
    </div>
  )
}

export default App
