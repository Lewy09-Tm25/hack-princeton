import { useEffect, useMemo, useState } from 'react'
import { fetchActiveModelId, fetchExamples, fetchModelMap, fetchResults } from './api'
import { CompareModels } from './charts/CompareModels'
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
import { EventsClient } from './events'
import { CloseIcon } from './icons'
import { ReportView } from './reports/ReportView'
import { ClaudeCodePanel } from './setup/ClaudeCodePanel'
import {
  HONESTY_BANNER_TEXT,
  getActivePaint,
  hasPlaceholderOrUnverifiedData,
  useAppStore,
  type DrawerTab,
} from './store'
import { ModelScene } from './view/ModelScene'
import { Timeline } from './view/Timeline'

export { HONESTY_BANNER_TEXT }

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
    let inFlight = false
    const syncActiveModel = async () => {
      // Never stack requests: if the engine is busy, wait for the previous check to return.
      if (inFlight) return
      inFlight = true
      try {
        const activeId = await fetchActiveModelId()
        if (stopped || !activeId || activeId === shownModelId) return
        const map = await fetchModelMap(activeId)
        if (stopped) return
        shownModelId = activeId
        setModelMap(map)
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
              <span className="floating-chip model-name-chip">
                <strong>{modelMap.model_id}</strong> ({modelMap.n_layers} layers
                {modelMap.arch_family ? `, ${modelMap.arch_family}` : ''})
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
          {activeJobEntries.length > 0 && (
            <span className="floating-chip">
              Job: {activeJobEntries[activeJobEntries.length - 1][1].status}
              {activeJobEntries[activeJobEntries.length - 1][1].step !== undefined &&
              activeJobEntries[activeJobEntries.length - 1][1].total !== undefined
                ? ` (${activeJobEntries[activeJobEntries.length - 1][1].step}/${activeJobEntries[activeJobEntries.length - 1][1].total})`
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

          <button
            type="button"
            className={`btn ${drawerTab === 'charts' ? 'btn-active' : ''}`}
            onClick={() => toggleDrawer('charts')}
          >
            Charts
          </button>
          <button
            type="button"
            className={`btn ${drawerTab === 'report' ? 'btn-active' : ''}`}
            onClick={() => toggleDrawer('report')}
          >
            Report
          </button>
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
                  <p style={{ margin: 0 }}>No model loaded.</p>
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
        <aside className="bottom-drawer" aria-label="Details drawer">
          <div className="drawer-header">
            <div className="drawer-tabs">
              <button
                type="button"
                className={`btn ${drawerTab === 'charts' ? 'btn-active' : ''}`}
                onClick={() => setDrawerTab('charts')}
              >
                Charts
              </button>
              <button
                type="button"
                className={`btn ${drawerTab === 'report' ? 'btn-active' : ''}`}
                onClick={() => setDrawerTab('report')}
              >
                Report
              </button>
              <button
                type="button"
                className={`btn ${drawerTab === 'claude-code' ? 'btn-active' : ''}`}
                onClick={() => setDrawerTab('claude-code')}
              >
                Claude Code setup
              </button>
            </div>
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
            {drawerTab === 'charts' && (
              <div>
                {selectedEvidenceFinding ? (
                  <>
                    <div className="evidence-summary-bar">
                      <div>
                        <div style={{ fontWeight: 600 }}>
                          Evidence for finding: <code>{selectedEvidenceFinding.finding_id}</code> (
                          {selectedEvidenceFinding.check}, run{' '}
                          <code>{selectedEvidenceFinding.run_id}</code>)
                        </div>
                        <div style={{ marginTop: '0.2rem' }}>{selectedEvidenceFinding.claim}</div>
                      </div>
                      <div className="badge-row">
                        {findings.length > 1 && (
                          <select
                            aria-label="Select finding for evidence"
                            className="select-compact"
                            value={selectedEvidenceFinding.finding_id}
                            onChange={(e) => openEvidenceForFinding(e.target.value)}
                          >
                            {findings.map((f) => (
                              <option key={f.finding_id} value={f.finding_id}>
                                {f.finding_id} ({f.check})
                              </option>
                            ))}
                          </select>
                        )}
                        {isEvidencePlaceholder && (
                          <span className="badge badge-placeholder">unverified / placeholder</span>
                        )}
                      </div>
                    </div>

                    <div className="evidence-grid">
                      <LayerCurve
                        results={resolvedEvidenceResults}
                        checkpointId={currentCheckpointId}
                      />
                      <CompareModels results={resolvedEvidenceResults} />
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
                ) : activePaint ? (
                  <div className="evidence-grid">
                    <LayerCurve
                      results={resolvedEvidenceResults}
                      checkpointId={currentCheckpointId}
                    />
                  </div>
                ) : (
                  <p className="drawer-empty">No charts yet.</p>
                )}
              </div>
            )}

            {drawerTab === 'report' && (
              <ReportView
                isFakeMode={isFakeMode}
                findings={findings}
                jobs={jobs}
                activeRunId={
                  selectedEvidenceFinding && !selectedEvidenceFinding.is_placeholder
                    ? selectedEvidenceFinding.run_id
                    : null
                }
              />
            )}

            {drawerTab === 'claude-code' && <ClaudeCodePanel isFakeMode={isFakeMode} />}
          </div>
        </aside>
      )}
    </div>
  )
}

export default App
