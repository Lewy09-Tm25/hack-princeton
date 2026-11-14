import { useEffect, useState } from 'react'
import { fetchClaudeCodeSetup, fetchModelMap, fetchModels, getReportUrl } from './api'
import {
  sharedFakePlayer,
  type FakeFixturePreset,
} from './dev/fakeEvents'
import { EventsClient } from './events'
import {
  HONESTY_BANNER_TEXT,
  formatComponentLabel,
  getActivePaint,
  hasPlaceholderOrUnverifiedData,
  useAppStore,
  type DrawerTab,
} from './store'

export { HONESTY_BANNER_TEXT }

export function App() {
  const modelMap = useAppStore((s) => s.modelMap)
  const series = useAppStore((s) => s.series)
  const currentCheckpointId = useAppStore((s) => s.currentCheckpointId)
  const paintByCheckpoint = useAppStore((s) => s.paintByCheckpoint)
  const selectedComponentId = useAppStore((s) => s.selectedComponentId)
  const viewMode = useAppStore((s) => s.viewMode)
  const jobs = useAppStore((s) => s.jobs)
  const plans = useAppStore((s) => s.plans)
  const findings = useAppStore((s) => s.findings)
  const chatMessages = useAppStore((s) => s.chatMessages)
  const isFakeMode = useAppStore((s) => s.isFakeMode)
  const connectionStatus = useAppStore((s) => s.connectionStatus)
  const drawerTab = useAppStore((s) => s.drawerTab)
  const activeEvidenceFindingId = useAppStore((s) => s.activeEvidenceFindingId)

  const setModelMap = useAppStore((s) => s.setModelMap)
  const setCurrentCheckpointId = useAppStore((s) => s.setCurrentCheckpointId)
  const setViewMode = useAppStore((s) => s.setViewMode)
  const setFakeMode = useAppStore((s) => s.setFakeMode)
  const setDrawerTab = useAppStore((s) => s.setDrawerTab)

  const [replaySpeed, setReplaySpeed] = useState<number>(1)
  const [preset, setPreset] = useState<FakeFixturePreset>('qwen25_sycophancy')
  const [setupCommand, setSetupCommand] = useState<string>(
    'claude mcp add model-safety-engine -- tool mcp-stdio',
  )

  const showHonestyBanner = hasPlaceholderOrUnverifiedData({
    isFakeMode,
    plans,
    findings,
    chatMessages,
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

    void (async () => {
      try {
        const models = await fetchModels()
        if (models.length > 0 && !useAppStore.getState().modelMap) {
          const map = await fetchModelMap(models[0].model_id)
          setModelMap(map)
        }
      } catch {
        // Engine server may not have a loaded model yet
      }
    })()

    return () => {
      client.stop()
    }
  }, [isFakeMode, preset, replaySpeed, setModelMap])

  useEffect(() => {
    sharedFakePlayer.setSpeed(replaySpeed)
  }, [replaySpeed])

  useEffect(() => {
    if (drawerTab === 'claude-code' && !isFakeMode) {
      void fetchClaudeCodeSetup()
        .then((res) => {
          if (res.command) {
            setSetupCommand(res.command)
          }
        })
        .catch(() => {
          // Keep default fallback command
        })
    }
  }, [drawerTab, isFakeMode])

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
  const selectedEvidenceFinding =
    findings.find((f) => f.finding_id === activeEvidenceFindingId) ?? findings[0] ?? null

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

      <header className="app-header">
        <div className="app-header-left">
          <h1 className="app-title">Model safety investigation tool</h1>
          <span className="status-pill">
            <span
              className={`status-dot ${
                connectionStatus === 'connected'
                  ? 'connected'
                  : connectionStatus === 'fake'
                    ? 'fake'
                    : ''
              }`}
            />
            {connectionStatus === 'connected' && 'Engine connected'}
            {connectionStatus === 'connecting' && 'Connecting to engine'}
            {connectionStatus === 'disconnected' && 'Engine disconnected'}
            {connectionStatus === 'fake' && 'Fixture replay mode (placeholder)'}
          </span>
          {activeJobEntries.length > 0 && (
            <span className="status-pill">
              Job: {activeJobEntries[activeJobEntries.length - 1][1].status}
              {activeJobEntries[activeJobEntries.length - 1][1].step !== undefined &&
              activeJobEntries[activeJobEntries.length - 1][1].total !== undefined
                ? ` (${activeJobEntries[activeJobEntries.length - 1][1].step}/${activeJobEntries[activeJobEntries.length - 1][1].total})`
                : ''}
            </span>
          )}
        </div>

        <div className="app-header-right">
          {(connectionStatus === 'disconnected' || isFakeMode) && (
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
      </header>

      <div className="workspace-split">
        <section className="pane-chat" aria-label="Conversation">
          <div className="chat-panel">
            <div className="chat-messages">
              {chatMessages.length === 0 ? (
                <div className="chat-empty">
                  Describe your model or ask a safety question in plain language to draft an
                  experiment plan.
                </div>
              ) : (
                chatMessages.map((m) => (
                  <div key={m.id} className={`msg-row ${m.role}`}>
                    <div className="msg-bubble">{m.text}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>

        <section className="pane-view" aria-label="Model structure view">
          <div className="view-toolbar">
            <div>
              {modelMap ? (
                <span>
                  <strong>{modelMap.model_id}</strong> ({modelMap.n_layers} layers
                  {modelMap.arch_family ? `, ${modelMap.arch_family}` : ''})
                </span>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>No model loaded yet</span>
              )}
            </div>
            <div style={{ display: 'flex', gap: '0.35rem' }}>
              <button
                type="button"
                className={`btn ${viewMode === '3d' ? 'btn-active' : ''}`}
                onClick={() => setViewMode('3d')}
              >
                3D view
              </button>
              <button
                type="button"
                className={`btn ${viewMode === '2d' ? 'btn-active' : ''}`}
                onClick={() => setViewMode('2d')}
              >
                2D grid
              </button>
            </div>
          </div>

          <div className="view-stage">
            {!modelMap ? (
              <div className="view-empty-overlay">
                <p style={{ margin: 0 }}>
                  Connect the engine server and load a model, or start the placeholder fixture
                  replay above to inspect a sample architecture map.
                </p>
              </div>
            ) : (
              <div className="view-empty-overlay">
                <p style={{ margin: 0 }}>
                  Loaded {modelMap.model_id} ({modelMap.n_layers} layers, view mode: {viewMode}).
                  {activePaint
                    ? ` Painted metric: ${activePaint.metric} (${Object.keys(activePaint.values).length} components).`
                    : ' No measurement painted yet.'}
                  {selectedComponentId
                    ? ` Selected: ${formatComponentLabel(selectedComponentId)}.`
                    : ''}
                </p>
              </div>
            )}
          </div>

          {series && (
            <div className="timeline-slot" aria-label="Checkpoint timeline">
              <strong>
                {series.kind === 'pretraining' ? 'Pretraining checkpoints:' : 'Training stages:'}
              </strong>
              <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                {series.items.map((item) => (
                  <button
                    key={item.checkpoint_id}
                    type="button"
                    className={`btn ${
                      currentCheckpointId === item.checkpoint_id ? 'btn-active' : ''
                    }`}
                    onClick={() => setCurrentCheckpointId(item.checkpoint_id)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          )}
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
                Charts and evidence
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
            <button type="button" className="btn" onClick={() => setDrawerTab('closed')}>
              Close
            </button>
          </div>

          <div className="drawer-body">
            {drawerTab === 'charts' && (
              <div>
                {selectedEvidenceFinding ? (
                  <div>
                    <p style={{ marginTop: 0, fontWeight: 600 }}>
                      Evidence for finding: {selectedEvidenceFinding.finding_id}{' '}
                      {(!selectedEvidenceFinding.verified ||
                        selectedEvidenceFinding.is_placeholder) && (
                        <span className="badge badge-placeholder">unverified / placeholder</span>
                      )}
                    </p>
                    <p>{selectedEvidenceFinding.claim}</p>
                    <p style={{ color: 'var(--text-muted)' }}>
                      Evidence rows: {selectedEvidenceFinding.evidence.join(', ')}
                    </p>
                  </div>
                ) : activePaint ? (
                  <div>
                    <p style={{ marginTop: 0, fontWeight: 600 }}>
                      Active painted metric: {activePaint.metric} ({activePaint.scale})
                    </p>
                    <p style={{ color: 'var(--text-muted)' }}>
                      {Object.keys(activePaint.values).length} component measurements available in
                      store. Detailed layer curves and comparison charts attach here in W5.
                    </p>
                  </div>
                ) : (
                  <p style={{ color: 'var(--text-muted)', margin: 0 }}>
                    Run an approved check or select a finding to inspect per-layer curves, control
                    comparisons, and shift-test outcomes.
                  </p>
                )}
              </div>
            )}

            {drawerTab === 'report' && (
              <div>
                {selectedEvidenceFinding && !selectedEvidenceFinding.is_placeholder ? (
                  <iframe
                    title="Experiment report"
                    src={getReportUrl(selectedEvidenceFinding.run_id)}
                    style={{ width: '100%', height: '240px', border: '1px solid var(--border-subtle)' }}
                  />
                ) : (
                  <p style={{ color: 'var(--text-muted)', margin: 0 }}>
                    {isFakeMode || selectedEvidenceFinding?.is_placeholder
                      ? 'Placeholder fixture mode does not generate a verified HTML report. Run a real experiment on the engine to view its saved HTML report.'
                      : 'No saved HTML report selected yet. Run a check to generate a report.'}
                  </p>
                )}
              </div>
            )}

            {drawerTab === 'claude-code' && (
              <div>
                <p style={{ marginTop: 0 }}>
                  Connect Claude Code to the same local engine and workspace so runs started from
                  terminal appear in this view:
                </p>
                <pre
                  style={{
                    padding: '0.6rem 0.8rem',
                    background: 'var(--bg-subtle)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '6px',
                    overflowX: 'auto',
                  }}
                >
                  <code>{setupCommand}</code>
                </pre>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  )
}

export default App
