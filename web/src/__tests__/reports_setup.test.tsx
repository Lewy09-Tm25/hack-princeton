import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import {
  buildReportForRun,
  fetchClaudeCodeSetup,
  fetchReports,
  getReportUrl,
  type ClaudeCodeSetupResponse,
  type ReportSummary,
} from '../api'
import {
  createPlaceholderSycophancyFinding,
  sharedFakePlayer,
} from '../dev/fakeEvents'
import {
  ReportView,
  SYNTHETIC_REPORT_BANNER_TEXT,
  extractBuildableRunIds,
  requestBuildReportForRun,
  resolveActiveReport,
} from '../reports/ReportView'
import {
  ClaudeCodePanel,
  DEFAULT_HTTP_COMMAND,
  DEFAULT_PLUGIN_COMMAND,
  DEFAULT_STDIO_COMMAND,
  copyCommandToClipboard,
  resolveSetupCommands,
} from '../setup/ClaudeCodePanel'
import { useAppStore } from '../store'

describe('W6 ReportView & report API helpers', () => {
  const savedInitial = { ...useAppStore.getInitialState() }

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    sharedFakePlayer.stop()
    Object.assign(useAppStore.getInitialState(), savedInitial)
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('fetches past reports from /api/reports, renders selectable list and iframe, and appends ?token= when local auth is active', async () => {
    const sampleReports: ReportSummary[] = [
      {
        report_id: 'rep_syco_001',
        run_id: 'run_syco_001',
        created_at: '2026-11-14T11:00:00Z',
        path: '/workspace/reports/rep_syco_001.html',
      },
      {
        report_id: 'rep_refusal_002',
        run_id: 'run_refusal_002',
        created_at: '2026-11-14T11:30:00Z',
        path: '/workspace/reports/rep_refusal_002.html',
      },
    ]

    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => ({
        ok: true,
        status: 200,
        json: async () => (url === '/api/reports' ? sampleReports : []),
        text: async () => '',
      })),
    )

    const fetched = await fetchReports()
    expect(fetched).toHaveLength(2)
    expect(fetched[0].report_id).toBe('rep_syco_001')

    // Verify iframe src defaults to first report and switches when selectedReportId changes
    const firstHtml = renderToStaticMarkup(
      <ReportView isFakeMode={false} initialReports={fetched} selectedReportId="rep_syco_001" />,
    )
    expect(firstHtml).toContain('>Reports<')
    expect(firstHtml).toContain('rep_syco_001')
    expect(firstHtml).toContain('rep_refusal_002')
    expect(firstHtml).toContain('2026-11-14 11:00:00 UTC')
    expect(firstHtml).toContain('title="Investigation report"')
    expect(firstHtml).toContain('src="/api/reports/rep_syco_001"')
    expect(firstHtml).toContain('Open in new tab')

    const switchedHtml = renderToStaticMarkup(
      <ReportView
        isFakeMode={false}
        initialReports={fetched}
        selectedReportId="rep_refusal_002"
      />,
    )
    expect(switchedHtml).toContain('src="/api/reports/rep_refusal_002"')
    expect(
      resolveActiveReport(fetched, 'rep_refusal_002', 'run_syco_001')?.report_id,
    ).toBe('rep_refusal_002')

    // Verify local bearer token is appended to iframe / new-tab URL when present
    vi.stubGlobal('window', { location: { search: '?token=local_secret_123' } })
    expect(getReportUrl('rep_syco_001')).toBe('/api/reports/rep_syco_001?token=local_secret_123')
    const tokenHtml = renderToStaticMarkup(
      <ReportView isFakeMode={false} initialReports={fetched} selectedReportId="rep_syco_001" />,
    )
    expect(tokenHtml).toContain('src="/api/reports/rep_syco_001?token=local_secret_123"')
  })

  it('triggers buildReportForRun (POST /api/reports/{runId}) and falls back gracefully when report is not built yet', async () => {
    const fetchMock = vi.fn(async (url: string, init?: { method?: string }) => {
      if (url === '/api/reports/run_live_101' && init?.method === 'POST') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            report_id: 'rep_live_101',
            run_id: 'run_live_101',
            created_at: '2026-11-14T11:45:00Z',
          }),
          text: async () => '',
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: 'Not found' }),
        text: async () => 'Not found',
      }
    })
    vi.stubGlobal('fetch', fetchMock)

    const built = await buildReportForRun('run_live_101')
    expect(built.report_id).toBe('rep_live_101')

    const successOutcome = await requestBuildReportForRun('run_live_101')
    expect(successOutcome.report?.report_id).toBe('rep_live_101')
    expect(successOutcome.fallbackMessage).toBeNull()

    // Fallback when POST /api/reports/{run_id} returns 404
    const missingOutcome = await requestBuildReportForRun('run_unbuilt_999')
    expect(missingOutcome.report).toBeNull()
    expect(missingOutcome.fallbackMessage).toContain('tool report run_unbuilt_999')

    // Verify Build report button renders for non-placeholder findings/runs
    const liveFinding = {
      ...createPlaceholderSycophancyFinding(),
      finding_id: 'finding_live_101',
      run_id: 'run_live_101',
      is_placeholder: false,
      verified: true,
    }
    expect(extractBuildableRunIds([liveFinding], {})).toEqual(['run_live_101'])
    const buildHtml = renderToStaticMarkup(
      <ReportView isFakeMode={false} findings={[liveFinding]} initialReports={[]} />,
    )
    expect(buildHtml).toContain('Build report')
    expect(buildHtml).toContain('run_live_101')
  })

  it('never presents fake data as a real report when isFakeMode=true', () => {
    const fakeHtml = renderToStaticMarkup(
      <ReportView
        isFakeMode={true}
        initialReports={[
          { report_id: 'rep_should_not_render', run_id: 'run_fixture_syco_001' },
        ]}
      />,
    )
    expect(fakeHtml).toContain(SYNTHETIC_REPORT_BANNER_TEXT)
    expect(fakeHtml).toContain('unverified / placeholder')
    expect(fakeHtml).toContain('tool report')
    expect(fakeHtml).toContain('build_report')
    expect(fakeHtml).not.toContain('<iframe')
    expect(fakeHtml).not.toContain('rep_should_not_render')
  })
})

describe('W7 ClaudeCodePanel & App drawer integration', () => {
  const savedInitial = { ...useAppStore.getInitialState() }

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    sharedFakePlayer.stop()
    Object.assign(useAppStore.getInitialState(), savedInitial)
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('fetches /api/setup/claude-code, renders HTTP / stdio / plugin commands and shared-workspace explanation, and copies to clipboard with Copied feedback', async () => {
    const apiPayload: ClaudeCodeSetupResponse = {
      command:
        'claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp --header "Authorization: Bearer tok_1"',
      http_command:
        'claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp --header "Authorization: Bearer tok_1"',
      stdio_command:
        'claude mcp add model-safety-engine -- tool mcp-stdio --workspace "C:/workspace"',
      mcp_url: 'http://127.0.0.1:8000/mcp',
      workspace: 'C:/workspace',
      notes: 'Connects Claude Code to the same local engine and workspace used by the web app.',
    }

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => apiPayload,
        text: async () => '',
      })),
    )

    const loaded = await fetchClaudeCodeSetup()
    const resolved = resolveSetupCommands(loaded)
    expect(resolved.httpCommand).toContain('--transport http model-safety-engine')
    expect(resolved.stdioCommand).toContain('tool mcp-stdio --workspace "C:/workspace"')
    expect(resolved.pluginCommand).toBe(DEFAULT_PLUGIN_COMMAND)

    // Verify offline defaults
    const defaults = resolveSetupCommands(null)
    expect(defaults.httpCommand).toBe(DEFAULT_HTTP_COMMAND)
    expect(defaults.stdioCommand).toBe(DEFAULT_STDIO_COMMAND)
    expect(defaults.pluginCommand).toBe(DEFAULT_PLUGIN_COMMAND)

    const html = renderToStaticMarkup(<ClaudeCodePanel setup={loaded} isFakeMode={false} />)
    expect(html).toContain('Live HTTP connection')
    expect(html).toContain('Stdio MCP server')
    expect(html).toContain('Claude Code plugin')
    expect(html).toContain(resolved.httpCommand.replace(/"/g, '&quot;'))
    expect(html).toContain(DEFAULT_PLUGIN_COMMAND)
    expect(html).toContain('same local engine, workspace, and')
    expect(html).toContain('in sync between terminal and')
    // The workspace path is no longer shown in the intro sentence.
    expect(html).not.toContain('workspace (C:/workspace)')

    // Copy to navigator.clipboard and check Copied state
    const writeTextMock = vi.fn(async () => undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText: writeTextMock } })

    const copied = await copyCommandToClipboard(resolved.httpCommand)
    expect(copied).toBe(true)
    expect(writeTextMock).toHaveBeenCalledWith(resolved.httpCommand)

    const copiedHtml = renderToStaticMarkup(
      <ClaudeCodePanel setup={loaded} isFakeMode={false} copiedKeyOverride="http" />,
    )
    expect(copiedHtml).toContain('>Copied<')
  })

  it('switches App bottom drawer cleanly between charts, report, and claude-code tabs', () => {
    sharedFakePlayer.replaySynchronously('qwen25_sycophancy')
    const store = useAppStore.getState()

    // 1. Charts tab
    store.setDrawerTab('charts')
    Object.assign(useAppStore.getInitialState(), useAppStore.getState())
    const chartsHtml = renderToStaticMarkup(<App />)
    expect(chartsHtml).toContain('Per-layer curve')
    expect(chartsHtml).toContain('Distribution shift tests')

    // 2. Report tab (in fake mode shows honest synthetic report preview, never a fake verified report)
    store.setDrawerTab('report')
    Object.assign(useAppStore.getInitialState(), useAppStore.getState())
    const reportHtml = renderToStaticMarkup(<App />)
    expect(reportHtml).toContain(SYNTHETIC_REPORT_BANNER_TEXT)

    // 3. Claude Code setup tab
    store.setDrawerTab('claude-code')
    Object.assign(useAppStore.getInitialState(), useAppStore.getState())
    const setupHtml = renderToStaticMarkup(<App />)
    expect(setupHtml).toContain('Live HTTP connection')
    expect(setupHtml).toContain('Stdio MCP server')
    expect(setupHtml).toContain('Claude Code plugin')
    expect(setupHtml).toContain(DEFAULT_PLUGIN_COMMAND)

    // 4. Closed drawer
    store.setDrawerTab('closed')
    Object.assign(useAppStore.getInitialState(), useAppStore.getState())
    const closedHtml = renderToStaticMarkup(<App />)
    expect(closedHtml).not.toContain('aria-label="Details drawer"')
  })
})
