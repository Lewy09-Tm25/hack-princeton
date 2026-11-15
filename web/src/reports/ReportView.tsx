import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ApiError,
  buildReportForRun,
  fetchReports,
  getReportUrl,
  type ReportSummary,
} from '../api'
import type { Finding } from '../contracts'
import { useAppStore, type JobState } from '../store'

export const SYNTHETIC_REPORT_BANNER_TEXT =
  'Synthetic fixture preview — placeholder test runs do not produce verified HTML reports.'

export interface ReportViewProps {
  isFakeMode?: boolean
  findings?: Finding[]
  jobs?: Record<string, JobState>
  activeRunId?: string | null
  initialReports?: ReportSummary[]
  selectedReportId?: string | null
  onSelectReport?: (reportId: string) => void
  onBuildReport?: (runId: string) => Promise<ReportSummary | null> | void
}

export function extractBuildableRunIds(
  findings: readonly Finding[],
  jobs: Record<string, JobState>,
): string[] {
  const seen = new Set<string>()
  const ordered: string[] = []

  for (const f of findings) {
    if (f.is_placeholder) continue
    if (f.run_id && !seen.has(f.run_id)) {
      seen.add(f.run_id)
      ordered.push(f.run_id)
    }
  }

  for (const [jobId, job] of Object.entries(jobs)) {
    if (job.status === 'completed' && jobId.startsWith('run_') && !seen.has(jobId)) {
      seen.add(jobId)
      ordered.push(jobId)
    }
  }

  return ordered
}

export function resolveActiveReport(
  reports: readonly ReportSummary[],
  selectedReportId: string | null | undefined,
  preferredRunId?: string | null,
): ReportSummary | null {
  if (reports.length === 0) {
    return null
  }
  if (selectedReportId) {
    const exact = reports.find((r) => r.report_id === selectedReportId)
    if (exact) return exact
  }
  if (preferredRunId) {
    const byRun = reports.find(
      (r) => r.run_id === preferredRunId || r.report_id === preferredRunId,
    )
    if (byRun) return byRun
  }
  return reports[0] ?? null
}

export async function requestBuildReportForRun(runId: string): Promise<{
  report: ReportSummary | null
  fallbackMessage: string | null
}> {
  try {
    const built = await buildReportForRun(runId)
    const normalized: ReportSummary = {
      ...built,
      report_id: built.report_id || runId,
      run_id: built.run_id ?? runId,
    }
    return { report: normalized, fallbackMessage: null }
  } catch (err) {
    const status = err instanceof ApiError ? err.status : null
    if (status === 404 || status === 405) {
      return {
        report: null,
        fallbackMessage: `Report for ${runId} has not been built yet. Run "tool report ${runId}" in your terminal or ask the assistant to call build_report.`,
      }
    }
    return {
      report: null,
      fallbackMessage: `Could not build report for ${runId} right now. You can generate it from stored workspace results with "tool report ${runId}" or build_report.`,
    }
  }
}

function formatReportTimestamp(iso?: string): string {
  if (!iso) return 'Saved report'
  const parsed = Date.parse(iso)
  if (Number.isNaN(parsed)) return iso
  return new Date(parsed).toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
}

export function ReportView({
  isFakeMode: isFakeModeProp,
  findings: findingsProp,
  jobs: jobsProp,
  activeRunId: activeRunIdProp,
  initialReports,
  selectedReportId: selectedReportIdProp,
  onSelectReport,
  onBuildReport,
}: ReportViewProps = {}) {
  const storeIsFakeMode = useAppStore((s) => s.isFakeMode)
  const storeFindings = useAppStore((s) => s.findings)
  const storeJobs = useAppStore((s) => s.jobs)
  const storeActiveFindingId = useAppStore((s) => s.activeEvidenceFindingId)

  const isFakeMode = isFakeModeProp !== undefined ? isFakeModeProp : storeIsFakeMode
  const findings = findingsProp !== undefined ? findingsProp : storeFindings
  const jobs = jobsProp !== undefined ? jobsProp : storeJobs

  const derivedActiveRunId = useMemo(() => {
    if (activeRunIdProp !== undefined) return activeRunIdProp
    const matched =
      findings.find((f) => f.finding_id === storeActiveFindingId) ?? findings[0] ?? null
    return matched && !matched.is_placeholder ? matched.run_id : null
  }, [activeRunIdProp, findings, storeActiveFindingId])

  const [fetchedReports, setFetchedReports] = useState<ReportSummary[]>(initialReports ?? [])
  const [localSelectedId, setLocalSelectedId] = useState<string | null>(
    selectedReportIdProp ?? null,
  )
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [buildingRunId, setBuildingRunId] = useState<string | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)

  const reports = initialReports !== undefined ? initialReports : fetchedReports
  const effectiveSelectedId =
    selectedReportIdProp !== undefined ? selectedReportIdProp : localSelectedId

  const refreshReports = useCallback(() => {
    if (isFakeMode || initialReports !== undefined) {
      return
    }
    void fetchReports()
      .then((list) => {
        setFetchedReports(list)
        setLoadFailed(false)
      })
      .catch(() => {
        setLoadFailed(true)
      })
  }, [isFakeMode, initialReports])

  useEffect(() => {
    if (isFakeMode || initialReports !== undefined) {
      return
    }
    let cancelled = false
    void fetchReports()
      .then((list) => {
        if (cancelled) return
        setFetchedReports(list)
        setLoadFailed(false)
      })
      .catch(() => {
        if (cancelled) return
        setLoadFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [isFakeMode, initialReports])

  const buildableRunIds = useMemo(
    () => extractBuildableRunIds(findings, jobs),
    [findings, jobs],
  )

  const activeReport = useMemo(
    () => resolveActiveReport(reports, effectiveSelectedId, derivedActiveRunId),
    [reports, effectiveSelectedId, derivedActiveRunId],
  )

  const handleSelectReport = (reportId: string) => {
    setStatusMessage(null)
    if (onSelectReport) {
      onSelectReport(reportId)
    } else {
      setLocalSelectedId(reportId)
    }
  }

  const handleBuildReport = async (runId: string) => {
    setBuildingRunId(runId)
    setStatusMessage(null)
    try {
      if (onBuildReport) {
        const custom = await onBuildReport(runId)
        if (custom) {
          setFetchedReports((prev) => [
            custom,
            ...prev.filter((r) => r.report_id !== custom.report_id),
          ])
          handleSelectReport(custom.report_id)
        }
        return
      }
      const outcome = await requestBuildReportForRun(runId)
      if (outcome.report) {
        const created = outcome.report
        setFetchedReports((prev) => [
          created,
          ...prev.filter((r) => r.report_id !== created.report_id),
        ])
        handleSelectReport(created.report_id)
      } else if (outcome.fallbackMessage) {
        setStatusMessage(outcome.fallbackMessage)
      }
    } finally {
      setBuildingRunId(null)
    }
  }

  if (isFakeMode) {
    return (
      <div className="report-view-synthetic" role="region" aria-label="Report viewer (fixture mode)">
        <div className="synthetic-report-banner" role="note">
          <strong>{SYNTHETIC_REPORT_BANNER_TEXT}</strong>
          <span className="badge badge-placeholder">unverified / placeholder</span>
        </div>
        <p className="report-synthetic-copy">
          Fixture replay mode (`?fake=1`) only plays back recorded placeholder events so the
          interface can be inspected offline. It never generates or displays a verified safety
          report.
        </p>
        <p className="report-synthetic-copy">
          To generate a real standalone HTML report with embedded measurements, controls, shift
          tests, and redacted examples, connect to the local engine (`tool serve`) and run{' '}
          <code>tool report &lt;run_id&gt;</code> or call <code>build_report</code> after an
          approved model check completes.
        </p>
      </div>
    )
  }

  const targetBuildRunId = derivedActiveRunId ?? buildableRunIds[0] ?? null
  const activeReportUrl = activeReport ? getReportUrl(activeReport.report_id) : null

  // Nothing to show and nothing to build: say so, and nothing more.
  if (reports.length === 0 && !targetBuildRunId && !statusMessage) {
    return (
      <div className="report-view" role="region" aria-label="Report viewer">
        <p className="drawer-empty">
          {loadFailed ? 'Could not load reports.' : 'No reports yet.'}
        </p>
      </div>
    )
  }

  return (
    <div className="report-view" role="region" aria-label="Report viewer">
      <div className="report-sidebar">
        <div className="report-sidebar-header">
          <strong>Reports</strong>
          <button
            type="button"
            className="btn"
            aria-label="Refresh saved reports"
            onClick={refreshReports}
          >
            Refresh
          </button>
        </div>

        {targetBuildRunId && (
          <div className="report-build-box">
            <span className="report-build-label">
              Known run: <code>{targetBuildRunId}</code>
            </span>
            <button
              type="button"
              className="btn btn-primary"
              disabled={buildingRunId === targetBuildRunId}
              onClick={() => {
                void handleBuildReport(targetBuildRunId)
              }}
            >
              {buildingRunId === targetBuildRunId ? 'Building report...' : 'Build report'}
            </button>
          </div>
        )}

        {statusMessage && (
          <div className="report-status-note" role="status">
            {statusMessage}
          </div>
        )}

        {reports.length === 0 ? (
          <p className="chart-empty-note">
            {loadFailed ? 'Could not load reports.' : 'No reports yet.'}
          </p>
        ) : (
          <ul className="report-list" aria-label="Past reports">
            {reports.map((rep) => {
              const isSelected = activeReport?.report_id === rep.report_id
              return (
                <li key={rep.report_id}>
                  <button
                    type="button"
                    className={`report-list-btn ${isSelected ? 'active' : ''}`}
                    aria-pressed={isSelected}
                    onClick={() => handleSelectReport(rep.report_id)}
                  >
                    <span className="report-list-id">
                      <code>{rep.report_id}</code>
                    </span>
                    {rep.run_id && rep.run_id !== rep.report_id && (
                      <span className="report-list-meta">run: {rep.run_id}</span>
                    )}
                    <span className="report-list-meta">
                      {formatReportTimestamp(rep.created_at)}
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      <div className="report-frame-pane">
        {activeReport && activeReportUrl ? (
          <>
            <div className="report-frame-toolbar">
              <span>
                Showing report <code>{activeReport.report_id}</code>
                {activeReport.run_id ? ` (run ${activeReport.run_id})` : ''}
              </span>
              <a
                href={activeReportUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="btn"
              >
                Open in new tab
              </a>
            </div>
            <iframe
              title="Investigation report"
              src={activeReportUrl}
              className="report-iframe"
            />
          </>
        ) : (
          <div className="report-frame-empty">
            <p style={{ margin: 0 }}>
              {reports.length > 0 ? 'Select a report.' : 'No report selected.'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
