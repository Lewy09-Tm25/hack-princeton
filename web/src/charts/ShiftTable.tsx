import type { ControlResult, Finding } from '../contracts'

export type ShiftOutcomeStatus = 'passed' | 'failed' | 'not run'

export interface ShiftTableProps {
  shiftTests: ControlResult[]
  finding?: Finding | null
  isPlaceholder?: boolean
}

export function resolveShiftOutcome(item: ControlResult): ShiftOutcomeStatus {
  if (item.passed === true) return 'passed'
  if (item.passed === false) return 'failed'
  return 'not run'
}

export function doesShiftOutcomeCapStrength(shiftTests: readonly ControlResult[]): boolean {
  if (shiftTests.length === 0) return true
  return shiftTests.some((item) => item.passed !== true)
}

function outcomeBadgeClass(status: ShiftOutcomeStatus): string {
  if (status === 'passed') return 'badge badge-pass'
  if (status === 'failed') return 'badge badge-fail'
  return 'badge badge-inconclusive'
}

export function ShiftTable({ shiftTests, finding, isPlaceholder }: ShiftTableProps) {
  const placeholder = Boolean(
    isPlaceholder || finding?.is_placeholder || (finding && !finding.verified),
  )
  const capsStrength = doesShiftOutcomeCapStrength(shiftTests)
  const failedCount = shiftTests.filter((t) => resolveShiftOutcome(t) === 'failed').length
  const notRunCount = shiftTests.filter((t) => resolveShiftOutcome(t) === 'not run').length

  return (
    <section className="chart-card" aria-label="Distribution shift tests">
      <div className="chart-card-header">
        <div>
          <h4 className="chart-card-title">Distribution shift tests</h4>
          <span className="chart-card-subtitle">
            Checks whether the measured effect holds on paraphrased, translated, or multi-turn
            variants.
          </span>
        </div>
        {placeholder && (
          <span className="badge badge-placeholder">unverified / placeholder</span>
        )}
      </div>

      {capsStrength && (
        <div className="shift-cap-notice" role="note">
          {shiftTests.length === 0 ? (
            <span>
              No distribution shift tests were recorded for this finding, so claim strength is
              capped at <strong>suggestive</strong>.
            </span>
          ) : (
            <span>
              {failedCount > 0 && notRunCount > 0
                ? `${failedCount} shift ${failedCount === 1 ? 'test' : 'tests'} failed and ${notRunCount} ${notRunCount === 1 ? 'was' : 'were'} not run`
                : failedCount > 0
                  ? `${failedCount} shift ${failedCount === 1 ? 'test' : 'tests'} failed`
                  : `${notRunCount} shift ${notRunCount === 1 ? 'test was' : 'tests were'} not run`}{' '}
              — claim strength is capped at <strong>suggestive</strong> because the effect did not
              pass every tested shift condition.
            </span>
          )}
        </div>
      )}

      {shiftTests.length === 0 ? (
        <p className="chart-empty-note">No shift tests recorded on this finding.</p>
      ) : (
        <table className="evidence-table" aria-label="Shift test outcomes">
          <thead>
            <tr>
              <th scope="col">Shift kind</th>
              <th scope="col">Outcome</th>
              <th scope="col">Measured value</th>
              <th scope="col">Notes</th>
            </tr>
          </thead>
          <tbody>
            {shiftTests.map((item, idx) => {
              const status = resolveShiftOutcome(item)
              const hasValue =
                item.value !== undefined && item.value !== null && Number.isFinite(item.value)
              return (
                <tr key={`${item.kind}_${idx}`}>
                  <td>
                    <code>{item.kind}</code>
                  </td>
                  <td>
                    <span className={outcomeBadgeClass(status)}>{status}</span>
                  </td>
                  <td>{hasValue ? (item.value as number).toFixed(2) : '—'}</td>
                  <td>{item.notes ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </section>
  )
}
