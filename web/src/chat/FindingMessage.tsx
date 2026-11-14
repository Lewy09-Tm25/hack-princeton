import type { ControlResult, Finding, StrengthLevel } from '../contracts'
import { useAppStore } from '../store'

export interface FindingMessageProps {
  finding: Finding
  onShowEvidence?: (findingId: string) => void
}

function formatOutcome(item: ControlResult): string {
  const statusLabel =
    item.passed === true ? 'Pass' : item.passed === false ? 'Fail' : 'Not run'
  const valPart =
    item.value !== undefined && item.value !== null && Number.isFinite(item.value)
      ? ` (${item.value.toFixed(2)})`
      : ''
  return `${item.kind}: ${statusLabel}${valPart}`
}

function strengthBadgeClass(strength: StrengthLevel): string {
  if (strength === 'established') return 'badge badge-established'
  if (strength === 'suggestive') return 'badge badge-suggestive'
  return 'badge badge-inconclusive'
}

export function FindingMessage({ finding, onShowEvidence }: FindingMessageProps) {
  const effectiveStrength: StrengthLevel =
    finding.reviewer_verdict?.lower_strength_to ?? finding.strength
  const displayClaim = finding.reviewer_verdict?.corrected_claim ?? finding.claim
  const isUnverifiedOrPlaceholder = !finding.verified || Boolean(finding.is_placeholder)

  return (
    <div className="msg-card" aria-label={`Finding for ${finding.check}`}>
      <div className="msg-card-header">
        <h3 className="msg-card-title">Finding: {finding.check}</h3>
        <div className="badge-row">
          <span className={strengthBadgeClass(effectiveStrength)}>
            strength: {effectiveStrength}
          </span>
          <span
            className={`badge ${
              finding.maturity === 'established' ? 'badge-established' : 'badge-suggestive'
            }`}
          >
            method: {finding.maturity}
          </span>
          {isUnverifiedOrPlaceholder && (
            <span className="badge badge-placeholder">unverified / placeholder</span>
          )}
        </div>
      </div>

      <div style={{ fontWeight: 500 }}>{displayClaim}</div>

      <dl className="kv-list">
        <dt>Controls</dt>
        <dd>
          {finding.controls.length > 0
            ? finding.controls.map(formatOutcome).join(' · ')
            : 'None recorded'}
        </dd>

        <dt>Shift tests</dt>
        <dd>
          {finding.shift_tests.length > 0
            ? finding.shift_tests.map(formatOutcome).join(' · ')
            : 'None recorded'}
        </dd>
      </dl>

      <div className="card-section">
        <div className="card-section-label">Coverage</div>
        <div>{finding.coverage}</div>
      </div>

      <div className="card-section">
        <div className="card-section-label">What this does not show</div>
        <div>{finding.not_shown}</div>
      </div>

      <div className="card-section">
        <div className="card-section-label">Reviewer notes</div>
        <div>
          {finding.reviewer_verdict
            ? finding.reviewer_verdict.notes ||
              (finding.reviewer_verdict.agrees
                ? 'Reviewer agreed that claim wording matches evidence and coverage.'
                : 'Reviewer adjusted claim strength or wording.')
            : 'Review pending'}
        </div>
      </div>

      <div
        className="card-section"
        style={{ display: 'flex', justifyContent: 'flex-end' }}
      >
        <button
          type="button"
          className="btn"
          onClick={() => {
            if (onShowEvidence) {
              onShowEvidence(finding.finding_id)
            } else {
              useAppStore.getState().openEvidenceForFinding(finding.finding_id)
            }
          }}
        >
          Show evidence
        </button>
      </div>
    </div>
  )
}
