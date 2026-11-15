import type { ControlResult, Finding, StrengthLevel } from '../contracts'
import { plainDatasetText } from '../datasetNames'
import { useAppStore } from '../store'

export interface FindingMessageProps {
  finding: Finding
  onShowEvidence?: (findingId: string) => void
}

const CHECK_TITLES: Readonly<Record<string, string>> = {
  sycophancy: 'Sycophancy',
  refusal: 'Weakened refusals',
  backdoor: 'Backdoor',
}

// What each strength means, so a reader does not need to know the scale in advance.
const STRENGTH_MEANING: Readonly<Record<StrengthLevel, string>> = {
  established: 'Strongest of three levels: passed every control and every robustness test.',
  suggestive:
    'Middle of three levels: passed its controls, but not every robustness test, so treat it as likely rather than proven.',
  inconclusive:
    'Weakest of three levels: a required control failed or the effect was too small, so this is not a finding.',
}

// Plain names for the controls and robustness tests.
const TEST_NAMES: Readonly<Record<string, string>> = {
  random_direction: 'stronger than random directions',
  shuffled_labels: 'stronger than shuffled labels',
  held_out: 'holds on unseen examples',
  reference_model: 'differs from the original model',
  surface_matched_harmless: 'ignores harmless look-alikes',
  clean_reference_model: 'absent in a clean model',
  behavioral_confirmation: 'changes the model’s behavior',
  paraphrase: 'reworded inputs',
  multilingual: 'another language',
  multi_turn: 'longer conversations',
}

function formatOutcome(item: ControlResult): string {
  const name = TEST_NAMES[item.kind] ?? item.kind.replace(/_/g, ' ')
  const status = item.passed === true ? 'passed' : item.passed === false ? 'failed' : 'not run'
  return `${name}: ${status}`
}

function strengthBadgeClass(strength: StrengthLevel): string {
  if (strength === 'established') return 'badge badge-established'
  if (strength === 'suggestive') return 'badge badge-suggestive'
  return 'badge badge-inconclusive'
}

/** Reviewer notes without internal prefixes such as "[offline rule audit; …]". */
function cleanReviewerNotes(notes: string): string {
  return plainDatasetText(notes.replace(/^\s*\[[^\]]*\]\s*/, '').trim())
}

export function FindingMessage({ finding, onShowEvidence }: FindingMessageProps) {
  const effectiveStrength: StrengthLevel =
    finding.reviewer_verdict?.lower_strength_to ?? finding.strength
  const displayClaim = plainDatasetText(
    finding.reviewer_verdict?.corrected_claim ?? finding.claim,
  )
  const isUnverifiedOrPlaceholder = !finding.verified || Boolean(finding.is_placeholder)
  const verdict = finding.reviewer_verdict
  const reviewerText = verdict
    ? (verdict.notes && cleanReviewerNotes(verdict.notes)) ||
      (verdict.agrees
        ? 'The wording matches the evidence.'
        : 'The reviewer adjusted the wording or strength.')
    : 'Being checked…'

  return (
    <div className="msg-card" aria-label={`Finding for ${finding.check}`}>
      <div className="msg-card-header">
        <h3 className="msg-card-title">{CHECK_TITLES[finding.check] ?? finding.check}</h3>
        <div className="badge-row">
          <span className={strengthBadgeClass(effectiveStrength)}>{effectiveStrength}</span>
          {isUnverifiedOrPlaceholder && (
            <span className="badge badge-placeholder">unverified / placeholder</span>
          )}
        </div>
      </div>

      <div style={{ fontWeight: 500 }}>{displayClaim}</div>
      <div className="strength-meaning">{STRENGTH_MEANING[effectiveStrength]}</div>

      <dl className="kv-list">
        <dt>Controls</dt>
        <dd>
          {finding.controls.length > 0
            ? finding.controls.map(formatOutcome).join(' · ')
            : 'None recorded'}
        </dd>

        <dt>Robustness tests</dt>
        <dd>
          {finding.shift_tests.length > 0
            ? finding.shift_tests.map(formatOutcome).join(' · ')
            : 'None recorded'}
        </dd>
      </dl>

      <div className="card-section">
        <div className="card-section-label">Coverage</div>
        <div>{plainDatasetText(finding.coverage)}</div>
      </div>

      <div className="card-section">
        <div className="card-section-label">What this does not show</div>
        <div>{plainDatasetText(finding.not_shown)}</div>
      </div>

      <div className="card-section">
        <div className="card-section-label">Independent review</div>
        <div>{reviewerText}</div>
      </div>

      <div className="card-section" style={{ display: 'flex', justifyContent: 'flex-end' }}>
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
