import type { SpotCheckExample } from '../api'
import { datasetLabel } from '../datasetNames'

export const REDACTED_BADGE_TEXT = 'text withheld'

export interface ExamplesListProps {
  examples: SpotCheckExample[]
  loading?: boolean
  isPlaceholder?: boolean
}

export interface SanitizedExampleView {
  datasetId: string
  index: number
  isRedacted: boolean
  isPlaceholder: boolean
  promptText: string
  responseText: string
  referenceResponseText: string | null
  /** For withheld examples: what the model did, when the engine recorded it. */
  outcome: string | null
}

function isBracketedRedactionSummary(text: string): boolean {
  const trimmed = text.trim()
  return trimmed.startsWith('[') && trimmed.endsWith(']')
}

/** What the model did on a withheld example, from the engine's safe summary fields. */
function outcomeOf(example: SpotCheckExample): string | null {
  if (typeof example.refused === 'boolean') {
    return example.refused ? 'refused' : 'did not refuse'
  }
  const resp = typeof example.response === 'string' ? example.response.trim() : ''
  const m = /^\[response:\s*(.+)\]$/i.exec(resp)
  if (m) {
    const word = m[1].trim().toLowerCase()
    if (word === 'complied' || word === 'answered') return 'did not refuse'
    return word
  }
  if (typeof example.label === 'string' && example.label) return example.label
  return null
}

export function sanitizeExampleForDisplay(
  example: SpotCheckExample,
  forcePlaceholder = false,
): SanitizedExampleView {
  const isRedacted = Boolean(example.redacted || example.contains_harmful)
  const isPlaceholder = Boolean(example.is_placeholder || forcePlaceholder)

  if (isRedacted) {
    // Harmful prompts and the model's answers to them never leave the engine. Show only
    // safe summaries: which dataset item it was, and what the model did.
    const safePrompt =
      example.redacted &&
      typeof example.prompt === 'string' &&
      isBracketedRedactionSummary(example.prompt)
        ? example.prompt.trim()
        : REDACTED_BADGE_TEXT
    return {
      datasetId: example.dataset_id,
      index: example.index,
      isRedacted: true,
      isPlaceholder,
      promptText: safePrompt,
      responseText: REDACTED_BADGE_TEXT,
      referenceResponseText: null,
      outcome: outcomeOf(example),
    }
  }

  return {
    datasetId: example.dataset_id,
    index: example.index,
    isRedacted: false,
    isPlaceholder,
    promptText: example.prompt,
    responseText: example.response,
    referenceResponseText:
      typeof example.reference_response === 'string' ? example.reference_response : null,
    outcome: null,
  }
}

export function ExamplesList({
  examples,
  loading = false,
  isPlaceholder = false,
}: ExamplesListProps) {
  const sanitized = examples.map((ex) => sanitizeExampleForDisplay(ex, isPlaceholder))
  const anyPlaceholder = isPlaceholder || sanitized.some((s) => s.isPlaceholder)
  const anyRedacted = sanitized.some((s) => s.isRedacted)

  return (
    <section className="chart-card" aria-label="Spot-check examples">
      <div className="chart-card-header">
        <div>
          <h4 className="chart-card-title">Spot-check examples</h4>
          <span className="chart-card-subtitle">
            {anyRedacted
              ? 'Examples from the run. Harmful requests and the model’s answers to them stay inside the engine, so only what the model did is shown.'
              : 'Examples from the run.'}
          </span>
        </div>
        <div className="badge-row">
          {anyPlaceholder && (
            <span className="badge badge-placeholder">unverified / placeholder</span>
          )}
        </div>
      </div>

      {loading ? (
        <p className="chart-empty-note">Loading examples…</p>
      ) : sanitized.length === 0 ? (
        <p className="chart-empty-note">No examples recorded for this run.</p>
      ) : (
        <ul className="examples-list">
          {sanitized.map((item, idx) =>
            item.isRedacted ? (
              <li
                key={`${item.datasetId}_${item.index}_${idx}`}
                className="example-item example-item-withheld"
                aria-label={`Example ${item.index} from ${datasetLabel(item.datasetId)}`}
              >
                <span>
                  {datasetLabel(item.datasetId)} #{item.index}
                  <span className="example-withheld-note"> · text withheld</span>
                </span>
                {item.outcome && (
                  <span
                    className={`badge ${
                      item.outcome === 'refused' ? 'badge-established' : 'badge-inconclusive'
                    }`}
                  >
                    {item.outcome}
                  </span>
                )}
              </li>
            ) : (
              <li
                key={`${item.datasetId}_${item.index}_${idx}`}
                className="example-item"
                aria-label={`Example ${item.index} from ${datasetLabel(item.datasetId)}`}
              >
                <div className="example-item-header">
                  <span>
                    {datasetLabel(item.datasetId)} #{item.index}
                  </span>
                  {item.isPlaceholder && (
                    <span className="badge badge-placeholder">unverified / placeholder</span>
                  )}
                </div>
                <div className="example-field">
                  <span className="example-field-label">Prompt:</span>{' '}
                  <span>{item.promptText}</span>
                </div>
                <div className="example-field">
                  <span className="example-field-label">Model response:</span>{' '}
                  <span>{item.responseText}</span>
                </div>
                {item.referenceResponseText && (
                  <div className="example-field">
                    <span className="example-field-label">Reference response:</span>{' '}
                    <span>{item.referenceResponseText}</span>
                  </div>
                )}
              </li>
            ),
          )}
        </ul>
      )}
    </section>
  )
}
