import type { SpotCheckExample } from '../api'

export const REDACTED_BADGE_TEXT = '[REDACTED — harmful prompt/output withheld]'

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
}

function isBracketedRedactionSummary(text: string): boolean {
  const trimmed = text.trim()
  return trimmed.startsWith('[') && trimmed.endsWith(']')
}

export function sanitizeExampleForDisplay(
  example: SpotCheckExample,
  forcePlaceholder = false,
): SanitizedExampleView {
  const isRedacted = Boolean(example.redacted || example.contains_harmful)
  const isPlaceholder = Boolean(example.is_placeholder || forcePlaceholder)

  if (isRedacted) {
    const safePrompt =
      example.redacted && typeof example.prompt === 'string' && isBracketedRedactionSummary(example.prompt)
        ? example.prompt.trim()
        : REDACTED_BADGE_TEXT
    const safeResponse =
      example.redacted && typeof example.response === 'string' && isBracketedRedactionSummary(example.response)
        ? example.response.trim()
        : REDACTED_BADGE_TEXT
    return {
      datasetId: example.dataset_id,
      index: example.index,
      isRedacted: true,
      isPlaceholder,
      promptText: safePrompt,
      responseText: safeResponse,
      referenceResponseText: null,
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
            Sampled items from the run. Harmful prompts and outputs are withheld by engine
            redaction.
          </span>
        </div>
        <div className="badge-row">
          {anyRedacted && <span className="badge badge-redacted">{REDACTED_BADGE_TEXT}</span>}
          {anyPlaceholder && (
            <span className="badge badge-placeholder">unverified / placeholder</span>
          )}
        </div>
      </div>

      {loading ? (
        <p className="chart-empty-note">Loading spot-check examples...</p>
      ) : sanitized.length === 0 ? (
        <p className="chart-empty-note">No spot-check examples recorded for this run.</p>
      ) : (
        <ul className="examples-list">
          {sanitized.map((item, idx) => (
            <li
              key={`${item.datasetId}_${item.index}_${idx}`}
              className="example-item"
              aria-label={`Example ${item.index} from ${item.datasetId}`}
            >
              <div className="example-item-header">
                <span>
                  <code>{item.datasetId}</code> · item #{item.index}
                </span>
                <div className="badge-row">
                  {item.isRedacted && (
                    <span className="badge badge-redacted">{REDACTED_BADGE_TEXT}</span>
                  )}
                  {item.isPlaceholder && (
                    <span className="badge badge-placeholder">unverified / placeholder</span>
                  )}
                </div>
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
          ))}
        </ul>
      )}
    </section>
  )
}
