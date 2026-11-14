import type { Plan } from '../contracts'
import { describeCheckMeasurement, formatDurationEstimate } from './chatLogic'

export interface PlanMessageProps {
  plan: Plan
  onApprove: (planId: string) => void
}

export function PlanMessage({ plan, onApprove }: PlanMessageProps) {
  const layersText =
    plan.layers === 'all' ? 'All layers' : `Layers ${plan.layers.join(', ')}`

  return (
    <div className="msg-card" aria-label={`Experiment plan for ${plan.check}`}>
      <div className="msg-card-header">
        <h3 className="msg-card-title">
          Experiment plan: {plan.check} (v{plan.version ?? 1})
        </h3>
        <div className="badge-row">
          <span
            className={`badge ${
              plan.maturity === 'established' ? 'badge-established' : 'badge-suggestive'
            }`}
          >
            method: {plan.maturity}
          </span>
          {plan.is_placeholder && (
            <span className="badge badge-placeholder">unverified / placeholder</span>
          )}
          {plan.approved ? (
            <span className="badge badge-established">approved</span>
          ) : (
            <span className="badge badge-inconclusive">awaiting approval</span>
          )}
        </div>
      </div>

      <dl className="kv-list">
        <dt>Model</dt>
        <dd>
          <code>{plan.model_id}</code>
          {plan.reference_model_id ? (
            <span>
              {' '}
              compared with <code>{plan.reference_model_id}</code>
            </span>
          ) : null}
        </dd>

        <dt>What is measured</dt>
        <dd>{describeCheckMeasurement(plan.check)}</dd>

        <dt>Layers</dt>
        <dd>{layersText}</dd>

        <dt>Datasets</dt>
        <dd>{plan.datasets.length > 0 ? plan.datasets.join(', ') : 'Default check datasets'}</dd>

        <dt>Required controls</dt>
        <dd>{plan.controls.join(', ')}</dd>

        <dt>Shift tests</dt>
        <dd>{plan.shift_tests.length > 0 ? plan.shift_tests.join(', ') : 'None'}</dd>

        <dt>Source papers</dt>
        <dd>
          {plan.papers.length > 0 ? (
            <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
              {plan.papers.map((paper) => (
                <li key={paper}>{paper}</li>
              ))}
            </ul>
          ) : (
            'None listed'
          )}
        </dd>

        <dt>Estimated time</dt>
        <dd>{formatDurationEstimate(plan.estimate_seconds)} on detected hardware</dd>
      </dl>

      <div className="card-section">
        <div className="card-section-label">Coverage</div>
        <div>{plan.coverage}</div>
      </div>

      <div className="card-section">
        <div className="card-section-label">What is not covered</div>
        <div>{plan.not_covered}</div>
      </div>

      {!plan.approved && (
        <div
          className="card-section"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.6rem',
            flexWrap: 'wrap',
          }}
        >
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Nothing runs until you approve. Reply in plain language (for example, &ldquo;yes, go
            ahead&rdquo; or request changes) or click Approve plan.
          </span>
          {plan.plan_id && (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                if (plan.plan_id) {
                  onApprove(plan.plan_id)
                }
              }}
            >
              Approve plan
            </button>
          )}
        </div>
      )}
    </div>
  )
}
