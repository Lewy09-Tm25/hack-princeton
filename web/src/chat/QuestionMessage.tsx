export interface QuestionMessageProps {
  jobId: string
  text: string
  isPlaceholder?: boolean
}

export function QuestionMessage({ jobId, text, isPlaceholder }: QuestionMessageProps) {
  return (
    <div className="msg-card" aria-label="Paused job question">
      <div className="msg-card-header">
        <h3 className="msg-card-title">Job paused for your decision</h3>
        <div className="badge-row">
          <span className="badge badge-suggestive">job: {jobId}</span>
          {isPlaceholder && (
            <span className="badge badge-placeholder">placeholder</span>
          )}
        </div>
      </div>
      <div>{text}</div>
      <div style={{ fontSize: '0.77rem', color: 'var(--text-muted)' }}>
        Reply in plain language below to continue or stop this job.
      </div>
    </div>
  )
}
