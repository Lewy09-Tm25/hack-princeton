import type { ReactNode } from 'react'
import type { Finding, Plan } from '../contracts'
import { formatComponentLabel, type ChatMessage } from '../store'
import { FindingMessage } from './FindingMessage'
import { PlanMessage } from './PlanMessage'
import { QuestionMessage } from './QuestionMessage'
import { ToolActivity } from './ToolActivity'

function renderInlineFormatting(line: string): ReactNode[] {
  const tokens = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return tokens.map((tok, idx) => {
    if (tok.startsWith('**') && tok.endsWith('**') && tok.length > 4) {
      return <strong key={idx}>{tok.slice(2, -2)}</strong>
    }
    if (tok.startsWith('`') && tok.endsWith('`') && tok.length > 2) {
      return <code key={idx}>{tok.slice(1, -1)}</code>
    }
    return <span key={idx}>{tok}</span>
  })
}

function renderSafeMarkdown(text: string): ReactNode {
  const lines = text.split('\n')
  const blocks: ReactNode[] = []
  let bulletBuffer: string[] = []

  const flushBullets = (keyPrefix: string) => {
    if (bulletBuffer.length === 0) return
    const items = bulletBuffer
    bulletBuffer = []
    blocks.push(
      <ul key={keyPrefix} style={{ margin: '0.25rem 0', paddingLeft: '1.2rem' }}>
        {items.map((item, i) => (
          <li key={i}>{renderInlineFormatting(item)}</li>
        ))}
      </ul>,
    )
  }

  lines.forEach((rawLine, lineIdx) => {
    const trimmed = rawLine.trim()
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      bulletBuffer.push(trimmed.slice(2))
      return
    }
    flushBullets(`ul_${lineIdx}`)
    if (trimmed.length === 0) {
      return
    }
    blocks.push(
      <p
        key={`p_${lineIdx}`}
        style={{ margin: blocks.length === 0 ? 0 : '0.4rem 0 0' }}
      >
        {renderInlineFormatting(rawLine)}
      </p>,
    )
  })
  flushBullets('ul_end')

  return blocks.length > 0 ? blocks : text
}

export interface MessageProps {
  message: ChatMessage
  resolvedPlan?: Plan
  resolvedFinding?: Finding
  onApprovePlan: (planId: string) => void
  onShowEvidence: (findingId: string) => void
}

export function Message({
  message,
  resolvedPlan,
  resolvedFinding,
  onApprovePlan,
  onShowEvidence,
}: MessageProps) {
  if (message.kind === 'tool_activity' && message.toolActivity) {
    return (
      <div className="msg-row assistant">
        <ToolActivity activity={message.toolActivity} />
      </div>
    )
  }

  if (message.kind === 'plan') {
    const plan = resolvedPlan ?? message.plan
    if (plan) {
      return (
        <div className="msg-row assistant">
          <PlanMessage plan={plan} onApprove={onApprovePlan} />
        </div>
      )
    }
  }

  if (message.kind === 'finding') {
    const finding = resolvedFinding ?? message.finding
    if (finding) {
      return (
        <div className="msg-row assistant">
          <FindingMessage finding={finding} onShowEvidence={onShowEvidence} />
        </div>
      )
    }
  }

  if (message.kind === 'question') {
    return (
      <div className="msg-row assistant">
        <QuestionMessage
          jobId={message.questionJobId ?? 'active_job'}
          text={message.text}
          isPlaceholder={message.isPlaceholder}
        />
      </div>
    )
  }

  // No "You" / "Assistant" labels: the bubble's side and colour show who is speaking.
  // A small line remains only when there is something extra to say.
  const metaParts: string[] = []
  if (message.contextComponentId) {
    metaParts.push(
      `context: ${formatComponentLabel(message.contextComponentId)} (${message.contextComponentId})`,
    )
  }
  if (message.streaming) {
    metaParts.push('streaming…')
  }

  return (
    <div className={`msg-row ${message.role}`}>
      {metaParts.length > 0 && <div className="msg-meta">{metaParts.join(' · ')}</div>}
      <div className="msg-bubble">{renderSafeMarkdown(message.text)}</div>
    </div>
  )
}
