import type { ToolActivityInfo } from '../store'

export interface ToolActivityProps {
  activity: ToolActivityInfo
}

export function ToolActivity({ activity }: ToolActivityProps) {
  const prefix = activity.done ? 'Completed:' : 'Running:'
  return (
    <details className="tool-activity">
      <summary>
        <strong>{prefix}</strong> {activity.summary}
      </summary>
      {activity.detail && <pre>{activity.detail}</pre>}
    </details>
  )
}
