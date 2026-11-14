import type { PaintEntry } from '../store'
import {
  NEUTRAL_MISSING_HEX,
  buildLegendGradientCss,
  computeDivergingMaxAbs,
} from './colors'

export interface LegendProps {
  paint: PaintEntry | null
  isPlaceholder?: boolean
}

export function Legend({ paint, isPlaceholder }: LegendProps) {
  if (!paint) {
    return (
      <div className="legend-overlay" aria-label="Measurement legend">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          <span
            className="grid-swatch"
            style={{ backgroundColor: NEUTRAL_MISSING_HEX }}
          />
          <span>No measurement painted yet (neutral gray)</span>
        </div>
      </div>
    )
  }

  const maxAbs =
    paint.scale === 'diverging' ? computeDivergingMaxAbs(paint.values) : 1

  return (
    <div className="legend-overlay" aria-label="Measurement legend">
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.4rem',
          fontWeight: 600,
        }}
      >
        <span>{paint.metric}</span>
        {isPlaceholder && (
          <span className="badge badge-placeholder">placeholder</span>
        )}
      </div>

      <div
        className="legend-bar"
        style={{ background: buildLegendGradientCss(paint.scale) }}
      />

      {paint.scale === 'sequential' ? (
        <div className="legend-ticks">
          <span>0.00</span>
          <span>0.50</span>
          <span>1.00</span>
        </div>
      ) : (
        <div className="legend-ticks">
          <span>-{maxAbs.toFixed(2)}</span>
          <span>0.00</span>
          <span>+{maxAbs.toFixed(2)}</span>
        </div>
      )}

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          marginTop: '0.35rem',
          color: 'var(--text-muted)',
          fontSize: '0.72rem',
        }}
      >
        <span
          className="grid-swatch"
          style={{ backgroundColor: NEUTRAL_MISSING_HEX }}
        />
        <span>No measurement</span>
      </div>
    </div>
  )
}
