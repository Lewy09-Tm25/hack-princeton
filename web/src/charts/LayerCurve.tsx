import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ResultRecord } from '../contracts'
import { useAppStore } from '../store'
import { OKABE_ITO, renderPlotlySafely } from './plotlyClient'

export interface LayerCurvePoint {
  layerIndex: number
  layerId: string
  value: number
  controlValue: number | null
  controlKind: string | null
  isPlaceholder: boolean
}

export interface LayerCurveSeries {
  metric: string
  availableMetrics: string[]
  checkpointId: string | null
  points: LayerCurvePoint[]
  isPlaceholder: boolean
}

export interface LayerCurveProps {
  results: ResultRecord[]
  metric?: string
  checkpointId?: string | null
  focusedComponentIds?: string[]
  onFocusLayer?: (layerId: string, reason: string) => void
}

export function extractLayerCurveSeries(
  results: readonly ResultRecord[],
  preferredMetric?: string,
  checkpointId?: string | null,
): LayerCurveSeries {
  const layerRows = results.filter((r) => /^L\d+$/.test(r.component_id) && Number.isFinite(r.value))
  if (layerRows.length === 0) {
    return {
      metric: preferredMetric ?? 'probe_accuracy',
      availableMetrics: [],
      checkpointId: checkpointId ?? null,
      points: [],
      isPlaceholder: results.some((r) => r.is_placeholder === true),
    }
  }

  let checkpointFiltered = layerRows
  let resolvedCheckpoint: string | null = checkpointId ?? null
  if (checkpointId) {
    const exact = layerRows.filter((r) => r.checkpoint_id === checkpointId)
    if (exact.length > 0) {
      checkpointFiltered = exact
    }
  } else {
    const nonNullCheckpoints = Array.from(
      new Set(layerRows.map((r) => r.checkpoint_id).filter((c): c is string => Boolean(c))),
    )
    if (nonNullCheckpoints.length > 0) {
      resolvedCheckpoint = nonNullCheckpoints[nonNullCheckpoints.length - 1]
      checkpointFiltered = layerRows.filter((r) => r.checkpoint_id === resolvedCheckpoint)
    }
  }

  const countsByMetric = new Map<string, number>()
  for (const row of checkpointFiltered) {
    countsByMetric.set(row.metric, (countsByMetric.get(row.metric) ?? 0) + 1)
  }
  const availableMetrics = Array.from(countsByMetric.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([m]) => m)

  const chosenMetric =
    preferredMetric && countsByMetric.has(preferredMetric)
      ? preferredMetric
      : countsByMetric.has('probe_accuracy')
        ? 'probe_accuracy'
        : (availableMetrics[0] ?? 'probe_accuracy')

  const metricRows = checkpointFiltered.filter((r) => r.metric === chosenMetric)
  const byLayer = new Map<number, LayerCurvePoint>()
  for (const row of metricRows) {
    const match = /^L(\d+)$/.exec(row.component_id)
    if (!match) continue
    const layerIndex = Number(match[1])
    const ctrlVal =
      row.control?.value !== undefined &&
      row.control?.value !== null &&
      Number.isFinite(row.control.value)
        ? row.control.value
        : null
    byLayer.set(layerIndex, {
      layerIndex,
      layerId: row.component_id,
      value: row.value,
      controlValue: ctrlVal,
      controlKind: row.control?.kind ?? null,
      isPlaceholder: Boolean(row.is_placeholder),
    })
  }

  const points = Array.from(byLayer.values()).sort((a, b) => a.layerIndex - b.layerIndex)
  return {
    metric: chosenMetric,
    availableMetrics,
    checkpointId: resolvedCheckpoint,
    points,
    isPlaceholder: points.some((p) => p.isPlaceholder),
  }
}

export function triggerLayerFocus(
  layerId: string,
  metric: string,
  value: number,
  onFocusLayer?: (layerId: string, reason: string) => void,
): void {
  const reason = `Selected ${layerId} on ${metric} curve (${value.toFixed(2)})`
  if (onFocusLayer) {
    onFocusLayer(layerId, reason)
  } else {
    useAppStore.getState().focusLayerFromChart(layerId, reason)
  }
}

export function LayerCurve({
  results,
  metric: metricProp,
  checkpointId: checkpointIdProp,
  focusedComponentIds: focusedProp,
  onFocusLayer,
}: LayerCurveProps) {
  const storeFocusedIds = useAppStore((s) => s.focusedComponentIds)
  const storeCheckpointId = useAppStore((s) => s.currentCheckpointId)

  const focusedComponentIds = focusedProp !== undefined ? focusedProp : storeFocusedIds
  const activeCheckpointId =
    checkpointIdProp !== undefined ? checkpointIdProp : storeCheckpointId

  const [selectedMetric, setSelectedMetric] = useState<string | undefined>(metricProp)
  const [plotlyReady, setPlotlyReady] = useState(false)
  const plotContainerRef = useRef<HTMLDivElement | null>(null)

  const series = useMemo(
    () => extractLayerCurveSeries(results, metricProp ?? selectedMetric, activeCheckpointId),
    [results, metricProp, selectedMetric, activeCheckpointId],
  )

  const handlePointFocus = useCallback(
    (point: LayerCurvePoint) => {
      triggerLayerFocus(point.layerId, series.metric, point.value, onFocusLayer)
    },
    [series.metric, onFocusLayer],
  )

  useEffect(() => {
    const el = plotContainerRef.current
    if (!el || series.points.length === 0) {
      return
    }
    let cancelled = false
    let cleanup: (() => void) | null = null

    const xLabels = series.points.map((p) => p.layerId)
    const yValues = series.points.map((p) => p.value)
    const controlPoints = series.points.filter((p) => p.controlValue !== null)

    const traces: Array<Record<string, unknown>> = []
    if (controlPoints.length > 0) {
      const ctrlUpper = controlPoints.map((p) => Math.min(1, (p.controlValue ?? 0.5) + 0.03))
      const ctrlLower = controlPoints.map((p) => Math.max(0, (p.controlValue ?? 0.5) - 0.03))
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Control upper bound',
        x: controlPoints.map((p) => p.layerId),
        y: ctrlUpper,
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
      })
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: `Control (${controlPoints[0].controlKind ?? 'baseline'})`,
        x: controlPoints.map((p) => p.layerId),
        y: ctrlLower,
        fill: 'tonexty',
        fillcolor: 'rgba(86, 180, 233, 0.18)',
        line: { width: 0 },
        hoverinfo: 'skip',
      })
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Control baseline',
        x: controlPoints.map((p) => p.layerId),
        y: controlPoints.map((p) => p.controlValue),
        line: { color: '#5c5c57', width: 1.5, dash: 'dash' },
      })
    }

    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: series.metric,
      x: xLabels,
      y: yValues,
      customdata: xLabels,
      line: { color: OKABE_ITO.blue, width: 2.5 },
      marker: {
        size: series.points.map((p) => (focusedComponentIds.includes(p.layerId) ? 11 : 7)),
        color: series.points.map((p) =>
          focusedComponentIds.includes(p.layerId) ? OKABE_ITO.vermilion : OKABE_ITO.blue,
        ),
      },
    })

    void renderPlotlySafely(
      el,
      traces,
      {
        height: 220,
        xaxis: { title: { text: 'Layer' } },
        yaxis: { title: { text: series.metric } },
        legend: { orientation: 'h', y: 1.16, x: 0 },
      },
      undefined,
      (clickEvt) => {
        const pt = clickEvt.points?.[0]
        const clickedId =
          typeof pt?.customdata === 'string'
            ? pt.customdata
            : typeof pt?.x === 'string'
              ? pt.x
              : null
        if (!clickedId) return
        const found = series.points.find((p) => p.layerId === clickedId)
        if (found) {
          handlePointFocus(found)
        }
      },
    ).then((teardown) => {
      if (cancelled) {
        teardown?.()
        return
      }
      cleanup = teardown
      setPlotlyReady(Boolean(teardown))
    })

    return () => {
      cancelled = true
      cleanup?.()
    }
  }, [series, focusedComponentIds, handlePointFocus])

  if (series.points.length === 0) {
    return (
      <section className="chart-card" aria-label="Per-layer metric curve">
        <div className="chart-card-header">
          <h4 className="chart-card-title">Per-layer curve</h4>
        </div>
        <p className="chart-empty-note">
          No per-layer measurements (`L0`, `L1`, ...) were recorded for this run.
        </p>
      </section>
    )
  }

  const minVal = Math.min(
    0,
    ...series.points.map((p) => p.value),
    ...series.points.map((p) => p.controlValue ?? p.value),
  )
  const maxVal = Math.max(
    1,
    ...series.points.map((p) => p.value),
    ...series.points.map((p) => p.controlValue ?? p.value),
  )
  const span = maxVal - minVal > 0 ? maxVal - minVal : 1
  const svgW = 520
  const svgH = 150
  const padL = 36
  const padR = 14
  const padT = 14
  const padB = 24
  const plotW = svgW - padL - padR
  const plotH = svgH - padT - padB

  const xForIdx = (i: number) =>
    series.points.length <= 1
      ? padL + plotW / 2
      : padL + (i / (series.points.length - 1)) * plotW
  const yForVal = (v: number) => padT + (1 - (v - minVal) / span) * plotH

  const polylinePoints = series.points
    .map((p, idx) => `${xForIdx(idx).toFixed(1)},${yForVal(p.value).toFixed(1)}`)
    .join(' ')

  const controlPts = series.points.filter((p) => p.controlValue !== null)
  const controlLinePoints = controlPts
    .map((p) => {
      const idx = series.points.indexOf(p)
      return `${xForIdx(idx).toFixed(1)},${yForVal(p.controlValue ?? 0.5).toFixed(1)}`
    })
    .join(' ')

  return (
    <section className="chart-card" aria-label="Per-layer metric curve">
      <div className="chart-card-header">
        <div>
          <h4 className="chart-card-title">
            Per-layer curve: <code>{series.metric}</code>
            {series.checkpointId ? ` (${series.checkpointId})` : ''}
          </h4>
          <span className="chart-card-subtitle">
            Click a layer point to focus that layer in the model view.
          </span>
        </div>
        <div className="badge-row">
          {series.availableMetrics.length > 1 && metricProp === undefined && (
            <select
              aria-label="Layer curve metric"
              className="select-compact"
              value={series.metric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              {series.availableMetrics.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          )}
          {series.isPlaceholder && (
            <span className="badge badge-placeholder">unverified / placeholder</span>
          )}
        </div>
      </div>

      <div
        ref={plotContainerRef}
        className="plotly-mount"
        style={{ display: plotlyReady ? 'block' : 'none' }}
        data-testid="layer-curve-plotly"
      />

      {!plotlyReady && (
        <svg
          className="chart-fallback-svg"
          viewBox={`0 0 ${svgW} ${svgH}`}
          role="img"
          aria-label={`Layer curve for ${series.metric}`}
        >
          <rect x={padL} y={padT} width={plotW} height={plotH} fill="#ffffff" stroke="#dcdcd5" />
          {controlPts.length > 1 && (
            <polyline
              fill="none"
              stroke="#5c5c57"
              strokeWidth="1.5"
              strokeDasharray="4 3"
              points={controlLinePoints}
            />
          )}
          {series.points.length > 1 && (
            <polyline
              fill="none"
              stroke={OKABE_ITO.blue}
              strokeWidth="2.2"
              points={polylinePoints}
            />
          )}
          {series.points.map((pt, idx) => {
            const isFocused = focusedComponentIds.includes(pt.layerId)
            return (
              <circle
                key={pt.layerId}
                cx={xForIdx(idx)}
                cy={yForVal(pt.value)}
                r={isFocused ? 5 : 3.5}
                fill={isFocused ? OKABE_ITO.vermilion : OKABE_ITO.blue}
              />
            )
          })}
          <text x={4} y={padT + 9} fontSize="10" fill="#5c5c57">
            {maxVal.toFixed(2)}
          </text>
          <text x={4} y={padT + plotH} fontSize="10" fill="#5c5c57">
            {minVal.toFixed(2)}
          </text>
        </svg>
      )}

      <div className="layer-point-chips" role="group" aria-label="Focus layer from curve">
        {series.points.map((pt) => {
          const isFocused = focusedComponentIds.includes(pt.layerId)
          return (
            <button
              key={pt.layerId}
              type="button"
              className={`layer-chip-btn ${isFocused ? 'focused' : ''}`}
              aria-pressed={isFocused}
              aria-label={`Focus ${pt.layerId} (${pt.value.toFixed(2)})`}
              onClick={() => handlePointFocus(pt)}
            >
              <span>{pt.layerId}</span>
              <span className="layer-chip-val">{pt.value.toFixed(2)}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
