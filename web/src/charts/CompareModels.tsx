import { useEffect, useMemo, useRef, useState } from 'react'
import type { ResultRecord } from '../contracts'
import { OKABE_ITO, renderPlotlySafely } from './plotlyClient'

export interface ModelComparisonPair {
  id: string
  label: string
  componentId: string
  metric: string
  modelValue: number
  referenceValue: number
  delta: number
  notes: string | null
  isPlaceholder: boolean
}

export interface CompareModelsProps {
  results: ResultRecord[]
  modelLabel?: string
  referenceLabel?: string
}

const REFERENCE_CONTROL_KINDS = new Set([
  'reference_model',
  'clean_reference_model',
  'clean_model',
  'base_model',
])

const DUAL_METRIC_PAIRS: ReadonlyArray<[string, string, string]> = [
  ['suspect_probe_auroc', 'clean_probe_auroc', 'probe_auroc'],
  ['suspect_probe_accuracy', 'clean_probe_accuracy', 'probe_accuracy'],
  ['finetuned_flip_rate', 'reference_flip_rate', 'flip_rate'],
  ['suspect_refusal_rate', 'reference_refusal_rate', 'refusal_rate'],
]

export function extractModelComparisonPairs(
  results: readonly ResultRecord[],
): ModelComparisonPair[] {
  const pairs: ModelComparisonPair[] = []
  const seenKeys = new Set<string>()

  for (const row of results) {
    const ctrl = row.control
    if (
      ctrl &&
      REFERENCE_CONTROL_KINDS.has(ctrl.kind) &&
      ctrl.value !== undefined &&
      ctrl.value !== null &&
      Number.isFinite(ctrl.value) &&
      Number.isFinite(row.value)
    ) {
      const key = `${row.component_id}:${row.metric}`
      if (seenKeys.has(key)) continue
      seenKeys.add(key)
      pairs.push({
        id: key,
        label:
          row.component_id === 'model'
            ? row.metric
            : `${row.metric} (${row.component_id})`,
        componentId: row.component_id,
        metric: row.metric,
        modelValue: row.value,
        referenceValue: ctrl.value,
        delta: row.value - ctrl.value,
        notes: ctrl.notes ?? null,
        isPlaceholder: Boolean(row.is_placeholder),
      })
    }
  }

  for (const [suspectMetric, cleanMetric, displayMetric] of DUAL_METRIC_PAIRS) {
    const suspectRows = results.filter((r) => r.metric === suspectMetric)
    for (const sRow of suspectRows) {
      const cRow = results.find(
        (r) =>
          r.metric === cleanMetric &&
          r.component_id === sRow.component_id &&
          r.checkpoint_id === sRow.checkpoint_id,
      )
      if (!cRow || !Number.isFinite(sRow.value) || !Number.isFinite(cRow.value)) continue
      const key = `${sRow.component_id}:${displayMetric}`
      if (seenKeys.has(key)) continue
      seenKeys.add(key)
      pairs.push({
        id: key,
        label:
          sRow.component_id === 'model'
            ? displayMetric
            : `${displayMetric} (${sRow.component_id})`,
        componentId: sRow.component_id,
        metric: displayMetric,
        modelValue: sRow.value,
        referenceValue: cRow.value,
        delta: sRow.value - cRow.value,
        notes: null,
        isPlaceholder: Boolean(sRow.is_placeholder || cRow.is_placeholder),
      })
    }
  }

  return pairs
}

export function CompareModels({
  results,
  modelLabel = 'Investigated model',
  referenceLabel = 'Reference / control model',
}: CompareModelsProps) {
  const pairs = useMemo(() => extractModelComparisonPairs(results), [results])
  const plotContainerRef = useRef<HTMLDivElement | null>(null)
  const [plotlyReady, setPlotlyReady] = useState(false)

  useEffect(() => {
    const el = plotContainerRef.current
    if (!el || pairs.length === 0) {
      return
    }
    let cancelled = false
    let cleanup: (() => void) | null = null

    const labels = pairs.map((p) => p.label)
    const traces: Array<Record<string, unknown>> = [
      {
        type: 'bar',
        name: modelLabel,
        x: labels,
        y: pairs.map((p) => p.modelValue),
        marker: { color: OKABE_ITO.blue },
      },
      {
        type: 'bar',
        name: referenceLabel,
        x: labels,
        y: pairs.map((p) => p.referenceValue),
        marker: { color: OKABE_ITO.orange },
      },
    ]

    void renderPlotlySafely(el, traces, {
      barmode: 'group',
      height: 210,
      yaxis: { title: { text: 'Value' } },
      legend: { orientation: 'h', y: 1.16, x: 0 },
    }).then((teardown) => {
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
  }, [pairs, modelLabel, referenceLabel])

  const isPlaceholder = pairs.some((p) => p.isPlaceholder)

  return (
    <section className="chart-card" aria-label="Model versus reference comparison">
      <div className="chart-card-header">
        <div>
          <h4 className="chart-card-title">Model vs. reference comparison</h4>
          <span className="chart-card-subtitle">
            Side-by-side metrics against the clean or base reference model.
          </span>
        </div>
        {isPlaceholder && (
          <span className="badge badge-placeholder">unverified / placeholder</span>
        )}
      </div>

      {pairs.length === 0 ? (
        <p className="chart-empty-note">
          No reference or clean comparison model was included in this run. Single-model runs
          compare against shuffled-label and random-direction controls instead.
        </p>
      ) : (
        <>
          <div
            ref={plotContainerRef}
            className="plotly-mount"
            style={{ display: plotlyReady ? 'block' : 'none' }}
            data-testid="compare-models-plotly"
          />

          <table className="evidence-table" aria-label="Model comparison metrics">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">{modelLabel}</th>
                <th scope="col">{referenceLabel}</th>
                <th scope="col">Difference</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((pair) => {
                const scaleMax = Math.max(1, Math.abs(pair.modelValue), Math.abs(pair.referenceValue))
                const modelPct = Math.min(100, Math.max(0, (pair.modelValue / scaleMax) * 100))
                const refPct = Math.min(100, Math.max(0, (pair.referenceValue / scaleMax) * 100))
                const signedDelta =
                  pair.delta >= 0 ? `+${pair.delta.toFixed(2)}` : pair.delta.toFixed(2)
                return (
                  <tr key={pair.id}>
                    <td>
                      <div>
                        <strong>{pair.label}</strong>
                      </div>
                      {pair.notes && <div className="table-row-note">{pair.notes}</div>}
                    </td>
                    <td>
                      <div className="bar-cell">
                        <span
                          className="bar-fill"
                          style={{
                            width: `${modelPct.toFixed(1)}%`,
                            backgroundColor: OKABE_ITO.blue,
                          }}
                          aria-hidden="true"
                        />
                        <span>{pair.modelValue.toFixed(2)}</span>
                      </div>
                    </td>
                    <td>
                      <div className="bar-cell">
                        <span
                          className="bar-fill"
                          style={{
                            width: `${refPct.toFixed(1)}%`,
                            backgroundColor: OKABE_ITO.orange,
                          }}
                          aria-hidden="true"
                        />
                        <span>{pair.referenceValue.toFixed(2)}</span>
                      </div>
                    </td>
                    <td>
                      <code>{signedDelta}</code>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </>
      )}
    </section>
  )
}
