import { useMemo } from 'react'
import type { Component, ModelMap } from '../contracts'
import type { PaintEntry } from '../store'
import {
  NEUTRAL_MISSING_HEX,
  colorHexForValue,
  computeDivergingMaxAbs,
} from './colors'

export interface Grid2DProps {
  modelMap: ModelMap
  paint: PaintEntry | null
  expandedLayerIds: string[]
  selectedComponentId: string | null
  focusedComponentIds: string[]
  onToggleAndSelectLayer: (layerId: string) => void
  onSelectComponent: (componentId: string) => void
}

function formatMetricVal(val: number | undefined): string {
  if (val === undefined || !Number.isFinite(val)) {
    return 'no measurement'
  }
  return val.toFixed(3)
}

export function Grid2D({
  modelMap,
  paint,
  expandedLayerIds,
  selectedComponentId,
  focusedComponentIds,
  onToggleAndSelectLayer,
  onSelectComponent,
}: Grid2DProps) {
  const byId = useMemo(() => {
    const map = new Map<string, Component>()
    for (const c of modelMap.components) {
      map.set(c.id, c)
    }
    return map
  }, [modelMap])

  const expandedSet = useMemo(() => new Set(expandedLayerIds), [expandedLayerIds])
  const focusedSet = useMemo(() => new Set(focusedComponentIds), [focusedComponentIds])

  const scale = paint?.scale ?? 'sequential'
  const maxAbs = useMemo(
    () => (paint && paint.scale === 'diverging' ? computeDivergingMaxAbs(paint.values) : 1),
    [paint],
  )

  const colorOf = (id: string): string => {
    if (!paint) return NEUTRAL_MISSING_HEX
    return colorHexForValue(paint.values[id], scale, maxAbs)
  }

  const valOf = (id: string): number | undefined => {
    if (!paint) return undefined
    return paint.values[id]
  }

  const embedComp = modelMap.components.find((c) => c.kind === 'embedding') ?? null
  const unembedComp = modelMap.components.find((c) => c.kind === 'unembedding') ?? null
  const layersDesc = useMemo(
    () =>
      modelMap.components
        .filter((c) => c.kind === 'layer')
        .slice()
        .sort((a, b) => (b.index ?? 0) - (a.index ?? 0)),
    [modelMap],
  )

  return (
    <div className="grid-2d-container" aria-label="2D model component grid">
      <table className="grid-2d-table">
        <thead>
          <tr>
            <th>Layer / block</th>
            <th>Attention</th>
            <th>MLP</th>
            <th>Other components</th>
          </tr>
        </thead>
        <tbody>
          {unembedComp && (
            <tr>
              <td>
                <button
                  type="button"
                  className={`grid-cell-btn ${selectedComponentId === unembedComp.id ? 'selected' : ''}`}
                  onClick={() => onSelectComponent(unembedComp.id)}
                >
                  <span
                    className="grid-swatch"
                    style={{ backgroundColor: colorOf(unembedComp.id) }}
                  />
                  <span>Unembedding ({unembedComp.id})</span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    [{formatMetricVal(valOf(unembedComp.id))}]
                  </span>
                </button>
              </td>
              <td>—</td>
              <td>—</td>
              <td>{focusedSet.has(unembedComp.id) ? 'Focused' : '—'}</td>
            </tr>
          )}

          {layersDesc.map((layer) => {
            const children = (layer.children ?? [])
              .map((cid) => byId.get(cid))
              .filter((c): c is Component => c !== undefined)
            const attn = children.find((c) => c.kind === 'attention') ?? null
            const mlp = children.find((c) => c.kind === 'mlp') ?? null
            const others = children.filter(
              (c) => c.kind !== 'attention' && c.kind !== 'mlp',
            )
            const isExpanded = expandedSet.has(layer.id)
            const heads = attn
              ? (attn.children ?? [])
                  .map((hid) => byId.get(hid))
                  .filter((c): c is Component => c !== undefined && c.kind === 'attention_head')
              : []

            return (
              <tr key={layer.id}>
                <td>
                  <button
                    type="button"
                    className={`grid-cell-btn ${selectedComponentId === layer.id ? 'selected' : ''}`}
                    onClick={() => onToggleAndSelectLayer(layer.id)}
                  >
                    <span
                      className="grid-swatch"
                      style={{ backgroundColor: colorOf(layer.id) }}
                    />
                    <span>
                      Layer {layer.index ?? ''} ({layer.id}) {isExpanded ? '▾' : '▸'}
                    </span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      [{formatMetricVal(valOf(layer.id))}]
                    </span>
                  </button>
                  {focusedSet.has(layer.id) && (
                    <span
                      className="badge badge-suggestive"
                      style={{ marginLeft: '0.4rem' }}
                    >
                      focused
                    </span>
                  )}
                </td>

                <td>
                  {attn ? (
                    <div>
                      <button
                        type="button"
                        className={`grid-cell-btn ${selectedComponentId === attn.id ? 'selected' : ''}`}
                        onClick={() => onSelectComponent(attn.id)}
                      >
                        <span
                          className="grid-swatch"
                          style={{ backgroundColor: colorOf(attn.id) }}
                        />
                        <span>{attn.id}</span>
                        <span style={{ color: 'var(--text-muted)' }}>
                          [{formatMetricVal(valOf(attn.id))}]
                        </span>
                      </button>
                      {isExpanded && heads.length > 0 && (
                        <div className="heads-strip" aria-label={`${attn.id} heads`}>
                          {heads.map((h) => (
                            <button
                              key={h.id}
                              type="button"
                              className="head-chip"
                              title={`${h.id}: ${formatMetricVal(valOf(h.id))}`}
                              style={{
                                backgroundColor: colorOf(h.id),
                                outline:
                                  selectedComponentId === h.id
                                    ? '2px solid var(--accent)'
                                    : focusedSet.has(h.id)
                                      ? '2px solid #d97706'
                                      : 'none',
                                color: '#fff',
                              }}
                              onClick={() => onSelectComponent(h.id)}
                            >
                              h{h.head ?? ''}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    '—'
                  )}
                </td>

                <td>
                  {mlp ? (
                    <button
                      type="button"
                      className={`grid-cell-btn ${selectedComponentId === mlp.id ? 'selected' : ''}`}
                      onClick={() => onSelectComponent(mlp.id)}
                    >
                      <span
                        className="grid-swatch"
                        style={{ backgroundColor: colorOf(mlp.id) }}
                      />
                      <span>{mlp.id}</span>
                      <span style={{ color: 'var(--text-muted)' }}>
                        [{formatMetricVal(valOf(mlp.id))}]
                      </span>
                    </button>
                  ) : (
                    '—'
                  )}
                </td>

                <td>
                  {others.length > 0 ? (
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                      {others.map((o) => (
                        <button
                          key={o.id}
                          type="button"
                          className={`grid-cell-btn ${selectedComponentId === o.id ? 'selected' : ''}`}
                          onClick={() => onSelectComponent(o.id)}
                        >
                          <span
                            className="grid-swatch"
                            style={{ backgroundColor: colorOf(o.id) }}
                          />
                          <span>{o.id}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    '—'
                  )}
                </td>
              </tr>
            )
          })}

          {embedComp && (
            <tr>
              <td>
                <button
                  type="button"
                  className={`grid-cell-btn ${selectedComponentId === embedComp.id ? 'selected' : ''}`}
                  onClick={() => onSelectComponent(embedComp.id)}
                >
                  <span
                    className="grid-swatch"
                    style={{ backgroundColor: colorOf(embedComp.id) }}
                  />
                  <span>Embedding ({embedComp.id})</span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    [{formatMetricVal(valOf(embedComp.id))}]
                  </span>
                </button>
              </td>
              <td>—</td>
              <td>—</td>
              <td>{focusedSet.has(embedComp.id) ? 'Focused' : '—'}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
