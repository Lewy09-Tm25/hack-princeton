import { Component as ReactComponent, type ReactNode, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import type { Component, ModelMap } from '../contracts'
import {
  detectWebGLSupport,
  formatComponentLabel,
  getActivePaint,
  useAppStore,
} from '../store'
import { CameraRig } from './CameraRig'
import { computeDivergingMaxAbs } from './colors'
import { Grid2D } from './Grid2D'
import { LayerBlock, TerminalBlock, usePrefersReducedMotion } from './LayerBlock'
import { Legend } from './Legend'
import { computeModelLayout } from './layout'
import { ResidualLine } from './ResidualLine'

interface CanvasErrorBoundaryProps {
  onError: () => void
  children: ReactNode
}

interface CanvasErrorBoundaryState {
  hasError: boolean
}

class CanvasErrorBoundary extends ReactComponent<
  CanvasErrorBoundaryProps,
  CanvasErrorBoundaryState
> {
  constructor(props: CanvasErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): CanvasErrorBoundaryState {
    return { hasError: true }
  }

  override componentDidCatch(): void {
    this.props.onError()
  }

  override render(): ReactNode {
    if (this.state.hasError) {
      return null
    }
    return this.props.children
  }
}

export interface ModelSceneProps {
  modelMap: ModelMap
}

export function ModelScene({ modelMap }: ModelSceneProps) {
  const currentCheckpointId = useAppStore((s) => s.currentCheckpointId)
  const paintByCheckpoint = useAppStore((s) => s.paintByCheckpoint)
  const focusedComponentIds = useAppStore((s) => s.focusedComponentIds)
  const selectedComponentId = useAppStore((s) => s.selectedComponentId)
  const expandedLayerIds = useAppStore((s) => s.expandedLayerIds)
  const viewMode = useAppStore((s) => s.viewMode)
  const isFakeMode = useAppStore((s) => s.isFakeMode)

  const selectComponent = useAppStore((s) => s.selectComponent)
  const toggleLayerExpanded = useAppStore((s) => s.toggleLayerExpanded)
  const setViewMode = useAppStore((s) => s.setViewMode)

  const [webglFailed, setWebglFailed] = useState<boolean>(() => !detectWebGLSupport())
  const reducedMotion = usePrefersReducedMotion()

  const activePaint = getActivePaint({ currentCheckpointId, paintByCheckpoint })
  const paintValues = activePaint?.values ?? null
  const scale = activePaint?.scale ?? 'sequential'
  const maxAbs = useMemo(
    () => (activePaint && activePaint.scale === 'diverging' ? computeDivergingMaxAbs(activePaint.values) : 1),
    [activePaint],
  )

  // The component with the strongest measurement gets a halo and a "Strongest signal" tag.
  // Prefer the check's own choice (it breaks ties the way the finding does); while a run is
  // still measuring, show the "Measuring" marker instead of guessing.
  const checkPeakId = useAppStore((s) => s.peakComponentId)
  const scanningComponentId = useAppStore((s) => s.scanningComponentId)
  const peakId = useMemo(() => {
    if (checkPeakId) return checkPeakId
    if (scanningComponentId) return null
    if (!paintValues) return null
    let best: string | null = null
    let bestVal = -Infinity
    for (const [id, raw] of Object.entries(paintValues)) {
      const v = scale === 'diverging' ? Math.abs(raw) : raw
      if (v > bestVal) {
        bestVal = v
        best = id
      }
    }
    return best
  }, [paintValues, scale, checkPeakId, scanningComponentId])

  const expandedSet = useMemo(() => new Set(expandedLayerIds), [expandedLayerIds])
  const focusedSet = useMemo(() => new Set(focusedComponentIds), [focusedComponentIds])

  const layout = useMemo(
    () => computeModelLayout(modelMap, expandedSet),
    [expandedSet, modelMap],
  )

  const selectedComp: Component | null = useMemo(() => {
    if (!selectedComponentId) return null
    return modelMap.components.find((c) => c.id === selectedComponentId) ?? null
  }, [modelMap, selectedComponentId])

  const handleToggleAndSelectLayer = (layerId: string) => {
    toggleLayerExpanded(layerId)
    selectComponent(layerId)
  }

  const handleSelectComponent = (componentId: string) => {
    selectComponent(componentId)
  }

  const show2D = viewMode === '2d' || webglFailed
  const selectedMeasuredValue =
    selectedComponentId && paintValues ? paintValues[selectedComponentId] : undefined

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {selectedComp && (
        <div className="component-detail-card" aria-label="Selected component details">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '0.4rem',
            }}
          >
            <strong>{formatComponentLabel(selectedComp.id)}</strong>
            <button
              type="button"
              className="btn"
              style={{ padding: '0.1rem 0.4rem', fontSize: '0.72rem' }}
              onClick={() => selectComponent(null)}
            >
              Clear
            </button>
          </div>

          <dl className="kv-list" style={{ marginTop: '0.45rem' }}>
            <dt>Component ID</dt>
            <dd>
              <code>{selectedComp.id}</code>
            </dd>
            <dt>Kind</dt>
            <dd>{selectedComp.kind}</dd>
            {selectedComp.module && (
              <>
                <dt>Module</dt>
                <dd>
                  <code style={{ fontSize: '0.74rem' }}>{selectedComp.module}</code>
                </dd>
              </>
            )}
            {selectedComp.n_heads !== undefined && selectedComp.n_heads !== null && (
              <>
                <dt>Heads</dt>
                <dd>{selectedComp.n_heads}</dd>
              </>
            )}
            <dt>Measurement</dt>
            <dd>
              {selectedMeasuredValue !== undefined && Number.isFinite(selectedMeasuredValue)
                ? `${selectedMeasuredValue.toFixed(4)} (${activePaint?.metric ?? ''})`
                : 'No measurement'}
            </dd>
          </dl>

          {selectedComp.kind === 'layer' && (
            <div style={{ marginTop: '0.5rem' }}>
              <button
                type="button"
                className="btn"
                onClick={() => toggleLayerExpanded(selectedComp.id)}
              >
                {expandedSet.has(selectedComp.id)
                  ? 'Collapse layer plates'
                  : 'Expand attention and MLP'}
              </button>
            </div>
          )}

          <div
            style={{
              marginTop: '0.45rem',
              fontSize: '0.73rem',
              color: 'var(--text-muted)',
            }}
          >
            Attached to chat input as context ({selectedComp.id}).
          </div>
        </div>
      )}

      {show2D ? (
        <Grid2D
          modelMap={modelMap}
          paint={activePaint}
          expandedLayerIds={expandedLayerIds}
          selectedComponentId={selectedComponentId}
          focusedComponentIds={focusedComponentIds}
          onToggleAndSelectLayer={handleToggleAndSelectLayer}
          onSelectComponent={handleSelectComponent}
        />
      ) : (
        <CanvasErrorBoundary
          onError={() => {
            setWebglFailed(true)
            setViewMode('2d')
          }}
        >
          <Canvas
            camera={{ position: [5.5, layout.totalHeight * 0.55, 8.5], fov: 42 }}
            onPointerMissed={() => {
              // Keep current selection unless explicitly cleared or re-selected
            }}
          >
            <ambientLight intensity={0.78} />
            <directionalLight position={[8, 18, 12]} intensity={0.95} />
            <directionalLight position={[-8, 6, -8]} intensity={0.35} />

            <ResidualLine
              start={layout.residualLine.start}
              end={layout.residualLine.end}
            />

            {layout.embedBlock && (
              <TerminalBlock
                block={layout.embedBlock}
                paintValues={paintValues}
                scale={scale}
                maxAbs={maxAbs}
                selectedComponentId={selectedComponentId}
                focusedSet={focusedSet}
                peakId={peakId}
                scanningId={scanningComponentId}
                reducedMotion={reducedMotion}
                onSelect={handleSelectComponent}
              />
            )}

            {layout.layers.map((layer) => (
              <LayerBlock
                key={layer.id}
                layer={layer}
                paintValues={paintValues}
                scale={scale}
                maxAbs={maxAbs}
                selectedComponentId={selectedComponentId}
                focusedSet={focusedSet}
                peakId={peakId}
                scanningId={scanningComponentId}
                reducedMotion={reducedMotion}
                onToggleAndSelectLayer={handleToggleAndSelectLayer}
                onSelectSubcomponent={handleSelectComponent}
              />
            ))}

            {layout.unembedBlock && (
              <TerminalBlock
                block={layout.unembedBlock}
                paintValues={paintValues}
                scale={scale}
                maxAbs={maxAbs}
                selectedComponentId={selectedComponentId}
                focusedSet={focusedSet}
                peakId={peakId}
                scanningId={scanningComponentId}
                reducedMotion={reducedMotion}
                onSelect={handleSelectComponent}
              />
            )}

            <CameraRig
              layout={layout}
              modelId={modelMap.model_id}
              focusedComponentIds={focusedComponentIds}
              reducedMotion={reducedMotion}
            />
          </Canvas>
        </CanvasErrorBoundary>
      )}

      <Legend paint={activePaint} isPlaceholder={isFakeMode} />
    </div>
  )
}
