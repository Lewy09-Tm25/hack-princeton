import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Series, SeriesItem } from '../contracts'
import { useAppStore, type PaintEntry } from '../store'
import { NEUTRAL_MISSING_HEX, colorHexForValue, computeDivergingMaxAbs } from './colors'

export const NO_MEASUREMENTS_CHECKPOINT_TEXT =
  'No measurements at this checkpoint — shown in neutral gray'

export interface TimelineProps {
  series?: Series | null
  currentCheckpointId?: string | null
  paintByCheckpoint?: Record<string, PaintEntry>
  onSelectCheckpoint?: (checkpointId: string) => void
  stepIntervalMs?: number
}

export interface ResolvedTimelineState {
  orderedItems: SeriesItem[]
  activeIndex: number
  activeItem: SeriesItem | null
  activePaint: PaintEntry | null
  measuredComponentCount: number
  hasMeasurements: boolean
  headerLabel: string
}

export function sortSeriesItems(items: readonly SeriesItem[]): SeriesItem[] {
  return [...items].sort((a, b) => a.order - b.order)
}

export function resolveTimelineState(
  series: Series | null,
  currentCheckpointId: string | null,
  paintByCheckpoint: Record<string, PaintEntry>,
): ResolvedTimelineState {
  if (!series || series.items.length === 0) {
    return {
      orderedItems: [],
      activeIndex: 0,
      activeItem: null,
      activePaint: null,
      measuredComponentCount: 0,
      hasMeasurements: false,
      headerLabel: '',
    }
  }

  const orderedItems = sortSeriesItems(series.items)
  const foundIdx = orderedItems.findIndex((item) => item.checkpoint_id === currentCheckpointId)
  const activeIndex = foundIdx >= 0 ? foundIdx : 0
  const activeItem = orderedItems[activeIndex] ?? null
  const activePaint = activeItem ? (paintByCheckpoint[activeItem.checkpoint_id] ?? null) : null
  const measuredComponentCount = activePaint
    ? Object.values(activePaint.values).filter(
        (v) => typeof v === 'number' && Number.isFinite(v),
      ).length
    : 0

  return {
    orderedItems,
    activeIndex,
    activeItem,
    activePaint,
    measuredComponentCount,
    hasMeasurements: measuredComponentCount > 0,
    headerLabel:
      series.kind === 'pretraining' ? 'Pretraining checkpoints' : 'Training stages',
  }
}

export function resolveCheckpointComponentHex(
  paintByCheckpoint: Record<string, PaintEntry>,
  checkpointId: string | null,
  componentId: string,
): string {
  if (!checkpointId) {
    return NEUTRAL_MISSING_HEX
  }
  const paint = paintByCheckpoint[checkpointId]
  if (!paint) {
    return NEUTRAL_MISSING_HEX
  }
  const rawValue = paint.values[componentId]
  if (rawValue === undefined || rawValue === null || !Number.isFinite(rawValue)) {
    return NEUTRAL_MISSING_HEX
  }
  const maxAbs = paint.scale === 'diverging' ? computeDivergingMaxAbs(paint.values) : 1
  return colorHexForValue(rawValue, paint.scale, maxAbs)
}

export function stepTimelineIndex(
  currentIndex: number,
  totalItems: number,
  direction: 'prev' | 'next',
): number {
  if (totalItems <= 0) return 0
  if (direction === 'prev') {
    return Math.max(0, currentIndex - 1)
  }
  return Math.min(totalItems - 1, currentIndex + 1)
}

export function startTimelinePlaybackInterval(options: {
  orderedItems: readonly SeriesItem[]
  getCurrentCheckpointId: () => string | null
  onSelectCheckpoint: (checkpointId: string) => void
  onStop: () => void
  stepIntervalMs?: number
}): () => void {
  const {
    orderedItems,
    getCurrentCheckpointId,
    onSelectCheckpoint,
    onStop,
    stepIntervalMs = 900,
  } = options
  if (orderedItems.length <= 1) {
    return () => {}
  }
  const timer = setInterval(() => {
    const latestId = getCurrentCheckpointId()
    const currentIdx = orderedItems.findIndex((item) => item.checkpoint_id === latestId)
    const safeIdx = currentIdx >= 0 ? currentIdx : 0
    if (safeIdx >= orderedItems.length - 1) {
      onStop()
      return
    }
    const nextItem = orderedItems[safeIdx + 1]
    if (nextItem) {
      onSelectCheckpoint(nextItem.checkpoint_id)
      if (safeIdx + 1 >= orderedItems.length - 1) {
        onStop()
      }
    }
  }, stepIntervalMs)

  return () => {
    clearInterval(timer)
  }
}

export function Timeline({
  series: seriesProp,
  currentCheckpointId: checkpointIdProp,
  paintByCheckpoint: paintProp,
  onSelectCheckpoint,
  stepIntervalMs = 900,
}: TimelineProps = {}) {
  const storeSeries = useAppStore((s) => s.series)
  const storeCheckpointId = useAppStore((s) => s.currentCheckpointId)
  const storePaintByCheckpoint = useAppStore((s) => s.paintByCheckpoint)
  const storeSetCheckpointId = useAppStore((s) => s.setCurrentCheckpointId)

  const series = seriesProp !== undefined ? seriesProp : storeSeries
  const currentCheckpointId =
    checkpointIdProp !== undefined ? checkpointIdProp : storeCheckpointId
  const paintByCheckpoint = paintProp !== undefined ? paintProp : storePaintByCheckpoint

  const selectCheckpoint = useCallback(
    (checkpointId: string) => {
      if (onSelectCheckpoint) {
        onSelectCheckpoint(checkpointId)
      } else {
        storeSetCheckpointId(checkpointId)
      }
    },
    [onSelectCheckpoint, storeSetCheckpointId],
  )

  const [isPlaying, setIsPlaying] = useState(false)

  const timelineState = useMemo(
    () => resolveTimelineState(series, currentCheckpointId, paintByCheckpoint),
    [series, currentCheckpointId, paintByCheckpoint],
  )

  const {
    orderedItems,
    activeIndex,
    activeItem,
    activePaint,
    measuredComponentCount,
    hasMeasurements,
    headerLabel,
  } = timelineState

  useEffect(() => {
    if (!isPlaying || orderedItems.length <= 1) {
      return
    }
    return startTimelinePlaybackInterval({
      orderedItems,
      getCurrentCheckpointId: () =>
        checkpointIdProp !== undefined
          ? checkpointIdProp
          : useAppStore.getState().currentCheckpointId,
      onSelectCheckpoint: selectCheckpoint,
      onStop: () => setIsPlaying(false),
      stepIntervalMs,
    })
  }, [isPlaying, orderedItems, checkpointIdProp, selectCheckpoint, stepIntervalMs])

  if (!series || orderedItems.length === 0 || !activeItem) {
    return null
  }

  const handleTogglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false)
      return
    }
    if (activeIndex >= orderedItems.length - 1 && orderedItems.length > 1) {
      selectCheckpoint(orderedItems[0].checkpoint_id)
    }
    setIsPlaying(true)
  }

  const handleStep = (direction: 'prev' | 'next') => {
    setIsPlaying(false)
    const nextIdx = stepTimelineIndex(activeIndex, orderedItems.length, direction)
    const target = orderedItems[nextIdx]
    if (target) {
      selectCheckpoint(target.checkpoint_id)
    }
  }

  const handleRangeChange = (rawIndex: number) => {
    setIsPlaying(false)
    const clamped = Math.max(0, Math.min(orderedItems.length - 1, Math.round(rawIndex)))
    const target = orderedItems[clamped]
    if (target) {
      selectCheckpoint(target.checkpoint_id)
    }
  }

  return (
    <div className="timeline-bar" role="region" aria-label="Checkpoint timeline">
      <div className="timeline-top-row">
        <div className="timeline-title-group">
          <strong>{headerLabel}</strong>
          <span className="timeline-family">{series.model_family}</span>
          <span className="status-pill">
            Active: {activeItem.label} ({activeIndex + 1} of {orderedItems.length})
          </span>
        </div>

        <div className="timeline-controls">
          <button
            type="button"
            className="btn"
            aria-label="Previous checkpoint"
            disabled={activeIndex <= 0}
            onClick={() => handleStep('prev')}
          >
            Previous
          </button>
          <button
            type="button"
            className={`btn ${isPlaying ? 'btn-active' : 'btn-primary'}`}
            aria-label={
              isPlaying ? 'Pause checkpoint timeline' : 'Play checkpoint timeline'
            }
            disabled={orderedItems.length <= 1}
            onClick={handleTogglePlay}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            className="btn"
            aria-label="Next checkpoint"
            disabled={activeIndex >= orderedItems.length - 1}
            onClick={() => handleStep('next')}
          >
            Next
          </button>
        </div>
      </div>

      <div className="timeline-slider-wrap">
        <input
          type="range"
          className="timeline-range"
          aria-label="Checkpoint timeline slider"
          aria-valuemin={0}
          aria-valuemax={orderedItems.length - 1}
          aria-valuenow={activeIndex}
          aria-valuetext={activeItem.label}
          min={0}
          max={Math.max(0, orderedItems.length - 1)}
          step={1}
          value={activeIndex}
          onChange={(e) => handleRangeChange(Number(e.target.value))}
        />

        <div className="timeline-ticks" role="group" aria-label="Checkpoint steps">
          {orderedItems.map((item) => {
            const isSelected = item.checkpoint_id === activeItem.checkpoint_id
            const itemPaint = paintByCheckpoint[item.checkpoint_id]
            const itemHasData = Boolean(
              itemPaint &&
                Object.values(itemPaint.values).some(
                  (v) => typeof v === 'number' && Number.isFinite(v),
                ),
            )
            return (
              <button
                key={item.checkpoint_id}
                type="button"
                className={`timeline-tick-btn ${isSelected ? 'active' : ''} ${
                  itemHasData ? 'measured' : 'unmeasured'
                }`}
                aria-pressed={isSelected}
                aria-label={`Select checkpoint ${item.label}`}
                onClick={() => {
                  setIsPlaying(false)
                  selectCheckpoint(item.checkpoint_id)
                }}
              >
                <span
                  className="timeline-tick-dot"
                  style={{
                    backgroundColor: itemHasData ? '#0072B2' : NEUTRAL_MISSING_HEX,
                  }}
                />
                <span>{item.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      <div className="timeline-status-row" role="status" aria-live="polite">
        {hasMeasurements && activePaint ? (
          <span className="timeline-status-measured">
            Painted metric: <strong>{activePaint.metric}</strong> ({measuredComponentCount}{' '}
            {measuredComponentCount === 1 ? 'component' : 'components'} measured at{' '}
            {activeItem.label})
          </span>
        ) : (
          <span className="timeline-status-missing">
            <span
              className="grid-swatch"
              style={{ backgroundColor: NEUTRAL_MISSING_HEX }}
              aria-hidden="true"
            />
            <span>{NO_MEASUREMENTS_CHECKPOINT_TEXT}</span>
          </span>
        )}
      </div>
    </div>
  )
}
