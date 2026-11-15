import { useEffect, useRef, useState } from 'react'
import { Edges, Html } from '@react-three/drei'
import { useFrame, type ThreeEvent } from '@react-three/fiber'
import { useSpring } from '@react-spring/three'
import * as THREE from 'three'
import type { PaintScale } from '../contracts'
import { AttentionHeads } from './AttentionHeads'
import { colorRgb01ForValue } from './colors'
import type { LayerLayout, TerminalBlockLayout } from './layout'

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return false
    }
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  })

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return
    }
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const listener = (e: MediaQueryListEvent) => setReduced(e.matches)
    mq.addEventListener('change', listener)
    return () => mq.removeEventListener('change', listener)
  }, [])

  return reduced
}

interface AnimatedColorBoxProps {
  size: [number, number, number]
  position: [number, number, number]
  targetRgb: [number, number, number]
  reducedMotion: boolean
  isSelected: boolean
  isFocused: boolean
  /** The component with the strongest measurement: drawn with a halo and a tag. */
  isPeak?: boolean
  /** The layer a running check is measuring right now. */
  isScanning?: boolean
  label: string
  labelPosition?: [number, number, number]
  onClick: (e: ThreeEvent<MouseEvent>) => void
}

// Blue stands apart from the yellow-to-red measurement colours.
const PEAK_COLOR = '#2f80ff'
// The layer being measured during a run.
const SCAN_COLOR = '#ffffff'

function AnimatedColorBox({
  size,
  position,
  targetRgb,
  reducedMotion,
  isSelected,
  isFocused,
  isPeak = false,
  isScanning = false,
  label,
  labelPosition,
  onClick,
}: AnimatedColorBoxProps) {
  const haloRef = useRef<THREE.MeshBasicMaterial | null>(null)
  const [hovered, setHovered] = useState(false)
  const groupRef = useRef<THREE.Group | null>(null)
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null)

  const spring = useSpring({
    pos: position,
    rgb: targetRgb,
    immediate: reducedMotion,
    config: { duration: reducedMotion ? 0 : 400 },
  })

  useFrame(() => {
    const grp = groupRef.current
    if (grp) {
      const [px, py, pz] = spring.pos.get()
      grp.position.set(px, py, pz)
    }
    const mat = materialRef.current
    if (mat) {
      const [r, g, b] = spring.rgb.get()
      mat.color.setRGB(r, g, b, THREE.SRGBColorSpace)
    }
    const halo = haloRef.current
    if (halo) {
      // Gentle pulse; steady when the user prefers reduced motion.
      halo.opacity = reducedMotion
        ? 0.3
        : 0.2 + 0.16 * (0.5 + 0.5 * Math.sin(performance.now() / 380))
    }
  })

  const edgeColor = isPeak
    ? PEAK_COLOR
    : isScanning
    ? SCAN_COLOR
    : isSelected
    ? '#1d5bd6'
    : isFocused
      ? '#d97706'
      : hovered
        ? '#52524e'
        : '#3d3d39'

  const showStrongOutline = isPeak || isScanning || isSelected || isFocused || hovered

  return (
    <group ref={groupRef} position={position}>
      <mesh
        onClick={onClick}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={size} />
        <meshStandardMaterial
          ref={materialRef}
          roughness={0.52}
          metalness={0.06}
        />
        <Edges
          scale={showStrongOutline ? 1.015 : 1.002}
          threshold={15}
          color={edgeColor}
        />
      </mesh>
      {isScanning && !isPeak && (
        <>
          <mesh raycast={() => null}>
            <boxGeometry args={[size[0] * 1.06 + 0.1, size[1] * 1.25 + 0.1, size[2] * 1.06 + 0.1]} />
            <meshBasicMaterial color={SCAN_COLOR} transparent opacity={0.22} depthWrite={false} />
          </mesh>
          <Html
            position={[size[0] / 2 + 0.2, 0, size[2] / 2]}
            center={false}
            style={{
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              fontSize: '12px',
              fontWeight: 600,
              color: '#1a1a19',
              background: SCAN_COLOR,
              padding: '2px 8px',
              borderRadius: '999px',
              transform: 'translate(0, -50%)',
              boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
            }}
          >
            Measuring…
          </Html>
        </>
      )}
      {isPeak && (
        <>
          <mesh raycast={() => null}>
            <boxGeometry args={[size[0] * 1.08 + 0.12, size[1] * 1.3 + 0.12, size[2] * 1.08 + 0.12]} />
            <meshBasicMaterial
              ref={haloRef}
              color={PEAK_COLOR}
              transparent
              opacity={0.3}
              depthWrite={false}
            />
          </mesh>
          <Html
            position={[size[0] / 2 + 0.2, 0, size[2] / 2]}
            center={false}
            style={{
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
              fontSize: '12px',
              fontWeight: 600,
              color: '#ffffff',
              background: PEAK_COLOR,
              padding: '2px 8px',
              borderRadius: '999px',
              transform: 'translate(0, -50%)',
              boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
            }}
          >
            Strongest signal
          </Html>
        </>
      )}
      <Html
        position={labelPosition ?? [-size[0] / 2 - 0.15, 0, size[2] / 2]}
        center={false}
        style={{
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
          fontSize: '11px',
          fontWeight: isPeak || isSelected || isFocused ? 600 : 400,
          color: 'var(--text-primary)',
          background: 'color-mix(in srgb, var(--bg-panel) 82%, transparent)',
          padding: '1px 4px',
          borderRadius: '3px',
          transform: 'translate(-100%, -50%)',
        }}
      >
        {label}
      </Html>
    </group>
  )
}

export interface TerminalBlockProps {
  block: TerminalBlockLayout
  paintValues: Record<string, number> | null
  scale: PaintScale
  maxAbs: number
  selectedComponentId: string | null
  focusedSet: ReadonlySet<string>
  peakId?: string | null
  scanningId?: string | null
  reducedMotion: boolean
  onSelect: (id: string) => void
}

export function TerminalBlock({
  block,
  paintValues,
  scale,
  maxAbs,
  selectedComponentId,
  focusedSet,
  peakId = null,
  scanningId = null,
  reducedMotion,
  onSelect,
}: TerminalBlockProps) {
  const val = paintValues ? paintValues[block.id] : undefined
  const rgb = colorRgb01ForValue(val, scale, maxAbs)

  return (
    <AnimatedColorBox
      size={block.size}
      position={block.position}
      targetRgb={rgb}
      reducedMotion={reducedMotion}
      isSelected={selectedComponentId === block.id}
      isFocused={focusedSet.has(block.id)}
      isPeak={peakId === block.id}
      isScanning={scanningId === block.id}
      label={`${block.label} (${block.id})`}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(block.id)
      }}
    />
  )
}

export interface LayerBlockProps {
  layer: LayerLayout
  paintValues: Record<string, number> | null
  scale: PaintScale
  maxAbs: number
  selectedComponentId: string | null
  focusedSet: ReadonlySet<string>
  peakId?: string | null
  scanningId?: string | null
  reducedMotion: boolean
  onToggleAndSelectLayer: (layerId: string) => void
  onSelectSubcomponent: (componentId: string) => void
}

export function LayerBlock({
  layer,
  paintValues,
  scale,
  maxAbs,
  selectedComponentId,
  focusedSet,
  peakId = null,
  scanningId = null,
  reducedMotion,
  onToggleAndSelectLayer,
  onSelectSubcomponent,
}: LayerBlockProps) {
  const layerVal = paintValues ? paintValues[layer.id] : undefined
  const layerRgb = colorRgb01ForValue(layerVal, scale, maxAbs)

  if (!layer.expanded) {
    return (
      <AnimatedColorBox
        size={layer.size}
        position={layer.position}
        targetRgb={layerRgb}
        reducedMotion={reducedMotion}
        isSelected={selectedComponentId === layer.id}
        isFocused={focusedSet.has(layer.id)}
        isPeak={peakId === layer.id}
        isScanning={scanningId === layer.id}
        label={`Layer ${layer.index} (${layer.id})`}
        onClick={(e) => {
          e.stopPropagation()
          onToggleAndSelectLayer(layer.id)
        }}
      />
    )
  }

  const attnPlate = layer.attentionPlate
  const mlpPlate = layer.mlpPlate

  const attnVal = attnPlate && paintValues ? paintValues[attnPlate.id] : undefined
  const attnRgb = colorRgb01ForValue(attnVal, scale, maxAbs)

  const mlpVal = mlpPlate && paintValues ? paintValues[mlpPlate.id] : undefined
  const mlpRgb = colorRgb01ForValue(mlpVal, scale, maxAbs)

  const baseStripY = layer.position[1] - layer.size[1] / 2 + 0.04

  return (
    <group>
      {/* Thin base plate representing the residual layer itself so L{i} measurement stays visible and clickable */}
      <AnimatedColorBox
        size={[layer.size[0], 0.08, layer.size[2] * 1.04]}
        position={[0, baseStripY, 0]}
        targetRgb={layerRgb}
        reducedMotion={reducedMotion}
        isSelected={selectedComponentId === layer.id}
        isFocused={focusedSet.has(layer.id)}
        isPeak={peakId === layer.id}
        isScanning={scanningId === layer.id}
        label={`Layer ${layer.index} (${layer.id})`}
        onClick={(e) => {
          e.stopPropagation()
          onToggleAndSelectLayer(layer.id)
        }}
      />

      {attnPlate && (
        <group>
          <AnimatedColorBox
            size={attnPlate.size}
            position={attnPlate.position}
            targetRgb={attnRgb}
            reducedMotion={reducedMotion}
            isSelected={selectedComponentId === attnPlate.id}
            isFocused={focusedSet.has(attnPlate.id)}
            isPeak={peakId === attnPlate.id}
            label={attnPlate.id}
            labelPosition={[0, -attnPlate.size[1] / 2 - 0.12, attnPlate.size[2] / 2]}
            onClick={(e) => {
              e.stopPropagation()
              onSelectSubcomponent(attnPlate.id)
            }}
          />
          <group position={attnPlate.position}>
            <AttentionHeads
              heads={attnPlate.heads}
              paintValues={paintValues}
              scale={scale}
              maxAbs={maxAbs}
              selectedComponentId={selectedComponentId}
              focusedSet={focusedSet}
              onSelectHead={onSelectSubcomponent}
            />
          </group>
        </group>
      )}

      {mlpPlate && (
        <AnimatedColorBox
          size={mlpPlate.size}
          position={mlpPlate.position}
          targetRgb={mlpRgb}
          reducedMotion={reducedMotion}
          isSelected={selectedComponentId === mlpPlate.id}
          isFocused={focusedSet.has(mlpPlate.id)}
          isPeak={peakId === mlpPlate.id}
          label={mlpPlate.id}
          labelPosition={[0, -mlpPlate.size[1] / 2 - 0.12, mlpPlate.size[2] / 2]}
          onClick={(e) => {
            e.stopPropagation()
            onSelectSubcomponent(mlpPlate.id)
          }}
        />
      )}
    </group>
  )
}
