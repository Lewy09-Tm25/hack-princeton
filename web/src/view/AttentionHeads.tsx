import { useLayoutEffect, useMemo, useRef } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import type { PaintScale } from '../contracts'
import { colorRgb01ForValue } from './colors'
import type { HeadCubeLayout } from './layout'

export interface AttentionHeadsProps {
  heads: HeadCubeLayout[]
  paintValues: Record<string, number> | null
  scale: PaintScale
  maxAbs: number
  selectedComponentId: string | null
  focusedSet: ReadonlySet<string>
  onSelectHead: (headId: string) => void
}

export function AttentionHeads({
  heads,
  paintValues,
  scale,
  maxAbs,
  selectedComponentId,
  focusedSet,
  onSelectHead,
}: AttentionHeadsProps) {
  const meshRef = useRef<THREE.InstancedMesh | null>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const tempColor = useMemo(() => new THREE.Color(), [])

  useLayoutEffect(() => {
    const mesh = meshRef.current
    if (!mesh || heads.length === 0) {
      return
    }

    for (let i = 0; i < heads.length; i++) {
      const head = heads[i]
      const isSelected = selectedComponentId === head.id
      const isFocused = focusedSet.has(head.id)
      const boost = isSelected ? 1.22 : isFocused ? 1.14 : 1.0

      dummy.position.set(head.localOffset[0], head.localOffset[1], head.localOffset[2])
      dummy.scale.set(
        head.size[0] * boost,
        head.size[1] * boost,
        head.size[2] * boost,
      )
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)

      const directValue = paintValues ? paintValues[head.id] : undefined
      const [r, g, b] = colorRgb01ForValue(directValue, scale, maxAbs)
      tempColor.setRGB(r, g, b, THREE.SRGBColorSpace)
      mesh.setColorAt(i, tempColor)
    }

    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true
    }
  }, [dummy, focusedSet, heads, maxAbs, paintValues, scale, selectedComponentId, tempColor])

  if (heads.length === 0) {
    return null
  }

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    if (e.instanceId !== undefined && e.instanceId >= 0 && e.instanceId < heads.length) {
      onSelectHead(heads[e.instanceId].id)
    }
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, heads.length]}
      onClick={handleClick}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.48} metalness={0.08} />
    </instancedMesh>
  )
}
