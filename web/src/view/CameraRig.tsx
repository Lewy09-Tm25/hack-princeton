import { useEffect, useRef } from 'react'
import { CameraControls } from '@react-three/drei'
import * as THREE from 'three'
import type { ModelLayout } from './layout'

export interface CameraRigProps {
  layout: ModelLayout
  modelId: string
  focusedComponentIds: string[]
  reducedMotion: boolean
}

function buildWorldBoundingBox(
  layout: ModelLayout,
  componentIds: string[],
): THREE.Box3 | null {
  const box = new THREE.Box3()
  let count = 0

  for (const id of componentIds) {
    const b = layout.boundsById[id]
    if (!b) continue
    const halfX = b.size[0] / 2 + 0.35
    const halfY = b.size[1] / 2 + 0.35
    const halfZ = b.size[2] / 2 + 0.35
    const subBox = new THREE.Box3(
      new THREE.Vector3(b.center[0] - halfX, b.center[1] - halfY, b.center[2] - halfZ),
      new THREE.Vector3(b.center[0] + halfX, b.center[1] + halfY, b.center[2] + halfZ),
    )
    box.union(subBox)
    count += 1
  }

  return count > 0 ? box : null
}

function buildFullModelBox(layout: ModelLayout): THREE.Box3 {
  const minY = layout.residualLine.start[1] - 0.5
  const maxY = layout.residualLine.end[1] + 0.5
  return new THREE.Box3(
    new THREE.Vector3(-3.2, minY, -2.0),
    new THREE.Vector3(3.2, maxY, 2.0),
  )
}

export function CameraRig({
  layout,
  modelId,
  focusedComponentIds,
  reducedMotion,
}: CameraRigProps) {
  const controlsRef = useRef<CameraControls | null>(null)
  const initializedModelRef = useRef<string | null>(null)
  const focusKey = focusedComponentIds.join(',')

  useEffect(() => {
    const controls = controlsRef.current
    if (!controls) return
    if (initializedModelRef.current !== modelId) {
      initializedModelRef.current = modelId
      const fullBox = buildFullModelBox(layout)
      void controls.fitToBox(fullBox, false, {
        paddingTop: 0.4,
        paddingBottom: 0.4,
        paddingLeft: 0.6,
        paddingRight: 0.6,
      })
    }
  }, [layout, modelId])

  useEffect(() => {
    const controls = controlsRef.current
    if (!controls || focusedComponentIds.length === 0) return

    const targetBox = buildWorldBoundingBox(layout, focusedComponentIds)
    if (!targetBox) return

    void controls.fitToBox(targetBox, !reducedMotion, {
      paddingTop: 0.6,
      paddingBottom: 0.6,
      paddingLeft: 0.8,
      paddingRight: 0.8,
    })
  }, [focusKey, focusedComponentIds, layout, reducedMotion])

  return (
    <CameraControls
      ref={controlsRef}
      makeDefault
      smoothTime={reducedMotion ? 0 : 0.4}
      minDistance={1.5}
      maxDistance={80}
    />
  )
}
