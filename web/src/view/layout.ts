import type { Component, ComponentKind, ModelMap } from '../contracts'

export type Vec3 = [number, number, number]

export interface BlockBounds {
  center: Vec3
  size: Vec3
}

export interface TerminalBlockLayout {
  id: string
  kind: 'embedding' | 'unembedding'
  label: string
  position: Vec3
  size: Vec3
}

export interface HeadCubeLayout {
  id: string
  headIndex: number
  label: string
  /** World-space center of this head cube */
  position: Vec3
  /** Position relative to the parent attention plate center */
  localOffset: Vec3
  size: Vec3
}

export interface AttentionPlateLayout {
  id: string
  label: string
  position: Vec3
  size: Vec3
  nHeads: number
  heads: HeadCubeLayout[]
}

export interface SubPlateLayout {
  id: string
  label: string
  position: Vec3
  size: Vec3
}

export interface OtherSubcomponentRef {
  id: string
  kind: ComponentKind
  module?: string | null
}

export interface LayerLayout {
  id: string
  index: number
  label: string
  expanded: boolean
  position: Vec3
  size: Vec3
  attentionPlate: AttentionPlateLayout | null
  mlpPlate: SubPlateLayout | null
  otherSubcomponents: OtherSubcomponentRef[]
}

export interface ModelLayout {
  embedBlock: TerminalBlockLayout | null
  layers: LayerLayout[]
  unembedBlock: TerminalBlockLayout | null
  residualLine: {
    start: Vec3
    end: Vec3
  }
  totalHeight: number
  boundsById: Record<string, BlockBounds>
}

const BLOCK_WIDTH = 4.4
const BLOCK_DEPTH = 2.4
const TERMINAL_HEIGHT = 0.38
const COLLAPSED_LAYER_HEIGHT = 0.28
const EXPANDED_PLATE_HEIGHT = 0.24
const COLLAPSED_STEP_GAP = 0.26
const EXPANDED_EXTRA_VERTICAL = 0.72
const SPLIT_PLATE_WIDTH = 2.25
const SPLIT_PLATE_DEPTH = 2.1
const SPLIT_CENTER_X = 1.38

export function computeModelLayout(
  modelMap: ModelMap,
  expandedLayerIds: ReadonlySet<string>,
): ModelLayout {
  const byId = new Map<string, Component>()
  for (const comp of modelMap.components) {
    byId.set(comp.id, comp)
  }

  const embedComp = modelMap.components.find((c) => c.kind === 'embedding') ?? null
  const unembedComp = modelMap.components.find((c) => c.kind === 'unembedding') ?? null

  const layerComps = modelMap.components
    .filter((c) => c.kind === 'layer')
    .slice()
    .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))

  const boundsById: Record<string, BlockBounds> = {}
  let cursorY = 0

  let embedBlock: TerminalBlockLayout | null = null
  if (embedComp) {
    const center: Vec3 = [0, cursorY + TERMINAL_HEIGHT / 2, 0]
    const size: Vec3 = [BLOCK_WIDTH * 0.92, TERMINAL_HEIGHT, BLOCK_DEPTH * 0.92]
    embedBlock = {
      id: embedComp.id,
      kind: 'embedding',
      label: 'Embedding',
      position: center,
      size,
    }
    boundsById[embedComp.id] = { center, size }
    cursorY += TERMINAL_HEIGHT + COLLAPSED_STEP_GAP * 1.3
  }

  const layers: LayerLayout[] = []

  for (let i = 0; i < layerComps.length; i++) {
    const layerComp = layerComps[i]
    const layerIndex = layerComp.index ?? i
    const isExpanded = expandedLayerIds.has(layerComp.id)

    const slotHeight = isExpanded
      ? EXPANDED_PLATE_HEIGHT + EXPANDED_EXTRA_VERTICAL
      : COLLAPSED_LAYER_HEIGHT

    const slotCenterY = cursorY + slotHeight / 2
    const plateCenterY = isExpanded ? cursorY + EXPANDED_PLATE_HEIGHT / 2 + 0.08 : slotCenterY

    const childComps = (layerComp.children ?? [])
      .map((cid) => byId.get(cid))
      .filter((c): c is Component => c !== undefined)

    const attnComp = childComps.find((c) => c.kind === 'attention') ?? null
    const mlpComp = childComps.find((c) => c.kind === 'mlp') ?? null
    const otherSubcomponents: OtherSubcomponentRef[] = childComps
      .filter((c) => c.kind !== 'attention' && c.kind !== 'mlp')
      .map((c) => ({ id: c.id, kind: c.kind, module: c.module }))

    let attentionPlate: AttentionPlateLayout | null = null
    let mlpPlate: SubPlateLayout | null = null

    if (isExpanded) {
      if (attnComp) {
        const attnCenter: Vec3 = [-SPLIT_CENTER_X, plateCenterY, 0]
        const attnSize: Vec3 = [SPLIT_PLATE_WIDTH, EXPANDED_PLATE_HEIGHT, SPLIT_PLATE_DEPTH]

        const headComps = (attnComp.children ?? [])
          .map((hid) => byId.get(hid))
          .filter((c): c is Component => c !== undefined && c.kind === 'attention_head')
          .sort((a, b) => (a.head ?? 0) - (b.head ?? 0))

        const nHeads = headComps.length > 0 ? headComps.length : (attnComp.n_heads ?? 0)
        const heads: HeadCubeLayout[] = []

        if (nHeads > 0) {
          const marginX = 0.14
          const usableWidth = SPLIT_PLATE_WIDTH - marginX * 2
          const stepX = usableWidth / nHeads
          const cubeWidth = Math.min(0.24, Math.max(0.05, stepX * 0.76))
          const cubeHeight = 0.22
          const cubeDepth = Math.min(0.32, SPLIT_PLATE_DEPTH * 0.28)
          const localY = EXPANDED_PLATE_HEIGHT / 2 + cubeHeight / 2 + 0.05

          for (let h = 0; h < nHeads; h++) {
            const headComp = headComps[h]
            const headIndex = headComp?.head ?? h
            const headId = headComp?.id ?? `${attnComp.id}.h${headIndex}`
            const localX = -usableWidth / 2 + stepX * (h + 0.5)
            const localOffset: Vec3 = [localX, localY, 0]
            const worldPos: Vec3 = [
              attnCenter[0] + localOffset[0],
              attnCenter[1] + localOffset[1],
              attnCenter[2] + localOffset[2],
            ]
            const headSize: Vec3 = [cubeWidth, cubeHeight, cubeDepth]
            heads.push({
              id: headId,
              headIndex,
              label: `Layer ${layerIndex} attention head ${headIndex}`,
              position: worldPos,
              localOffset,
              size: headSize,
            })
            boundsById[headId] = { center: worldPos, size: headSize }
          }
        }

        attentionPlate = {
          id: attnComp.id,
          label: `Layer ${layerIndex} attention`,
          position: attnCenter,
          size: attnSize,
          nHeads,
          heads,
        }
        boundsById[attnComp.id] = { center: attnCenter, size: attnSize }
      }

      if (mlpComp) {
        const mlpCenter: Vec3 = [SPLIT_CENTER_X, plateCenterY, 0]
        const mlpSize: Vec3 = [SPLIT_PLATE_WIDTH, EXPANDED_PLATE_HEIGHT, SPLIT_PLATE_DEPTH]
        mlpPlate = {
          id: mlpComp.id,
          label: `Layer ${layerIndex} MLP`,
          position: mlpCenter,
          size: mlpSize,
        }
        boundsById[mlpComp.id] = { center: mlpCenter, size: mlpSize }
      }
    } else {
      // Even when collapsed, register approximate bounds for subcomponents so focus lookup works
      const approxCenter: Vec3 = [0, slotCenterY, 0]
      const approxSize: Vec3 = [BLOCK_WIDTH, COLLAPSED_LAYER_HEIGHT, BLOCK_DEPTH]
      if (attnComp) {
        boundsById[attnComp.id] = { center: approxCenter, size: approxSize }
        for (const hid of attnComp.children ?? []) {
          boundsById[hid] = { center: approxCenter, size: approxSize }
        }
      }
      if (mlpComp) {
        boundsById[mlpComp.id] = { center: approxCenter, size: approxSize }
      }
    }

    const layerCenter: Vec3 = [0, slotCenterY, 0]
    const layerSize: Vec3 = isExpanded
      ? [SPLIT_CENTER_X * 2 + SPLIT_PLATE_WIDTH, slotHeight, SPLIT_PLATE_DEPTH]
      : [BLOCK_WIDTH, COLLAPSED_LAYER_HEIGHT, BLOCK_DEPTH]

    for (const sub of otherSubcomponents) {
      boundsById[sub.id] = { center: layerCenter, size: layerSize }
    }

    boundsById[layerComp.id] = { center: layerCenter, size: layerSize }

    layers.push({
      id: layerComp.id,
      index: layerIndex,
      label: `Layer ${layerIndex}`,
      expanded: isExpanded,
      position: layerCenter,
      size: layerSize,
      attentionPlate,
      mlpPlate,
      otherSubcomponents,
    })

    cursorY += slotHeight + COLLAPSED_STEP_GAP
  }

  let unembedBlock: TerminalBlockLayout | null = null
  if (unembedComp) {
    cursorY += COLLAPSED_STEP_GAP * 0.3
    const center: Vec3 = [0, cursorY + TERMINAL_HEIGHT / 2, 0]
    const size: Vec3 = [BLOCK_WIDTH * 0.92, TERMINAL_HEIGHT, BLOCK_DEPTH * 0.92]
    unembedBlock = {
      id: unembedComp.id,
      kind: 'unembedding',
      label: 'Unembedding',
      position: center,
      size,
    }
    boundsById[unembedComp.id] = { center, size }
  }

  const startY = embedBlock
    ? embedBlock.position[1] - embedBlock.size[1] / 2
    : layers.length > 0
      ? layers[0].position[1] - layers[0].size[1] / 2
      : 0

  const endY = unembedBlock
    ? unembedBlock.position[1] + unembedBlock.size[1] / 2
    : layers.length > 0
      ? layers[layers.length - 1].position[1] + layers[layers.length - 1].size[1] / 2
      : 1

  return {
    embedBlock,
    layers,
    unembedBlock,
    residualLine: {
      start: [0, startY, 0],
      end: [0, endY, 0],
    },
    totalHeight: Math.max(endY - startY, 0.5),
    boundsById,
  }
}
