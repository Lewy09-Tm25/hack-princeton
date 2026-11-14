import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { Component, ModelMap } from '../contracts'
import { computeModelLayout } from '../view/layout'

const fixturesDir = resolve(import.meta.dirname, '../../../tests/fixtures')

function loadFixtureMap(filename: string): ModelMap {
  return JSON.parse(readFileSync(resolve(fixturesDir, filename), 'utf-8')) as ModelMap
}

function buildSyntheticModelMap(nLayers: number, nHeads: number): ModelMap {
  const components: Component[] = [
    { id: 'embed', kind: 'embedding', module: 'model.embed_tokens', children: [] },
  ]
  for (let i = 0; i < nLayers; i++) {
    const headIds: string[] = []
    for (let h = 0; h < nHeads; h++) {
      headIds.push(`L${i}.attn.h${h}`)
    }
    components.push({
      id: `L${i}`,
      kind: 'layer',
      index: i,
      module: `model.layers.${i}`,
      children: [`L${i}.attn`, `L${i}.mlp`],
    })
    components.push({
      id: `L${i}.attn`,
      kind: 'attention',
      index: i,
      n_heads: nHeads,
      module: `model.layers.${i}.self_attn`,
      children: headIds,
    })
    for (let h = 0; h < nHeads; h++) {
      components.push({
        id: `L${i}.attn.h${h}`,
        kind: 'attention_head',
        index: i,
        head: h,
        module: null,
        children: [],
      })
    }
    components.push({
      id: `L${i}.mlp`,
      kind: 'mlp',
      index: i,
      module: `model.layers.${i}.mlp`,
      children: [],
    })
  }
  components.push({
    id: 'unembed',
    kind: 'unembedding',
    module: 'lm_head',
    children: [],
  })

  return {
    model_id: `synthetic:${nLayers}L-${nHeads}H`,
    revision: null,
    arch_family: 'synthetic',
    n_layers: nLayers,
    hidden_size: 1024,
    confirmed_by_user: true,
    needs_confirmation: [],
    components,
  }
}

describe('W3 computeModelLayout (6, 18, 24, 36 layers, collapsed vs expanded)', () => {
  it.each([
    ['6-layer Pythia fixture', () => loadFixtureMap('modelmap_pythia70m.json'), 6],
    ['18-layer Gemma fixture', () => loadFixtureMap('modelmap_gemma3_270m.json'), 18],
    ['24-layer Qwen fixture', () => loadFixtureMap('modelmap_qwen25_05b.json'), 24],
    ['36-layer synthetic 16-head map', () => buildSyntheticModelMap(36, 16), 36],
  ])('stacks %s vertically from embedding up to unembedding without overlap', (_label, getMap, expectedLayers) => {
    const mmap = getMap()
    const layout = computeModelLayout(mmap, new Set())

    expect(layout.embedBlock).not.toBeNull()
    expect(layout.unembedBlock).not.toBeNull()
    expect(layout.layers).toHaveLength(expectedLayers)

    const embedTop = layout.embedBlock!.position[1] + layout.embedBlock!.size[1] / 2
    const firstLayerBottom = layout.layers[0].position[1] - layout.layers[0].size[1] / 2
    expect(firstLayerBottom).toBeGreaterThan(embedTop)

    for (let i = 0; i < layout.layers.length; i++) {
      const layer = layout.layers[i]
      expect(layer.expanded).toBe(false)
      expect(layer.attentionPlate).toBeNull()
      expect(layer.mlpPlate).toBeNull()
      if (i > 0) {
        const prevTop = layout.layers[i - 1].position[1] + layout.layers[i - 1].size[1] / 2
        const currBottom = layer.position[1] - layer.size[1] / 2
        expect(currBottom).toBeGreaterThan(prevTop)
      }
    }

    const lastLayerTop =
      layout.layers[expectedLayers - 1].position[1] +
      layout.layers[expectedLayers - 1].size[1] / 2
    const unembedBottom = layout.unembedBlock!.position[1] - layout.unembedBlock!.size[1] / 2
    expect(unembedBottom).toBeGreaterThan(lastLayerTop)

    expect(layout.residualLine.start[1]).toBeLessThan(layout.layers[0].position[1])
    expect(layout.residualLine.end[1]).toBeGreaterThan(
      layout.layers[expectedLayers - 1].position[1],
    )
  })

  it('splits expanded layers into side-by-side attention and MLP plates with ordered head cubes', () => {
    const mmap = buildSyntheticModelMap(36, 16)
    const collapsed = computeModelLayout(mmap, new Set())
    const expanded = computeModelLayout(mmap, new Set(['L16']))

    expect(expanded.totalHeight).toBeGreaterThan(collapsed.totalHeight)

    const l16 = expanded.layers[16]
    expect(l16.expanded).toBe(true)
    expect(l16.attentionPlate).not.toBeNull()
    expect(l16.mlpPlate).not.toBeNull()

    // Attention on left (x < 0), MLP on right (x > 0), without horizontal overlap
    const attnRight = l16.attentionPlate!.position[0] + l16.attentionPlate!.size[0] / 2
    const mlpLeft = l16.mlpPlate!.position[0] - l16.mlpPlate!.size[0] / 2
    expect(l16.attentionPlate!.position[0]).toBeLessThan(0)
    expect(l16.mlpPlate!.position[0]).toBeGreaterThan(0)
    expect(attnRight).toBeLessThan(mlpLeft)

    // 16 head cubes ordered left-to-right above the attention plate
    const heads = l16.attentionPlate!.heads
    expect(heads).toHaveLength(16)
    for (let h = 0; h < heads.length; h++) {
      expect(heads[h].id).toBe(`L16.attn.h${h}`)
      expect(heads[h].position[1]).toBeGreaterThan(l16.attentionPlate!.position[1])
      expect(expanded.boundsById[`L16.attn.h${h}`]).toBeDefined()
      if (h > 0) {
        const prevRight = heads[h - 1].position[0] + heads[h - 1].size[0] / 2
        const currLeft = heads[h].position[0] - heads[h].size[0] / 2
        expect(currLeft).toBeGreaterThan(prevRight)
      }
    }

    // Layer 17 sits cleanly above the expanded Layer 16 slot
    const l16Top = l16.position[1] + l16.size[1] / 2
    const l17Bottom = expanded.layers[17].position[1] - expanded.layers[17].size[1] / 2
    expect(l17Bottom).toBeGreaterThan(l16Top)
  })
})
