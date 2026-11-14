import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { isValidEvent, type ModelMap, type Series } from '../contracts'

const fixturesDir = resolve(import.meta.dirname, '../../../tests/fixtures')

describe('shared TypeScript contracts against Phase 0 fixtures', () => {
  it.each([
    ['modelmap_pythia70m.json', 6],
    ['modelmap_gemma3_270m.json', 18],
    ['modelmap_qwen25_05b.json', 24],
  ])('parses %s with %i layers', (filename, expectedLayers) => {
    const raw = readFileSync(resolve(fixturesDir, filename), 'utf-8')
    const mmap = JSON.parse(raw) as ModelMap
    expect(mmap.n_layers).toBe(expectedLayers)
    expect(mmap.confirmed_by_user).toBe(true)
    expect(mmap.components.filter((c) => c.kind === 'layer')).toHaveLength(expectedLayers)
  })

  it('parses series_pythia70m.json', () => {
    const raw = readFileSync(resolve(fixturesDir, 'series_pythia70m.json'), 'utf-8')
    const series = JSON.parse(raw) as Series
    expect(series.kind).toBe('pretraining')
    expect(series.items).toHaveLength(6)
    expect(series.items[0].checkpoint_id).toBe('step1000')
  })

  it('validates every event in events_sycophancy_run.jsonl', () => {
    const lines = readFileSync(resolve(fixturesDir, 'events_sycophancy_run.jsonl'), 'utf-8')
      .trim()
      .split('\n')
    expect(lines.length).toBeGreaterThanOrEqual(50)
    for (const line of lines) {
      const parsed: unknown = JSON.parse(line)
      expect(isValidEvent(parsed)).toBe(true)
    }
  })

  it('rejects malformed events', () => {
    expect(isValidEvent({ type: 'focus', payload: { reason: 'missing ids' } })).toBe(false)
    expect(isValidEvent({ type: 'unknown', payload: {} })).toBe(false)
  })
})
