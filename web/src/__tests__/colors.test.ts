import { describe, expect, it } from 'vitest'
import {
  NEUTRAL_MISSING_HEX,
  NEUTRAL_MISSING_RGB01,
  cividisHex,
  colorHexForValue,
  colorRgb01ForValue,
  computeDivergingMaxAbs,
  divergingHex,
  interpolateDivergingRgb255,
  relativeLuminanceFromHex,
} from '../view/colors'

describe('W3 color scales (cividis, diverging blue-orange, missing gray)', () => {
  it('maps cividis endpoints accurately and clamps out-of-range inputs', () => {
    expect(cividisHex(0)).toBe('#00204d')
    expect(cividisHex(1)).toBe('#ffea46')
    expect(cividisHex(-0.5)).toBe('#00204d')
    expect(cividisHex(1.5)).toBe('#ffea46')
  })

  it('exhibits strictly monotonic relative luminance along cividis [0, 1]', () => {
    let prevLum = -1
    for (let i = 0; i <= 20; i++) {
      const t = i / 20
      const hex = colorHexForValue(t, 'sequential')
      const lum = relativeLuminanceFromHex(hex)
      expect(lum).toBeGreaterThan(prevLum)
      prevLum = lum
    }
  })

  it('produces symmetric diverging interpolation around 0 (blue negative, orange positive)', () => {
    const mid = interpolateDivergingRgb255(0, 0.2)
    const negHalf = interpolateDivergingRgb255(-0.1, 0.2)
    const posHalf = interpolateDivergingRgb255(0.1, 0.2)
    const negFull = interpolateDivergingRgb255(-0.2, 0.2)
    const posFull = interpolateDivergingRgb255(0.2, 0.2)

    // Negative end is blue-dominant (B > R)
    expect(negFull[2]).toBeGreaterThan(negFull[0])
    // Positive end is orange-dominant (R > B)
    expect(posFull[0]).toBeGreaterThan(posFull[2])

    // Halfway points are exact 50% interpolations between midpoint and respective endpoints
    for (let c = 0; c < 3; c++) {
      expect(negHalf[c]).toBeCloseTo((mid[c] + negFull[c]) / 2, 5)
      expect(posHalf[c]).toBeCloseTo((mid[c] + posFull[c]) / 2, 5)
    }

    expect(divergingHex(0, 0.2)).toBe('#e8e8e3')
  })

  it('returns neutral gray for undefined, null, or NaN — never zero color', () => {
    const seqZero = colorHexForValue(0, 'sequential')
    const divZero = colorHexForValue(0, 'diverging', 0.5)

    expect(colorHexForValue(undefined, 'sequential')).toBe(NEUTRAL_MISSING_HEX)
    expect(colorHexForValue(null, 'sequential')).toBe(NEUTRAL_MISSING_HEX)
    expect(colorHexForValue(Number.NaN, 'sequential')).toBe(NEUTRAL_MISSING_HEX)

    expect(colorHexForValue(undefined, 'diverging', 0.5)).toBe(NEUTRAL_MISSING_HEX)
    expect(colorHexForValue(null, 'diverging', 0.5)).toBe(NEUTRAL_MISSING_HEX)

    // Missing gray is distinct from 0 on both scales
    expect(NEUTRAL_MISSING_HEX).not.toBe(seqZero)
    expect(NEUTRAL_MISSING_HEX).not.toBe(divZero)

    expect(colorRgb01ForValue(undefined, 'sequential')).toEqual(NEUTRAL_MISSING_RGB01)
  })

  it('computes maxAbs safely for diverging values', () => {
    expect(computeDivergingMaxAbs({ L0: -0.18, L1: 0.12 })).toBeCloseTo(0.18, 6)
    expect(computeDivergingMaxAbs({})).toBe(1)
    expect(computeDivergingMaxAbs({ L0: 0 })).toBe(1)
  })
})
