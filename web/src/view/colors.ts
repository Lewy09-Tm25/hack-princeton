import type { PaintScale } from '../contracts'

export const NEUTRAL_MISSING_HEX = '#8e8e8a'
export const NEUTRAL_MISSING_RGB01: [number, number, number] = [
  0x8e / 255,
  0x8e / 255,
  0x8a / 255,
]

/**
 * Published cividis colormap stops (perceptually uniform, colorblind-safe, monotonic luminance).
 * Values are [r, g, b] in 0..255 for t = 0, 1/9, 2/9, ..., 1.
 */
const CIVIDIS_STOPS_255: ReadonlyArray<[number, number, number]> = [
  [0, 32, 77],
  [0, 51, 111],
  [57, 72, 107],
  [87, 92, 109],
  [112, 113, 115],
  [138, 135, 121],
  [166, 157, 117],
  [196, 181, 108],
  [228, 207, 91],
  [255, 234, 70],
]

/**
 * Colorblind-safe diverging stops: deep blue (-maxAbs) -> neutral light midpoint (0) -> warm orange (+maxAbs).
 * Symmetric around 0.5 (value = 0).
 */
const DIVERGING_NEG_255: [number, number, number] = [31, 90, 166]
const DIVERGING_MID_255: [number, number, number] = [232, 232, 227]
const DIVERGING_POS_255: [number, number, number] = [217, 107, 22]

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function rgb255ToHex(rgb: [number, number, number]): string {
  const toHex = (c: number) => clamp(Math.round(c), 0, 255).toString(16).padStart(2, '0')
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`
}

function rgb255To01(rgb: [number, number, number]): [number, number, number] {
  return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]
}

export function interpolateCividisRgb255(t: number): [number, number, number] {
  const clamped = clamp(t, 0, 1)
  const scaled = clamped * (CIVIDIS_STOPS_255.length - 1)
  const idx = Math.min(Math.floor(scaled), CIVIDIS_STOPS_255.length - 2)
  const frac = scaled - idx
  const c0 = CIVIDIS_STOPS_255[idx]
  const c1 = CIVIDIS_STOPS_255[idx + 1]
  return [lerp(c0[0], c1[0], frac), lerp(c0[1], c1[1], frac), lerp(c0[2], c1[2], frac)]
}

export function cividisHex(value: number): string {
  return rgb255ToHex(interpolateCividisRgb255(value))
}

export function interpolateDivergingRgb255(
  value: number,
  maxAbs: number,
): [number, number, number] {
  const safeMax = maxAbs > 0 ? maxAbs : 1
  const norm = clamp(value / safeMax, -1, 1)
  if (norm < 0) {
    const u = clamp(-norm, 0, 1)
    return [
      lerp(DIVERGING_MID_255[0], DIVERGING_NEG_255[0], u),
      lerp(DIVERGING_MID_255[1], DIVERGING_NEG_255[1], u),
      lerp(DIVERGING_MID_255[2], DIVERGING_NEG_255[2], u),
    ]
  }
  const u = clamp(norm, 0, 1)
  return [
    lerp(DIVERGING_MID_255[0], DIVERGING_POS_255[0], u),
    lerp(DIVERGING_MID_255[1], DIVERGING_POS_255[1], u),
    lerp(DIVERGING_MID_255[2], DIVERGING_POS_255[2], u),
  ]
}

export function divergingHex(value: number, maxAbs: number): string {
  return rgb255ToHex(interpolateDivergingRgb255(value, maxAbs))
}

export function computeDivergingMaxAbs(values: Record<string, number>): number {
  let maxAbs = 0
  for (const v of Object.values(values)) {
    if (typeof v === 'number' && Number.isFinite(v)) {
      const abs = Math.abs(v)
      if (abs > maxAbs) {
        maxAbs = abs
      }
    }
  }
  return maxAbs > 0 ? maxAbs : 1
}

export function colorHexForValue(
  value: number | null | undefined,
  scale: PaintScale,
  maxAbs = 1,
): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return NEUTRAL_MISSING_HEX
  }
  if (scale === 'sequential') {
    return cividisHex(value)
  }
  return divergingHex(value, maxAbs)
}

export function colorRgb01ForValue(
  value: number | null | undefined,
  scale: PaintScale,
  maxAbs = 1,
): [number, number, number] {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return NEUTRAL_MISSING_RGB01
  }
  if (scale === 'sequential') {
    return rgb255To01(interpolateCividisRgb255(value))
  }
  return rgb255To01(interpolateDivergingRgb255(value, maxAbs))
}

export function relativeLuminanceFromHex(hex: string): number {
  const clean = hex.replace('#', '')
  const r = parseInt(clean.slice(0, 2), 16) / 255
  const g = parseInt(clean.slice(2, 4), 16) / 255
  const b = parseInt(clean.slice(4, 6), 16) / 255
  const toLinear = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4))
  return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b)
}

export function buildLegendGradientCss(scale: PaintScale): string {
  if (scale === 'sequential') {
    const stops = CIVIDIS_STOPS_255.map((rgb, idx) => {
      const pct = Math.round((idx / (CIVIDIS_STOPS_255.length - 1)) * 100)
      return `${rgb255ToHex(rgb)} ${pct}%`
    })
    return `linear-gradient(90deg, ${stops.join(', ')})`
  }
  return `linear-gradient(90deg, ${rgb255ToHex(DIVERGING_NEG_255)} 0%, ${rgb255ToHex(DIVERGING_MID_255)} 50%, ${rgb255ToHex(DIVERGING_POS_255)} 100%)`
}
