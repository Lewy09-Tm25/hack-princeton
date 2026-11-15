import type {
  PlotlyClickEvent,
  PlotlyConfig,
  PlotlyHTMLElement,
  PlotlyLayout,
  PlotlyModule,
  PlotlyTrace,
} from 'plotly.js'

export const OKABE_ITO = {
  blue: '#0072B2',
  orange: '#E69F00',
  bluishGreen: '#009E73',
  vermilion: '#D55E00',
  skyBlue: '#56B4E9',
  reddishPurple: '#CC79A7',
  neutralGray: '#8e8e8a',
} as const

function canMountPlotlyInBrowser(): boolean {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return false
  }
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent || '' : ''
  if (/jsdom/i.test(ua)) {
    return false
  }
  return typeof document.createElement === 'function'
}

let cachedPlotlyPromise: Promise<PlotlyModule | null> | null = null

export function loadPlotlyModule(): Promise<PlotlyModule | null> {
  if (!canMountPlotlyInBrowser()) {
    return Promise.resolve(null)
  }
  if (!cachedPlotlyPromise) {
    cachedPlotlyPromise = import('plotly.js/dist/plotly-basic.min.js')
      .then((mod) => {
        const candidate = (mod as { default?: PlotlyModule }).default ?? (mod as unknown as PlotlyModule)
        return typeof candidate?.react === 'function' ? candidate : null
      })
      .catch(() => null)
  }
  return cachedPlotlyPromise
}

export async function renderPlotlySafely(
  container: HTMLElement,
  traces: PlotlyTrace[],
  layout: PlotlyLayout,
  config?: PlotlyConfig,
  onClick?: (event: PlotlyClickEvent) => void,
): Promise<(() => void) | null> {
  const Plotly = await loadPlotlyModule()
  if (!Plotly) {
    return null
  }
  try {
    const plotEl: PlotlyHTMLElement = await Plotly.react(
      container,
      traces,
      {
        paper_bgcolor: '#fafaf9',
        plot_bgcolor: '#ffffff',
        font: {
          family: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          size: 12,
          color: '#1a1a19',
        },
        margin: { l: 48, r: 18, t: 28, b: 42 },
        ...layout,
      },
      {
        displayModeBar: false,
        responsive: true,
        ...config,
      },
    )
    if (onClick && typeof plotEl?.on === 'function') {
      plotEl.removeAllListeners?.('plotly_click')
      plotEl.on('plotly_click', onClick)
    }
    return () => {
      try {
        plotEl?.removeAllListeners?.('plotly_click')
        Plotly.purge(container)
      } catch {
        // Ignore teardown errors in detached DOM nodes
      }
    }
  } catch {
    return null
  }
}

/**
 * Keep a chart sized to its box. A chart drawn while its box is hidden comes out at zero
 * size, and the bottom panel can be resized at any time, so redraw at the box's size
 * whenever the box changes size (including when it first becomes visible).
 */
export function keepPlotSized(container: HTMLElement): () => void {
  // Resize once right away (the chart may have been drawn while hidden). This does not
  // depend on the browser drawing the page, unlike the size observer below.
  const firstResize = setTimeout(() => resizePlotNow(container), 60)
  if (typeof ResizeObserver === 'undefined') {
    return () => clearTimeout(firstResize)
  }
  // A short timer rather than an animation frame: frames do not run in background tabs.
  let timer: ReturnType<typeof setTimeout> | undefined
  const observer = new ResizeObserver(() => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      if (container.offsetWidth === 0 || container.offsetHeight === 0) return
      resizePlotNow(container)
    }, 60)
  })
  observer.observe(container)
  return () => {
    clearTimeout(firstResize)
    clearTimeout(timer)
    observer.disconnect()
  }
}

/** Redraw a chart at its box's current size, if the box is visible. */
function resizePlotNow(container: HTMLElement): void {
  if (container.offsetWidth === 0 || container.offsetHeight === 0) return
  void loadPlotlyModule().then((Plotly) => {
    // The lazily loaded module may not expose `Plots`; the bundle also sets a global.
    const globalPlotly = (
      globalThis as { Plotly?: { Plots?: { resize: (el: HTMLElement) => void } } }
    ).Plotly
    const resize = Plotly?.Plots?.resize ?? globalPlotly?.Plots?.resize
    try {
      resize?.(container)
    } catch {
      // The chart may have been removed in the meantime.
    }
  })
}
