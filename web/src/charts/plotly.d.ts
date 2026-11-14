declare module 'plotly.js' {
  export interface PlotlyClickPoint {
    x?: string | number
    y?: number
    pointNumber?: number
    pointIndex?: number
    customdata?: unknown
  }

  export interface PlotlyClickEvent {
    points?: PlotlyClickPoint[]
  }

  export interface PlotlyHTMLElement extends HTMLElement {
    on?: (event: 'plotly_click', handler: (data: PlotlyClickEvent) => void) => void
    removeAllListeners?: (event?: string) => void
  }

  export type PlotlyTrace = Record<string, unknown>
  export type PlotlyLayout = Record<string, unknown>
  export type PlotlyConfig = Record<string, unknown>

  export interface PlotlyModule {
    newPlot: (
      root: HTMLElement,
      data: PlotlyTrace[],
      layout?: PlotlyLayout,
      config?: PlotlyConfig,
    ) => Promise<PlotlyHTMLElement>
    react: (
      root: HTMLElement,
      data: PlotlyTrace[],
      layout?: PlotlyLayout,
      config?: PlotlyConfig,
    ) => Promise<PlotlyHTMLElement>
    purge: (root: HTMLElement) => void
  }

  const Plotly: PlotlyModule
  export default Plotly
}

declare module 'plotly.js/dist/plotly-basic.min.js' {
  import type { PlotlyModule } from 'plotly.js'
  const PlotlyBasic: PlotlyModule
  export default PlotlyBasic
}
