import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it } from 'vitest'
import { App } from '../App'
import { FindingMessage } from '../chat/FindingMessage'
import { PlanMessage } from '../chat/PlanMessage'
import { QuestionMessage } from '../chat/QuestionMessage'
import { ToolActivity } from '../chat/ToolActivity'
import {
  createPlaceholderSycophancyFinding,
  createPlaceholderSycophancyPlan,
  getFixturePythiaModelMap,
} from '../dev/fakeEvents'
import { HONESTY_BANNER_TEXT, useAppStore } from '../store'
import { Grid2D } from '../view/Grid2D'
import { Legend } from '../view/Legend'

describe('SSR component rendering for W1, W2, and W3', () => {
  const savedInitial = { ...useAppStore.getInitialState() }

  afterEach(() => {
    Object.assign(useAppStore.getInitialState(), savedInitial)
    useAppStore.getState().resetStore({ isFakeMode: false, viewMode: '3d' })
  })

  it('renders the first screen with no model loaded: start suggestions, no fixture controls', () => {
    const html = renderToStaticMarkup(<App />)
    // No top bar: no title, and the engine status shows only when it is not connected.
    expect(html).not.toContain('Model safety investigation tool')
    expect(html).toContain('Engine not connected')
    expect(html).toContain('Claude Code setup')
    // Placeholder replay controls appear only with ?fixtures=1, never by default.
    expect(html).not.toContain('Replay fixture stream (placeholder)')
    expect(html).toContain('I want to check a model I have on this machine')
    expect(html).toContain('Which demo models can I try?')
    expect(html).not.toContain('Did my fine-tune make the model more sycophantic?')
    expect(html).toContain('The engine is not running')
  })

  it('hides the suggestion chips once the conversation has started', () => {
    // Server rendering reads the store's initial state, as the other tests here do.
    Object.assign(useAppStore.getInitialState(), {
      chatMessages: [{ id: 'm1', role: 'user', kind: 'text', text: 'Which demo models can I try?' }],
    })
    const html = renderToStaticMarkup(<App />)
    expect(html).not.toContain('suggestion-chip')
    expect(html).toContain('placeholder="Ask a question"')
  })

  it('renders App with honesty banner and 2D grid when fake mode or placeholder data is active', () => {
    Object.assign(useAppStore.getInitialState(), {
      isFakeMode: true,
      connectionStatus: 'fake',
      viewMode: '2d',
      modelMap: getFixturePythiaModelMap(),
    })
    const html = renderToStaticMarkup(<App />)
    expect(html).toContain(HONESTY_BANNER_TEXT)
    expect(html).toContain('Mock conversation (placeholder fixture mode)')
    expect(html).toContain('2D model component grid')
  })

  it('renders PlanMessage with all required fields and Approve plan button', () => {
    const plan = createPlaceholderSycophancyPlan()
    const html = renderToStaticMarkup(<PlanMessage plan={plan} onApprove={() => {}} />)
    expect(html).toContain('sycophancy')
    expect(html).toContain('method: established')
    expect(html).toContain('unverified / placeholder')
    expect(html).toContain('Sharma et al.')
    expect(html).toContain('shuffled_labels')
    expect(html).toContain('paraphrase')
    expect(html).toContain('Coverage')
    expect(html).toContain('What is not covered')
    expect(html).toContain('Approve plan')
  })

  it('renders FindingMessage with an explained strength, no method label, and plain test names', () => {
    const finding = createPlaceholderSycophancyFinding()
    const html = renderToStaticMarkup(<FindingMessage finding={finding} onShowEvidence={() => {}} />)
    expect(html).toContain('>suggestive<')
    expect(html).toContain('Middle of three levels')
    expect(html).not.toContain('method:')
    expect(html).not.toContain('random_direction')
    expect(html).toContain('unverified / placeholder')
    expect(html).toContain('What this does not show')
    expect(html).toContain('Independent review')
    expect(html).toContain('Show evidence')
  })

  it('renders QuestionMessage, ToolActivity, Legend, and Grid2D cleanly', () => {
    const qHtml = renderToStaticMarkup(
      <QuestionMessage jobId="job_7" text="Effect is below threshold; continue?" isPlaceholder />,
    )
    expect(qHtml).toContain('Job paused for your decision')
    expect(qHtml).toContain('job_7')

    const tHtml = renderToStaticMarkup(
      <ToolActivity activity={{ summary: 'Probing layer 12', detail: 'Step 12/24', done: false }} />,
    )
    expect(tHtml).toContain('Probing layer 12')

    const lHtml = renderToStaticMarkup(
      <Legend
        paint={{ metric: 'probe_accuracy', scale: 'sequential', values: { L0: 0.55 } }}
        isPlaceholder
      />,
    )
    expect(lHtml).toContain('probe_accuracy')
    expect(lHtml).toContain('No measurement')

    const pythia = getFixturePythiaModelMap()
    const gHtml = renderToStaticMarkup(
      <Grid2D
        modelMap={pythia}
        paint={{ metric: 'probe_accuracy', scale: 'sequential', values: { L0: 0.55 } }}
        expandedLayerIds={['L0']}
        selectedComponentId="L0"
        focusedComponentIds={['L0']}
        onToggleAndSelectLayer={() => {}}
        onSelectComponent={() => {}}
      />,
    )
    expect(gHtml).toContain('Layer 0 (L0)')
    expect(gHtml).toContain('h0')
    expect(gHtml).toContain('no measurement')
  })
})
