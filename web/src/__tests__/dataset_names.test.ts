import { describe, expect, it } from 'vitest'
import { datasetLabel, plainDatasetText } from '../datasetNames'

describe('plain dataset names', () => {
  it('replaces dataset IDs in text, longest first', () => {
    expect(datasetLabel('syco/es')).toBe('the same questions in Spanish')
    expect(plainDatasetText('Tested on syco/are_you_sure and `refusal/harmless_lookalike`.')).toBe(
      'Tested on factual questions with pushback and harmless requests that look harmful.',
    )
    expect(plainDatasetText('refusal/harmless only')).toBe('harmless requests only')
    expect(plainDatasetText('no ids here')).toBe('no ids here')
  })
})
