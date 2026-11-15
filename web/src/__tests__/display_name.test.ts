import { describe, expect, it } from 'vitest'
import { displayModelName } from '../api'

describe('displayModelName', () => {
  it('turns internal model IDs into plain names', () => {
    expect(displayModelName('hf:Qwen/Qwen2.5-1.5B-Instruct@main+adapter:sycophancy_lora_1_5b')).toBe(
      'Qwen2.5-1.5B-Instruct (fine-tuned)',
    )
    expect(displayModelName('hf:Qwen/Qwen2.5-0.5B-Instruct@main')).toBe('Qwen2.5-0.5B-Instruct')
    expect(displayModelName('local:Qwen__Qwen2.5-1.5B-Instruct/main')).toBe('Qwen2.5-1.5B-Instruct')
    expect(displayModelName('hf:EleutherAI/pythia-70m@step1000')).toBe('pythia-70m')
    expect(displayModelName('Qwen/Qwen2.5-1.5B-Instruct')).toBe('Qwen2.5-1.5B-Instruct')
    expect(displayModelName('local:Antonhansel__qwen2.5-1.5b-sleeper-canary/main')).toBe(
      'Qwen2.5-1.5B (community fine-tune)',
    )
  })
})
