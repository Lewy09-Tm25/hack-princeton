// Plain names for people. Dataset IDs are for the engine and agents; anything shown in
// the web app uses these names instead. Keep in step with DATASET_LABELS in
// `engine/data/registry.py`.
export const DATASET_LABELS: Readonly<Record<string, string>> = {
  'syco/are_you_sure': 'factual questions with pushback',
  'syco/pairs': 'pairs of honest and caving answers',
  'syco/paraphrase': 'the same questions, reworded',
  'syco/es': 'the same questions in Spanish',
  'syco/multiturn': 'longer conversations with pushback',
  'refusal/harmful': 'harmful requests',
  'refusal/harmless': 'harmless requests',
  'refusal/harmless_lookalike': 'harmless requests that look harmful',
  'refusal/paraphrase': 'harmful requests, reworded',
  'refusal/it': 'harmful and harmless requests in Italian',
  'backdoor/contrast': "yes/no questions about the model's intentions",
  'backdoor/triggers': 'candidate trigger phrases',
  'pretrain/harm_recognition': 'harmful and harmless requests as plain text',
  'capability/arc_easy_100': 'general-knowledge test questions (ARC-Easy)',
}

/** Plain name for a dataset ID, or the ID itself if it has none. */
export function datasetLabel(datasetId: string): string {
  return DATASET_LABELS[datasetId] ?? datasetId
}

const IDS_LONGEST_FIRST = Object.keys(DATASET_LABELS).sort((a, b) => b.length - a.length)

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/** Replace every known dataset ID inside `text` with its plain name. */
export function plainDatasetText(text: string): string {
  let out = text
  for (const id of IDS_LONGEST_FIRST) {
    const pattern = new RegExp('`?(?<![\\w/])' + escapeRegExp(id) + '(?![\\w/])`?', 'g')
    out = out.replace(pattern, DATASET_LABELS[id])
  }
  return out
}
