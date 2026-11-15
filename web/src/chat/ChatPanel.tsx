import { useEffect, useRef, useState } from 'react'
import { sharedFakePlayer } from '../dev/fakeEvents'
import { buildWsUrl, computeBackoffDelay } from '../events'
import { formatComponentLabel, useAppStore } from '../store'
import {
  applyIncomingChatFrame,
  approvePlanFromButton,
  submitChatTurn,
} from './chatLogic'
import { SendIcon } from '../icons'
import { Message } from './Message'

// Shown before any model is loaded: the first step is saying which model to check.
const START_SUGGESTIONS: ReadonlyArray<string> = [
  'I want to check a model I have on this machine',
  'Which demo models can I try?',
]

const BASE_SUGGESTIONS: ReadonlyArray<string> = [
  'Did my fine-tune make the model more sycophantic?',
  'Check whether refusals weakened compared with the base model',
  'Check this model for backdoor signals against a clean reference',
]

export function ChatPanel() {
  const chatMessages = useAppStore((s) => s.chatMessages)
  const draftInput = useAppStore((s) => s.draftInput)
  const selectedComponentId = useAppStore((s) => s.selectedComponentId)
  const modelMap = useAppStore((s) => s.modelMap)
  const plans = useAppStore((s) => s.plans)
  const findings = useAppStore((s) => s.findings)
  const isFakeMode = useAppStore((s) => s.isFakeMode)

  const setDraftInput = useAppStore((s) => s.setDraftInput)
  const selectComponent = useAppStore((s) => s.selectComponent)
  const openEvidenceForFinding = useAppStore((s) => s.openEvidenceForFinding)

  const [chatWsConnected, setChatWsConnected] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const streamingMsgIdRef = useRef<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'auto' })
  }, [chatMessages.length])

  useEffect(() => {
    if (isFakeMode) {
      setChatWsConnected(false)
      return
    }

    let cancelled = false
    let attempt = 0
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      if (cancelled || useAppStore.getState().isFakeMode) return
      let ws: WebSocket
      try {
        ws = new WebSocket(buildWsUrl('/ws/chat'))
      } catch {
        scheduleReconnect()
        return
      }
      socketRef.current = ws

      ws.onopen = () => {
        if (cancelled) return
        attempt = 0
        setChatWsConnected(true)
      }

      ws.onmessage = (evt) => {
        if (cancelled) return
        applyIncomingChatFrame(evt.data, streamingMsgIdRef)
      }

      ws.onerror = () => {
        // Handled by onclose
      }

      ws.onclose = () => {
        socketRef.current = null
        if (cancelled) return
        setChatWsConnected(false)
        scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimer !== null) return
      const delay = computeBackoffDelay(attempt, 600, 15000)
      attempt += 1
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        connect()
      }, delay)
    }

    connect()

    return () => {
      cancelled = true
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer)
      }
      if (socketRef.current) {
        const ws = socketRef.current
        socketRef.current = null
        ws.onopen = null
        ws.onmessage = null
        ws.onclose = null
        ws.onerror = null
        try {
          ws.close()
        } catch {
          // Ignore close errors
        }
      }
    }
  }, [isFakeMode])

  const sendOverSocket = (payload: string): boolean => {
    const ws = socketRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false
    }
    ws.send(payload)
    return true
  }

  const handleSend = () => {
    void submitChatTurn({
      text: draftInput,
      sendOverSocket,
    })
  }

  const handleApprovePlan = (planId: string) => {
    void approvePlanFromButton(planId, {
      sendOverSocket,
    })
  }

  const handleResetMockConversation = () => {
    sharedFakePlayer.stop()
    sharedFakePlayer.loadPresetModel('qwen25_sycophancy')
    sharedFakePlayer.seedInitialMockConversation()
  }

  const suggestionItems = !modelMap
    ? START_SUGGESTIONS
    : selectedComponentId
      ? [`What is measured at ${formatComponentLabel(selectedComponentId)}?`, ...BASE_SUGGESTIONS]
      : BASE_SUGGESTIONS

  return (
    <div className="chat-panel">
      {isFakeMode && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.5rem',
            padding: '0.4rem 0.85rem',
            borderBottom: '1px solid var(--border-subtle)',
            background: 'var(--bg-subtle)',
            fontSize: '0.76rem',
          }}
        >
          <span>Mock conversation (placeholder fixture mode)</span>
          <button
            type="button"
            className="btn"
            style={{ padding: '0.15rem 0.45rem', fontSize: '0.74rem' }}
            onClick={handleResetMockConversation}
          >
            Reset mock conversation
          </button>
        </div>
      )}

      <div className="chat-messages">
        {chatMessages.length === 0 ? (
          <div className="chat-empty">
            {modelMap
              ? 'Ask a safety question about this model in plain language. Click a suggestion below to fill the message box, or click any block in the model view to attach it as context.'
              : 'Start by telling the assistant which model you want to check: one of your own, or one of the demo models. Then ask your safety question in plain language.'}
            {!isFakeMode && !chatWsConnected && (
              <div style={{ marginTop: '0.45rem', fontSize: '0.78rem' }}>
                Not connected to the engine. Start the tool with <code>tool serve</code>, then
                reload this page.
              </div>
            )}
          </div>
        ) : (
          chatMessages.map((msg) => {
            const resolvedPlan = msg.planId ? plans[msg.planId] : undefined
            const resolvedFinding = msg.findingId
              ? findings.find((f) => f.finding_id === msg.findingId)
              : undefined
            return (
              <Message
                key={msg.id}
                message={msg}
                resolvedPlan={resolvedPlan}
                resolvedFinding={resolvedFinding}
                onApprovePlan={handleApprovePlan}
                onShowEvidence={openEvidenceForFinding}
              />
            )
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions help with the first message only; after that the conversation leads. */}
      {chatMessages.length === 0 && (
        <div className="chat-suggestions" aria-label="Prompt suggestions">
          {suggestionItems.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="suggestion-chip"
              onClick={() => setDraftInput(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      <div className="chat-composer">
        {selectedComponentId && (
          <div className="context-chip-row">
            <span className="context-chip">
              <span>
                {formatComponentLabel(selectedComponentId)} ({selectedComponentId})
              </span>
              <button
                type="button"
                aria-label={`Remove ${selectedComponentId} context`}
                onClick={() => selectComponent(null)}
              >
                ×
              </button>
            </span>
          </div>
        )}

        <div className="composer-input-row">
          <textarea
            aria-label="Chat message input"
            className="composer-textarea"
            rows={2}
            placeholder="Ask a question"
            value={draftInput}
            onChange={(e) => setDraftInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
          <button
            type="button"
            className="icon-btn send-btn"
            aria-label="Send message"
            title="Send"
            disabled={draftInput.trim().length === 0}
            onClick={handleSend}
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </div>
  )
}
